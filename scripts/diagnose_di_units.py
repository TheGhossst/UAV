"""Diagnostic: D_i = 8 S_i / r  vs  D_i = S_i / r. Does not change production code.

Temporarily patches ``src.aodt`` and the copy imported by ``src.repair`` so the
bandwidth LP and evaluator see the same units. Restores both on exit.

    python -m scripts.diagnose_di_units
"""

from __future__ import annotations

import json
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import src.aodt as aodt_mod
import src.repair as repair_mod
from src.aodt import queueing_term
from src.config import DEFAULT, PAPER_SCENARIO_SEEDS
from src.experiments.aodt_parameter_search import (
    FIG9_THRESHOLDS,
    PAPER_FIG9_MBPS,
    SearchSpec,
    _write_csv,
    run_pairs,
    score_pair,
)
from src.logutil import banner, configure_logging, log

SI = 12000.0
L = 3.75e6
OUT_DEFAULT = Path("results") / "aodt_parameter_search" / "di_units_diagnostic"

VARIANTS = (
    {"name": "bytes_to_bits", "bits_per_byte": 8.0, "formula": "8*S_i/r_ij", "label": "current (dimensionally correct)"},
    {"name": "paper_as_written", "bits_per_byte": 1.0, "formula": "S_i/r_ij", "label": "paper-implementation hypothesis"},
)


def _upload_times_with_factor(bits_per_byte: float):
    def upload_times(association, processing, rates, cfg):
        if cfg.task_size_bytes is None:
            raise ValueError("task_size_bytes is required for AoDT")
        i_idx = np.arange(association.shape[0])
        j_assoc = association.argmax(axis=1)
        r_assoc = np.maximum(rates[i_idx, j_assoc], 1e-12)
        s_num = cfg.task_size_bytes * float(bits_per_byte)
        d = s_num / r_assoc
        same = processing[i_idx, j_assoc] > 0.5
        d = np.where(same, d, d + cfg.t_u2u)
        return d

    return upload_times


def _delay_floors_with_factor(bits_per_byte: float):
    def delay_rate_floors(scenario, association, processing, mu):
        cfg = scenario.cfg
        i = association.shape[0]
        floors = np.zeros(i)
        if cfg.task_size_bytes is None:
            return floors
        s_num = cfg.task_size_bytes * float(bits_per_byte)
        i_idx = np.arange(i)
        j_assoc = association.argmax(axis=1)
        forwarded = processing[i_idx, j_assoc] < 0.5
        for members in scenario.groups:
            if members.size == 0:
                continue
            slack = cfg.aodt_threshold - queueing_term(scenario, members, mu)
            for idx in members:
                budget = slack - (cfg.t_u2u if forwarded[idx] else 0.0)
                if budget <= 1e-12:
                    floors[idx] = 1e18
                else:
                    floors[idx] = s_num / budget
        return floors

    return delay_rate_floors


@contextmanager
def di_bits_per_byte(factor: float):
    """Patch upload delay and AoDT floors, then restore production functions."""
    orig_ut = aodt_mod.upload_times
    orig_df = aodt_mod.delay_rate_floors
    orig_repair_df = repair_mod.delay_rate_floors
    ut = _upload_times_with_factor(factor)
    df = _delay_floors_with_factor(factor)
    aodt_mod.upload_times = ut
    aodt_mod.delay_rate_floors = df
    repair_mod.delay_rate_floors = df
    try:
        yield
    finally:
        aodt_mod.upload_times = orig_ut
        aodt_mod.delay_rate_floors = orig_df
        repair_mod.delay_rate_floors = orig_repair_df


def _mbps(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:.3f}"


def _plot(out_dir: Path, by_variant: dict[str, dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tks = list(FIG9_THRESHOLDS)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    paper_x = sorted(PAPER_FIG9_MBPS)
    ax.plot(paper_x, [PAPER_FIG9_MBPS[t]["sca"] for t in paper_x], "k--", marker="x", label="paper SCA (approx.)")
    styles = {"bytes_to_bits": ("C0", "o"), "paper_as_written": ("C1", "s")}
    for name, rec in by_variant.items():
        color, marker = styles.get(name, ("C2", "d"))
        ys = [rec.get(f"sca_mbps_{tk:g}") for tk in tks]
        if any(v is None or (isinstance(v, float) and v != v) for v in ys):
            continue
        ax.plot(tks, ys, color=color, marker=marker, label=f"SCA {VARIANTS[0]['formula'] if name == 'bytes_to_bits' else VARIANTS[1]['formula']}")
    ax.set_xlabel("AoDT threshold T_k (s)")
    ax.set_ylabel("SCA sum rate (Mbps)")
    ax.set_title(r"Diagnostic: $D_i=8S_i/r$ vs $S_i/r$  ($S_i=12000$, $L=3.75\times10^6$)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_di_units.png", dpi=120)
    plt.close(fig)


def write_diagnostic_report(out_dir: Path, by_variant: dict[str, dict], n_seeds: int) -> None:
    cur = by_variant["bytes_to_bits"]
    hyp = by_variant["paper_as_written"]
    feas08_cur = float(cur.get("feas_sca_0.8") or 0.0)
    feas08_hyp = float(hyp.get("feas_sca_0.8") or 0.0)
    late_cur = float(cur.get("late_frac") or 0.0)
    late_hyp = float(hyp.get("late_frac") or 0.0)
    mae_cur = float(cur.get("mae_sca_fig9") or float("nan"))
    mae_hyp = float(hyp.get("mae_sca_fig9") or float("nan"))

    made_feasible = feas08_hyp > feas08_cur + 0.25
    closer = (mae_hyp == mae_hyp) and (mae_cur == mae_cur) and (mae_hyp + 0.15 < mae_cur)
    stronger_slope = late_hyp > late_cur + 0.1
    verdict_strong = made_feasible and closer and (late_hyp >= 0.25 or stronger_slope)
    if verdict_strong:
        verdict = (
            "The paper-as-written units (D_i = S_i / r_ij) make T_k = 0.8 s substantially more feasible "
            "and pull the SCA Fig. 9 curve closer to the paper. That is a strong candidate for the missing implementation assumption."
        )
    elif made_feasible and not closer:
        verdict = (
            "The paper-as-written units make T_k = 0.8 s more feasible, but the Fig. 9 *shape* does not get closer to the paper "
            "(typically the loose-T_k end flattens and SCA at 3 s jumps toward the unconstrained ~7 Mbps). "
            "Units-as-written are therefore only a partial explanation."
        )
    elif not made_feasible:
        verdict = (
            "The paper-as-written units do **not** make T_k = 0.8 s feasible at this (S_i, L). "
            "The missing-×8 hypothesis is not sufficient on its own."
        )
    else:
        verdict = "See the tables; neither variant is a clear Fig. 9 match."

    lines = [
        "# Diagnostic: D_i units (8 S_i / r vs S_i / r)",
        "",
        "Production `src/aodt.py` was **not** changed. This run monkey-patches upload delay and AoDT floors, then restores them.",
        "",
        f"- S_i = {SI:.0f} bytes, L = {L:.4g} cycles (20-seed search winner).",
        f"- Seeds {PAPER_SCENARIO_SEEDS[0]}–{PAPER_SCENARIO_SEEDS[-1]} ({n_seeds}).",
        f"- T_k ∈ {', '.join(str(t) for t in FIG9_THRESHOLDS)} s. Methods: random, kmeans, SCA. No TD3.",
        "- Radio unchanged (calibrated, B_sys = 8.8e6 Hz).",
        "",
        "## Variants",
        "",
        "| Variant | Formula | bits/byte |",
        "|---|---|---:|",
        "| current / dimensionally correct | $D_i = 8 S_i / r_{ij}$ | 8 |",
        "| paper-implementation hypothesis | $D_i = S_i / r_{ij}$ | 1 |",
        "",
        "## SCA Fig. 9",
        "",
        "| T_k (s) | Current 8S/r | Paper S/r | Paper figure | Current feas | Paper-units feas |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for tk in FIG9_THRESHOLDS:
        paper = PAPER_FIG9_MBPS.get(tk, {}).get("sca")
        paper_s = f"{paper:.1f}" if paper is not None else "—"
        lines.append(
            f"| {tk:g} | {_mbps(cur.get(f'sca_mbps_{tk:g}'))} | {_mbps(hyp.get(f'sca_mbps_{tk:g}'))} | {paper_s} | "
            f"{_mbps(cur.get(f'sca_feas_{tk:g}'))} | {_mbps(hyp.get(f'sca_feas_{tk:g}'))} |"
        )
    lines.extend(
        [
            "",
            "## Snapshot",
            "",
            "| Metric | Current 8S/r | Paper S/r | Paper |",
            "|---|---:|---:|---:|",
            f"| SCA T_k=0.8 (Mbps) | {_mbps(cur.get('sca_mbps_0.8'))} | {_mbps(hyp.get('sca_mbps_0.8'))} | ~4.0 |",
            f"| SCA T_k=2.0 (Mbps) | {_mbps(cur.get('sca_mbps_2'))} | {_mbps(hyp.get('sca_mbps_2'))} | ~6.3 |",
            f"| SCA T_k=3.0 (Mbps) | {_mbps(cur.get('sca_mbps_3'))} | {_mbps(hyp.get('sca_mbps_3'))} | ~7.4 |",
            f"| SCA feasible frac at 0.8 s | {_mbps(cur.get('feas_sca_0.8'))} | {_mbps(hyp.get('feas_sca_0.8'))} | drawn as a curve |",
            f"| SCA feasible frac at 3.0 s | {_mbps(cur.get('feas_sca_3.0'))} | {_mbps(hyp.get('feas_sca_3.0'))} | |",
            f"| Rise 0.8→3.0 (Mbps) | {_mbps(cur.get('rise_08_to_30'))} | {_mbps(hyp.get('rise_08_to_30'))} | ~3.4 |",
            f"| Late fraction after 1.6 s | {_mbps(cur.get('late_frac'))} | {_mbps(hyp.get('late_frac'))} | ~0.5 |",
            f"| K-means at 3 s | {_mbps(cur.get('kmeans_mbps_3'))} | {_mbps(hyp.get('kmeans_mbps_3'))} | ~4.4 |",
            f"| Random at 3 s | {_mbps(cur.get('random_mbps_3'))} | {_mbps(hyp.get('random_mbps_3'))} | ~3.2 |",
            f"| SCA Fig. 9 MAE (Mbps) | {_mbps(mae_cur)} | {_mbps(mae_hyp)} | 0 |",
            "",
            "## Verdict",
            "",
            verdict,
            "",
            "Production delay remains `D_i = 8 S_i / r_ij`. This folder is diagnostic only.",
            "",
        ]
    )
    (out_dir / "DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")


def run(out_dir: Path | None = None, seeds: tuple[int, ...] | None = None) -> dict:
    out_dir = Path(out_dir or OUT_DEFAULT)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or PAPER_SCENARIO_SEEDS
    spec = SearchSpec(
        task_sizes=(SI,),
        task_cycles=(L,),
        thresholds=FIG9_THRESHOLDS,
        seeds=seeds,
        methods=("random", "kmeans", "sca"),
        skip_td3=True,
    )
    args = Namespace(particles=20, iters=100, td3_steps=1, with_td3=False)
    cfg = DEFAULT
    if abs(cfg.b_sys - 8.8e6) > 1.0:
        raise SystemExit("diagnostic must keep the calibrated radio")

    banner("D_i units diagnostic (production aodt.py unchanged)")
    log.info("  S_i=%g L=%g seeds=%s–%s", SI, L, seeds[0], seeds[-1])
    all_raw = []
    by_variant: dict[str, dict] = {}
    for v in VARIANTS:
        banner(f"{v['label']}: D_i = {v['formula']}")
        with di_bits_per_byte(v["bits_per_byte"]):
            rows = run_pairs(cfg, spec, [(SI, L)], seeds, args, f"D_i={v['formula']}")
        for r in rows:
            r["variant"] = v["name"]
            r["bits_per_byte"] = v["bits_per_byte"]
            r["formula"] = v["formula"]
        all_raw.extend(rows)
        rec = score_pair(rows)
        rec["variant"] = v["name"]
        rec["bits_per_byte"] = v["bits_per_byte"]
        rec["formula"] = v["formula"]
        by_variant[v["name"]] = rec
        log.info(
            "  SCA 0.8=%.3f feas=%.2f  SCA 3.0=%.3f feas=%.2f  rise=%.3f late=%.3f",
            rec.get("sca_mbps_0.8") or float("nan"),
            rec.get("feas_sca_0.8") or float("nan"),
            rec.get("sca_mbps_3") or float("nan"),
            rec.get("feas_sca_3.0") or float("nan"),
            rec.get("rise_08_to_30") or float("nan"),
            rec.get("late_frac") or float("nan"),
        )

    _write_csv(out_dir / "raw.csv", all_raw)
    _write_csv(out_dir / "summary.csv", list(by_variant.values()))
    _plot(out_dir, by_variant)
    write_diagnostic_report(out_dir, by_variant, len(seeds))
    meta = {
        "diagnostic_only": True,
        "production_unchanged": "src/aodt.py still uses 8 * S_i / r",
        "task_size_bytes": SI,
        "task_cycles": L,
        "seeds": list(seeds),
        "variants": VARIANTS,
        "summary": {k: {kk: vv for kk, vv in rec.items() if not str(kk).startswith("err_")} for k, rec in by_variant.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("wrote D_i diagnostic under %s", out_dir)
    return {"out": str(out_dir), "by_variant": by_variant}


if __name__ == "__main__":
    configure_logging("INFO")
    run()
