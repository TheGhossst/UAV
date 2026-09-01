"""Replay leftover policies A/B/C on frozen Fig. 9 SCA (and K-means) placements.

Does not re-run solvers and does not change production defaults.
Association and processing are recovered from the stored UAV coordinates
(nearest association + process-consistent processing + stabilize). Bandwidth
is the only variable.

    python scripts/run_fig9_abc_leftover_replay.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT
from src.evaluator import evaluate
from src.experiments.sweeps import _summarize, _write_csv
from src.logutil import Counter, banner, configure_logging, log
from src.repair import (
    allocate_constrained_bandwidth,
    link_bandwidth_cap,
    nearest_association,
    process_consistent_processing,
    stabilize_processing,
)
from src.scenario import generate_scenario

SRC_RAW = Path("results") / "run_20260831" / "fig9_final_validation" / "raw_sumrate_vs_aodt.csv"
OUT_DEFAULT = Path("results") / "run_20260831" / "fig9_abc_leftover_replay"

TASK_SIZE_BYTES = 12000.0
TASK_CYCLES = 3.75e6
THRESHOLDS = (0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0)
USABLE = (1.2, 1.6, 2.0, 2.4, 2.8, 3.0)
ABS_ERR_LIMIT = 0.5
PAPER_SCA = {
    0.8: 4.0,
    1.2: 5.0,
    1.6: 5.7,
    2.0: 6.3,
    2.4: 6.8,
    2.8: 7.2,
    3.0: 7.4,
}

# A/B/C as specified. Placement is frozen; only leftover + cap change.
POLICIES = (
    ("A", True, 0.25),   # AoDT-first, 25% cap (production leftover)
    ("B", False, 0.25),  # SE-first, 25% cap
    ("C", False, None),  # SE-first, no cap
)


def _parse_xy(blob: str) -> np.ndarray:
    coords = json.loads(blob)
    return np.asarray([[p[0], p[1]] for p in coords], dtype=float)


def _load_placements(path: Path, method: str) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["method"] != method:
            continue
        out.append(
            {
                "seed": int(r["seed"]),
                "aodt_threshold": float(r["aodt_threshold"]),
                "csv_sum_rate": float(r["sum_rate"]),
                "csv_feasible": int(float(r["feasible"])),
                "csv_aodt_mean": None
                if r.get("aodt_mean") in ("", None, "None")
                else float(r["aodt_mean"]),
                "xy": _parse_xy(r["uav_xy"]),
            }
        )
    return out


def _frozen_binaries(scenario, xy: np.ndarray):
    a = nearest_association(scenario.iot_xy, xy)
    proc = process_consistent_processing(scenario, a)
    proc = stabilize_processing(scenario, proc)
    return a, proc


def _bw_stats(bw: np.ndarray, a: np.ndarray, cap: float, pool: float) -> dict:
    mask = a > 0.5
    b = bw[mask]
    used = float(b.sum())
    return {
        "used_hz": used,
        "unused_hz": float(pool - used),
        "min_hz": float(b.min()) if b.size else 0.0,
        "max_hz": float(b.max()) if b.size else 0.0,
        "mean_hz": float(b.mean()) if b.size else 0.0,
        "n_at_cap": int(np.sum(b >= cap - 1.0)) if np.isfinite(cap) else 0,
        "n_links": int(mask.sum()),
    }


def _cfg(tk: float, max_bw_share):
    return replace(
        DEFAULT.with_compute(task_size_bytes=TASK_SIZE_BYTES, task_cycles=TASK_CYCLES),
        aodt_threshold=float(tk),
        max_bw_share=max_bw_share,
    )


def _mbps_at(summary: list[dict], method: str, tk: float) -> float:
    for r in summary:
        if r["method"] == method and abs(float(r["aodt_threshold"]) - tk) < 1e-9:
            return float(r["sum_rate_mean"]) / 1e6
    return float("nan")


def _field_at(summary: list[dict], method: str, tk: float, field: str) -> float:
    for r in summary:
        if r["method"] == method and abs(float(r["aodt_threshold"]) - tk) < 1e-9:
            v = r.get(field)
            return float(v) if v is not None else float("nan")
    return float("nan")


def plot_curves(out_dir: Path, summary: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib missing; skip plot")
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    xs = list(THRESHOLDS)
    ax.plot(xs, [PAPER_SCA[t] for t in xs], "k--", marker="x", label="paper SCA")
    styles = {
        "sca_A": ("#1f77b4", "-", "*"),
        "sca_B": ("#2ca02c", "-", "s"),
        "sca_C": ("#d62728", "-", "D"),
        "kmeans_A": ("#1f77b4", ":", "o"),
        "kmeans_B": ("#2ca02c", ":", "^"),
        "kmeans_C": ("#d62728", ":", "v"),
    }
    for method, (color, ls, mk) in styles.items():
        sub = [r for r in summary if r["method"] == method]
        if not sub:
            continue
        order = np.argsort([float(r["aodt_threshold"]) for r in sub])
        ax.plot(
            np.array([float(sub[i]["aodt_threshold"]) for i in order]),
            np.array([float(sub[i]["sum_rate_mean"]) / 1e6 for i in order]),
            color=color,
            ls=ls,
            marker=mk,
            label=method.replace("sca_", "SCA ").replace("kmeans_", "K-means "),
        )
    ax.axvline(0.8, color="0.5", ls=":", lw=1)
    ax.set_xlabel("AoDT threshold T_k (s)")
    ax.set_ylabel("Sum rate (Mbps)")
    ax.set_title("Fig. 9 leftover A/B/C replay (frozen placements)")
    ax.set_xlim(0.6, 3.2)
    ax.set_ylim(0.0, 9.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_abc_vs_paper.png", dpi=140)
    plt.close(fig)


def write_report(
    out_dir: Path,
    summary: list[dict],
    sanity: dict,
    meta: dict,
) -> None:
    def fmt(x: float, d: int = 3) -> str:
        if x != x:
            return "—"
        return f"{x:.{d}f}"

    sca_mae = {}
    sca_n_within = {}
    for pol in ("A", "B", "C"):
        errs = []
        n = 0
        for tk in USABLE:
            impl = _mbps_at(summary, f"sca_{pol}", tk)
            err = abs(impl - PAPER_SCA[tk])
            errs.append(err)
            n += int(err <= ABS_ERR_LIMIT)
        sca_mae[pol] = float(np.mean(errs))
        sca_n_within[pol] = n

    lines = [
        "# Fig. 9 leftover A/B/C replay",
        "",
        "Frozen SCA (and K-means) placements from the 20-seed Fig. 9 validation.",
        "Solvers were **not** re-run. Production leftover defaults were **not** changed.",
        "Association / processing recovered from stored \(xy\) (nearest + process-consistent).",
        "",
        "## Policies",
        "",
        "| Version | Leftover | Cap |",
        "|---|---|---|",
        "| A | AoDT-first (`aodt_first=True`) | 25% |",
        "| B | spectral-efficiency-first (`aodt_first=False`) | 25% |",
        "| C | spectral-efficiency-first | none |",
        "",
        f"- Source placements: `{SRC_RAW.as_posix()}`",
        f"- Output: `{out_dir.as_posix()}`",
        f"- Config: S_i={TASK_SIZE_BYTES:g} bytes, L={TASK_CYCLES:.4g}, "
        f"B_sys={meta['b_sys']:g} Hz",
        f"- Wall time: {meta['wall_s']:.1f} s",
        "",
        "## Sanity: version A vs original SCA CSV",
        "",
        "If recovered binaries match the stored solution, A should reproduce the",
        "validation SCA rates.",
        "",
        f"- Mean |A − CSV| over all SCA (seed, T_k): **{sanity['mean_abs_mbps']:.4f} Mbps**",
        f"- Max |A − CSV|: **{sanity['max_abs_mbps']:.4f} Mbps** "
        f"(seed {sanity['max_seed']}, T_k={sanity['max_tk']:g} s)",
        f"- Fraction within 1e-3 Mbps: **{sanity['frac_match']:.2%}**",
        "",
        "## SCA placements (primary)",
        "",
        "| T_k (s) | A (Mbps) | B (Mbps) | C (Mbps) | Paper SCA | |A−paper| | |B−paper| | |C−paper| | A feas | B feas | C feas |",
        "|--------:|---------:|---------:|---------:|----------:|----------:|----------:|----------:|-------:|-------:|-------:|",
    ]
    for tk in THRESHOLDS:
        a = _mbps_at(summary, "sca_A", tk)
        b = _mbps_at(summary, "sca_B", tk)
        c = _mbps_at(summary, "sca_C", tk)
        p = PAPER_SCA[tk]
        lines.append(
            f"| {tk:g} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {p:.1f} | "
            f"{fmt(abs(a - p))} | {fmt(abs(b - p))} | {fmt(abs(c - p))} | "
            f"{fmt(_field_at(summary, 'sca_A', tk, 'feasible_frac'), 2)} | "
            f"{fmt(_field_at(summary, 'sca_B', tk, 'feasible_frac'), 2)} | "
            f"{fmt(_field_at(summary, 'sca_C', tk, 'feasible_frac'), 2)} |"
        )

    lines.extend(
        [
            "",
            "AoDT mean (s):",
            "",
            "| T_k (s) | A | B | C |",
            "|--------:|--:|--:|--:|",
        ]
    )
    for tk in THRESHOLDS:
        lines.append(
            "| {tk:g} | {a} | {b} | {c} |".format(
                tk=tk,
                a=fmt(_field_at(summary, "sca_A", tk, "aodt_mean")),
                b=fmt(_field_at(summary, "sca_B", tk, "aodt_mean")),
                c=fmt(_field_at(summary, "sca_C", tk, "aodt_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## Decision vs paper SCA (six usable T_k)",
            "",
            f"| Policy | MAE (Mbps) | Points within 0.5 Mbps | Trend (nondecr., 0.05) | Rate at T_k=3 |",
            "|---|---:|---:|---|---:|",
        ]
    )

    def _trend(pol: str) -> bool:
        vals = [_mbps_at(summary, f"sca_{pol}", tk) for tk in USABLE]
        return all(vals[i + 1] + 0.05 >= vals[i] for i in range(len(vals) - 1))

    for pol in ("A", "B", "C"):
        lines.append(
            f"| {pol} | {sca_mae[pol]:.3f} | {sca_n_within[pol]}/6 | "
            f"{'yes' if _trend(pol) else 'no'} | "
            f"{fmt(_mbps_at(summary, f'sca_{pol}', 3.0))} |"
        )

    a3 = _mbps_at(summary, "sca_A", 3.0)
    b3 = _mbps_at(summary, "sca_B", 3.0)
    c3 = _mbps_at(summary, "sca_C", 3.0)
    # Hypothesis: leftover policy is the main leak if C (or B) closes most of
    # the A-vs-paper gap at T_k=3 and MAE drops sharply.
    gap_a = abs(a3 - 7.4)
    gap_c = abs(c3 - 7.4)
    closed = (gap_a - gap_c) / gap_a if gap_a > 1e-9 else 0.0
    leftover_is_main = sca_mae["C"] < 0.8 or (sca_n_within["C"] >= 4) or (
        closed >= 0.6 and sca_mae["C"] < 0.5 * sca_mae["A"]
    )
    placement_is_main = sca_mae["C"] >= 0.8 and sca_n_within["C"] < 4 and closed < 0.6

    if leftover_is_main and not placement_is_main:
        verdict = (
            "**Leftover policy is the main discrepancy.** B/C on frozen SCA "
            "placements move the curve toward the paper SCA line. Do not treat "
            "this as a production change; it only confirms the hypothesis."
        )
    elif placement_is_main:
        verdict = (
            "**Placement/SCA search remains the main gap.** Even SE-first leftover "
            "(B/C) on the stored SCA coordinates does not reach the paper curve. "
            "The leftover heuristic is still a real Problem (P) mismatch, but it "
            "is not sufficient by itself."
        )
    else:
        verdict = (
            "**Mixed.** Leftover policy recovers a substantial fraction of the "
            "gap, but B/C still miss the paper curve by enough that SCA "
            "placement/CVX remains an open issue."
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict,
            "",
            f"- Paper SCA at T_k=3 s: 7.4 Mbps",
            f"- Frozen SCA + A: {fmt(a3)} Mbps (gap {fmt(gap_a)})",
            f"- Frozen SCA + B: {fmt(b3)} Mbps",
            f"- Frozen SCA + C: {fmt(c3)} Mbps (gap {fmt(gap_c)}; "
            f"{closed:.0%} of the A-vs-paper gap closed)",
            "",
            "## K-means placements (control)",
            "",
            "Same leftover rules on frozen K-means coordinates. If C on K-means",
            "also climbs toward 7 Mbps, leftover—not SCA's extra placement search—",
            "is doing the work.",
            "",
            "| T_k (s) | A | B | C | Paper K-means |",
            "|--------:|--:|--:|--:|--------------:|",
        ]
    )
    paper_km = {0.8: 1.8, 1.2: 2.4, 1.6: 2.8, 2.0: 3.3, 2.4: 3.8, 2.8: 4.2, 3.0: 4.4}
    for tk in THRESHOLDS:
        lines.append(
            "| {tk:g} | {a} | {b} | {c} | {p:.1f} |".format(
                tk=tk,
                a=fmt(_mbps_at(summary, "kmeans_A", tk)),
                b=fmt(_mbps_at(summary, "kmeans_B", tk)),
                c=fmt(_mbps_at(summary, "kmeans_C", tk)),
                p=paper_km[tk],
            )
        )

    lines.extend(
        [
            "",
            "Production files were not modified. Next step is a decision on whether",
            "to change `_give_aodt_chunk` / `max_bw_share`, not a retune of S_i or L.",
            "",
            "Plot: `fig9_abc_vs_paper.png`.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(out_dir: Path) -> None:
    src = ROOT / SRC_RAW
    if not src.exists():
        raise SystemExit(f"missing placements CSV: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)

    placements = {
        "sca": _load_placements(src, "sca"),
        "kmeans": _load_placements(src, "kmeans"),
    }
    n_jobs = sum(len(v) for v in placements.values()) * len(POLICIES)
    jobs = Counter("Fig.9 A/B/C leftover replay", n_jobs)

    rows: list[dict] = []
    sanity_err: list[float] = []
    sanity_meta: list[tuple[int, float, float]] = []

    t0 = time.perf_counter()
    for source, items in placements.items():
        for rec in items:
            tk = rec["aodt_threshold"]
            seed = rec["seed"]
            xy = rec["xy"]
            # Binaries from production-cap scenario (cap does not affect geometry).
            s_prod = generate_scenario(seed, _cfg(tk, 0.25))
            a, proc = _frozen_binaries(s_prod, xy)
            for pol, aodt_first, share in POLICIES:
                cfg = _cfg(tk, share)
                s = generate_scenario(seed, cfg)
                # Same IoT geometry; only the cap flag differs.
                bw = allocate_constrained_bandwidth(
                    s, xy, a, proc, aodt_first=aodt_first
                )
                result = evaluate(s, xy, a, proc, bw)
                pool = float(cfg.b_sys)
                cap = link_bandwidth_cap(cfg, pool)
                st = _bw_stats(bw, a, cap, pool)
                method = f"{source}_{pol}"
                aodt_mean = (
                    float(np.nanmean(result.aodt)) if result.compute_available else None
                )
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "source_placement": source,
                        "policy": pol,
                        "aodt_first": int(aodt_first),
                        "max_bw_share": "" if share is None else share,
                        "aodt_threshold": tk,
                        "sum_rate": result.sum_rate,
                        "qos": result.qos_violations,
                        "feasible": int(result.feasible),
                        "aodt_viol": result.aodt_violations,
                        "aodt_excess": result.aodt_excess,
                        "cpu_unstable": result.cpu_unstable,
                        "runtime": 0.0,
                        "aodt_mean": aodt_mean,
                        "unused_hz": st["unused_hz"],
                        "bw_min_hz": st["min_hz"],
                        "bw_max_hz": st["max_hz"],
                        "bw_mean_hz": st["mean_hz"],
                        "n_at_cap": st["n_at_cap"],
                        "csv_sum_rate": rec["csv_sum_rate"],
                    }
                )
                if source == "sca" and pol == "A":
                    err = abs(result.sum_rate - rec["csv_sum_rate"]) / 1e6
                    sanity_err.append(err)
                    sanity_meta.append((seed, tk, err))
                jobs.tick(f"{source}_{pol}  Tk={tk:g}  seed={seed}", result, 0.0)

    summary = _summarize(rows, "aodt_threshold")
    _write_csv(out_dir / "raw_abc.csv", rows)
    _write_csv(out_dir / "sumrate_abc.csv", summary)

    worst = max(sanity_meta, key=lambda t: t[2]) if sanity_meta else (None, float("nan"), float("nan"))
    sanity = {
        "mean_abs_mbps": float(np.mean(sanity_err)) if sanity_err else float("nan"),
        "max_abs_mbps": float(np.max(sanity_err)) if sanity_err else float("nan"),
        "frac_match": float(np.mean(np.array(sanity_err) < 1e-3)) if sanity_err else float("nan"),
        "max_seed": worst[0],
        "max_tk": worst[1],
    }
    meta = {
        "note": "A/B/C leftover replay on frozen Fig. 9 placements. Production unchanged.",
        "source_csv": str(src),
        "b_sys": float(DEFAULT.b_sys),
        "task_size_bytes": TASK_SIZE_BYTES,
        "task_cycles": TASK_CYCLES,
        "n_rows": len(rows),
        "sanity": sanity,
        "wall_s": time.perf_counter() - t0,
        "policies": [
            {"id": p, "aodt_first": af, "max_bw_share": sh} for p, af, sh in POLICIES
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_curves(out_dir, summary)
    write_report(out_dir, summary, sanity, meta)
    log.info("wrote %s", out_dir / "REPORT.md")
    log.info(
        "sanity A vs CSV: mean=%.4f Mbps  max=%.4f Mbps  match=%.1f%%",
        sanity["mean_abs_mbps"],
        sanity["max_abs_mbps"],
        100.0 * sanity["frac_match"],
    )
    for pol in ("A", "B", "C"):
        log.info(
            "SCA-%s  Tk=3  %.3f Mbps  (paper 7.4)",
            pol,
            _mbps_at(summary, f"sca_{pol}", 3.0),
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay A/B/C leftover on frozen Fig. 9 placements")
    p.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, log_file=str(out_dir / "run.log"))
    banner("Fig. 9 leftover A/B/C replay (frozen placements)")
    run(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
