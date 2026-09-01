"""Coarse-to-fine search over undocumented AoDT parameters S_i and L.

Radio, channel, solver, and evaluator knobs stay at the calibrated defaults.
Only ``task_size_bytes`` (S_i) and ``task_cycles`` (L) vary. Upload delay keeps
the current implementation ``D_i = 8 S_i / r_ij`` (bytes × 8 → bits).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.compute import service_rate
from src.config import (
    B_SYS,
    EXPERIMENTAL_TASK_CYCLES,
    EXPERIMENTAL_TASK_SIZE_BYTES,
    PAPER_SCENARIO_SEEDS,
    SimConfig,
)
from src.experiments.sweeps import _solve, _write_csv
from src.logutil import Counter, banner, log
from src.scenario import generate_scenario

# Fig. 9 analogue. T_k = 3 s quotes are from the IEEE body; 0.8 s and 2.0 s
# SCA / K-means / Random levels are approximate figure reads. Mbps.
PAPER_FIG9_MBPS: dict[float, dict[str, float]] = {
    0.8: {"sca": 4.0, "td3": 3.2, "kmeans": 1.8, "random": 0.8},
    2.0: {"sca": 6.3, "td3": 5.0, "kmeans": 3.3, "random": 2.2},
    3.0: {"sca": 7.4, "td3": 6.0, "kmeans": 4.4, "random": 3.2},
}

COARSE_TASK_SIZES = (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 64000.0, 128000.0, 256000.0)
COARSE_TASK_CYCLES = (1e5, 2.5e5, 5e5, 1e6, 2e6, 3e6, 5e6, 7.5e6, 1e7)
FIG9_THRESHOLDS = (0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0)
DEFAULT_SNAPSHOT_TK = 2.8
CALIBRATED_B_SYS = 8.8e6
COARSE_METHODS = ("random", "kmeans", "sca")
TD3_THRESHOLDS = (0.8, 2.0, 3.0)
TD3_TRAIN_SEEDS = tuple(range(200, 220))


@dataclass(frozen=True)
class SearchSpec:
    task_sizes: tuple[float, ...] = COARSE_TASK_SIZES
    task_cycles: tuple[float, ...] = COARSE_TASK_CYCLES
    thresholds: tuple[float, ...] = FIG9_THRESHOLDS
    seeds: tuple[int, ...] = tuple(range(100, 105))
    methods: tuple[str, ...] = COARSE_METHODS
    n_iot: int = 10
    n_uav: int = 3
    refine_count: int = 3
    shortlist_count: int = 5
    shortlist_seeds: tuple[int, ...] = PAPER_SCENARIO_SEEDS
    td3_top_n: int = 3
    td3_thresholds: tuple[float, ...] = TD3_THRESHOLDS
    skip_td3: bool = False


def pair_key(si: float, cycles: float) -> tuple[float, float]:
    return (round(float(si), 6), float(f"{float(cycles):.8g}"))


def coarse_pairs(spec: SearchSpec | None = None) -> list[tuple[float, float]]:
    spec = spec or SearchSpec()
    return [(float(si), float(L)) for si in spec.task_sizes for L in spec.task_cycles]


def refine_around(si: float, cycles: float) -> list[tuple[float, float]]:
    """Local grid with steps ~4× finer than the coarse doubling grid."""
    si = float(si)
    cycles = float(cycles)
    si_step = max(si / 4.0, 50.0)
    L_step = max(cycles / 4.0, 1e4)
    sizes = [si + k * si_step for k in (-2, -1, 0, 1, 2)]
    cycles_vals = [cycles + k * L_step for k in (-2, -1, 0, 1, 2)]
    out = []
    seen = set()
    for s in sizes:
        if s < 100.0:
            continue
        for L in cycles_vals:
            if L < 2e4:
                continue
            key = pair_key(s, L)
            if key in seen:
                continue
            seen.add(key)
            out.append((float(s), float(L)))
    return out


def refine_pairs(ranked: list[dict], n: int = 3) -> list[tuple[float, float]]:
    seen = set()
    out = []
    for rec in ranked[: max(n, 0)]:
        for pair in refine_around(rec["task_size_bytes"], rec["task_cycles"]):
            key = pair_key(*pair)
            if key in seen:
                continue
            seen.add(key)
            out.append(pair)
    return out


def cpu_metrics(cfg: SimConfig) -> dict[str, float]:
    mu = service_rate(cfg)
    if mu is None or cfg.task_cycles is None:
        return {"mu": float("nan"), "queue_term": float("nan"), "rho_one_process": float("nan")}
    n_per = cfg.iots_per_process
    lam = cfg.lambda_i
    # Eq. (17) queueing term for one process of n_per IoTs at equal λ.
    q = (1.0 / max(lam, 1e-12)) * (1.0 + (n_per * lam) / mu)
    rho_one = (n_per * lam) / mu
    return {"mu": float(mu), "queue_term": float(q), "rho_one_process": float(rho_one)}


def _cfg_for_pair(base: SimConfig, si: float, cycles: float, tk: float, spec: SearchSpec) -> SimConfig:
    cfg = base.with_compute(task_size_bytes=float(si), task_cycles=float(cycles))
    k = cfg.num_processes
    per = spec.n_iot // k
    return replace(
        cfg,
        aodt_threshold=float(tk),
        num_iot=spec.n_iot,
        num_uav=spec.n_uav,
        iots_per_process=per if per * k == spec.n_iot else cfg.iots_per_process,
    )


def _raw_row(seed, method, result, rt, si, cycles, tk, mu, queue_q) -> dict:
    aodt_mean = float(np.nanmean(result.aodt)) if result.compute_available else None
    aodt_max = float(np.nanmax(result.aodt)) if result.compute_available else None
    rho = np.asarray(result.rho, dtype=float)
    return {
        "task_size_bytes": float(si),
        "task_cycles": float(cycles),
        "aodt_threshold": float(tk),
        "seed": int(seed),
        "method": method,
        "sum_rate": float(result.sum_rate),
        "feasible": int(result.feasible),
        "qos": int(result.qos_violations),
        "aodt_violations": int(result.aodt_violations),
        "aodt_mean": aodt_mean,
        "aodt_max": aodt_max,
        "cpu_unstable": int(result.cpu_unstable),
        "rho_mean": float(np.mean(rho)) if rho.size else None,
        "rho_max": float(np.max(rho)) if rho.size else None,
        "mu": float(mu) if mu is not None else None,
        "queue_term": float(queue_q) if queue_q is not None else None,
        "runtime": float(rt),
    }


def run_pairs(
    base: SimConfig,
    spec: SearchSpec,
    pairs: list[tuple[float, float]],
    seeds: tuple[int, ...],
    args,
    label: str,
    methods: tuple[str, ...] | None = None,
    thresholds: tuple[float, ...] | None = None,
    existing_raw: list[dict] | None = None,
) -> list[dict]:
    methods = methods or spec.methods
    thresholds = thresholds or spec.thresholds
    done = set()
    rows = list(existing_raw or [])
    for r in rows:
        done.add((pair_key(r["task_size_bytes"], r["task_cycles"]), int(r["seed"]), r["method"], float(r["aodt_threshold"])))
    jobs = Counter(label, max(len(pairs), 1))
    for si, cycles in pairs:
        cfg_metrics = _cfg_for_pair(base, si, cycles, DEFAULT_SNAPSHOT_TK, spec)
        extras = cpu_metrics(cfg_metrics)
        mu, queue_q = extras["mu"], extras["queue_term"]
        n_done = 0
        n_need = len(seeds) * len(methods) * len(thresholds)
        for tk in thresholds:
            cfg_t = _cfg_for_pair(base, si, cycles, tk, spec)
            for seed in seeds:
                scenario = generate_scenario(seed, cfg_t)
                for name in methods:
                    key = (pair_key(si, cycles), int(seed), name, float(tk))
                    if key in done:
                        n_done += 1
                        continue
                    xy, result, rt = _solve(name, scenario, seed, args, n_uav=spec.n_uav)
                    rows.append(_raw_row(seed, name, result, rt, si, cycles, tk, mu, queue_q))
                    done.add(key)
                    n_done += 1
        jobs.tick(
            f"S_i={si:g} L={cycles:g} mu={mu:.3g} Q={queue_q:.3f}  {n_done}/{n_need}",
        )
    return rows


def _group_raw(rows: list[dict]) -> dict[tuple[float, float], list[dict]]:
    grouped: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        grouped.setdefault(pair_key(r["task_size_bytes"], r["task_cycles"]), []).append(r)
    return grouped


def _mean(vals: list[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _subset_stats(rows: list[dict], method: str, tk: float) -> dict[str, float]:
    sub = [r for r in rows if r["method"] == method and abs(float(r["aodt_threshold"]) - tk) < 1e-9]
    return {
        "sum_rate": _mean([r["sum_rate"] for r in sub]),
        "feasible": _mean([float(r["feasible"]) for r in sub]),
        "aodt_mean": _mean([r["aodt_mean"] for r in sub if r.get("aodt_mean") is not None]),
        "aodt_max": _mean([r["aodt_max"] for r in sub if r.get("aodt_max") is not None]),
        "aodt_viol": _mean([r["aodt_violations"] for r in sub]),
        "qos": _mean([r["qos"] for r in sub]),
        "rho_mean": _mean([r["rho_mean"] for r in sub if r.get("rho_mean") is not None]),
        "rho_max": _mean([r["rho_max"] for r in sub if r.get("rho_max") is not None]),
        "n": float(len(sub)),
    }


def _clip01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def score_pair(rows: list[dict]) -> dict:
    """Score one (S_i, L) T_k curve against Fig. 9 behaviour, not default sum rate."""
    si = float(rows[0]["task_size_bytes"])
    cycles = float(rows[0]["task_cycles"])
    mu = _mean([r["mu"] for r in rows if r.get("mu") is not None])
    queue_q = _mean([r["queue_term"] for r in rows if r.get("queue_term") is not None])
    methods = sorted({r["method"] for r in rows})
    tks = sorted({float(r["aodt_threshold"]) for r in rows})

    rec: dict = {
        "task_size_bytes": si,
        "task_cycles": cycles,
        "mu": mu,
        "queue_term": queue_q,
        "n_seeds": int(max((_subset_stats(rows, m, tks[0])["n"] if tks else 0) for m in methods) if methods and tks else 0),
        "methods": ",".join(methods),
    }

    snap = _subset_stats(rows, "sca", DEFAULT_SNAPSHOT_TK) if "sca" in methods else {}
    rec["sca_sum_rate"] = snap.get("sum_rate", float("nan"))
    rec["sca_feasible"] = snap.get("feasible", float("nan"))
    rec["sca_aodt_mean"] = snap.get("aodt_mean", float("nan"))
    rec["sca_aodt_max"] = snap.get("aodt_max", float("nan"))
    rec["sca_aodt_viol"] = snap.get("aodt_viol", float("nan"))
    rec["sca_qos_viol"] = snap.get("qos", float("nan"))
    rec["sca_rho_mean"] = snap.get("rho_mean", float("nan"))
    rec["sca_rho_max"] = snap.get("rho_max", float("nan"))

    if "td3" in methods:
        td = _subset_stats(rows, "td3", DEFAULT_SNAPSHOT_TK)
        rec["td3_sum_rate"] = td["sum_rate"]
        rec["td3_feasible"] = td["feasible"]
        rec["td3_aodt_mean"] = td["aodt_mean"]
        rec["td3_aodt_viol"] = td["aodt_viol"]
        rec["td3_qos_viol"] = td["qos"]
    else:
        rec["td3_sum_rate"] = float("nan")
        rec["td3_feasible"] = float("nan")
        rec["td3_aodt_mean"] = float("nan")
        rec["td3_aodt_viol"] = float("nan")
        rec["td3_qos_viol"] = float("nan")

    curves: dict[str, dict[float, dict[str, float]]] = {}
    for m in methods:
        curves[m] = {tk: _subset_stats(rows, m, tk) for tk in tks}
        for tk in tks:
            rec[f"{m}_mbps_{tk:g}"] = curves[m][tk]["sum_rate"] / 1e6
            rec[f"{m}_feas_{tk:g}"] = curves[m][tk]["feasible"]
            rec[f"{m}_aodt_{tk:g}"] = curves[m][tk]["aodt_mean"]
            rec[f"{m}_aodt_viol_{tk:g}"] = curves[m][tk]["aodt_viol"]

    sca_tks = [tk for tk in tks if "sca" in curves]
    sca_mbps = [curves["sca"][tk]["sum_rate"] / 1e6 for tk in sca_tks] if "sca" in curves else []
    if len(sca_mbps) >= 2:
        diffs = np.diff(sca_mbps)
        rec["monotonic"] = float(np.mean(diffs >= -0.05))
        rec["rise_08_to_30"] = sca_mbps[-1] - sca_mbps[0]
        rec["first_step"] = float(diffs[0])
        rec["last_step"] = float(diffs[-1])
    else:
        rec["monotonic"] = float("nan")
        rec["rise_08_to_30"] = float("nan")
        rec["first_step"] = float("nan")
        rec["last_step"] = float("nan")

    def _at(method: str, tk: float, field: str = "sum_rate") -> float:
        if method not in curves or tk not in curves[method]:
            return float("nan")
        val = curves[method][tk][field]
        return val / 1e6 if field == "sum_rate" else val

    rec["late_rise_20_to_30"] = _at("sca", 3.0) - _at("sca", 2.0)
    rise = rec["rise_08_to_30"]
    after_16 = _at("sca", 3.0) - _at("sca", 1.6)
    rec["late_frac"] = after_16 / rise if math.isfinite(rise) and abs(rise) > 0.05 else 0.0
    rec["flatten_early"] = float(rec["late_frac"] < 0.25) if math.isfinite(rec["late_frac"]) else 1.0

    mae_parts = []
    for tk, paper in PAPER_FIG9_MBPS.items():
        for method, target in paper.items():
            if method == "td3" and "td3" not in curves:
                continue
            got = _at(method, tk)
            if math.isfinite(got):
                mae_parts.append(abs(got - target))
                rec[f"err_{method}_{tk:g}"] = abs(got - target)
    rec["mae_fig9"] = float(np.mean(mae_parts)) if mae_parts else float("nan")
    rec["mae_sca_fig9"] = _mean([abs(_at("sca", tk) - PAPER_FIG9_MBPS[tk]["sca"]) for tk in PAPER_FIG9_MBPS])
    rec["mae_kmeans_fig9"] = _mean([abs(_at("kmeans", tk) - PAPER_FIG9_MBPS[tk]["kmeans"]) for tk in PAPER_FIG9_MBPS])
    rec["mae_random_fig9"] = _mean([abs(_at("random", tk) - PAPER_FIG9_MBPS[tk]["random"]) for tk in PAPER_FIG9_MBPS])

    rec["ranking_tk3"] = float(
        _at("sca", 3.0) > _at("kmeans", 3.0) + 0.05 and _at("kmeans", 3.0) > _at("random", 3.0) + 0.05
    )
    rec["feas_sca_0.8"] = _at("sca", 0.8, "feasible")
    rec["feas_sca_3.0"] = _at("sca", 3.0, "feasible")
    rec["aodt_binds"] = float(
        (_at("sca", 0.8, "aodt_viol") > _at("sca", 3.0, "aodt_viol") + 0.05)
        or (math.isfinite(rise) and rise > 0.4)
    )
    rec["cpu_stable"] = float(rec.get("sca_rho_max", 1.0) < 0.99) if math.isfinite(rec.get("sca_rho_max", float("nan"))) else 0.0

    # Composite: Fig. 9 shape first. Explicitly do NOT reward matching the already
    # calibrated ~7.1 Mbps SCA point at T_k = 2.8.
    paper_rise = PAPER_FIG9_MBPS[3.0]["sca"] - PAPER_FIG9_MBPS[0.8]["sca"]
    paper_late = PAPER_FIG9_MBPS[3.0]["sca"] - PAPER_FIG9_MBPS[2.0]["sca"]
    rec["score"] = (
        3.0 * (1.0 - min(rec["mae_sca_fig9"] if math.isfinite(rec["mae_sca_fig9"]) else 3.0, 3.0) / 3.0)
        + 2.0 * _clip01(rec["late_frac"] / 0.50)
        + 1.5 * (1.0 - min(abs((rec["late_rise_20_to_30"] if math.isfinite(rec["late_rise_20_to_30"]) else 0.0) - paper_late) / 2.0, 1.0))
        + 1.0 * (1.0 - min(abs((rise if math.isfinite(rise) else 0.0) - paper_rise) / 4.0, 1.0))
        + 1.0 * (rec["monotonic"] if math.isfinite(rec["monotonic"]) else 0.0)
        + 1.0 * (1.0 - min(rec["mae_kmeans_fig9"] if math.isfinite(rec["mae_kmeans_fig9"]) else 3.0, 3.0) / 3.0)
        + 0.5 * (1.0 - min(rec["mae_random_fig9"] if math.isfinite(rec["mae_random_fig9"]) else 3.0, 3.0) / 3.0)
        + 1.0 * (rec["feas_sca_3.0"] if math.isfinite(rec["feas_sca_3.0"]) else 0.0)
        + 0.7 * (rec["feas_sca_0.8"] if math.isfinite(rec["feas_sca_0.8"]) else 0.0)
        + 0.5 * rec["ranking_tk3"]
        + 0.5 * rec["aodt_binds"]
        + 0.5 * rec["cpu_stable"]
        - 2.0 * rec["flatten_early"]
    )
    rec["is_default"] = float(
        abs(si - EXPERIMENTAL_TASK_SIZE_BYTES) < 1e-6 and abs(cycles - EXPERIMENTAL_TASK_CYCLES) < 1.0
    )
    return rec


def rank_pairs(raw_rows: list[dict]) -> list[dict]:
    ranked = [score_pair(group) for group in _group_raw(raw_rows).values()]
    ranked.sort(key=lambda r: (-float(r["score"]), r["mae_sca_fig9"], r["task_size_bytes"], r["task_cycles"]))
    for i, rec in enumerate(ranked, start=1):
        rec["rank"] = i
    return ranked


def _mbps(x: float) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _fmt(x: float, nd=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def write_top_table(path: Path, ranked: list[dict], n: int = 10, title: str = "Top candidates") -> None:
    lines = [
        f"# {title}",
        "",
        "Ranked by Fig. 9 AoDT-threshold behaviour (shape, late rise, method levels), not by default sum rate.",
        "",
        "| Rank | S_i (bytes) | L (cycles) | Score | SCA 0.8 | SCA 2.0 | SCA 3.0 | K-means 3.0 | Random 3.0 | Rise | Late frac | Feas 0.8 | Feas 3.0 | μ | ρ_max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in ranked[:n]:
        lines.append(
            f"| {rec.get('rank', '')} | {rec['task_size_bytes']:.0f} | {rec['task_cycles']:.4g} | "
            f"{_fmt(rec['score'])} | {_fmt(rec.get('sca_mbps_0.8'))} | {_fmt(rec.get('sca_mbps_2'))} | "
            f"{_fmt(rec.get('sca_mbps_3'))} | {_fmt(rec.get('kmeans_mbps_3'))} | {_fmt(rec.get('random_mbps_3'))} | "
            f"{_fmt(rec.get('rise_08_to_30'))} | {_fmt(rec.get('late_frac'))} | "
            f"{_fmt(rec.get('feas_sca_0.8'))} | {_fmt(rec.get('feas_sca_3.0'))} | "
            f"{_fmt(rec.get('mu'), 2)} | {_fmt(rec.get('sca_rho_max'))} |"
        )
    lines.extend(
        [
            "",
            "Paper Fig. 9 (approx.): SCA 4.0 → 6.3 → 7.4 Mbps at T_k = 0.8, 2.0, 3.0 s; "
            "K-means ≈ 4.4 and Random ≈ 3.2 at 3 s; TD3 ≈ 6 Mbps at 3 s.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_top_curves(out_dir: Path, ranked: list[dict], n: int = 5) -> None:
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
    ax.plot(paper_x, [PAPER_FIG9_MBPS[t]["kmeans"] for t in paper_x], color="0.4", ls=":", marker="o", label="paper K-means")
    ax.plot(paper_x, [PAPER_FIG9_MBPS[t]["random"] for t in paper_x], color="0.6", ls=":", marker="s", label="paper Random")
    for rec in ranked[:n]:
        ys = [rec.get(f"sca_mbps_{tk:g}") for tk in tks]
        if any(v is None or not math.isfinite(v) for v in ys):
            continue
        label = f"S_i={rec['task_size_bytes']:.0f}, L={rec['task_cycles']:.2g}"
        ax.plot(tks, ys, marker="o", label=label)
    ax.set_xlabel("AoDT threshold T_k (s)")
    ax.set_ylabel("SCA sum rate (Mbps)")
    ax.set_title("Fig. 9 analogue: top (S_i, L) vs paper")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_top_candidates.png", dpi=120)
    plt.close(fig)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for r in rows:
        out = dict(r)
        for k, v in r.items():
            if k == "method" or k == "methods":
                continue
            try:
                if v == "" or v is None:
                    out[k] = None
                elif "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = v
        parsed.append(out)
    return parsed


def _print_top(ranked: list[dict], n: int = 10, label: str = "top candidates") -> None:
    banner(f"{label} (n={min(n, len(ranked))})")
    log.info(
        "%4s %10s %10s %7s %8s %8s %8s %7s %7s %6s",
        "rk", "S_i", "L", "score", "SCA0.8", "SCA2.0", "SCA3.0", "rise", "late", "feas08",
    )
    for rec in ranked[:n]:
        log.info(
            "%4s %10.0f %10.4g %7.3f %8s %8s %8s %7s %7s %6s",
            rec.get("rank", ""),
            rec["task_size_bytes"],
            rec["task_cycles"],
            rec["score"],
            _fmt(rec.get("sca_mbps_0.8")),
            _fmt(rec.get("sca_mbps_2")),
            _fmt(rec.get("sca_mbps_3")),
            _fmt(rec.get("rise_08_to_30")),
            _fmt(rec.get("late_frac")),
            _fmt(rec.get("feas_sca_0.8")),
        )


def write_report(
    out_dir: Path,
    spec: SearchSpec,
    coarse_ranked: list[dict],
    refine_ranked: list[dict],
    shortlist_ranked: list[dict],
    td3_raw: list[dict],
) -> None:
    best = (shortlist_ranked or refine_ranked or coarse_ranked or [None])[0]
    default = next((r for r in (shortlist_ranked or coarse_ranked) if r.get("is_default")), None)
    if default is None:
        default = next((r for r in coarse_ranked if r.get("is_default")), None)

    def _pair_line(rec: dict | None) -> str:
        if rec is None:
            return "(none)"
        return (
            f"S_i = {rec['task_size_bytes']:.0f} bytes, L = {rec['task_cycles']:.4g} cycles "
            f"(score {rec['score']:.3f}; SCA { _fmt(rec.get('sca_mbps_0.8'))} → {_fmt(rec.get('sca_mbps_3'))} Mbps)"
        )

    td3_note = "TD3 was skipped (`--skip-td3`)."
    if td3_raw:
        td3_bits = []
        for rec in shortlist_ranked[: spec.td3_top_n]:
            sub = [
                r
                for r in td3_raw
                if pair_key(r["task_size_bytes"], r["task_cycles"]) == pair_key(rec["task_size_bytes"], rec["task_cycles"])
            ]
            if not sub:
                continue
            at3 = [r for r in sub if r["method"] == "td3" and abs(float(r["aodt_threshold"]) - 3.0) < 1e-9]
            td3_bits.append(
                f"- S_i={rec['task_size_bytes']:.0f}, L={rec['task_cycles']:.4g}: "
                f"TD3 at T_k=3 s is {_mean([r['sum_rate'] for r in at3])/1e6:.3f} Mbps "
                f"(paper ≈ 6.0); feasible frac {_mean([float(r['feasible']) for r in at3]):.2f}."
            )
        td3_note = "\n".join(td3_bits) if td3_bits else "TD3 ran but produced no shortlist rows."

    can_reproduce = False
    caveats = []
    if best is not None:
        mae = best.get("mae_sca_fig9", float("nan"))
        late = best.get("late_frac", 0.0)
        feas08 = best.get("feas_sca_0.8", 0.0)
        rise = best.get("rise_08_to_30", 0.0)
        can_reproduce = bool(
            math.isfinite(mae) and mae < 0.8 and late >= 0.35 and feas08 >= 0.5 and 2.0 <= rise <= 5.0
        )
        if not (math.isfinite(mae) and mae < 0.8):
            caveats.append("SCA Fig. 9 level error stays large (MAE ≥ 0.8 Mbps).")
        if late < 0.35:
            caveats.append("The T_k curve still rises mostly at the tight end (late_frac < 0.35); the paper’s Fig. 9 looks closer to linear through 3 s.")
        if feas08 < 0.5:
            caveats.append("T_k = 0.8 s is mostly infeasible, but the paper plots a curve from 0.8 s.")
        if best["task_size_bytes"] >= 8 * EXPERIMENTAL_TASK_SIZE_BYTES:
            caveats.append(
                "The best S_i is several times the experimental default. That is consistent with D_i mattering at Mbps-scale rates, "
                "and also with the paper writing D_i = S_i / r_ij while calling S_i bytes (bit/s rates). This search did **not** drop the ×8 conversion."
            )
        if best.get("cpu_stable", 1) < 0.5:
            caveats.append("CPU utilisation is high enough that M/M/1 stability is questionable.")

    assumption = (
        "A plausible (S_i, L) pair can reproduce Fig. 9’s qualitative slope and method order under the current model."
        if can_reproduce
        else "No searched (S_i, L) pair fully reproduces the paper. Matching Fig. 9 likely needs another undocumented assumption "
        "(for example treating S_i as bits in D_i = S_i / r_ij, a different queueing model, or plotting infeasible points)."
    )

    lines = [
        "# AoDT parameter search: S_i and L",
        "",
        "This directory is a **new** result set. It does not overwrite `results/run_20260831/`, "
        "`results/bandwidth_2p4mhz/`, or `results/bandwidth_2p4mhz_nocap/`.",
        "",
        "Project defaults `EXPERIMENTAL_TASK_SIZE_BYTES = 2000` and `EXPERIMENTAL_TASK_CYCLES = 2e6` were **not** changed.",
        "",
        "## Search space",
        "",
        f"- S_i (bytes): {', '.join(str(int(x)) if float(x).is_integer() else str(x) for x in spec.task_sizes)}",
        f"- L (cycles): {', '.join(f'{x:.4g}' for x in spec.task_cycles)}",
        f"- T_k (s): {', '.join(str(x) for x in spec.thresholds)}",
        f"- Scene: I = {spec.n_iot}, J = {spec.n_uav}, --compute, radio unchanged (B_sys = {B_SYS:g} Hz).",
        f"- Coarse seeds: {spec.seeds[0]}–{spec.seeds[-1]} ({len(spec.seeds)}).",
        f"- Shortlist seeds: {spec.shortlist_seeds[0]}–{spec.shortlist_seeds[-1]} ({len(spec.shortlist_seeds)}).",
        f"- Methods (coarse/refine/shortlist): {', '.join(spec.methods)}. TD3 only on the final shortlist"
        + (" (skipped)." if spec.skip_td3 else "."),
        "- Upload delay: current implementation `D_i = 8 S_i / r_ij` (bytes → bits). Not changed.",
        "",
        "## Ranking target",
        "",
        "The calibrated radio already gives SCA ≈ 7.1 Mbps at the default point, so **sum-rate agreement at T_k = 2.8 s is not the objective**.",
        "Pairs are ranked on IEEE Fig. 9 behaviour:",
        "",
        "- Sum rate should keep rising as T_k goes from 0.8 s to 3 s (not flatten after the first step).",
        "- AoDT must actually change bandwidth/placement (tight T_k ≠ relaxed T_k).",
        "- T_k = 3 s: TD3 ≈ 6 Mbps, K-means ≈ 4.4 Mbps, Random ≈ 3.2 Mbps, SCA highest (≈ 7.4 Mbps from the figure).",
        "- Prefer physical feasibility where the paper draws a curve, including 0.8 s if possible.",
        "",
        "## Best candidates",
        "",
        f"- **Selected pair:** {_pair_line(best)}",
        f"- Default (2000, 2e6) for comparison: {_pair_line(default)}",
        "",
        "Top 10 after the last completed ranking stage (shortlist if present, else refine, else coarse):",
        "",
    ]
    table_src = shortlist_ranked or refine_ranked or coarse_ranked
    # Embed the table body without the heading.
    tmp = out_dir / "_tmp_top.md"
    write_top_table(tmp, table_src, n=10, title="unused")
    body = tmp.read_text(encoding="utf-8").split("Ranked by", 1)[-1]
    tmp.unlink(missing_ok=True)
    lines.append("Ranked by" + body)
    lines.extend(
        [
            "## Why they were selected",
            "",
            "The score weights SCA Fig. 9 MAE, continued rise after T_k = 1.6 s (`late_frac`), the 2.0→3.0 s increment, "
            "monotonicity, K-means/Random levels at the paper’s quoted points, feasibility at 0.8 s and 3 s, method order, "
            "and whether AoDT still binds. Early flattening is penalised. Matching the already-calibrated 7.1 Mbps SCA "
            "point is not rewarded.",
            "",
            f"Coarse grid had {len(coarse_ranked)} pairs. Refinement used ~4× finer steps around the top {spec.refine_count} regions. "
            f"The shortlist re-ran the top {spec.shortlist_count} unique pairs with {len(spec.shortlist_seeds)} seeds.",
            "",
            "## Comparison against the paper",
            "",
        ]
    )
    if best is not None:
        lines.append(
            f"Best SCA curve: {_fmt(best.get('sca_mbps_0.8'))} Mbps at 0.8 s, "
            f"{_fmt(best.get('sca_mbps_2'))} at 2.0 s, {_fmt(best.get('sca_mbps_3'))} at 3.0 s "
            f"(paper ≈ 4.0, 6.3, 7.4). Rise {_fmt(best.get('rise_08_to_30'))} Mbps (paper ≈ 3.4). "
            f"Late fraction {_fmt(best.get('late_frac'))} (linear Fig. 9 ≈ 0.5). "
            f"K-means / Random at 3 s: {_fmt(best.get('kmeans_mbps_3'))} / {_fmt(best.get('random_mbps_3'))} "
            f"(paper 4.4 / 3.2). Feasible fraction at 0.8 s: {_fmt(best.get('feas_sca_0.8'))}."
        )
        lines.append("")
        if default is not None and best is not default:
            lines.append(
                f"The experimental default S_i=2000, L=2e6 has late_frac={_fmt(default.get('late_frac'))} "
                f"and rise={_fmt(default.get('rise_08_to_30'))} Mbps: the constraint goes slack well before 3 s "
                f"because D_i = 8·2000 / r is only tens of milliseconds at Mbps-scale rates, so AoDT is dominated by the queue term."
            )
            lines.append("")
    lines.extend(
        [
            "### TD3 on the shortlist",
            "",
            td3_note,
            "",
            "## Can any (S_i, L) pair reproduce the paper?",
            "",
            assumption,
            "",
        ]
    )
    if caveats:
        lines.append("Caveats:")
        lines.extend(f"- {c}" for c in caveats)
        lines.append("")
    lines.extend(
        [
            "## Undocumented assumptions",
            "",
            "This search only moved S_i and L. If the best pair still cannot keep Fig. 9 rising through 3 s **and** feasible at 0.8 s, "
            "the gap is not a missing Table II number in the 500–256000 byte × 1e5–1e7 cycle box. Candidates then include:",
            "",
            "1. The paper’s Eq. (11) `D_i = S_i / r_ij` with mixed units (bytes over bit/s) used as written, i.e. without ×8. "
            "This reconstruction keeps ×8 and did not switch units to chase a fit.",
            "2. A different L (or heterogeneous CPU) so the queue term sits near 0.8–3 s instead of ~0.5 s.",
            "3. Plotting raw (possibly infeasible) sum rates at T_k = 0.8 s.",
            "4. Bandwidth / association modelling differences already documented (per-link cap, SCA surrogate).",
            "",
            "μ = f_j / L and ρ are recorded in the pair CSVs. Defaults in `src/config.py` were left at 2000 bytes and 2e6 cycles.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_td3_shortlist(base: SimConfig, spec: SearchSpec, pairs: list[tuple[float, float]], args) -> list[dict]:
    from src.solvers.td3 import solve_td3, train_td3_across_scenarios

    rows = []
    jobs = Counter("TD3 shortlist", max(len(pairs) * len(spec.td3_thresholds), 1))
    for si, cycles in pairs:
        for tk in spec.td3_thresholds:
            cfg_t = _cfg_for_pair(base, si, cycles, tk, spec)
            extras = cpu_metrics(cfg_t)
            log.info("Training TD3  S_i=%g L=%g Tk=%g steps=%d", si, cycles, tk, args.td3_steps)
            agent, _env, _log = train_td3_across_scenarios(
                cfg_t,
                TD3_TRAIN_SEEDS,
                seed=0,
                n_uav=spec.n_uav,
                total_steps=args.td3_steps,
            )
            for seed in spec.shortlist_seeds:
                scenario = generate_scenario(seed, cfg_t)
                _xy, result, rt, _train_log = solve_td3(
                    scenario, seed=seed, n_uav=spec.n_uav, total_steps=args.td3_steps, agent=agent
                )
                rows.append(_raw_row(seed, "td3", result, rt, si, cycles, tk, extras["mu"], extras["queue_term"]))
            jobs.tick(f"S_i={si:g} L={cycles:g} Tk={tk:g}")
    return rows


def run_aodt_parameter_search(cfg: SimConfig, args, spec: SearchSpec | None = None) -> dict:
    spec = spec or SearchSpec(skip_td3=bool(getattr(args, "skip_td3", False)))
    if abs(cfg.b_sys - CALIBRATED_B_SYS) > 1.0:
        raise SystemExit(
            f"aodt-param-search must keep the calibrated radio (B_sys={CALIBRATED_B_SYS:g} Hz); got {cfg.b_sys:g}"
        )
    out_dir = Path(getattr(args, "out", "results/aodt_parameter_search"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = getattr(args, "aodt_search_stage", "all")
    resume = bool(getattr(args, "resume", False))

    banner("AoDT (S_i, L) parameter search")
    log.info("  out       %s", out_dir)
    log.info("  radio     B_sys=%g Hz (unchanged)  noise=%g", cfg.b_sys, cfg.noise_power)
    log.info("  units     D_i = 8*S_i/r_ij  (current implementation, not modified)")
    log.info("  defaults  S_i=%g L=%g  (not written back to config)", EXPERIMENTAL_TASK_SIZE_BYTES, EXPERIMENTAL_TASK_CYCLES)

    coarse_raw_path = out_dir / "coarse_raw.csv"
    refine_raw_path = out_dir / "refine_raw.csv"
    shortlist_raw_path = out_dir / "shortlist_raw.csv"
    td3_raw_path = out_dir / "td3_raw.csv"

    coarse_raw: list[dict] = _load_csv(coarse_raw_path) if resume else []
    refine_raw: list[dict] = _load_csv(refine_raw_path) if resume else []
    shortlist_raw: list[dict] = _load_csv(shortlist_raw_path) if resume else []
    td3_raw: list[dict] = _load_csv(td3_raw_path) if resume else []
    coarse_ranked: list[dict] = []
    refine_ranked: list[dict] = []
    shortlist_ranked: list[dict] = []

    if stage in ("all", "coarse"):
        pairs = coarse_pairs(spec)
        coarse_raw = run_pairs(
            cfg, spec, pairs, spec.seeds, args, "coarse (S_i, L)", existing_raw=coarse_raw if resume else None
        )
        _write_csv(coarse_raw_path, coarse_raw)
        coarse_ranked = rank_pairs(coarse_raw)
        _write_csv(out_dir / "coarse_pairs.csv", coarse_ranked)
        write_top_table(out_dir / "table_coarse_top10.md", coarse_ranked, n=10, title="Coarse search — top 10")
        _print_top(coarse_ranked, 10, "coarse top 10")

    if not coarse_ranked and coarse_raw_path.exists():
        coarse_ranked = rank_pairs(_load_csv(coarse_raw_path))

    if stage in ("all", "refine"):
        local = refine_pairs(coarse_ranked, n=spec.refine_count)
        already = {pair_key(s, L) for s, L in coarse_pairs(spec)}
        new_local = [p for p in local if pair_key(*p) not in already]
        log.info("refine grid %d pairs (%d new, %d overlap with coarse)", len(local), len(new_local), len(local) - len(new_local))
        refine_raw = run_pairs(
            cfg, spec, new_local, spec.seeds, args, "refine (S_i, L)", existing_raw=refine_raw if resume else None
        )
        _write_csv(refine_raw_path, refine_raw)
        combined = list(_load_csv(coarse_raw_path)) + list(refine_raw)
        refine_ranked = rank_pairs(combined)
        _write_csv(out_dir / "refine_pairs.csv", refine_ranked)
        write_top_table(out_dir / "table_refine_top10.md", refine_ranked, n=10, title="After refinement — top 10")
        _print_top(refine_ranked, 10, "refine top 10")

    if not refine_ranked:
        if (out_dir / "refine_pairs.csv").exists():
            refine_ranked = _load_csv(out_dir / "refine_pairs.csv")
        else:
            refine_ranked = coarse_ranked

    if stage in ("all", "shortlist"):
        top = refine_ranked[: spec.shortlist_count]
        pairs = [(r["task_size_bytes"], r["task_cycles"]) for r in top]
        shortlist_raw = run_pairs(
            cfg,
            spec,
            pairs,
            spec.shortlist_seeds,
            args,
            "shortlist 20-seed",
            existing_raw=shortlist_raw if resume else None,
        )
        _write_csv(shortlist_raw_path, shortlist_raw)
        shortlist_ranked = rank_pairs(shortlist_raw)
        _write_csv(out_dir / "shortlist_pairs.csv", shortlist_ranked)
        write_top_table(out_dir / "table_shortlist.md", shortlist_ranked, n=len(shortlist_ranked), title="20-seed shortlist")
        _print_top(shortlist_ranked, 10, "20-seed shortlist")
        plot_top_curves(out_dir, shortlist_ranked, n=min(5, len(shortlist_ranked)))

    if not shortlist_ranked and shortlist_raw_path.exists():
        shortlist_ranked = rank_pairs(_load_csv(shortlist_raw_path))

    if stage in ("all", "td3") and not spec.skip_td3 and not getattr(args, "skip_td3", False):
        source = shortlist_ranked or refine_ranked or coarse_ranked
        td3_pairs = [(r["task_size_bytes"], r["task_cycles"]) for r in source[: spec.td3_top_n]]
        td3_raw = run_td3_shortlist(cfg, spec, td3_pairs, args)
        _write_csv(td3_raw_path, td3_raw)

    ranking_src = shortlist_ranked or refine_ranked or coarse_ranked
    _write_csv(out_dir / "ranking_all.csv", ranking_src)
    write_top_table(out_dir / "table_top10.md", ranking_src, n=10, title="Top 10 (S_i, L) candidates")
    if ranking_src and not (out_dir / "fig9_top_candidates.png").exists():
        plot_top_curves(out_dir, ranking_src, n=min(5, len(ranking_src)))

    write_report(out_dir, spec, coarse_ranked, refine_ranked, shortlist_ranked, td3_raw)
    meta = {
        "radio": {
            "b_sys_hz": cfg.b_sys,
            "noise_power_w": cfg.noise_power,
            "bandwidth_scope": cfg.bandwidth_scope,
            "max_bw_share": cfg.max_bw_share,
            "note": "calibrated profile held fixed; only S_i and L vary",
        },
        "units": "D_i = 8 * S_i / r_ij with S_i in bytes (current implementation)",
        "defaults_unchanged": {
            "EXPERIMENTAL_TASK_SIZE_BYTES": EXPERIMENTAL_TASK_SIZE_BYTES,
            "EXPERIMENTAL_TASK_CYCLES": EXPERIMENTAL_TASK_CYCLES,
        },
        "coarse_task_sizes": list(spec.task_sizes),
        "coarse_task_cycles": list(spec.task_cycles),
        "thresholds": list(spec.thresholds),
        "coarse_seeds": list(spec.seeds),
        "shortlist_seeds": list(spec.shortlist_seeds),
        "methods": list(spec.methods),
        "skip_td3": spec.skip_td3 or bool(getattr(args, "skip_td3", False)),
        "paper_fig9_mbps": PAPER_FIG9_MBPS,
        "best": ranking_src[0] if ranking_src else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("wrote AoDT parameter search under %s", out_dir)
    if ranking_src:
        b = ranking_src[0]
        log.info("Best candidate: S_i = %g, L = %g  (score %.3f)", b["task_size_bytes"], b["task_cycles"], b["score"])
    return {
        "out": str(out_dir),
        "coarse": coarse_ranked,
        "refine": refine_ranked,
        "shortlist": shortlist_ranked,
        "td3": td3_raw,
    }
