"""Final Fig. 9 validation with a frozen configuration.

Does not change production solvers, AoDT equations, S_i/L defaults, or B_sys.
Overrides are applied only on the SimConfig instance used for this run.

    python scripts/run_fig9_final_validation.py
    python scripts/run_fig9_final_validation.py --report-only
    python scripts/run_fig9_final_validation.py --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import (
    DEFAULT,
    PAPER_SCENARIO_SEEDS,
    PSO_N_ITER,
    PSO_N_PARTICLES,
    TD3_TOTAL_STEPS,
)
from src.experiments.sweeps import _row, _solve, _summarize, _write_csv, expand_positions
from src.logutil import Counter, banner, configure_logging, log
from src.scenario import generate_scenario
from src.solvers.td3 import set_default_device

# ---------------------------------------------------------------------------
# Locked before the solver loop. Do not edit after seeing results.
# ---------------------------------------------------------------------------

THRESHOLDS = (0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0)
USABLE_THRESHOLDS = (1.2, 1.6, 2.0, 2.4, 2.8, 3.0)
INFEASIBLE_TK = 0.8
ABS_ERR_LIMIT_MBPS = 0.5
TREND_TOL_MBPS = 0.05
MIN_POINTS_WITHIN = 4
TASK_SIZE_BYTES = 12000.0
TASK_CYCLES = 3.75e6
METHODS = ("random", "kmeans", "sca", "td3")
OUT_DEFAULT = Path("results") / "run_20260831" / "fig9_final_validation"

# IEEE Fig. 9 approximate curve (Mbps). See CRITERION.md.
PAPER_FIG9_APPROX_MBPS: dict[float, dict[str, float]] = {
    0.8: {"sca": 4.0, "td3": 3.2, "kmeans": 1.8, "random": 0.8},
    1.2: {"sca": 5.0, "td3": 4.0, "kmeans": 2.4, "random": 1.3},
    1.6: {"sca": 5.7, "td3": 4.6, "kmeans": 2.8, "random": 1.8},
    2.0: {"sca": 6.3, "td3": 5.0, "kmeans": 3.3, "random": 2.2},
    2.4: {"sca": 6.8, "td3": 5.5, "kmeans": 3.8, "random": 2.7},
    2.8: {"sca": 7.2, "td3": 5.8, "kmeans": 4.2, "random": 3.0},
    3.0: {"sca": 7.4, "td3": 6.0, "kmeans": 4.4, "random": 3.2},
}

CRITERION_MD = Path(__file__).resolve().parents[1] / OUT_DEFAULT / "CRITERION.md"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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


def _fmt(x: float, digits: int = 3) -> str:
    if x != x:  # NaN
        return "—"
    return f"{x:.{digits}f}"


def _nondecreasing(values: list[float], tol: float) -> bool:
    if any(v != v for v in values):
        return False
    return all(values[i + 1] + tol >= values[i] for i in range(len(values) - 1))


def done_keys(rows: list[dict]) -> set[tuple[int, str, float]]:
    return {
        (int(r["seed"]), str(r["method"]), float(r["aodt_threshold"]))
        for r in rows
    }


def compare_rows(summary: list[dict]) -> list[dict]:
    out = []
    for tk in THRESHOLDS:
        for method in METHODS:
            impl = _mbps_at(summary, method, tk)
            paper = PAPER_FIG9_APPROX_MBPS[tk][method]
            err = abs(impl - paper) if impl == impl else float("nan")
            out.append(
                {
                    "aodt_threshold": tk,
                    "method": method,
                    "impl_mbps": impl,
                    "paper_mbps": paper,
                    "abs_error_mbps": err,
                    "within_0p5": int(err == err and err <= ABS_ERR_LIMIT_MBPS),
                    "usable_for_gate": int(tk in USABLE_THRESHOLDS),
                    "feasibility_frac": _field_at(summary, method, tk, "feasible_frac"),
                    "qos_mean": _field_at(summary, method, tk, "qos_mean"),
                    "aodt_viol_mean": _field_at(summary, method, tk, "aodt_viol_mean"),
                    "cpu_unstable_mean": _field_at(summary, method, tk, "cpu_unstable_mean"),
                    "aodt_mean": _field_at(summary, method, tk, "aodt_mean"),
                    "sum_rate_std_mbps": _field_at(summary, method, tk, "sum_rate_std") / 1e6,
                    "n": _field_at(summary, method, tk, "n"),
                }
            )
    return out


def evaluate_gate(cmp_rows: list[dict], summary: list[dict]) -> dict:
    usable = [r for r in cmp_rows if r["usable_for_gate"]]
    sca = [r for r in usable if r["method"] == "sca"]
    td3 = [r for r in usable if r["method"] == "td3"]
    sca_n = int(sum(r["within_0p5"] for r in sca))
    td3_n = int(sum(r["within_0p5"] for r in td3))
    sca_mae = float(np.mean([r["abs_error_mbps"] for r in sca]))
    td3_mae = float(np.mean([r["abs_error_mbps"] for r in td3]))
    sca_curve = [_mbps_at(summary, "sca", tk) for tk in USABLE_THRESHOLDS]
    td3_curve = [_mbps_at(summary, "td3", tk) for tk in USABLE_THRESHOLDS]
    trend_ok = _nondecreasing(sca_curve, TREND_TOL_MBPS) and _nondecreasing(
        td3_curve, TREND_TOL_MBPS
    )
    at3 = {m: _mbps_at(summary, m, 3.0) for m in METHODS}
    order_ok = (
        at3["sca"] + 1e-9 >= at3["td3"]
        and at3["td3"] + 1e-9 >= at3["kmeans"]
        and at3["kmeans"] + 1e-9 >= at3["random"]
    )
    passed = (
        sca_n >= MIN_POINTS_WITHIN
        and td3_n >= MIN_POINTS_WITHIN
        and trend_ok
        and order_ok
    )
    return {
        "sca_n_within": sca_n,
        "td3_n_within": td3_n,
        "sca_mae": sca_mae,
        "td3_mae": td3_mae,
        "combined_mae": float(np.mean([sca_mae, td3_mae])),
        "trend_ok": trend_ok,
        "order_ok": order_ok,
        "pass": passed,
        "verdict": "PASS" if passed else "FAIL",
        "at3": at3,
        "sca_curve": sca_curve,
        "td3_curve": td3_curve,
    }


def plot_vs_paper(out_dir: Path, summary: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib missing; skip plot")
        return

    colors = {"sca": "#1f77b4", "td3": "#ffbf00", "kmeans": "#d62728", "random": "#2ca02c"}
    markers = {"sca": "*", "td3": "D", "kmeans": "o", "random": "s"}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    paper_x = list(THRESHOLDS)
    for method in METHODS:
        ax.plot(
            paper_x,
            [PAPER_FIG9_APPROX_MBPS[t][method] for t in paper_x],
            color=colors[method],
            ls="--",
            marker=markers[method],
            alpha=0.55,
            label=f"paper {method}",
        )
    for method in METHODS:
        sub = [r for r in summary if r["method"] == method]
        xs = np.array([float(r["aodt_threshold"]) for r in sub])
        ys = np.array([float(r["sum_rate_mean"]) / 1e6 for r in sub])
        es = np.array([float(r["sum_rate_std"]) / 1e6 for r in sub])
        order = np.argsort(xs)
        ax.errorbar(
            xs[order],
            ys[order],
            yerr=es[order],
            color=colors[method],
            marker=markers[method],
            ls="-",
            label=f"ours {method}",
        )
    ax.axvline(INFEASIBLE_TK, color="0.4", ls=":", lw=1.0)
    ax.annotate(
        "0.8 s: infeasible under\nour implemented constraints",
        xy=(INFEASIBLE_TK, 0.35),
        xytext=(1.05, 0.6),
        fontsize=8,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )
    ax.set_xlabel("AoDT threshold T_k (s)")
    ax.set_ylabel("Sum rate (Mbps)")
    ax.set_title("Fig. 9 final validation (frozen config vs IEEE curve)")
    ax.set_xlim(0.6, 3.2)
    ax.set_ylim(0.0, 9.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_vs_paper.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for method in METHODS:
        sub = [r for r in summary if r["method"] == method]
        xs = np.array([float(r["aodt_threshold"]) for r in sub])
        ys = np.array([float(r["feasible_frac"]) for r in sub])
        order = np.argsort(xs)
        ax.plot(
            xs[order],
            ys[order],
            color=colors[method],
            marker=markers[method],
            label=method,
        )
    ax.axvline(INFEASIBLE_TK, color="0.4", ls=":", lw=1.0)
    ax.set_xlabel("AoDT threshold T_k (s)")
    ax.set_ylabel("Feasibility fraction")
    ax.set_title("Feasibility vs T_k (20 seeds)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_feasibility.png", dpi=140)
    plt.close(fig)


def _results_table(summary: list[dict]) -> list[str]:
    lines = [
        "| T_k (s) | Method | Sum rate (Mbps) | Std | Feasible | QoS viol. | AoDT viol. | CPU unstable | AoDT mean (s) | Runtime (s) |",
        "|--------:|--------|----------------:|----:|---------:|----------:|-----------:|-------------:|--------------:|------------:|",
    ]
    for tk in THRESHOLDS:
        for method in METHODS:
            note = " *(infeasible under our constraints)*" if tk == INFEASIBLE_TK else ""
            lines.append(
                "| {tk:g}{note} | {m} | {sr} | {sd} | {fe} | {qos} | {av} | {cpu} | {am} | {rt} |".format(
                    tk=tk,
                    note=note,
                    m=method,
                    sr=_fmt(_mbps_at(summary, method, tk)),
                    sd=_fmt(_field_at(summary, method, tk, "sum_rate_std") / 1e6),
                    fe=_fmt(_field_at(summary, method, tk, "feasible_frac"), 2),
                    qos=_fmt(_field_at(summary, method, tk, "qos_mean"), 2),
                    av=_fmt(_field_at(summary, method, tk, "aodt_viol_mean"), 2),
                    cpu=_fmt(_field_at(summary, method, tk, "cpu_unstable_mean"), 2),
                    am=_fmt(_field_at(summary, method, tk, "aodt_mean")),
                    rt=_fmt(_field_at(summary, method, tk, "runtime_mean"), 2),
                )
            )
    return lines


def write_report(
    out_dir: Path,
    cfg,
    seeds: tuple[int, ...],
    summary: list[dict],
    cmp_rows: list[dict],
    gate: dict,
    meta: dict,
) -> None:
    sca_pts = [r for r in cmp_rows if r["method"] == "sca" and r["usable_for_gate"]]
    td3_pts = [r for r in cmp_rows if r["method"] == "td3" and r["usable_for_gate"]]
    verdict = gate["verdict"]
    if gate["pass"]:
        conclusion = (
            "reasonably validated, with documented known gaps. "
            "Parameter tuning on this Fig. 9 axis **stops** here."
        )
        fail_next = ""
    else:
        conclusion = (
            "not yet validated against the predefined criterion. "
            "S_i and L are **not** retuned. The remaining discrepancy is treated "
            "as evidence for a possible undocumented assumption in the paper or "
            "in this reconstruction."
        )
        fail_next = "\n".join(
            [
                "",
                "## Next investigation (FAIL only; no S_i / L retune)",
                "",
                "Most likely areas, in order, without changing production equations:",
                "",
                "1. **Forwarding delay T_u2u = 0.3 s** — whether the paper's Fig. 9",
                "   placements keep association = processing so the hop is rarely charged.",
                "2. **What Fig. 9 plots at infeasible points** — raw associated sum rate",
                "   vs a repaired / penalized objective (the 0.8 s endpoint).",
                "3. **Per-link bandwidth cap (0.25)** — undocumented in Problem (P); it",
                "   changes how leftover spectrum is spent as T_k relaxes.",
                "4. **SCA surrogate vs CVX+MOSEK** — this repo uses a trust-region",
                "   numerical SCA, not the paper's convexified (P).",
                "5. **Upload-delay units** — D_i = 8 S_i / r_ij (bytes to bits) vs a",
                "   reading that treats S_i as already in bits.",
                "",
                "Do not search new S_i or L values, change B_sys, drop forwarding,",
                "or edit production AoDT / QoS / association to chase the curve.",
            ]
        )

    lines = [
        "# Fig. 9 final validation report",
        "",
        f"**Verdict: {verdict}** — {conclusion}",
        "",
        "This is a validation experiment, not a tuning experiment. Production",
        "AoDT equations, radio, repair, and solver code were not modified for this run.",
        "",
        "## 1. Frozen configuration",
        "",
        "Locked for this run (instance overrides only; project defaults untouched):",
        "",
        f"- Radio profile: calibrated (`B_sys = {cfg.b_sys:.1e}` Hz, "
        f"`noise_power = {cfg.noise_power}` W, `max_bw_share = {cfg.max_bw_share}`)",
        f"- `S_i` = {cfg.task_size_bytes:g} bytes",
        f"- `L` = {cfg.task_cycles:.4g} CPU cycles",
        f"- AoDT equations: production `src/aodt.py` (unchanged)",
        f"- Forwarding: `T_u2u = {cfg.t_u2u}` s (unchanged)",
        f"- QoS: `R_min = {cfg.r_min:g}` bit/s (unchanged)",
        f"- Arrival / CPU: `λ_i = {cfg.lambda_i}`, `f_j = {cfg.uav_cpu:.3g}` cycles/s",
        f"- Scene: I = {cfg.num_iot}, J = {cfg.num_uav}, K = {cfg.num_processes}, "
        f"area {cfg.area_x:.0f}×{cfg.area_y:.0f} m², UAV height {cfg.uav_height:.0f} m",
        "- Bandwidth policy:",
        "  - Random: **equal split** of each pool",
        "  - K-means: **equal split** of each pool",
        "  - SCA: production constrained LP (`allocate_constrained_bandwidth`)",
        "  - TD3: production constrained LP via `complete_solution` (default)",
        "- Association / processing / queueing / repair: production, unchanged",
        "",
        f"Git HEAD at launch: `{meta.get('git_head', 'unknown')}`",
        "",
        "## 2. Exact experiment settings",
        "",
        f"- Seeds: {len(seeds)} paper-style seeds `{seeds[0]}…{seeds[-1]}`",
        f"- T_k grid: {list(THRESHOLDS)} s",
        f"- Methods: {', '.join(METHODS)}",
        f"- TD3: {TD3_TOTAL_STEPS} steps per (seed, T_k), greedy evaluation, device auto",
        f"- SCA: production trust-region SCA (not MATLAB CVX+MOSEK)",
        f"- PSO particles/iters (unused here): {PSO_N_PARTICLES}×{PSO_N_ITER}",
        "- Methodology matches `src/experiments/sweeps.py` `sweep_aodt` "
        "(per-cell retrain, shared evaluator), restricted to Fig. 9 methods",
        f"- Output directory: `{out_dir.as_posix()}`",
        f"- Wall time: {meta.get('wall_s', 'n/a')} s",
        "",
        "Pre-registered criterion: `CRITERION.md` in this directory.",
        "",
        "## 3. Results table (all 7 T_k values)",
        "",
        "T_k = 0.8 s is labelled infeasible under our implemented constraints.",
        "The raw sum-rate column is still the associated Shannon sum, as plotted",
        "by the paper at that point.",
        "",
    ]
    lines.extend(_results_table(summary))
    lines.extend(
        [
            "",
            "Raw per-seed rows: `raw_sumrate_vs_aodt.csv`. Seed means: `sumrate_vs_aodt.csv`.",
            "",
            "## 4. Paper-vs-implementation (six usable points)",
            "",
            "Usable T_k = 1.2, 1.6, 2.0, 2.4, 2.8, 3.0 s. Paper values from `CRITERION.md`.",
            "",
            "| T_k (s) | Method | Ours (Mbps) | Paper (Mbps) | Abs. error (Mbps) | ≤ 0.5 Mbps |",
            "|--------:|--------|------------:|-------------:|------------------:|:----------:|",
        ]
    )
    for r in cmp_rows:
        if not r["usable_for_gate"]:
            continue
        flag = "yes" if r["within_0p5"] else "no"
        lines.append(
            f"| {r['aodt_threshold']:g} | {r['method']} | {_fmt(r['impl_mbps'])} | "
            f"{_fmt(r['paper_mbps'], 1)} | {_fmt(r['abs_error_mbps'])} | {flag} |"
        )
    lines.extend(
        [
            "",
            "## 5. Absolute error for each usable point",
            "",
            "SCA:",
            "",
        ]
    )
    for r in sca_pts:
        lines.append(
            f"- T_k = {r['aodt_threshold']:g} s: |{_fmt(r['impl_mbps'])} − {r['paper_mbps']:.1f}| "
            f"= **{_fmt(r['abs_error_mbps'])} Mbps**"
        )
    lines.extend(["", "TD3:", ""])
    for r in td3_pts:
        lines.append(
            f"- T_k = {r['aodt_threshold']:g} s: |{_fmt(r['impl_mbps'])} − {r['paper_mbps']:.1f}| "
            f"= **{_fmt(r['abs_error_mbps'])} Mbps**"
        )
    lines.extend(
        [
            "",
            "## 6. MAE (six usable points)",
            "",
            f"- SCA MAE = **{_fmt(gate['sca_mae'])} Mbps**",
            f"- TD3 MAE = **{_fmt(gate['td3_mae'])} Mbps**",
            f"- Mean of SCA and TD3 MAE = {_fmt(gate['combined_mae'])} Mbps",
            "",
            "Random / K-means MAE is not a gate (equal-split limitation).",
            "",
            "## 7. Points within the 0.5 Mbps threshold",
            "",
            f"- SCA: **{gate['sca_n_within']} / 6**",
            f"- TD3: **{gate['td3_n_within']} / 6**",
            f"- Required for PASS: at least {MIN_POINTS_WITHIN} / 6 for **each** of SCA and TD3",
            "",
            f"Trend (non-decreasing, {TREND_TOL_MBPS} Mbps tolerance): "
            f"{'yes' if gate['trend_ok'] else 'no'}",
            "",
            "Ordering at T_k = 3.0 s (SCA ≥ TD3 ≥ K-means ≥ Random): "
            f"{'yes' if gate['order_ok'] else 'no'} "
            f"(SCA {_fmt(gate['at3']['sca'])}, TD3 {_fmt(gate['at3']['td3'])}, "
            f"K-means {_fmt(gate['at3']['kmeans'])}, Random {_fmt(gate['at3']['random'])})",
            "",
            "## 8. T_k = 0.8 s infeasibility",
            "",
            "**0.8 s: infeasible under our implemented constraints.**",
            "",
            "This point is excluded from the PASS/FAIL gate. Constraints, forwarding,",
            "queueing, S_i, L, and AoDT equations were not altered to force feasibility.",
            "The paper still draws a marker at 0.8 s; we therefore report the raw",
            "associated sum rate below, together with violation counts.",
            "",
            "| Method | Raw sum rate (Mbps) | Paper (Mbps) | Feasible | QoS viol. | AoDT viol. |",
            "|--------|--------------------:|-------------:|---------:|----------:|-----------:|",
        ]
    )
    for method in METHODS:
        r = next(
            x
            for x in cmp_rows
            if x["method"] == method and abs(x["aodt_threshold"] - INFEASIBLE_TK) < 1e-9
        )
        lines.append(
            f"| {method} | {_fmt(r['impl_mbps'])} | {_fmt(r['paper_mbps'], 1)} | "
            f"{_fmt(r['feasibility_frac'], 2)} | {_fmt(r['qos_mean'], 2)} | "
            f"{_fmt(r['aodt_viol_mean'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## 9. PASS / FAIL against the predefined criterion",
            "",
            f"**{verdict}**",
            "",
            "Gate checklist (locked in `CRITERION.md`):",
            "",
            f"1. SCA ≥ {MIN_POINTS_WITHIN}/6 within 0.5 Mbps: "
            f"{'yes' if gate['sca_n_within'] >= MIN_POINTS_WITHIN else 'no'} "
            f"({gate['sca_n_within']}/6)",
            f"2. TD3 ≥ {MIN_POINTS_WITHIN}/6 within 0.5 Mbps: "
            f"{'yes' if gate['td3_n_within'] >= MIN_POINTS_WITHIN else 'no'} "
            f"({gate['td3_n_within']}/6)",
            f"3. SCA and TD3 non-decreasing on the six points: "
            f"{'yes' if gate['trend_ok'] else 'no'}",
            f"4. Ordering at 3.0 s: {'yes' if gate['order_ok'] else 'no'}",
            "",
            "## 10–11. Conclusion",
            "",
            conclusion,
            "",
            "Known gaps that remain even if the gate passes:",
            "",
            "- T_k = 0.8 s is infeasible here and is not claimed as a feasible operating point.",
            "- Random / K-means equal splitting is less T_k-responsive than the paper curves.",
            "- Paper y-values are figure reads (±0.2 Mbps), not tabulated numbers.",
            "- SCA is a Python trust-region surrogate, not CVX+MOSEK on a convexified (P).",
            "- S_i and L are not in Table II; this run freezes the current experimental pair.",
            "",
            "Comparison plot: `fig9_vs_paper.png`. Feasibility: `fig9_feasibility.png`.",
        ]
    )
    if fail_next:
        lines.append(fail_next)
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", out_dir / "REPORT.md")


def run_sweep(cfg, args_ns: Namespace, out_dir: Path, resume: bool) -> list[dict]:
    seeds = PAPER_SCENARIO_SEEDS
    raw_path = out_dir / "raw_sumrate_vs_aodt.csv"
    rows: list[dict] = []
    if resume and raw_path.exists():
        import csv

        with raw_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["seed"] = int(r["seed"])
            r["sum_rate"] = float(r["sum_rate"])
            r["qos"] = float(r["qos"])
            r["feasible"] = int(float(r["feasible"]))
            r["aodt_viol"] = float(r["aodt_viol"])
            r["aodt_excess"] = float(r["aodt_excess"]) if r.get("aodt_excess") not in ("", None) else 0.0
            r["cpu_unstable"] = float(r["cpu_unstable"])
            r["runtime"] = float(r["runtime"])
            r["aodt_mean"] = None if r.get("aodt_mean") in ("", None, "None") else float(r["aodt_mean"])
            r["aodt_threshold"] = float(r["aodt_threshold"])
        log.info("resume: %d rows already in %s", len(rows), raw_path)
    seen = done_keys(rows)
    total = len(THRESHOLDS) * len(seeds) * len(METHODS)
    remaining = total - len(seen)
    jobs = Counter("Fig.9 final validation", max(remaining, 1))
    if remaining <= 0:
        log.info("all %d jobs already present", total)
        return rows
    for tk in THRESHOLDS:
        cfg_t = replace(cfg, aodt_threshold=tk)
        for seed in seeds:
            scenario = generate_scenario(seed, cfg_t)
            for name in METHODS:
                key = (seed, name, float(tk))
                if key in seen:
                    continue
                _xy, result, rt = _solve(name, scenario, seed, args_ns)
                rows.append(
                    _row(seed, name, result, rt, _xy, cfg_t.uav_height, aodt_threshold=tk)
                )
                jobs.tick(f"Tk={tk:g}s  seed={seed}  {name:8}", result, rt)
                _write_csv(raw_path, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Final Fig. 9 validation (frozen config)")
    p.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--report-only", action="store_true", help="Rebuild REPORT.md from existing CSVs")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--td3-steps", type=int, default=TD3_TOTAL_STEPS)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, log_file=str(out_dir / "run.log"))
    set_default_device(args.device)

    cfg = DEFAULT.with_compute(task_size_bytes=TASK_SIZE_BYTES, task_cycles=TASK_CYCLES)
    if abs(cfg.b_sys - 8.8e6) > 1.0:
        raise SystemExit(f"B_sys is {cfg.b_sys}, expected 8.8e6 (calibrated profile)")
    if cfg.task_size_bytes != TASK_SIZE_BYTES or cfg.task_cycles != TASK_CYCLES:
        raise SystemExit("frozen S_i / L did not attach to the config instance")

    args_ns = Namespace(
        particles=PSO_N_PARTICLES,
        iters=PSO_N_ITER,
        td3_steps=args.td3_steps,
        with_td3=True,
        paper_runs=True,
        device=args.device,
    )

    banner("Fig. 9 final validation (frozen)")
    log.info("criterion file: %s", CRITERION_MD)
    log.info("S_i=%g  L=%g  B_sys=%g  methods=%s  seeds=%d",
             cfg.task_size_bytes, cfg.task_cycles, cfg.b_sys, METHODS, len(PAPER_SCENARIO_SEEDS))

    t0 = time.perf_counter()
    if not args.report_only:
        rows = run_sweep(cfg, args_ns, out_dir, resume=args.resume or (out_dir / "raw_sumrate_vs_aodt.csv").exists())
    else:
        import csv

        raw_path = out_dir / "raw_sumrate_vs_aodt.csv"
        if not raw_path.exists():
            raise SystemExit(f"--report-only needs {raw_path}")
        with raw_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["seed"] = int(r["seed"])
            r["sum_rate"] = float(r["sum_rate"])
            r["qos"] = float(r["qos"])
            r["feasible"] = int(float(r["feasible"]))
            r["aodt_viol"] = float(r["aodt_viol"])
            r["cpu_unstable"] = float(r["cpu_unstable"])
            r["runtime"] = float(r["runtime"])
            r["aodt_mean"] = None if r.get("aodt_mean") in ("", None, "None") else float(r["aodt_mean"])
            r["aodt_threshold"] = float(r["aodt_threshold"])

    wall = time.perf_counter() - t0
    summary = _summarize(rows, "aodt_threshold")
    _write_csv(out_dir / "sumrate_vs_aodt.csv", summary)
    _write_csv(out_dir / "positions_sumrate_vs_aodt.csv", expand_positions(rows))
    cmp_rows = compare_rows(summary)
    _write_csv(out_dir / "comparison_vs_paper.csv", cmp_rows)
    plot_vs_paper(out_dir, summary)
    gate = evaluate_gate(cmp_rows, summary)
    meta = {
        "note": "Final Fig. 9 validation. Frozen config. Production code not modified.",
        "git_head": _git_head(),
        "b_sys": cfg.b_sys,
        "task_size_bytes": cfg.task_size_bytes,
        "task_cycles": cfg.task_cycles,
        "noise_power": cfg.noise_power,
        "max_bw_share": cfg.max_bw_share,
        "t_u2u": cfg.t_u2u,
        "lambda_i": cfg.lambda_i,
        "uav_cpu": cfg.uav_cpu,
        "r_min": cfg.r_min,
        "n_seeds": len(PAPER_SCENARIO_SEEDS),
        "seeds": list(PAPER_SCENARIO_SEEDS),
        "thresholds": list(THRESHOLDS),
        "methods": list(METHODS),
        "td3_steps": args.td3_steps,
        "abs_err_limit_mbps": ABS_ERR_LIMIT_MBPS,
        "paper_fig9_approx_mbps": PAPER_FIG9_APPROX_MBPS,
        "bandwidth_policy": {
            "random": "equal_split",
            "kmeans": "equal_split",
            "sca": "constrained_lp",
            "td3": "constrained_lp",
        },
        "gate": {k: v for k, v in gate.items() if k not in ("at3", "sca_curve", "td3_curve")},
        "wall_s": wall if not args.report_only else None,
        "verdict": gate["verdict"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(out_dir, cfg, PAPER_SCENARIO_SEEDS, summary, cmp_rows, gate, meta)
    log.info("verdict %s  SCA %d/6  TD3 %d/6  MAE_sca=%.3f  MAE_td3=%.3f",
             gate["verdict"], gate["sca_n_within"], gate["td3_n_within"],
             gate["sca_mae"], gate["td3_mae"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
