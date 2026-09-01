"""Test whether baseline vs paper gaps are a bandwidth-sharing rule, not S_i / L.

Random and K-means split each pool equally (no LP). SCA keeps the constrained
LP. A second SCA variant drops the undocumented 25% per-link cap. Surplus
after floors goes to the current AoDT bottleneck first (production LP).
"""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import DEV_SCENARIO_SEEDS, PAPER_SCENARIO_SEEDS, SimConfig
from src.experiments.aodt_parameter_search import FIG9_THRESHOLDS, PAPER_FIG9_MBPS
from src.experiments.sweeps import _row, _summarize, _write_csv
from src.logutil import Counter, banner, log
from src.evaluator import evaluate
from src.repair import (
    allocate_constrained_bandwidth,
    complete_solution,
    equal_bandwidth,
)
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca
from src.solvers.td3 import solve_td3


def _seeds(args) -> tuple[int, ...]:
    if getattr(args, "paper_runs", False):
        return PAPER_SCENARIO_SEEDS
    return DEV_SCENARIO_SEEDS


def _methods(args) -> tuple[str, ...]:
    names = ["random", "kmeans", "random_lp", "kmeans_lp", "sca", "sca_nocap"]
    if getattr(args, "with_td3", False):
        names = names + ["td3"]
    return tuple(names)


def _solve(name: str, scenario, seed: int, args):
    if name == "random":
        return solve_random(scenario, seed=seed)
    if name == "kmeans":
        return solve_kmeans(scenario, seed=seed)
    if name == "random_lp":
        return solve_random(scenario, seed=seed, equal_split=False)
    if name == "kmeans_lp":
        return solve_kmeans(scenario, seed=seed, equal_split=False)
    if name == "sca":
        return solve_sca(scenario, seed=seed)
    if name == "sca_nocap":
        sc = generate_scenario(seed, replace(scenario.cfg, max_bw_share=None))
        return solve_sca(sc, seed=seed)
    if name == "td3":
        xy, result, rt, _ = solve_td3(
            scenario, seed=seed, total_steps=args.td3_steps
        )
        return xy, result, rt
    raise ValueError(name)


def _mbps_at(summary: list[dict], method: str, tk: float) -> float:
    for r in summary:
        if r["method"] == method and abs(float(r["aodt_threshold"]) - tk) < 1e-9:
            return float(r["sum_rate_mean"]) / 1e6
    return float("nan")


def _aodt_at(summary: list[dict], method: str, tk: float) -> float:
    for r in summary:
        if r["method"] == method and abs(float(r["aodt_threshold"]) - tk) < 1e-9:
            v = r.get("aodt_mean")
            return float(v) if v is not None else float("nan")
    return float("nan")


def frozen_aodt_first_delta(cfg: SimConfig, seed: int = 100) -> dict:
    """Same geometry: SE-only surplus vs AoDT-first surplus (no placement search)."""
    s = generate_scenario(seed, cfg)
    xy = np.array([[120.0, 130.0], [380.0, 200.0], [250.0, 400.0]])
    xy, a, proc, _ = complete_solution(s, xy, equal_split=True)
    bw_se = allocate_constrained_bandwidth(s, xy, a, proc, aodt_first=False)
    bw_af = allocate_constrained_bandwidth(s, xy, a, proc, aodt_first=True)
    r_se = evaluate(s, xy, a, proc, bw_se)
    r_af = evaluate(s, xy, a, proc, bw_af)
    return {
        "seed": seed,
        "se_sum_rate_mbps": r_se.sum_rate / 1e6,
        "aodt_first_sum_rate_mbps": r_af.sum_rate / 1e6,
        "se_aodt_mean": float(np.mean(r_se.aodt)),
        "aodt_first_aodt_mean": float(np.mean(r_af.aodt)),
        "se_aodt_excess": r_se.aodt_excess,
        "aodt_first_aodt_excess": r_af.aodt_excess,
        "equal_split_mbps": evaluate(s, xy, a, proc, equal_bandwidth(a, s.cfg)).sum_rate / 1e6,
    }


def plot_fig9(out_dir: Path, summary: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    paper_x = sorted(PAPER_FIG9_MBPS)
    ax.plot(
        paper_x,
        [PAPER_FIG9_MBPS[t]["sca"] for t in paper_x],
        "k--",
        marker="x",
        label="paper SCA",
    )
    ax.plot(
        paper_x,
        [PAPER_FIG9_MBPS[t]["kmeans"] for t in paper_x],
        color="0.35",
        ls=":",
        marker="o",
        label="paper K-means",
    )
    ax.plot(
        paper_x,
        [PAPER_FIG9_MBPS[t]["random"] for t in paper_x],
        color="0.55",
        ls=":",
        marker="s",
        label="paper Random",
    )
    style = {
        "random": ("C0", "o", "-"),
        "kmeans": ("C1", "s", "-"),
        "random_lp": ("C0", "o", "--"),
        "kmeans_lp": ("C1", "s", "--"),
        "sca": ("C3", "^", "-"),
        "sca_nocap": ("C3", "^", "--"),
        "td3": ("C2", "D", "-"),
    }
    methods = sorted({r["method"] for r in summary})
    for m in methods:
        sub = [r for r in summary if r["method"] == m]
        xs = np.array([float(r["aodt_threshold"]) for r in sub])
        ys = np.array([float(r["sum_rate_mean"]) / 1e6 for r in sub])
        order = np.argsort(xs)
        color, marker, ls = style.get(m, ("C7", ".", "-"))
        ax.plot(xs[order], ys[order], color=color, marker=marker, ls=ls, label=m)
    ax.set_xlabel("AoDT threshold T_k (s)")
    ax.set_ylabel("Sum rate (Mbps)")
    ax.set_title("Fig. 9: equal-split baselines vs LP (SCA ± cap)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_bandwidth_sharing.png", dpi=120)
    plt.close(fig)


def write_report(
    out_dir: Path,
    summary: list[dict],
    frozen: dict,
    cfg: SimConfig,
    n_seeds: int,
) -> None:
    def mbps(method: str, tk: float) -> str:
        v = _mbps_at(summary, method, tk)
        return "n/a" if v != v else f"{v:.2f}"

    k3 = _mbps_at(summary, "kmeans", 3.0)
    r3 = _mbps_at(summary, "random", 3.0)
    kl3 = _mbps_at(summary, "kmeans_lp", 3.0)
    rl3 = _mbps_at(summary, "random_lp", 3.0)
    s3 = _mbps_at(summary, "sca", 3.0)
    sn3 = _mbps_at(summary, "sca_nocap", 3.0)
    paper_k, paper_r, paper_s = 4.4, 3.2, 7.4

    def closer(equal_v: float, lp_v: float, paper_v: float) -> str:
        if equal_v != equal_v or lp_v != lp_v:
            return "n/a"
        return "equal-split closer to paper" if abs(equal_v - paper_v) < abs(lp_v - paper_v) else "LP closer to paper"

    lines = [
        "# Bandwidth-sharing and SCA cap tests",
        "",
        "Production defaults: Random / K-means equal-split bandwidth even with AoDT on.",
        "SCA / TD3 / PSO keep the constrained LP. After floors, leftover goes to the",
        "current AoDT bottleneck, then to highest spectral-efficiency links.",
        "",
        f"- Seeds: {n_seeds} (I={cfg.num_iot}, J={cfg.num_uav})",
        f"- S_i = {cfg.task_size_bytes} bytes, L = {cfg.task_cycles} cycles (experimental defaults, not retuned)",
        f"- Radio: B_sys={cfg.b_sys:g} Hz, max_bw_share={cfg.max_bw_share} (dropped only for sca_nocap)",
        "",
        "## Fig. 9 at T_k = 3.0 s (Mbps)",
        "",
        "| Method | Ours | Paper |",
        "|---|---:|---:|",
        f"| Random (equal split) | {mbps('random', 3.0)} | {paper_r} |",
        f"| K-means (equal split) | {mbps('kmeans', 3.0)} | {paper_k} |",
        f"| Random + LP | {mbps('random_lp', 3.0)} | {paper_r} |",
        f"| K-means + LP | {mbps('kmeans_lp', 3.0)} | {paper_k} |",
        f"| SCA (25% cap) | {mbps('sca', 3.0)} | {paper_s} |",
        f"| SCA (no cap) | {mbps('sca_nocap', 3.0)} | {paper_s} |",
        f"| TD3 | {mbps('td3', 3.0)} | 6.0 |",
        "",
        "### Bandwidth-sharing theory",
        "",
        f"- K-means equal vs LP at 3 s: {k3:.2f} vs {kl3:.2f} Mbps. Paper ≈ {paper_k}. Verdict: **{closer(k3, kl3, paper_k)}**.",
        f"- Random equal vs LP at 3 s: {r3:.2f} vs {rl3:.2f} Mbps. Paper ≈ {paper_r}. Verdict: **{closer(r3, rl3, paper_r)}**.",
        "",
        "If equal-split K-means / Random sit near 4.4 / 3.2 and the LP variants stay high,",
        "the paper gap was the sharing rule, not S_i or L.",
        "",
        "### SCA 25% cap",
        "",
        f"- SCA with cap: {s3:.2f} Mbps. Without cap: {sn3:.2f} Mbps. Paper ≈ {paper_s}.",
        f"- Cap removal {'raises' if sn3 > s3 else 'does not raise'} SCA toward 7.4.",
        "",
        "## Frozen-geometry AoDT-first surplus (seed 100, no search)",
        "",
        f"- SE-only leftover: {frozen['se_sum_rate_mbps']:.3f} Mbps, AoDT mean {frozen['se_aodt_mean']:.3f} s, excess {frozen['se_aodt_excess']:.4f}",
        f"- AoDT-first leftover: {frozen['aodt_first_sum_rate_mbps']:.3f} Mbps, AoDT mean {frozen['aodt_first_aodt_mean']:.3f} s, excess {frozen['aodt_first_aodt_excess']:.4f}",
        f"- Equal split on the same placement: {frozen['equal_split_mbps']:.3f} Mbps",
        "",
        "## Full T_k sweep (mean Mbps)",
        "",
        "| T_k | Random | K-means | Random+LP | K-means+LP | SCA | SCA no cap | Paper SCA | Paper K-means | Paper Random |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tk in FIG9_THRESHOLDS:
        p = PAPER_FIG9_MBPS.get(tk, {})
        lines.append(
            f"| {tk:g} | {mbps('random', tk)} | {mbps('kmeans', tk)} | {mbps('random_lp', tk)} | "
            f"{mbps('kmeans_lp', tk)} | {mbps('sca', tk)} | {mbps('sca_nocap', tk)} | "
            f"{p.get('sca', '')} | {p.get('kmeans', '')} | {p.get('random', '')} |"
        )
    lines.extend(["", "SCA / TD3 AoDT mean at T_k = 3.0 s:", ""])
    lines.append(f"- SCA AoDT mean: {_aodt_at(summary, 'sca', 3.0):.3f} s")
    lines.append(f"- SCA no-cap AoDT mean: {_aodt_at(summary, 'sca_nocap', 3.0):.3f} s")
    lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bandwidth_sharing(cfg: SimConfig, args: Namespace) -> list[dict]:
    if not cfg.use_compute_model:
        cfg = cfg.with_compute()
    out_dir = Path(getattr(args, "out", "results/bandwidth_sharing"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _seeds(args)
    methods = _methods(args)
    tks = FIG9_THRESHOLDS
    banner("bandwidth sharing: equal-split baselines vs SCA ± cap")
    log.info("seeds=%s methods=%s T_k=%s", seeds, methods, tks)

    frozen = frozen_aodt_first_delta(replace(cfg, aodt_threshold=2.8))
    (out_dir / "frozen_aodt_first.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    rows: list[dict] = []
    jobs = Counter("Fig.9 bandwidth sharing", len(tks) * len(seeds) * len(methods))
    for tk in tks:
        cfg_t = replace(cfg, aodt_threshold=tk)
        for seed in seeds:
            scenario = generate_scenario(seed, cfg_t)
            for name in methods:
                xy, result, rt = _solve(name, scenario, seed, args)
                rows.append(
                    _row(seed, name, result, rt, xy, cfg_t.uav_height, aodt_threshold=tk)
                )
                jobs.tick(f"Tk={tk:g}s  seed={seed}  {name:10}", result, rt)

    _write_csv(out_dir / "raw_sumrate_vs_aodt.csv", rows)
    summary = _summarize(rows, "aodt_threshold")
    _write_csv(out_dir / "sumrate_vs_aodt.csv", summary)
    plot_fig9(out_dir, summary)
    write_report(out_dir, summary, frozen, cfg, len(seeds))
    meta = {
        "task_size_bytes": cfg.task_size_bytes,
        "task_cycles": cfg.task_cycles,
        "max_bw_share": cfg.max_bw_share,
        "b_sys": cfg.b_sys,
        "seeds": list(seeds),
        "methods": list(methods),
        "paper_fig9_mbps": PAPER_FIG9_MBPS,
        "frozen_aodt_first": frozen,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s", out_dir / "REPORT.md")
    return rows
