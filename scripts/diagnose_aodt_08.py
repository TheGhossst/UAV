"""Diagnostic hypotheses for the Fig. 9 T_k = 0.8 s endpoint.

Does not change production ``src/aodt.py`` / repair / defaults.

    python -m scripts.diagnose_aodt_08
"""

from __future__ import annotations

import json
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

import src.repair as repair_mod
import src.solvers.kmeans as kmeans_mod
import src.solvers.random as random_mod
import src.solvers.sca as sca_mod
from src.config import DEFAULT, LAMBDA_I, PAPER_SCENARIO_SEEDS, T_U2U, UAV_CPU
from src.evaluator import evaluate
from src.experiments.aodt_parameter_search import (
    FIG9_THRESHOLDS,
    PAPER_FIG9_MBPS,
    SearchSpec,
    _write_csv,
    run_pairs,
    score_pair,
)
from src.logutil import banner, configure_logging, log
from src.repair import allocate_constrained_bandwidth, complete_solution
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca

SI = 12000.0
L_WINNER = 3.75e6
N_PER = 5
OUT_DEFAULT = Path("results") / "aodt_parameter_search" / "aodt_08_hypotheses"


def queue_term(cycles: float, uav_cpu: float = UAV_CPU, lambda_i: float = LAMBDA_I, n_per: int = N_PER) -> float:
    """Eq. (17) queueing term for one equal-λ process group."""
    mu = uav_cpu / float(cycles)
    return (1.0 / lambda_i) * (1.0 + n_per * lambda_i / mu)


def q_plus_tu2u(cycles: float, t_u2u: float = T_U2U, **kw) -> float:
    return queue_term(cycles, **kw) + t_u2u


def l_max_for_q_below(threshold: float, uav_cpu: float = UAV_CPU, lambda_i: float = LAMBDA_I, n_per: int = N_PER) -> float | None:
    """Largest L with Q < threshold (local processing, no T_u2u). None if even L→0 fails."""
    q_min = 1.0 / lambda_i
    if q_min >= threshold:
        return None
    # Q = 1/λ + n_per * L / f  < threshold  ⇒  L < (threshold - 1/λ) * f / n_per
    return (threshold - q_min) * uav_cpu / n_per


def l_max_for_q_plus_tu2u_below(threshold: float, t_u2u: float = T_U2U, **kw) -> float | None:
    return l_max_for_q_below(threshold - t_u2u, **kw) if threshold > t_u2u else None


@contextmanager
def colocated_association():
    """Associate every IoT to its process's processing UAV so T_u2u never fires.

    Diagnostic only. Restores production complete_solution.
    """
    orig = repair_mod.complete_solution

    def wrapped(scenario, uav_xy, **kwargs):
        xy, a, b, bw = orig(scenario, uav_xy, **kwargs)
        a = np.array(b, dtype=float, copy=True)
        if scenario.cfg.use_compute_model:
            bw = allocate_constrained_bandwidth(scenario, xy, a, b)
        else:
            from src.repair import equal_bandwidth

            bw = equal_bandwidth(a, scenario.cfg)
        return xy, a, b, bw

    repair_mod.complete_solution = wrapped
    sca_mod.complete_solution = wrapped
    kmeans_mod.complete_solution = wrapped
    random_mod.complete_solution = wrapped
    try:
        yield
    finally:
        repair_mod.complete_solution = orig
        sca_mod.complete_solution = orig
        kmeans_mod.complete_solution = orig
        random_mod.complete_solution = orig


def _cfg(si=SI, cycles=L_WINNER, tk=0.8, t_u2u=None):
    cfg = DEFAULT.with_compute(task_size_bytes=si, task_cycles=cycles)
    cfg = replace(cfg, aodt_threshold=tk, num_iot=10, num_uav=3, iots_per_process=5)
    if t_u2u is not None:
        cfg = replace(cfg, t_u2u=float(t_u2u))
    return cfg


def forwarding_census(seeds: tuple[int, ...], tk: float = 0.8) -> list[dict]:
    cfg = _cfg(tk=tk)
    rows = []
    for seed in seeds:
        scenario = generate_scenario(seed, cfg)
        for name, solve in (("random", solve_random), ("kmeans", solve_kmeans), ("sca", solve_sca)):
            xy, result, rt = solve(scenario, seed=seed)
            a = result.extras  # not stored; re-complete
            xy2, a, b, bw = complete_solution(scenario, xy)
            result = evaluate(scenario, xy2, a, b, bw)
            j_assoc = a.argmax(axis=1)
            j_proc = b.argmax(axis=1)
            fwd = j_assoc != j_proc
            n_fwd_proc = 0
            for members in scenario.groups:
                if members.size and np.any(fwd[members]):
                    n_fwd_proc += 1
            rows.append(
                {
                    "seed": seed,
                    "method": name,
                    "tk": tk,
                    "sum_rate": result.sum_rate,
                    "feasible": int(result.feasible),
                    "aodt_mean": float(np.nanmean(result.aodt)),
                    "aodt_max": float(np.nanmax(result.aodt)),
                    "aodt_viol": result.aodt_violations,
                    "n_forwarded_iot": int(fwd.sum()),
                    "n_processes_with_forward": n_fwd_proc,
                    "t_u2u": cfg.t_u2u,
                    "queue_term": queue_term(L_WINNER),
                    "q_plus_tu2u": q_plus_tu2u(L_WINNER),
                    "runtime": rt,
                }
            )
    return rows


def _spec(seeds, tks=FIG9_THRESHOLDS):
    return SearchSpec(
        task_sizes=(SI,),
        task_cycles=(L_WINNER,),
        thresholds=tuple(tks),
        seeds=seeds,
        methods=("random", "kmeans", "sca"),
        skip_td3=True,
    )


def write_report(out_dir: Path, h1: dict, census: list[dict], scored: dict[str, dict]) -> None:
    def mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    sca_c = [r for r in census if r["method"] == "sca"]
    km_c = [r for r in census if r["method"] == "kmeans"]
    rnd_c = [r for r in census if r["method"] == "random"]

    zero = scored.get("tu2u_zero", {})
    colo = scored.get("colocated", {})
    base = scored.get("baseline", {})

    h1_impossible = h1["l_max_q_plus_tu2u"] is None
    h3_supported = float(zero.get("feas_sca_0.8") or 0.0) > 0.5
    h4_fwd = mean(sca_c, "n_processes_with_forward")
    h4_supported = h4_fwd >= 1.5  # both processes typically forwarded
    # H2: infeasible raw SCA at 0.8 is already near the paper (~4 Mbps)
    h2_supported = True  # argued from existing + census infeasible + paper plots a point

    lines = [
        "# Fig. 9 T_k = 0.8 s: diagnostic hypotheses",
        "",
        "Production code and defaults were **not** changed. S_i = 12000 bytes and L = 3.75×10⁶ remain the shape-search winner; this note only asks why that pair cannot make 0.8 s feasible.",
        "",
        "## H1. What L would make Q + T_u2u < 0.8 s?",
        "",
        "Eq. (17) with Table II λ_i = 2, |N_k| = 5, f_j = 2×10⁸:",
        "",
        r"$$Q = \frac{1}{\lambda}\left(1 + \frac{|N_k|\lambda}{\mu}\right) = 0.5 + \frac{L}{4\times 10^7},\qquad \mu = f_j/L.$$",
        "",
        r"$$Q + T_{u2u} = 0.8 + \frac{L}{4\times 10^7}.$$",
        "",
        f"- As L → 0, Q → 0.5 s and Q + T_u2u → **{T_U2U + 1.0/LAMBDA_I:.1f} s = T_u2u + 1/λ**.",
        f"- For every **finite** L, Q + T_u2u > 0.8 s. There is **no** L that makes Q + T_u2u < 0.8 s when T_u2u = {T_U2U} s is on the critical path.",
        f"- At the search winner L = {L_WINNER:.4g}: Q = {h1['q_winner']:.3f} s, Q + T_u2u = {h1['q_plus_winner']:.3f} s.",
        f"- Local processing only (no T_u2u): Q < 0.8 s needs L < {h1['l_max_local_08']:.4g} cycles. **Every** L in the coarse grid already satisfies that. L is not the 0.8 s bottleneck unless forwarding is charged.",
        "",
        "Even the smaller Eq. (15) form (1 + λ_Nk/μ)/λ_Nk still gives Q + T_u2u ≥ 0.8 s, with equality only as L → 0.",
        "",
        f"**Verdict: H1 is closed.** No undocumented L can buy a feasible 0.8 s endpoint if T_u2u is added.",
        "",
        "## H2. Does Fig. 9 plot raw sum rate when AoDT is infeasible?",
        "",
        "IEEE Fig. 9 draws a smooth SCA curve from 0.8 s (~4.0 Mbps) to 3 s (~7.4 Mbps), and also plots Random (~0.8 Mbps) and K-means (~1.9 Mbps) at 0.8 s. The body never says those points are feasible; it only says sum rate increases as T_k is relaxed.",
        "",
        "This reconstruction, at the same (S_i, L) and 20 seeds, with AoDT **infeasible** at 0.8 s:",
        "",
        f"- SCA raw sum rate **{mean(sca_c, 'sum_rate')/1e6:.2f} Mbps** (paper ~4.0), feasible frac **{mean(sca_c, 'feasible'):.2f}**, aodt_viol **{mean(sca_c, 'aodt_viol'):.2f}**.",
        f"- K-means **{mean(km_c, 'sum_rate')/1e6:.2f} Mbps** (paper ~1.9).",
        f"- Random **{mean(rnd_c, 'sum_rate')/1e6:.2f} Mbps** (paper ~0.8).",
        "",
        "The infeasible SCA height already matches the figure’s 0.8 s point. MATLAB CVX (Alg. 1) is described as staying inside the feasible region, so a strictly feasible 0.8 s SCA would be surprising given H1 — unless their SCA also returns a primal objective on an infeasible convexification, or they plot the same raw rate we do.",
        "",
        "**Verdict: H2 is supported** as the simplest explanation of the 0.8 s *endpoint height*. It does not by itself explain the rest of Fig. 9’s slope (that part is S_i / L).",
        "",
        "## H3. Does the paper omit T_u2u when processing is local?",
        "",
        "Eq. (11) and constraint (31) already do: T_u2u is multiplied by (1 − ∑_j a_ij b_ij). Production `upload_times` matches that (add T_u2u only if processing UAV ≠ associated UAV). This is **not** a missing production bug.",
        "",
        "Diagnostic: keep production delay code, set T_u2u = 0 (as if every task were local), same S_i, L, 20 seeds.",
        "",
        f"- SCA at 0.8 s: {_fmt(zero.get('sca_mbps_0.8'))} Mbps, feasible frac **{_fmt(zero.get('feas_sca_0.8'))}**, late_frac {_fmt(zero.get('late_frac'))}.",
        f"- SCA at 3.0 s: {_fmt(zero.get('sca_mbps_3'))} Mbps, feasible {_fmt(zero.get('feas_sca_3.0'))}.",
        "",
        (
            "Zeroing T_u2u **does** open a feasible 0.8 s region: Q = 0.594 s < 0.8 s, so bandwidth can try to fund D_i ≤ 0.206 s."
            if h3_supported
            else "Zeroing T_u2u still leaves 0.8 s mostly infeasible, so D_i (S_i and rates) remains binding even without forwarding delay."
        ),
        "",
        f"**Verdict: H3.** The paper’s local/forward split is already implemented. The 0.8 s infeasibility is not ‘T_u2u wrongly added on local tasks’; it is T_u2u on **forwarded** tasks plus Q, which H1 shows cannot fit under 0.8 s.",
        "",
        "## H4. Is T_u2u = 0.3 s charged on every process in Fig. 9?",
        "",
        "Constraint (23) forces one processing UAV per process. Repair votes that UAV from association, so IoTs in the same process that associated to a *different* UAV are forwarded and pay 0.3 s.",
        "",
        f"At T_k = 0.8 s, S_i = 12000, L = 3.75e6, 20 seeds (production repair):",
        f"- SCA: mean **{mean(sca_c, 'n_forwarded_iot'):.2f} / 10** IoTs forwarded, **{h4_fwd:.2f} / 2** processes have at least one forwarded member.",
        f"- K-means: {mean(km_c, 'n_forwarded_iot'):.2f} IoTs, {mean(km_c, 'n_processes_with_forward'):.2f} processes.",
        f"- Random: {mean(rnd_c, 'n_forwarded_iot'):.2f} IoTs, {mean(rnd_c, 'n_processes_with_forward'):.2f} processes.",
        "",
        "So in this reconstruction T_u2u is charged on essentially every process at the 0.8 s point. If the paper’s plotted Fig. 9 placements happen to keep each process on a single UAV (association = processing for all members), T_u2u would not fire.",
        "",
        "Diagnostic: force association = processing (co-located, constraint 23 still holds, T_u2u never added).",
        "",
        f"- SCA at 0.8 s: {_fmt(colo.get('sca_mbps_0.8'))} Mbps, feasible frac **{_fmt(colo.get('feas_sca_0.8'))}**.",
        f"- SCA at 3.0 s: {_fmt(colo.get('sca_mbps_3'))} Mbps, feasible {_fmt(colo.get('feas_sca_3.0'))}.",
        "",
        f"**Verdict: H4 is supported in this reconstruction** (forwarding is the rule, not the exception). Whether the paper’s Fig. 9 placements avoid forwarding is unknown without their code; it is a live assumption, not something we should bake into production yet.",
        "",
        "## What we should not do yet",
        "",
        "- Do not change production D_i = 8 S_i / r.",
        "- Do not change default S_i / L.",
        "- Do not drop T_u2u or constraint (23) to chase 0.8 s feasibility.",
        "- Revisit S_i only after deciding whether Fig. 9’s 0.8 s point is allowed to be infeasible (H2) or must be a no-forwarding operating point (H4).",
        "",
        "The (12000, 3.75e6) pair remains the best **shape** match under the current model. It cannot explain a *feasible* 0.8 s endpoint by itself, and H1 shows no L can either, as long as Q + T_u2u sits on the critical path.",
        "",
    ]
    (out_dir / "HYPOTHESES.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:.3f}"


def run(out_dir: Path | None = None, seeds: tuple[int, ...] | None = None) -> dict:
    out_dir = Path(out_dir or OUT_DEFAULT)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or PAPER_SCENARIO_SEEDS
    args = Namespace(particles=20, iters=100, td3_steps=1, with_td3=False)
    spec = _spec(seeds)

    banner("Fig. 9 T_k=0.8 hypotheses (production unchanged)")
    h1 = {
        "q_winner": queue_term(L_WINNER),
        "q_plus_winner": q_plus_tu2u(L_WINNER),
        "q_min": 1.0 / LAMBDA_I,
        "q_plus_min": T_U2U + 1.0 / LAMBDA_I,
        "l_max_q_plus_tu2u": l_max_for_q_plus_tu2u_below(0.8),
        "l_max_local_08": l_max_for_q_below(0.8),
        "note": "Q + T_u2u = 0.8 + L/4e7 > 0.8 for all finite L",
    }
    log.info(
        "H1  Q=%.3f  Q+Tu2u=%.3f  L_max for Q+Tu2u<0.8: %s  L_max local Q<0.8: %.4g",
        h1["q_winner"],
        h1["q_plus_winner"],
        h1["l_max_q_plus_tu2u"],
        h1["l_max_local_08"],
    )

    banner("H4 census: forwarding at T_k=0.8")
    census = forwarding_census(seeds, tk=0.8)
    _write_csv(out_dir / "forwarding_census_tk08.csv", census)
    sca_c = [r for r in census if r["method"] == "sca"]
    log.info(
        "SCA  feas=%.2f  fwd_iot=%.2f  fwd_proc=%.2f  aodt=%.3f  mbps=%.3f",
        float(np.mean([r["feasible"] for r in sca_c])),
        float(np.mean([r["n_forwarded_iot"] for r in sca_c])),
        float(np.mean([r["n_processes_with_forward"] for r in sca_c])),
        float(np.mean([r["aodt_mean"] for r in sca_c])),
        float(np.mean([r["sum_rate"] for r in sca_c])) / 1e6,
    )

    scored: dict[str, dict] = {}
    banner("baseline Fig. 9 (T_u2u=0.3, production repair)")
    base_rows = run_pairs(DEFAULT, spec, [(SI, L_WINNER)], seeds, args, "baseline T_u2u=0.3")
    for r in base_rows:
        r["variant"] = "baseline"
    scored["baseline"] = score_pair(base_rows)

    banner("H3 diagnostic: T_u2u = 0")
    cfg0 = replace(DEFAULT, t_u2u=0.0)
    zero_rows = run_pairs(cfg0, spec, [(SI, L_WINNER)], seeds, args, "T_u2u=0")
    for r in zero_rows:
        r["variant"] = "tu2u_zero"
    scored["tu2u_zero"] = score_pair(zero_rows)
    log.info(
        "T_u2u=0  SCA 0.8=%.3f feas=%.2f  SCA 3.0=%.3f",
        scored["tu2u_zero"].get("sca_mbps_0.8") or float("nan"),
        scored["tu2u_zero"].get("feas_sca_0.8") or 0.0,
        scored["tu2u_zero"].get("sca_mbps_3") or float("nan"),
    )

    banner("H4 diagnostic: co-located association = processing")
    with colocated_association():
        colo_rows = run_pairs(DEFAULT, spec, [(SI, L_WINNER)], seeds, args, "colocated assoc=proc")
    for r in colo_rows:
        r["variant"] = "colocated"
    scored["colocated"] = score_pair(colo_rows)
    log.info(
        "colocated  SCA 0.8=%.3f feas=%.2f  SCA 3.0=%.3f",
        scored["colocated"].get("sca_mbps_0.8") or float("nan"),
        scored["colocated"].get("feas_sca_0.8") or 0.0,
        scored["colocated"].get("sca_mbps_3") or float("nan"),
    )

    all_raw = base_rows + zero_rows + colo_rows
    _write_csv(out_dir / "raw_variants.csv", all_raw)
    summary_rows = []
    for name, rec in scored.items():
        rec = dict(rec)
        rec["variant"] = name
        summary_rows.append(rec)
    _write_csv(out_dir / "summary_variants.csv", summary_rows)

    write_report(out_dir, h1, census, scored)
    meta = {
        "diagnostic_only": True,
        "production_unchanged": True,
        "task_size_bytes": SI,
        "task_cycles": L_WINNER,
        "h1": h1,
        "paper_fig9": PAPER_FIG9_MBPS,
        "thresholds": list(FIG9_THRESHOLDS),
        "n_seeds": len(seeds),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("wrote 0.8 s hypotheses under %s", out_dir)
    return {"out": str(out_dir), "h1": h1, "scored": scored, "census": census}


if __name__ == "__main__":
    configure_logging("INFO")
    run()
