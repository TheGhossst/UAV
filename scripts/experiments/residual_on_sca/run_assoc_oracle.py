"""Experiment C: association oracle at frozen SCA q.

1-opt (+ optional random maps, + best-SE) over legal a at the frozen
SCA geometry, then one SCA polish of the winner. Same evaluate() and
primary 8.8 MHz / 25% cap / T_k=2.8 s as the headline campaign.

Falsifies “beat SCA on default Mbps by searching a at frozen q”.
Does not overwrite campaign_8.8mhz_cap25_si12k.json.
Does not edit src/uavdt/sca/.

Usage:
  python scripts/experiments/residual_on_sca/run_assoc_oracle.py
  python scripts/experiments/residual_on_sca/run_assoc_oracle.py --n-runs 2 --n-random 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from _paths import OUT_ASSOC_ORACLE, PROTECTED, bootstrap  # noqa: E402

bootstrap()

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

from uavdt.assoc_search import (  # noqa: E402
    forwarding_slack_s,
    hamming_association,
    must_group_mask,
    search_legal_association,
)
from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.models import Allocation  # noqa: E402
from uavdt.sca.algorithm import solve_sca  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

OUT_DEFAULT = OUT_ASSOC_ORACLE
PRACTICAL_MBPS = 0.05
FLAT_MBPS = 0.01


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _cfg() -> SimConfig:
    return SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _map_record(scored) -> dict:
    return {
        "sum_rate_Mbps": float(scored.sum_rate_Mbps),
        "feasible": bool(scored.feasible),
        "reward": float(scored.reward),
        "n_forwarding": int(scored.n_forwarding),
        "hamming_from_start": int(scored.hamming_from_start),
        "reason": scored.reason,
        "association": np.argmax(scored.association, axis=1).tolist(),
        "processing": np.argmax(scored.processing, axis=1).tolist(),
    }


def run_seed(
    seed: int,
    cfg: SimConfig,
    *,
    n_random: int,
    max_rounds: int,
    polish: bool,
) -> dict:
    sc = generate_scenario(seed, cfg)
    settings = SCASettings(solver=None, max_iterations=30)
    t0 = perf_counter()
    frozen = solve_sca(sc, seed, settings=settings)
    q = np.asarray(frozen.uav_xyz_m, dtype=float)
    a0 = frozen.allocation.hard_association()
    slack = forwarding_slack_s(sc)
    must = must_group_mask(sc)
    t1 = perf_counter()
    search = search_legal_association(
        sc,
        q,
        a0,
        max_rounds=int(max_rounds),
        n_random=int(n_random),
        seed=int(seed),
        include_best_se=True,
    )
    t2 = perf_counter()
    winner = search.incumbent
    a_star = winner.association
    b_star = winner.processing
    a_changed = hamming_association(a_star, a0) > 0
    polish_rec = None
    polish_s = 0.0
    if polish and winner.feasible:
        t_p = perf_counter()
        polished = solve_sca(
            sc,
            seed,
            settings=settings,
            uav_xyz_m=q,
            allocation=Allocation(a_star, b_star, frozen.allocation.bandwidth_hz),
        )
        polish_s = perf_counter() - t_p
        polish_rec = {
            "sum_rate_Mbps": float(polished.true_eval.sum_rate_mbps),
            "feasible": bool(polished.true_eval.feasible),
            "n_iterations": int(polished.n_iterations),
            "stop_reason": polished.diagnostics.get("stop_reason"),
            "wall_clock_s": float(polish_s),
            "a_unchanged": not a_changed,
        }
    elif polish:
        polish_rec = {
            "sum_rate_Mbps": float(winner.sum_rate_Mbps),
            "feasible": False,
            "n_iterations": 0,
            "stop_reason": "search_infeasible_skip_polish",
            "wall_clock_s": 0.0,
            "a_unchanged": not a_changed,
        }
    d_lp = float(winner.sum_rate_Mbps - frozen.true_eval.sum_rate_mbps)
    polish_mbps = (
        float(polish_rec["sum_rate_Mbps"])
        if polish_rec is not None
        else float(winner.sum_rate_Mbps)
    )
    d_pol = float(polish_mbps - frozen.true_eval.sum_rate_mbps)
    return {
        "seed": int(seed),
        "frozen_Mbps": float(frozen.true_eval.sum_rate_mbps),
        "frozen_feasible": bool(frozen.true_eval.feasible),
        "frozen_n_forwarding": int(
            np.sum(
                np.argmax(a0, axis=1)
                != np.argmax(frozen.allocation.hard_processing(), axis=1)
            )
        ),
        "q_k_s": slack.tolist(),
        "must_group": must.tolist(),
        "search": _map_record(winner),
        "search_n_lp_evals": int(search.n_lp_evals),
        "search_n_accepted": int(search.n_accepted),
        "search_n_neighbors": int(search.n_neighbors_scored),
        "search_n_random": int(search.n_random_scored),
        "search_n_illegal": int(search.n_illegal),
        "search_rounds": int(search.rounds),
        "a_changed": bool(a_changed),
        "delta_lp_Mbps": d_lp,
        "polish": polish_rec,
        "delta_polish_Mbps": d_pol,
        "wall_clock_s": {
            "frozen_sca": float(t1 - t0),
            "search": float(t2 - t1),
            "polish": float(polish_s),
            "total": float(perf_counter() - t0),
        },
    }


def summarize(rows: list[dict]) -> dict:
    seeds = [int(r["seed"]) for r in rows]
    frozen = np.array([r["frozen_Mbps"] for r in rows], dtype=float)
    lp = np.array([r["search"]["sum_rate_Mbps"] for r in rows], dtype=float)
    pol = np.array(
        [
            (r["polish"]["sum_rate_Mbps"] if r.get("polish") else r["search"]["sum_rate_Mbps"])
            for r in rows
        ],
        dtype=float,
    )
    d_lp = lp - frozen
    d_pol = pol - frozen
    w_lp = wilcoxon_signed_rank(d_lp)
    w_pol = wilcoxon_signed_rank(d_pol)
    t_lp = paired_t(d_lp)
    t_pol = paired_t(d_pol)
    win_lp = [int(s) for s, d in zip(seeds, d_lp) if d > 1e-9]
    win_pol = [int(s) for s, d in zip(seeds, d_pol) if d > 1e-9]
    practical = [int(s) for s, d in zip(seeds, d_pol) if d > PRACTICAL_MBPS]
    a_changed = [int(r["seed"]) for r in rows if r.get("a_changed")]
    mean_d = float(np.mean(d_pol))
    max_d = float(np.max(d_pol)) if d_pol.size else 0.0
    if max_d > PRACTICAL_MBPS:
        readout = "goal_a_real"
    elif mean_d > FLAT_MBPS and w_pol["p_greater"] < 0.05:
        readout = "stat_not_practical"
    elif abs(float(np.mean(d_lp))) <= FLAT_MBPS:
        readout = "flat_at_frozen_q"
    else:
        readout = "lp_moved_polish_unclear"
    return {
        "n": len(rows),
        "seeds": seeds,
        "mean_frozen_Mbps": float(np.mean(frozen)),
        "mean_search_lp_Mbps": float(np.mean(lp)),
        "mean_polish_Mbps": float(np.mean(pol)),
        "mean_delta_lp_Mbps": float(np.mean(d_lp)),
        "std_delta_lp_Mbps": float(np.std(d_lp, ddof=1)) if d_lp.size > 1 else 0.0,
        "mean_delta_polish_Mbps": mean_d,
        "std_delta_polish_Mbps": float(np.std(d_pol, ddof=1)) if d_pol.size > 1 else 0.0,
        "max_delta_lp_Mbps": float(np.max(d_lp)) if d_lp.size else 0.0,
        "max_delta_polish_Mbps": max_d,
        "min_delta_polish_Mbps": float(np.min(d_pol)) if d_pol.size else 0.0,
        "n_a_changed": len(a_changed),
        "a_changed_seeds": a_changed,
        "n_lp_wins": len(win_lp),
        "n_polish_wins": len(win_pol),
        "win_seeds_lp": win_lp,
        "win_seeds_polish": win_pol,
        "practical_win_seeds": practical,
        "mean_hamming": float(np.mean([r["search"]["hamming_from_start"] for r in rows])),
        "mean_n_lp_evals": float(np.mean([r["search_n_lp_evals"] for r in rows])),
        "wilcoxon_lp": w_lp,
        "wilcoxon_polish": w_pol,
        "paired_t_lp": t_lp,
        "paired_t_polish": t_pol,
        "readout": readout,
        "flat_threshold_Mbps": FLAT_MBPS,
        "practical_threshold_Mbps": PRACTICAL_MBPS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experiment C: 1-opt legal a at frozen SCA q, then SCA polish"
    )
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.resolve() == PROTECTED.resolve():
        raise SystemExit(f"refusing to overwrite protected {PROTECTED}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _cfg()
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.n_runs)))
    polish = not bool(args.no_polish)
    _log(
        f"Experiment C  seeds={seeds[0]}..{seeds[-1]}  "
        f"B_sys=8.8 MHz  cap=25%  T_k={cfg.aodt_threshold_s:g}s  "
        f"1-opt + best-SE + n_random={args.n_random}  polish={polish}"
    )
    rows = []
    for seed in seeds:
        _log(f"  seed {seed}")
        row = run_seed(
            seed,
            cfg,
            n_random=int(args.n_random),
            max_rounds=int(args.max_rounds),
            polish=polish,
        )
        pol = row["polish"]["sum_rate_Mbps"] if row.get("polish") else row["search"]["sum_rate_Mbps"]
        _log(
            f"    frozen={row['frozen_Mbps']:.4f}  "
            f"lp={row['search']['sum_rate_Mbps']:.4f}  "
            f"polish={pol:.4f}  "
            f"d_lp={row['delta_lp_Mbps']:+.4f}  "
            f"d_pol={row['delta_polish_Mbps']:+.4f}  "
            f"H={row['search']['hamming_from_start']}  "
            f"LPs={row['search_n_lp_evals']}"
        )
        rows.append(row)
    summary = summarize(rows)
    payload = {
        "experiment": "C",
        "label": (
            "association oracle at frozen SCA q: 1-opt + best-SE + random "
            "legal maps, then SCA polish"
        ),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "aodt_threshold_s": cfg.aodt_threshold_s,
        "n_runs": int(args.n_runs),
        "seed_start": int(args.seed_start),
        "n_random": int(args.n_random),
        "max_rounds": int(args.max_rounds),
        "polish": polish,
        "seeds": list(seeds),
        "per_seed": rows,
        "summary": summary,
        "notes": [
            "Does not overwrite campaign_8.8mhz_cap25_si12k.json.",
            "Does not edit src/uavdt/sca/.",
            "Legal a: (21)(23)(24) + cohesive on processes with T_k-Q_k < T_u2u.",
            "If LP Δ≈0 at frozen q, default-Mbps win is joint (q,a) (Experiment A), not a-only.",
        ],
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    _log(f"wrote {out}")
    s = summary
    _log(
        f"mean d LP = {s['mean_delta_lp_Mbps']:+.4f} +/- {s['std_delta_lp_Mbps']:.4f}  "
        f"wins={s['n_lp_wins']}/{s['n']}  "
        f"Wilcoxon p_greater={s['wilcoxon_lp']['p_greater']:.4g}"
    )
    _log(
        f"mean d polish = {s['mean_delta_polish_Mbps']:+.4f} +/- {s['std_delta_polish_Mbps']:.4f}  "
        f"max={s['max_delta_polish_Mbps']:+.4f}  "
        f"wins={s['n_polish_wins']}/{s['n']}  "
        f"practical={s['practical_win_seeds']}  "
        f"Wilcoxon p_greater={s['wilcoxon_polish']['p_greater']:.4g}  "
        f"readout={s['readout']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
