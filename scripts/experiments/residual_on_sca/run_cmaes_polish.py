"""CMA-ES residual on SCA, then SCA polish — non-RL control for Experiment B.

Same fitness as residual-on-SCA: freeze SCA a,b, frozen-q LP, feasible Mbps.
Then Algorithm 1 polish from the CMA q with those binaries.

Usage:
  python scripts/experiments/residual_on_sca/run_cmaes_polish.py
  python scripts/experiments/residual_on_sca/run_cmaes_polish.py --n-runs 1 --max-evals 40
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
from _paths import OUT_CMAES_POLISH, PROTECTED, bootstrap  # noqa: E402

bootstrap()

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.models import Allocation  # noqa: E402
from uavdt.placement.cmaes import (  # noqa: E402
    apply_residual_xy,
    cmaes_maximize,
    feasible_lp_score,
    residual_bounds,
)
from uavdt.sca.algorithm import solve_sca  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

OUT_DEFAULT = OUT_CMAES_POLISH
PRACTICAL_MBPS = 0.05


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


def run_seed(seed: int, cfg: SimConfig, *, max_evals: int, sigma0: float) -> dict:
    sc = generate_scenario(seed, cfg)
    settings = SCASettings(solver=None, max_iterations=30)
    t0 = perf_counter()
    frozen = solve_sca(sc, seed, settings=settings)
    a = frozen.allocation.hard_association()
    b = frozen.allocation.hard_processing()
    origin = np.asarray(frozen.uav_xyz_m, dtype=float)
    lo, hi = residual_bounds(origin, sc.cfg)
    n = 2 * sc.cfg.num_uav

    def objective(delta: np.ndarray) -> float:
        uav = apply_residual_xy(origin, delta, sc.cfg)
        reward, _ev = feasible_lp_score(sc, uav, a, b)
        return reward

    t1 = perf_counter()
    cma = cmaes_maximize(
        objective,
        np.zeros(n, dtype=float),
        sigma0=float(sigma0),
        max_evals=int(max_evals),
        seed=int(seed),
        lower=lo,
        upper=hi,
    )
    cma_uav = apply_residual_xy(origin, cma.x, sc.cfg)
    cma_reward, cma_ev = feasible_lp_score(sc, cma_uav, a, b)
    t2 = perf_counter()
    polished = solve_sca(
        sc,
        seed,
        settings=settings,
        uav_xyz_m=cma_uav,
        allocation=Allocation(a, b, frozen.allocation.bandwidth_hz),
    )
    t3 = perf_counter()
    d_cma = float(cma_ev.sum_rate_mbps - frozen.true_eval.sum_rate_mbps)
    d_pol = float(polished.true_eval.sum_rate_mbps - frozen.true_eval.sum_rate_mbps)
    return {
        "seed": int(seed),
        "frozen_Mbps": float(frozen.true_eval.sum_rate_mbps),
        "frozen_feasible": bool(frozen.true_eval.feasible),
        "cma_Mbps": float(cma_ev.sum_rate_mbps),
        "cma_feasible": bool(cma_ev.feasible),
        "cma_fitness": float(cma_reward),
        "cma_n_evals": int(cma.n_evals),
        "cma_best_value": float(cma.value),
        "polish_Mbps": float(polished.true_eval.sum_rate_mbps),
        "polish_feasible": bool(polished.true_eval.feasible),
        "delta_cma_minus_sca_Mbps": d_cma,
        "delta_polish_minus_sca_Mbps": d_pol,
        "wall_sca_s": float(t1 - t0),
        "wall_cma_s": float(t2 - t1),
        "wall_polish_s": float(t3 - t2),
        "cma_uav_xyz_m": np.asarray(cma_uav, dtype=float).tolist(),
        "polish_uav_xyz_m": np.asarray(polished.uav_xyz_m, dtype=float).tolist(),
        "frozen_uav_xyz_m": origin.tolist(),
    }


def summarize(rows: list[dict], *, key: str) -> dict:
    sca = np.array([r["frozen_Mbps"] for r in rows], dtype=float)
    other = np.array([r[key] for r in rows], dtype=float)
    deltas = other - sca
    seeds = [int(r["seed"]) for r in rows]
    w = wilcoxon_signed_rank(deltas)
    return {
        "n": len(rows),
        "seeds": seeds,
        "mean_sca_Mbps": float(np.mean(sca)),
        "mean_other_Mbps": float(np.mean(other)),
        "mean_delta_Mbps": float(np.mean(deltas)),
        "std_delta_Mbps": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
        "min_delta_Mbps": float(np.min(deltas)) if deltas.size else 0.0,
        "max_delta_Mbps": float(np.max(deltas)) if deltas.size else 0.0,
        "n_wins": int(np.sum(deltas > 1e-9)),
        "n_losses": int(np.sum(deltas < -1e-9)),
        "seeds_worse_than_0p05": [
            int(s) for s, d in zip(seeds, deltas) if d < -PRACTICAL_MBPS
        ],
        "wilcoxon": w,
        "paired_t": paired_t(deltas),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CMA-ES residual + SCA polish vs frozen SCA")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=21)
    parser.add_argument("--max-evals", type=int, default=400)
    parser.add_argument("--sigma0", type=float, default=5.0)
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.resolve() == PROTECTED.resolve():
        raise SystemExit(f"refusing to overwrite protected {PROTECTED}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _cfg()
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.n_runs)))
    _log(
        f"CMA-ES residual  seeds={seeds[0]}..{seeds[-1]}  "
        f"max_evals={args.max_evals}  sigma0={args.sigma0} m"
    )
    rows = []
    for seed in seeds:
        _log(f"  seed {seed}")
        row = run_seed(seed, cfg, max_evals=int(args.max_evals), sigma0=float(args.sigma0))
        _log(
            f"    sca={row['frozen_Mbps']:.4f}  cma={row['cma_Mbps']:.4f}  "
            f"polish={row['polish_Mbps']:.4f}  "
            f"d_cma={row['delta_cma_minus_sca_Mbps']:+.4f}  "
            f"d_pol={row['delta_polish_minus_sca_Mbps']:+.4f}"
        )
        rows.append(row)
    payload = {
        "experiment": "cmaes_sibling",
        "label": "CMA-ES residual on SCA + SCA polish (non-RL control)",
        "held_out": bool(int(args.seed_start) == 21 and int(args.n_runs) == 20),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "max_evals": int(args.max_evals),
        "sigma0_m": float(args.sigma0),
        "per_seed": rows,
        "summary_cma": summarize(rows, key="cma_Mbps"),
        "summary_polish": summarize(rows, key="polish_Mbps"),
        "notes": [
            "Does not overwrite campaign_8.8mhz_cap25_si12k.json.",
            "If CMA-ES matches residual-TD3, the neural net is not load-bearing.",
        ],
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    _log(f"wrote {out}")
    sp = payload["summary_polish"]
    _log(
        f"polish-SCA mean d={sp['mean_delta_Mbps']:+.4f} +/- {sp['std_delta_Mbps']:.4f}  "
        f"wins={sp['n_wins']}/{sp['n']}  "
        f"Wilcoxon p_greater={sp['wilcoxon']['p_greater']:.4g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
