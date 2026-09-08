"""Pairwise CVXPY vs MATLAB CVX+MOSEK SCA on the headline default point (J=3).

Compares per-seed true evaluate() Mbps. CVXPY rates can be loaded from an
existing campaign JSON to avoid re-running Python SCA.

Usage:
    $env:PYTHONPATH="src"
    python scripts/compare_sca_cvxpy_matlab.py
    python scripts/compare_sca_cvxpy_matlab.py --rerun-cvxpy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import SimConfig
from uavdt.experiments.grids import config_for_counts
from uavdt.sca.algorithm import solve_sca
from uavdt.sca.matlab_bridge import matlab_available
from uavdt.sca.settings import SCASettings
from uavdt.scenario import generate_scenario

CVXPY_CAMPAIGN = Path("results/campaign_8.8mhz_cap25_si12k.json")  # saturation-test 20-seed pair
OUT = Path("results/sca_cvxpy_vs_matlab_j3.json")


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(deltas: list[float]) -> float:
    """Two-sided Wilcoxon p (normal approx with tie correction)."""
    x = np.asarray(deltas, dtype=float)
    nz = x[x != 0.0]
    n = len(nz)
    if n == 0:
        return 1.0
    ranks = _rankdata(np.abs(nz))
    w_plus = float(np.sum(ranks[nz > 0]))
    mu = n * (n + 1) / 4.0
    # tie correction
    _, counts = np.unique(np.abs(nz), return_counts=True)
    tie = sum(c * (c * c - 1) for c in counts if c > 1)
    sigma2 = (n * (n + 1) * (2 * n + 1) - tie / 2.0) / 24.0
    if sigma2 <= 0:
        return 1.0
    z = (w_plus - mu) / math.sqrt(sigma2)
    return min(1.0, 2.0 * _norm_sf(abs(z)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + 1 + j + 1)
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def cvxpy_from_campaign() -> dict[int, float]:
    payload = json.loads(CVXPY_CAMPAIGN.read_text(encoding="utf-8"))
    for pt in payload["points"]:
        if pt["axis"] == "uavs" and abs(float(pt["x"]) - 3.0) < 1e-9:
            sca = pt["by_method"]["sca"]
            return {
                int(s): float(m)
                for s, m in zip(sca["seeds"], sca["per_seed_Mbps"])
            }
    raise KeyError("J=3 point not found in campaign JSON")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--rerun-cvxpy", action="store_true")
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()

    if not matlab_available():
        print("MATLAB CVX/MOSEK not available", file=sys.stderr)
        return 1

    cfg = config_for_counts(
        num_iot=10,
        num_uav=3,
        cfg=SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25),
    )
    seeds = list(range(args.seed_start, args.seed_start + args.n_runs))
    cvx_map = cvxpy_from_campaign() if not args.rerun_cvxpy else {}

    rows = []
    for seed in seeds:
        scenario = generate_scenario(seed, cfg)
        if args.rerun_cvxpy:
            py = solve_sca(
                scenario,
                seed,
                settings=SCASettings(solver=None, max_iterations=30, step_size_m=20.0),
            )
            cvx_mbps = py.true_eval.sum_rate_mbps
            cvx_feas = py.true_eval.feasible
            cvx_stop = py.diagnostics.get("stop_reason")
        else:
            cvx_mbps = cvx_map[seed]
            cvx_feas = True
            cvx_stop = "from_campaign"

        print(f"seed {seed:2d}  MATLAB...", flush=True)
        ml = solve_sca(
            scenario,
            seed,
            settings=SCASettings(solver="matlab", max_iterations=30, step_size_m=20.0),
        )
        ml_mbps = ml.true_eval.sum_rate_mbps
        delta = ml_mbps - cvx_mbps
        rows.append(
            {
                "seed": seed,
                "cvxpy_Mbps": cvx_mbps,
                "matlab_Mbps": ml_mbps,
                "delta_matlab_minus_cvxpy_Mbps": delta,
                "cvxpy_feasible": cvx_feas,
                "matlab_feasible": ml.true_eval.feasible,
                "cvxpy_stop": cvx_stop,
                "matlab_stop": ml.diagnostics.get("stop_reason"),
                "matlab_accepted_steps": ml.diagnostics.get("accepted_steps"),
                "matlab_solver_status": ml.solver_status,
                "se_max_abs_diff": ml.diagnostics.get("se_max_abs_diff"),
            }
        )
        print(
            f"  cvxpy={cvx_mbps:.6f}  matlab={ml_mbps:.6f}  d={delta:+.6f}",
            flush=True,
        )

    deltas = [r["delta_matlab_minus_cvxpy_Mbps"] for r in rows]
    abs_d = [abs(d) for d in deltas]
    payload = {
        "label": "PAPER Fig. 6  I=10  J=3  8.8 MHz cap 25%",
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "n_seeds": len(rows),
        "cvxpy_source": "rerun" if args.rerun_cvxpy else str(CVXPY_CAMPAIGN),
        "summary": {
            "mean_cvxpy_Mbps": float(np.mean([r["cvxpy_Mbps"] for r in rows])),
            "mean_matlab_Mbps": float(np.mean([r["matlab_Mbps"] for r in rows])),
            "mean_delta_Mbps": float(np.mean(deltas)),
            "std_delta_Mbps": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "max_abs_delta_Mbps": float(max(abs_d)),
            "median_abs_delta_Mbps": float(np.median(abs_d)),
            "matlab_wins": int(sum(1 for d in deltas if d > 0)),
            "cvxpy_wins": int(sum(1 for d in deltas if d < 0)),
            "ties": int(sum(1 for d in deltas if d == 0)),
            "wilcoxon_p_two_sided": wilcoxon_signed_rank(deltas),
            "all_feasible_both": all(r["cvxpy_feasible"] and r["matlab_feasible"] for r in rows),
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    s = payload["summary"]
    print("\n=== SUMMARY ===")
    print(f"mean CVXPY  {s['mean_cvxpy_Mbps']:.6f} Mbps")
    print(f"mean MATLAB {s['mean_matlab_Mbps']:.6f} Mbps")
    print(f"mean delta  {s['mean_delta_Mbps']:+.6f} Mbps (MATLAB - CVXPY)")
    print(f"std delta   {s['std_delta_Mbps']:.6f} Mbps")
    print(f"max |delta| {s['max_abs_delta_Mbps']:.6f} Mbps")
    print(f"wins        MATLAB {s['matlab_wins']}/{len(rows)}  CVXPY {s['cvxpy_wins']}/{len(rows)}")
    print(f"Wilcoxon p  {s['wilcoxon_p_two_sided']:.4g}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
