"""One-off local direction diagnostic (seed=1, 2.4 MHz). Not part of the package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uavdt.config import SimConfig
from uavdt.constraints import pairwise_uav_distance_m
from uavdt.evaluator import evaluate
from uavdt.models import Allocation
from uavdt.sca.algorithm import (
    _apply_step,
    position_ascent_direction,
    true_gate_ok,
)
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
from uavdt.sca.initialize import initialize_sca
from uavdt.sca.settings import SCASettings
from uavdt.scenario import generate_scenario, make_uav_xyz_m

SEED = 1
B_HZ = 2.4e6
STEP_SIZES = (0.5, 1.0, 2.0, 5.0)
CARDINALS = {
    "+x": (1.0, 0.0),
    "-x": (-1.0, 0.0),
    "+y": (0.0, 1.0),
    "-y": (0.0, -1.0),
}
DIAGONALS = {
    "+x+y": (1.0, 1.0),
    "+x-y": (1.0, -1.0),
    "-x+y": (-1.0, 1.0),
    "-x-y": (-1.0, -1.0),
}


def _min_sep(uav: np.ndarray) -> float:
    d = pairwise_uav_distance_m(uav)
    finite = d[np.isfinite(d)]
    return float(np.min(finite)) if finite.size else float("inf")


def _try_candidate(
    scenario,
    uav0: np.ndarray,
    xy_new: np.ndarray,
    a,
    b,
    settings,
    baseline: float,
) -> dict:
    cfg = scenario.cfg
    uav = make_uav_xyz_m(xy_new, cfg.uav_height_m)
    bw_res = solve_bandwidth_at_fixed_q(scenario, uav, a, b, settings)
    if bw_res.infeasible:
        return {
            "feasible": False,
            "true_sum_rate": None,
            "delta_vs_baseline": None,
            "improves": False,
            "bw_status": bw_res.status,
            "gate": False,
            "qos": -1,
            "aodt": -1,
            "sep": -1,
            "min_sep": _min_sep(uav),
        }
    ev = evaluate(scenario, uav, Allocation(a, b, bw_res.bandwidth_hz))
    gate = true_gate_ok(ev)
    rate = float(ev.sum_rate_bit_per_s)
    return {
        "feasible": ev.feasible,
        "true_sum_rate": rate,
        "true_sum_rate_Mbps": rate / 1e6,
        "delta_vs_baseline": rate - baseline,
        "delta_Mbps": (rate - baseline) / 1e6,
        "improves": gate and rate > baseline + 1.0,
        "bw_status": bw_res.status,
        "gate": gate,
        "qos": ev.constraints.qos_violations,
        "aodt": ev.constraints.aodt_violations,
        "sep": ev.constraints.sep_violations,
        "min_sep": _min_sep(uav),
        "uav_xy": uav[:, :2].tolist(),
    }


def _move_one_uav(uav0: np.ndarray, j: int, ux: float, uy: float, step: float, cfg) -> np.ndarray:
  xy = uav0[:, :2].copy()
  # L-inf step: each axis moves by step * unit component (cardinal = step in one axis).
  xy[j, 0] = np.clip(xy[j, 0] + ux * step, 0.0, cfg.area_x_m)
  xy[j, 1] = np.clip(xy[j, 1] + uy * step, 0.0, cfg.area_y_m)
  return xy


def main() -> None:
    cfg = SimConfig(b_sys_hz=B_HZ)
    scenario = generate_scenario(SEED, cfg)
    settings = SCASettings()
    uav0, alloc = initialize_sca(scenario, SEED)
    a = alloc.hard_association()
    b = alloc.hard_processing()
    bw0 = solve_bandwidth_at_fixed_q(scenario, uav0, a, b, settings)
    base_ev = evaluate(scenario, uav0, Allocation(a, b, bw0.bandwidth_hz))
    baseline = float(base_ev.sum_rate_bit_per_s)
    print(f"baseline true sum rate = {baseline / 1e6:.6f} Mbps ({baseline:.3f} bit/s)")
    print(f"baseline gate ok = {true_gate_ok(base_ev)}  min_sep = {_min_sep(uav0):.4f} m")
    print(f"uav0 xy:\n{uav0[:, :2]}")
    print()

    rows = []
    best = {"delta": 0.0, "label": "baseline"}

    # Algorithm ascent direction (all UAVs jointly)
    direction = position_ascent_direction(scenario, uav0, bw0.bandwidth_hz, a, settings.fd_step_m)
    print("=== algorithm ascent direction (all UAVs, normalized L-inf step) ===")
    print(f"raw direction (dx,dy) per UAV:\n{direction}")
    for step in STEP_SIZES:
        xy = _apply_step(uav0, direction, step, cfg.area_x_m, cfg.area_y_m)
        r = _try_candidate(scenario, uav0, xy, a, b, settings, baseline)
        label = f"algo_dir step={step}m"
        rows.append({"kind": "algorithm", "uav": "all", "dir": "ascent", "step_m": step, **r})
        if r["true_sum_rate"] is None:
            print(
                f"  step={step:4.1f} m  INFEASIBLE bw_status={r['bw_status']}  "
                f"min_sep={r['min_sep']:.3f}"
            )
            continue
        print(
            f"  step={step:4.1f} m  rate={r['true_sum_rate_Mbps']:.6f} Mbps  "
            f"delta={r['delta_Mbps']:+.6f}  gate={r['gate']}  "
            f"qos={r['qos']} sep={r['sep']} min_sep={r['min_sep']:.3f}  "
            f"improves={r['improves']}"
        )
        if r["delta_vs_baseline"] is not None and r["delta_vs_baseline"] > best["delta"]:
            best = {"delta": r["delta_vs_baseline"], "label": label}
    print()

    for j in range(uav0.shape[0]):
        print(f"=== UAV {j} at {uav0[j, :2]} ===")
        for step in STEP_SIZES:
            for name, (ux, uy) in {**CARDINALS, **DIAGONALS}.items():
                xy = _move_one_uav(uav0, j, ux, uy, step, cfg)
                r = _try_candidate(scenario, uav0, xy, a, b, settings, baseline)
                label = f"uav{j} {name} step={step}m"
                rows.append(
                    {
                        "kind": "cardinal" if name in CARDINALS else "diagonal",
                        "uav": j,
                        "dir": name,
                        "step_m": step,
                        **r,
                    }
                )
                if r["true_sum_rate"] is None:
                    continue
                if r["improves"]:
                    print(
                        f"  ** IMPROVES ** step={step} {name:5s}  "
                        f"rate={r['true_sum_rate_Mbps']:.6f}  delta={r['delta_Mbps']:+.6f}  "
                        f"gate={r['gate']}"
                    )
                if r["delta_vs_baseline"] is not None and r["delta_vs_baseline"] > best["delta"]:
                    best = {"delta": r["delta_vs_baseline"], "label": label}
        # compact summary: best per step for this UAV
        for step in STEP_SIZES:
            subset = [
                x
                for x in rows
                if x.get("uav") == j
                and x.get("step_m") == step
                and x.get("kind") in ("cardinal", "diagonal")
            ]
            feasible = [x for x in subset if x.get("gate")]
            if not feasible:
                print(f"  step={step} m: no true-feasible direction")
                continue
            top = max(feasible, key=lambda x: x["true_sum_rate"])
            d_mbps = top["delta_vs_baseline"] / 1e6
            print(
                f"  step={step} m best: {top['dir']:5s}  "
                f"rate={top['true_sum_rate'] / 1e6:.6f}  delta={d_mbps:+.6f}  "
                f"improves={top['improves']}"
            )
        print()

    improving = [x for x in rows if x.get("improves")]
    gate_feasible = [x for x in rows if x.get("gate")]
    print("=== summary ===")
    print(f"total probes: {len(rows)}")
    print(f"true-feasible (gate): {len(gate_feasible)}")
    print(f"improve baseline by >1 bit/s: {len(improving)}")
    if improving:
        print("improving candidates:")
        for x in sorted(improving, key=lambda z: -z["delta_vs_baseline"])[:15]:
            print(
                f"  {x.get('kind')} uav={x.get('uav')} dir={x.get('dir')} "
                f"step={x.get('step_m')} delta_Mbps={x['delta_Mbps']:+.6f}"
            )
    else:
        print("No nearby probe improved the true feasible sum rate.")
    print(f"best overall (any label): {best['label']}  delta={best['delta'] / 1e6:+.6f} Mbps")

    out = Path("results/local_direction_diag_seed1_2p4mhz.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "seed": SEED,
                "B_sys_hz": B_HZ,
                "baseline_bit_per_s": baseline,
                "baseline_Mbps": baseline / 1e6,
                "best": best,
                "n_improving": len(improving),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
