"""Sequential SCA-style debug logs. Not the discarded joint Taylor audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from uavdt.config import SimConfig
from uavdt.evaluator import evaluate
from uavdt.sca.algorithm import solve_sca, true_gate_ok
from uavdt.sca.linearize import spectral_efficiency
from uavdt.sca.settings import SCASettings
from uavdt.scenario import generate_scenario


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return obj


def run_sca_seq_debug(
    seed: int,
    cfg: SimConfig,
    *,
    max_iterations: int = 30,
    step_size_m: float = 20.0,
    out_json: str = "results/sca_seq_debug.json",
    solver: str | None = None,
) -> dict:
    scenario = generate_scenario(seed, cfg)
    settings = SCASettings(
        max_iterations=max_iterations,
        epsilon=1e-4,
        step_size_m=step_size_m,
        solver=solver,
    )
    result = solve_sca(scenario, seed, settings=settings)
    a = result.allocation.hard_association()
    bw = result.allocation.bandwidth_hz
    se = spectral_efficiency(scenario.iot_xyz_m, result.uav_xyz_m, cfg)
    true = evaluate(scenario, result.uav_xyz_m, result.allocation)
    j_assoc = np.argmax(a, axis=1)
    links = []
    for i in range(a.shape[0]):
        j = int(j_assoc[i])
        links.append(
            {
                "iot": i,
                "uav": j,
                "SE_ij": float(se[i, j]),
                "B_ij_hz": float(bw[i, j]),
                "true_r_ij": float(true.rates_bit_per_s[i, j]),
            }
        )
    accepted = [row for row in result.history if row.accepted and row.iteration > 0]
    rejected = [row for row in result.history if not row.accepted]
    payload = {
        "seed": seed,
        "B_sys_hz": cfg.b_sys_hz,
        "method": result.diagnostics.get("method"),
        "stop_reason": result.diagnostics.get("stop_reason"),
        "converged": result.diagnostics.get("converged"),
        "initial_true_sum_rate_bit_per_s": result.history[0].true_objective
        if result.history
        else None,
        "final_true_sum_rate_bit_per_s": result.true_objective,
        "improvement_bit_per_s": (
            result.true_objective - result.history[0].true_objective
            if result.history
            else 0.0
        ),
        "accepted_steps": result.diagnostics.get("accepted_steps"),
        "rejected_steps": result.diagnostics.get("rejected_steps"),
        "step_size_reductions": result.diagnostics.get("step_size_reductions"),
        "uav_xyz_m": result.uav_xyz_m.tolist(),
        "bandwidth_hz": bw.tolist(),
        "association": a.tolist(),
        "processing": result.allocation.hard_processing().tolist(),
        "links": links,
        "aodt_s": true.aodt_s.tolist(),
        "rho": true.rho.tolist(),
        "min_uav_separation_m": result.history[-1].current_min_separation
        if result.history
        else None,
        "violations": {
            "qos": true.constraints.qos_violations,
            "aodt": true.constraints.aodt_violations,
            "sep": true.constraints.sep_violations,
            "cpu": true.constraints.cpu_unstable_count,
            "bw_excess_hz": true.constraints.bw_excess_hz,
        },
        "true_gate_ok": true_gate_ok(true),
        "evaluator_sum_rate_bit_per_s": true.sum_rate_bit_per_s,
        "reported_equals_evaluator": abs(
            result.true_objective - true.sum_rate_bit_per_s
        )
        <= 1e-6,
        "every_accepted_true_feasible": all(
            row.qos_violations == 0 and row.aodt_violations == 0 for row in accepted
        ),
        "n_accepted_logged": len(accepted),
        "n_rejected_logged": len(rejected),
        "history": [row.__dict__ for row in result.history],
        "diagnostics": result.diagnostics,
    }
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "iteration",
            "current_true_objective",
            "candidate_true_objective",
            "current_max_AoDT",
            "candidate_max_AoDT",
            "current_min_separation",
            "candidate_min_separation",
            "step_size",
            "bandwidth_objective",
            "true_objective",
            "accepted",
            "rejection_reason",
            "solver_status",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.history:
            writer.writerow({k: getattr(row, k) for k in fieldnames})
    payload["_json"] = str(path)
    payload["_csv"] = str(csv_path)
    return payload


def print_human_table(payload: dict) -> None:
    print("=== Algorithm 1 SCA debug ===")
    print("joint convexified (P); fixed association and processing")
    print(f"seed                   {payload['seed']}")
    print(f"B_sys_Hz               {payload['B_sys_hz']}")
    print(f"solver_backend        {payload['diagnostics'].get('solver_backend')}")
    if payload["diagnostics"].get("se_max_abs_diff") is not None:
        print(f"se_max_abs_diff       {payload['diagnostics'].get('se_max_abs_diff')}")
    print(f"stop_reason            {payload['stop_reason']}")
    if payload["stop_reason"] == "CONVERGED":
        print("CONVERGED")
    elif payload["stop_reason"] == "STEP_SIZE_LIMIT":
        print("STEP_SIZE_LIMIT")
    elif payload["stop_reason"] == "MAX_ITERATIONS":
        print("MAX_ITERATIONS")
    print(f"initial_true_Mbps      {payload['initial_true_sum_rate_bit_per_s'] / 1e6:.6g}")
    print(f"final_true_Mbps        {payload['final_true_sum_rate_bit_per_s'] / 1e6:.6g}")
    print(f"improvement_Mbps       {payload['improvement_bit_per_s'] / 1e6:.6g}")
    print(f"accepted_steps         {payload['accepted_steps']}")
    print(f"rejected_steps         {payload['rejected_steps']}")
    print(f"step_size_reductions   {payload['step_size_reductions']}")
    if payload["diagnostics"].get("final_step_m") is not None:
        print(f"final_step_m           {payload['diagnostics'].get('final_step_m')}")
    print(f"true_gate_ok           {payload['true_gate_ok']}")
    print(f"reported_equals_eval   {payload['reported_equals_evaluator']}")
    print("uav_xyz_m")
    for row in payload["uav_xyz_m"]:
        print(f"  {row}")
    print(f"AoDT_s                 {payload['aodt_s']}")
    print(f"rho                    {payload['rho']}")
    print(f"min_sep_m              {payload['min_uav_separation_m']}")
    print(f"violations             {payload['violations']}")
    print()
    print(
        f"{'iter':>4} {'step':>8} {'acc':>5} {'true_rate':>14} "
        f"{'maxAoDT':>9} {'min_sep':>8} {'reason':<22}"
    )
    for row in payload["history"]:
        acc = "Y" if row["accepted"] else "N"
        reason = row["rejection_reason"] or ("accepted" if row["accepted"] else "")
        if row["iteration"] == 0:
            reason = "init"
        cand = row["candidate_true_objective"]
        shown = cand if cand is not None else row["true_objective"]
        max_a = row["candidate_max_AoDT"]
        if max_a is None:
            max_a = row["current_max_AoDT"]
        min_s = row["candidate_min_separation"]
        if min_s is None:
            min_s = row["current_min_separation"]
        print(
            f"{row['iteration']:4d} {row['step_size']:8.4g} {acc:>5} "
            f"{shown:14.8g} {max_a:9.5g} {float(min_s):8.4f} {reason:<22}"
        )
    print()
    print("=== per-link SE, B, true r (final) ===")
    print(f"{'i':>3} {'j':>3} {'SE':>12} {'B_Hz':>14} {'true_r':>14}")
    for L in payload["links"]:
        print(
            f"{L['iot']:3d} {L['uav']:3d} {L['SE_ij']:12.6g} "
            f"{L['B_ij_hz']:14.6g} {L['true_r_ij']:14.6g}"
        )
    print()
    print(f"json  {payload.get('_json')}")
    print(f"csv   {payload.get('_csv')}")
