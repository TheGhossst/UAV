"""CVX/MOSEK spot-validation of frozen SCA. Does not change the solver."""

from __future__ import annotations

from uavdt.config import SimConfig
from uavdt.sca.matlab_bridge import matlab_available
from uavdt.sca.settings import SCASettings
from uavdt.sca.algorithm import solve_sca
from uavdt.scenario import generate_scenario


def spot_validate_sca(
    seed: int,
    cfg: SimConfig,
    *,
    max_iterations: int = 12,
    step_size_m: float = 20.0,
) -> dict:
    """Solve the same scenario with CVXPY and MATLAB CVX/MOSEK.

    Compares published true-evaluator scores. Raises if MATLAB SE disagrees
    (solve_sca already enforces se_max_abs_diff). A large objective gap is
    reported, not silently ignored.
    """
    scenario = generate_scenario(seed, cfg)
    py = solve_sca(
        scenario,
        seed,
        settings=SCASettings(
            solver=None,
            max_iterations=max_iterations,
            step_size_m=step_size_m,
        ),
    )
    out = {
        "seed": seed,
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "matlab_available": matlab_available(),
        "cvxpy": {
            "true_sum_rate_bit_per_s": py.true_objective,
            "true_sum_rate_Mbps": py.true_eval.sum_rate_mbps,
            "feasible": py.true_eval.feasible,
            "stop_reason": py.diagnostics.get("stop_reason"),
            "accepted_steps": py.diagnostics.get("accepted_steps"),
            "solver_status": py.solver_status,
        },
        "matlab": None,
        "abs_obj_diff_bit_per_s": None,
        "agreement": None,
    }
    if not matlab_available():
        out["agreement"] = "matlab_unavailable"
        return out
    ml = solve_sca(
        scenario,
        seed,
        settings=SCASettings(
            solver="matlab",
            max_iterations=max_iterations,
            step_size_m=step_size_m,
        ),
    )
    diff = abs(float(ml.true_objective) - float(py.true_objective))
    out["matlab"] = {
        "true_sum_rate_bit_per_s": ml.true_objective,
        "true_sum_rate_Mbps": ml.true_eval.sum_rate_mbps,
        "feasible": ml.true_eval.feasible,
        "stop_reason": ml.diagnostics.get("stop_reason"),
        "accepted_steps": ml.diagnostics.get("accepted_steps"),
        "solver_status": ml.solver_status,
        "se_max_abs_diff": ml.diagnostics.get("se_max_abs_diff"),
    }
    out["abs_obj_diff_bit_per_s"] = diff
    rel = diff / max(abs(float(py.true_objective)), 1.0)
    # Different inner solvers may take different accepted paths; flag a
    # large gap, do not hide it. 1% or 10 kbit/s, whichever is looser.
    out["agreement"] = "ok" if (diff <= 1.0e4 or rel <= 0.01) else "objective_mismatch"
    return out
