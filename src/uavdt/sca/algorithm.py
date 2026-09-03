"""Algorithm 1 SCA for Problem (P). Physics stay in the core evaluator.

Each iteration solves one convexified (P) for UAV positions and bandwidth
(paper §V: first-order Taylor of (3)–(5) and (25)). Association and
processing are not in Algorithm 1's update list and stay at initialization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from uavdt.constraints import pairwise_uav_distance_m
from uavdt.evaluator import EvalResult, evaluate
from uavdt.models import Allocation, Scenario
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q, solve_joint_convex_step
from uavdt.sca.initialize import initialize_sca
from uavdt.sca.linearize import se_jacobian
from uavdt.sca.settings import SCASettings
from uavdt.scenario import make_uav_xyz_m


@dataclass
class SCAIterationLog:
    iteration: int
    current_true_objective: float
    candidate_true_objective: float | None
    current_max_AoDT: float
    candidate_max_AoDT: float | None
    current_min_separation: float
    candidate_min_separation: float | None
    step_size: float
    bandwidth_objective: float | None
    true_objective: float
    accepted: bool
    rejection_reason: str
    solver_status: str
    qos_violations: int
    aodt_violations: int
    max_cpu_load: float
    bandwidth_usage: float
    solve_time_s: float


@dataclass
class SCAResult:
    uav_xyz_m: np.ndarray
    allocation: Allocation
    surrogate_objective: float
    true_objective: float
    true_eval: EvalResult
    history: list[SCAIterationLog] = field(default_factory=list)
    solver_status: str = ""
    solver_name: str = ""
    n_iterations: int = 0
    diagnostics: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        hist = [asdict(row) for row in self.history]
        ev = self.true_eval
        return {
            "uav_xyz_m": self.uav_xyz_m.tolist(),
            "association": self.allocation.hard_association().tolist(),
            "processing": self.allocation.hard_processing().tolist(),
            "bandwidth_hz": self.allocation.bandwidth_hz.tolist(),
            "surrogate_objective": self.surrogate_objective,
            "true_objective": self.true_objective,
            "true_sum_rate_bit_per_s": ev.sum_rate_bit_per_s,
            "true_sum_rate_Mbps": ev.sum_rate_mbps,
            "aodt_s": ev.aodt_s.tolist(),
            "rho": ev.rho.tolist(),
            "feasible": ev.feasible,
            "n_iterations": self.n_iterations,
            "solver_status": self.solver_status,
            "solver_name": self.solver_name,
            "history": hist,
            "diagnostics": self.diagnostics,
        }


def _min_sep(uav_xyz_m: np.ndarray) -> float:
    d = pairwise_uav_distance_m(uav_xyz_m)
    finite = d[np.isfinite(d)]
    return float(np.min(finite)) if finite.size else float("inf")


def _max_aodt(ev: EvalResult) -> float:
    finite = ev.aodt_s[np.isfinite(ev.aodt_s)]
    return float(np.max(finite)) if finite.size else float("inf")


def true_gate_ok(ev: EvalResult) -> bool:
    """Item 5: QoS, AoDT, CPU, bandwidth, and separation must all be clean."""
    c = ev.constraints
    return (
        c.qos_violations == 0
        and c.aodt_violations == 0
        and c.cpu_unstable_count == 0
        and c.bandwidth_budget_ok
        and c.bandwidth_support_ok
        and c.sep_violations == 0
        and c.uav_in_field_ok
    )


def position_ascent_direction(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    bandwidth_hz: np.ndarray,
    association: np.ndarray,
    step_m: float,
) -> np.ndarray:
    """Local first-order direction of true sum rate with B held fixed.

    d/d q_j of sum_i a_ij B_ij SE_ij(q_j) = sum_i a_ij B_ij grad SE_ij.
    Jacobian of SE is numerical from the core channel.
    """
    a = association > 0.5
    _se, g_x, g_y = se_jacobian(
        scenario.iot_xyz_m, uav_xyz_m, scenario.cfg, step_m=step_m
    )
    w = np.where(a, bandwidth_hz, 0.0)
    dx = np.sum(w * g_x, axis=0)
    dy = np.sum(w * g_y, axis=0)
    return np.column_stack([dx, dy])


def _apply_step(
    uav_xyz_m: np.ndarray,
    direction: np.ndarray,
    step_size_m: float,
    area_x_m: float,
    area_y_m: float,
) -> np.ndarray:
    scale = float(np.max(np.abs(direction)))
    xy = uav_xyz_m[:, :2].copy()
    if scale > 1e-18:
        delta = direction * (step_size_m / scale)
        xy = xy + delta
    xy[:, 0] = np.clip(xy[:, 0], 0.0, area_x_m)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, area_y_m)
    return xy


def _use_matlab(settings: SCASettings) -> bool:
    return settings.solver in {"matlab", "MATLAB", "mosek", "MOSEK"}


def _effective_step_size(settings: SCASettings) -> float:
    if settings.trust_region_m is not None and settings.trust_region_m <= 0.0:
        return 0.0
    return float(settings.step_size_m)


def classify_stop_reason(
    proposed: str,
    *,
    accepted_steps: int,
    step_size: float,
    min_step_size: float,
) -> str:
    """Top-level SCA verdict. Independent of LP solver status.

    MOSEK/CVXPY `Solved` is not convergence. Line-search exhaustion with
    no accepted true-feasible move is `STEP_SIZE_LIMIT`.
    CONVERGED requires accepted_steps >= 1. MAX_ITERATIONS only when the
    iteration cap is hit with step_size still above min_step_size.

    Keep in sync with matlab/sca_seq.m `classify_stop_reason`.
    """
    if proposed.startswith("init_"):
        return proposed
    if (
        proposed == "MAX_ITERATIONS"
        and int(accepted_steps) > 0
        and float(step_size) <= float(min_step_size)
    ):
        return "CONVERGED"
    if int(accepted_steps) == 0 and proposed != "MAX_ITERATIONS":
        if float(step_size) <= float(min_step_size) or proposed == "CONVERGED":
            return "STEP_SIZE_LIMIT"
    return proposed


def _opt_float(value) -> float | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, (list, tuple, np.ndarray)) and np.size(value) == 0:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _f(value, default: float = 0.0) -> float:
    x = _opt_float(value)
    return default if x is None else x


def _as_row_list(rows) -> list[dict]:
    if rows is None:
        return []
    if isinstance(rows, dict):
        return [rows]
    return [row for row in rows if isinstance(row, dict)]


def _accepted(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "accepted"}
    return bool(value)


def _history_from_matlab(rows) -> list[SCAIterationLog]:
    history: list[SCAIterationLog] = []
    for row in _as_row_list(rows):
        history.append(
            SCAIterationLog(
                iteration=int(_f(row.get("iteration"), 0.0)),
                current_true_objective=_f(row.get("current_true_objective")),
                candidate_true_objective=_opt_float(row.get("candidate_true_objective")),
                current_max_AoDT=_f(row.get("current_max_AoDT"), float("inf")),
                candidate_max_AoDT=_opt_float(row.get("candidate_max_AoDT")),
                current_min_separation=_f(row.get("current_min_separation"), float("inf")),
                candidate_min_separation=_opt_float(row.get("candidate_min_separation")),
                step_size=_f(row.get("step_size")),
                bandwidth_objective=_opt_float(row.get("bandwidth_objective")),
                true_objective=_f(row.get("true_objective")),
                accepted=_accepted(row.get("accepted", False)),
                rejection_reason=str(row.get("rejection_reason", "")),
                solver_status=str(row.get("solver_status", "")),
                qos_violations=int(_f(row.get("qos_violations"), 0.0)),
                aodt_violations=int(_f(row.get("aodt_violations"), 0.0)),
                max_cpu_load=_f(row.get("max_cpu_load")),
                bandwidth_usage=_f(row.get("bandwidth_usage")),
                solve_time_s=_f(row.get("solve_time_s")),
            )
        )
    return history


def _common_diagnostics(seed: int, cfg, settings: SCASettings, extra: dict) -> dict:
    out = {
        "method": "algorithm_1_sca_problem_p",
        "label": (
            "Algorithm 1 SCA of Problem (P): joint convexified (q, B) "
            "with first-order Taylor of (3)–(5) and (25). a_ij, b_ij fixed."
        ),
        "binaries": "a_ij and b_ij fixed after initialization (IMPLEMENTATION CHOICE)",
        "seed": seed,
        "b_sys_hz": cfg.b_sys_hz,
        "step_size_m": settings.step_size_m,
        "association_init_equals_final": True,
        "processing_init_equals_final": True,
        "solver_backend": settings.solver or "cvxpy",
    }
    out.update(extra)
    return out


def _solve_sca_matlab(
    scenario: Scenario,
    seed: int,
    settings: SCASettings,
    uav: np.ndarray,
    alloc: Allocation,
) -> SCAResult:
    from uavdt.sca.matlab_bridge import matlab_available, run_sequential_matlab

    if not matlab_available():
        raise RuntimeError(
            "MATLAB CVX/MOSEK requested but matlab.exe or matlab/*.m was not found"
        )
    a = alloc.hard_association()
    b = alloc.hard_processing()
    sol = run_sequential_matlab(scenario, uav, a, b, settings)
    se_diff = _opt_float(sol.get("se_max_abs_diff"))
    if se_diff is not None and se_diff > 1e-6:
        raise RuntimeError(
            "MATLAB channel_se disagrees with Python spectral_efficiency: "
            f"max abs diff {se_diff:.3e}"
        )

    if bool(sol.get("infeasible")):
        ev0 = evaluate(scenario, uav, alloc)
        stop = str(sol.get("stop_reason", "init_bandwidth_infeasible"))
        history0 = _history_from_matlab(sol.get("history"))
        if not history0:
            history0 = [
                SCAIterationLog(
                    iteration=0,
                    current_true_objective=ev0.sum_rate_bit_per_s,
                    candidate_true_objective=None,
                    current_max_AoDT=_max_aodt(ev0),
                    candidate_max_AoDT=None,
                    current_min_separation=_min_sep(uav),
                    candidate_min_separation=None,
                    step_size=_effective_step_size(settings),
                    bandwidth_objective=None,
                    true_objective=ev0.sum_rate_bit_per_s,
                    accepted=False,
                    rejection_reason=stop,
                    solver_status=str(sol.get("cvx_status", "")),
                    qos_violations=ev0.constraints.qos_violations,
                    aodt_violations=ev0.constraints.aodt_violations,
                    max_cpu_load=float(np.max(ev0.rho)) if ev0.rho.size else 0.0,
                    bandwidth_usage=float(np.sum(alloc.bandwidth_hz)),
                    solve_time_s=0.0,
                )
            ]
        return SCAResult(
            uav_xyz_m=uav,
            allocation=Allocation(a, b, alloc.bandwidth_hz),
            surrogate_objective=float("nan"),
            true_objective=float(ev0.sum_rate_bit_per_s),
            true_eval=ev0,
            history=history0,
            solver_status=str(sol.get("cvx_status", "")),
            solver_name=str(sol.get("solver_name", "MATLAB-CVX-MOSEK")),
            n_iterations=0,
            diagnostics=_common_diagnostics(
                seed,
                scenario.cfg,
                settings,
                {
                    "stop_reason": stop,
                    "converged": False,
                    "accepted_steps": 0,
                    "rejected_steps": 1,
                    "step_size_reductions": 0,
                    "se_max_abs_diff": se_diff,
                    "solver_backend": "MATLAB-CVX-MOSEK",
                },
            ),
        )

    i_n = int(sol.get("I", a.shape[0]))
    j_n = int(sol.get("J", a.shape[1]))
    uav_out = np.asarray(sol["uav_xyz"], dtype=float).reshape(-1).reshape((j_n, 3), order="F")
    uav_out = np.column_stack(
        [
            np.clip(uav_out[:, 0], 0.0, scenario.cfg.area_x_m),
            np.clip(uav_out[:, 1], 0.0, scenario.cfg.area_y_m),
            np.full(j_n, scenario.cfg.uav_height_m),
        ]
    )
    bw = np.asarray(sol["B"], dtype=float).reshape(-1).reshape((i_n, j_n), order="F")
    bw = np.maximum(bw, 0.0)
    bw[a < 0.5] = 0.0
    if bw.sum() > scenario.cfg.b_sys_hz + 1e-6:
        bw *= scenario.cfg.b_sys_hz / bw.sum()

    true_final = evaluate(scenario, uav_out, Allocation(a, b, bw))
    if not true_gate_ok(true_final):
        c = true_final.constraints
        raise RuntimeError(
            "MATLAB SCA returned a point that Python evaluate() rejects: "
            f"qos={c.qos_violations} aodt={c.aodt_violations} "
            f"cpu={c.cpu_unstable_count} sep={c.sep_violations} "
            f"bw_excess={c.bw_excess_hz} matlab_feasible={sol.get('feasible')}"
        )

    history = _history_from_matlab(sol.get("history"))
    if not history:
        history = [
            SCAIterationLog(
                iteration=0,
                current_true_objective=true_final.sum_rate_bit_per_s,
                candidate_true_objective=true_final.sum_rate_bit_per_s,
                current_max_AoDT=_max_aodt(true_final),
                candidate_max_AoDT=_max_aodt(true_final),
                current_min_separation=_min_sep(uav_out),
                candidate_min_separation=_min_sep(uav_out),
                step_size=_effective_step_size(settings),
                bandwidth_objective=None,
                true_objective=true_final.sum_rate_bit_per_s,
                accepted=True,
                rejection_reason="init",
                solver_status=str(sol.get("cvx_status", "")),
                qos_violations=0,
                aodt_violations=0,
                max_cpu_load=float(np.max(true_final.rho)) if true_final.rho.size else 0.0,
                bandwidth_usage=float(np.sum(bw)),
                solve_time_s=0.0,
            )
        ]
    # Published score is always the core evaluator, not MATLAB's copy of Eq. (6).
    history[-1].true_objective = float(true_final.sum_rate_bit_per_s)
    matlab_obj = _opt_float(sol.get("true_obj"))
    accepted = int(_f(sol.get("accepted_steps"), 0.0))
    final_step = _opt_float(sol.get("final_step_m"))
    if final_step is None:
        final_step = float(settings.step_size_m) * (
            float(settings.step_size_shrink) ** int(_f(sol.get("step_size_reductions"), 0.0))
        )
    stop = classify_stop_reason(
        str(sol.get("stop_reason", "MAX_ITERATIONS")),
        accepted_steps=accepted,
        step_size=final_step,
        min_step_size=float(settings.min_step_size_m),
    )
    return SCAResult(
        uav_xyz_m=uav_out,
        allocation=Allocation(a, b, bw),
        surrogate_objective=float("nan"),
        true_objective=float(true_final.sum_rate_bit_per_s),
        true_eval=true_final,
        history=history,
        solver_status=str(sol.get("cvx_status", "")),
        solver_name=str(sol.get("solver_name", "MATLAB-CVX-MOSEK")),
        n_iterations=int(history[-1].iteration),
        diagnostics=_common_diagnostics(
            seed,
            scenario.cfg,
            settings,
            {
                "stop_reason": stop,
                "converged": stop == "CONVERGED",
                "accepted_steps": accepted,
                "rejected_steps": int(_f(sol.get("rejected_steps"), 0.0)),
                "step_size_reductions": int(_f(sol.get("step_size_reductions"), 0.0)),
                "final_step_m": final_step,
                "se_max_abs_diff": se_diff,
                "matlab_true_obj": matlab_obj,
                "matlab_python_obj_abs_diff": (
                    None
                    if matlab_obj is None
                    else abs(matlab_obj - true_final.sum_rate_bit_per_s)
                ),
                "n_lp_solves": sol.get("n_lp_solves"),
                "solver_backend": "MATLAB-CVX-MOSEK",
            },
        ),
    )


def solve_sca(
    scenario: Scenario,
    seed: int,
    settings: SCASettings | None = None,
    uav_xyz_m: np.ndarray | None = None,
    allocation: Allocation | None = None,
) -> SCAResult:
    """Sequential SCA-style solver. Final scores always come from evaluate()."""
    settings = settings or SCASettings()
    cfg = scenario.cfg
    if uav_xyz_m is None or allocation is None:
        uav, alloc = initialize_sca(scenario, seed)
    else:
        uav = np.asarray(uav_xyz_m, dtype=float)
        alloc = allocation
    a = alloc.hard_association()
    b = alloc.hard_processing()
    if _use_matlab(settings):
        return _solve_sca_matlab(scenario, seed, settings, uav, alloc)

    bw_init = solve_bandwidth_at_fixed_q(scenario, uav, a, b, settings)
    if bw_init.infeasible:
        ev0 = evaluate(scenario, uav, Allocation(a, b, alloc.bandwidth_hz))
        history0 = [
            SCAIterationLog(
                iteration=0,
                current_true_objective=ev0.sum_rate_bit_per_s,
                candidate_true_objective=None,
                current_max_AoDT=_max_aodt(ev0),
                candidate_max_AoDT=None,
                current_min_separation=_min_sep(uav),
                candidate_min_separation=None,
                step_size=_effective_step_size(settings),
                bandwidth_objective=None,
                true_objective=ev0.sum_rate_bit_per_s,
                accepted=False,
                rejection_reason="init_bandwidth_infeasible",
                solver_status=bw_init.status,
                qos_violations=ev0.constraints.qos_violations,
                aodt_violations=ev0.constraints.aodt_violations,
                max_cpu_load=float(np.max(ev0.rho)) if ev0.rho.size else 0.0,
                bandwidth_usage=float(np.sum(alloc.bandwidth_hz)),
                solve_time_s=bw_init.solve_time_s,
            )
        ]
        return SCAResult(
            uav_xyz_m=uav,
            allocation=Allocation(a, b, alloc.bandwidth_hz),
            surrogate_objective=float("nan"),
            true_objective=float(ev0.sum_rate_bit_per_s),
            true_eval=ev0,
            history=history0,
            solver_status=bw_init.status,
            solver_name=bw_init.solver_name,
            n_iterations=0,
            diagnostics={
                "method": "sequential_sca_style_fixed_association_processing",
                "stop_reason": "init_bandwidth_infeasible",
                "converged": False,
                "accepted_steps": 0,
                "rejected_steps": 1,
                "step_size_reductions": 0,
            },
        )

    bw = bw_init.bandwidth_hz
    current_eval = evaluate(scenario, uav, Allocation(a, b, bw))
    if not true_gate_ok(current_eval):
        raise RuntimeError(
            "sequential SCA-style solver requires a true-feasible start; "
            f"violations qos={current_eval.constraints.qos_violations} "
            f"aodt={current_eval.constraints.aodt_violations} "
            f"cpu={current_eval.constraints.cpu_unstable_count} "
            f"sep={current_eval.constraints.sep_violations} "
            f"bw_excess={current_eval.constraints.bw_excess_hz}"
        )

    history = [
        SCAIterationLog(
            iteration=0,
            current_true_objective=current_eval.sum_rate_bit_per_s,
            candidate_true_objective=current_eval.sum_rate_bit_per_s,
            current_max_AoDT=_max_aodt(current_eval),
            candidate_max_AoDT=_max_aodt(current_eval),
            current_min_separation=_min_sep(uav),
            candidate_min_separation=_min_sep(uav),
            step_size=_effective_step_size(settings),
            bandwidth_objective=bw_init.objective,
            true_objective=current_eval.sum_rate_bit_per_s,
            accepted=True,
            rejection_reason="init",
            solver_status=bw_init.status,
            qos_violations=0,
            aodt_violations=0,
            max_cpu_load=float(np.max(current_eval.rho)) if current_eval.rho.size else 0.0,
            bandwidth_usage=float(np.sum(bw)),
            solve_time_s=bw_init.solve_time_s,
        )
    ]

    step = _effective_step_size(settings)
    accepted_steps = 0
    rejected_steps = 0
    n_reductions = 0
    stop_reason = "MAX_ITERATIONS"
    last_status = bw_init.status
    solver_name = bw_init.solver_name
    n = 0

    if step <= 0.0:
        stop_reason = "STEP_SIZE_LIMIT"
    else:
        for n in range(1, settings.max_iterations + 1):
            if step <= settings.min_step_size_m:
                stop_reason = (
                    "STEP_SIZE_LIMIT" if accepted_steps == 0 else "CONVERGED"
                )
                n = n - 1
                break
            joint = solve_joint_convex_step(
                scenario, uav, bw, a, b, step, settings
            )
            last_status = joint.status
            solver_name = joint.solver_name
            if joint.infeasible:
                rejected_steps += 1
                n_reductions += 1
                step *= settings.step_size_shrink
                history.append(
                    SCAIterationLog(
                        iteration=n,
                        current_true_objective=current_eval.sum_rate_bit_per_s,
                        candidate_true_objective=None,
                        current_max_AoDT=_max_aodt(current_eval),
                        candidate_max_AoDT=None,
                        current_min_separation=_min_sep(uav),
                        candidate_min_separation=_min_sep(uav),
                        step_size=step / settings.step_size_shrink,
                        bandwidth_objective=None,
                        true_objective=current_eval.sum_rate_bit_per_s,
                        accepted=False,
                        rejection_reason="convex_p_infeasible",
                        solver_status=joint.status,
                        qos_violations=0,
                        aodt_violations=0,
                        max_cpu_load=float(np.max(current_eval.rho))
                        if current_eval.rho.size
                        else 0.0,
                        bandwidth_usage=float(np.sum(bw)),
                        solve_time_s=joint.solve_time_s,
                    )
                )
                continue
            xy_cand = np.column_stack([joint.x_m, joint.y_m])
            move = float(np.max(np.abs(xy_cand - uav[:, :2])))
            if move < settings.min_step_size_m:
                stop_reason = "CONVERGED"
                history.append(
                    SCAIterationLog(
                        iteration=n,
                        current_true_objective=current_eval.sum_rate_bit_per_s,
                        candidate_true_objective=current_eval.sum_rate_bit_per_s,
                        current_max_AoDT=_max_aodt(current_eval),
                        candidate_max_AoDT=_max_aodt(current_eval),
                        current_min_separation=_min_sep(uav),
                        candidate_min_separation=_min_sep(uav),
                        step_size=step,
                        bandwidth_objective=joint.objective,
                        true_objective=current_eval.sum_rate_bit_per_s,
                        accepted=False,
                        rejection_reason="stationary_convex_p",
                        solver_status=joint.status,
                        qos_violations=0,
                        aodt_violations=0,
                        max_cpu_load=float(np.max(current_eval.rho))
                        if current_eval.rho.size
                        else 0.0,
                        bandwidth_usage=float(np.sum(bw)),
                        solve_time_s=joint.solve_time_s,
                    )
                )
                break
            uav_cand = make_uav_xyz_m(xy_cand, cfg.uav_height_m)
            bw_res = solve_bandwidth_at_fixed_q(scenario, uav_cand, a, b, settings)
            last_status = bw_res.status
            solver_name = bw_res.solver_name

            if bw_res.infeasible:
                rejected_steps += 1
                n_reductions += 1
                step *= settings.step_size_shrink
                history.append(
                    SCAIterationLog(
                        iteration=n,
                        current_true_objective=current_eval.sum_rate_bit_per_s,
                        candidate_true_objective=None,
                        current_max_AoDT=_max_aodt(current_eval),
                        candidate_max_AoDT=None,
                        current_min_separation=_min_sep(uav),
                        candidate_min_separation=_min_sep(uav_cand),
                        step_size=step / settings.step_size_shrink,
                        bandwidth_objective=None,
                        true_objective=current_eval.sum_rate_bit_per_s,
                        accepted=False,
                        rejection_reason="bandwidth_infeasible",
                        solver_status=bw_res.status,
                        qos_violations=-1,
                        aodt_violations=-1,
                        max_cpu_load=float(np.max(current_eval.rho))
                        if current_eval.rho.size
                        else 0.0,
                        bandwidth_usage=float(np.sum(bw)),
                        solve_time_s=bw_res.solve_time_s,
                    )
                )
                continue

            cand_eval = evaluate(scenario, uav_cand, Allocation(a, b, bw_res.bandwidth_hz))
            cand_ok = true_gate_ok(cand_eval)
            improved = cand_eval.sum_rate_bit_per_s > (
                current_eval.sum_rate_bit_per_s + settings.improvement_tolerance
            )
            if (not cand_ok) or (not improved):
                rejected_steps += 1
                n_reductions += 1
                reason = "true_infeasible" if not cand_ok else "no_true_improvement"
                step *= settings.step_size_shrink
                history.append(
                    SCAIterationLog(
                        iteration=n,
                        current_true_objective=current_eval.sum_rate_bit_per_s,
                        candidate_true_objective=cand_eval.sum_rate_bit_per_s,
                        current_max_AoDT=_max_aodt(current_eval),
                        candidate_max_AoDT=_max_aodt(cand_eval),
                        current_min_separation=_min_sep(uav),
                        candidate_min_separation=_min_sep(uav_cand),
                        step_size=step / settings.step_size_shrink,
                        bandwidth_objective=bw_res.objective,
                        true_objective=current_eval.sum_rate_bit_per_s,
                        accepted=False,
                        rejection_reason=reason,
                        solver_status=bw_res.status,
                        qos_violations=cand_eval.constraints.qos_violations,
                        aodt_violations=cand_eval.constraints.aodt_violations,
                        max_cpu_load=float(np.max(cand_eval.rho))
                        if cand_eval.rho.size
                        else 0.0,
                        bandwidth_usage=float(np.sum(bw_res.bandwidth_hz)),
                        solve_time_s=bw_res.solve_time_s,
                    )
                )
                if accepted_steps > 0 and step <= settings.min_step_size_m:
                    stop_reason = "CONVERGED"
                    break
                continue

            delta = cand_eval.sum_rate_bit_per_s - current_eval.sum_rate_bit_per_s
            prev_obj = current_eval.sum_rate_bit_per_s
            prev_aodt = _max_aodt(current_eval)
            prev_sep = _min_sep(uav)
            uav = uav_cand
            bw = bw_res.bandwidth_hz
            current_eval = cand_eval
            accepted_steps += 1
            used_step = step
            history.append(
                SCAIterationLog(
                    iteration=n,
                    current_true_objective=prev_obj,
                    candidate_true_objective=cand_eval.sum_rate_bit_per_s,
                    current_max_AoDT=prev_aodt,
                    candidate_max_AoDT=_max_aodt(cand_eval),
                    current_min_separation=prev_sep,
                    candidate_min_separation=_min_sep(uav),
                    step_size=used_step,
                    bandwidth_objective=bw_res.objective,
                    true_objective=current_eval.sum_rate_bit_per_s,
                    accepted=True,
                    rejection_reason="",
                    solver_status=bw_res.status,
                    qos_violations=0,
                    aodt_violations=0,
                    max_cpu_load=float(np.max(current_eval.rho))
                    if current_eval.rho.size
                    else 0.0,
                    bandwidth_usage=float(np.sum(bw)),
                    solve_time_s=bw_res.solve_time_s,
                )
            )
            if abs(delta) <= settings.epsilon:
                stop_reason = "CONVERGED"
                break
        else:
            stop_reason = "MAX_ITERATIONS"

    true_final = evaluate(scenario, uav, Allocation(a, b, bw))
    if abs(true_final.sum_rate_bit_per_s - current_eval.sum_rate_bit_per_s) > 1e-6:
        current_eval = true_final

    stop_reason = classify_stop_reason(
        stop_reason,
        accepted_steps=accepted_steps,
        step_size=step,
        min_step_size=float(settings.min_step_size_m),
    )

    return SCAResult(
        uav_xyz_m=uav,
        allocation=Allocation(a, b, bw),
        surrogate_objective=float("nan"),
        true_objective=float(current_eval.sum_rate_bit_per_s),
        true_eval=current_eval,
        history=history,
        solver_status=last_status,
        solver_name=solver_name,
        n_iterations=int(history[-1].iteration),
        diagnostics={
            "method": "algorithm_1_sca_problem_p",
            "label": (
                "Algorithm 1 SCA of Problem (P): joint convexified (q, B) "
                "with first-order Taylor of (3)–(5) and (25). a_ij, b_ij fixed."
            ),
            "binaries": "a_ij and b_ij fixed after initialization (IMPLEMENTATION CHOICE)",
            "stop_reason": stop_reason,
            "converged": stop_reason == "CONVERGED",
            "accepted_steps": accepted_steps,
            "final_step_m": step,
            "rejected_steps": rejected_steps,
            "step_size_reductions": n_reductions,
            "seed": seed,
            "b_sys_hz": cfg.b_sys_hz,
            "step_size_m": settings.step_size_m,
            "association_init_equals_final": True,
            "processing_init_equals_final": True,
            "solver_backend": settings.solver or "cvxpy",
        },
    )


def write_history(result: SCAResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_jsonable()
    if path.suffix.lower() == ".csv":
        import csv

        rows = payload["history"]
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
