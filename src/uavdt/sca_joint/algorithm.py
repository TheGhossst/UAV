"""SCA-joint: frozen Algorithm 1 (q, B) plus a discrete a_ij/b_ij re-match.

Frozen SCA (`uavdt.sca.algorithm.solve_sca`) is not imported for the
loop body and is not modified. Same stopping rule and max_iterations.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from uavdt.evaluator import evaluate
from uavdt.models import Allocation, Scenario
from uavdt.sca.algorithm import (
    SCAIterationLog,
    SCAResult,
    _effective_step_size,
    _max_aodt,
    _min_sep,
    classify_stop_reason,
    true_gate_ok,
)
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q, solve_joint_convex_step
from uavdt.sca.initialize import initialize_sca
from uavdt.sca.settings import SCASettings
from uavdt.scenario import make_uav_xyz_m
from uavdt.sca_joint.rematch import (
    association_process_cohesive,
    iter_rematch_candidates,
    n_forwarding,
)


def _same_one_hot(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left > 0.5, right > 0.5))


def _use_matlab(settings: SCASettings) -> bool:
    return settings.solver in {"matlab", "MATLAB", "mosek", "MOSEK"}


def _diag_base(seed: int, cfg, settings: SCASettings, extra: dict) -> dict:
    out = {
        "method": "sca_joint",
        "label": (
            "SCA-joint methodology probe: Algorithm 1 (q, B) Taylor-LP "
            "plus discrete a_ij re-match (best-SE"
            + (
                " and process-cohesive"
                if getattr(settings, "process_cohesive_candidate", False)
                else ""
            )
            + ") with cpu_stable_processing for (23)/(24). "
            "Not a replacement for frozen SCA."
        ),
        "binaries": "a_ij and b_ij re-matched each outer iteration (METHODOLOGY PROBE)",
        "seed": seed,
        "b_sys_hz": cfg.b_sys_hz,
        "step_size_m": settings.step_size_m,
        "solver_backend": settings.solver or "cvxpy",
    }
    out.update(extra)
    return out


def _history_row(
    *,
    iteration: int,
    current_eval,
    cand_eval,
    uav: np.ndarray,
    uav_cand: np.ndarray | None,
    step: float,
    bandwidth_objective: float | None,
    accepted: bool,
    rejection_reason: str,
    solver_status: str,
    solve_time_s: float,
    bandwidth_hz: np.ndarray,
) -> SCAIterationLog:
    cand_obj = None if cand_eval is None else cand_eval.sum_rate_bit_per_s
    cand_aodt = None if cand_eval is None else _max_aodt(cand_eval)
    cand_sep = None if uav_cand is None else _min_sep(uav_cand)
    ev_for_rho = current_eval if cand_eval is None else cand_eval
    qos = 0 if cand_eval is None else cand_eval.constraints.qos_violations
    aodt = 0 if cand_eval is None else cand_eval.constraints.aodt_violations
    return SCAIterationLog(
        iteration=iteration,
        current_true_objective=current_eval.sum_rate_bit_per_s,
        candidate_true_objective=cand_obj,
        current_max_AoDT=_max_aodt(current_eval),
        candidate_max_AoDT=cand_aodt,
        current_min_separation=_min_sep(uav),
        candidate_min_separation=cand_sep if cand_sep is not None else _min_sep(uav),
        step_size=step,
        bandwidth_objective=bandwidth_objective,
        true_objective=current_eval.sum_rate_bit_per_s,
        accepted=accepted,
        rejection_reason=rejection_reason,
        solver_status=solver_status,
        qos_violations=qos,
        aodt_violations=aodt,
        max_cpu_load=float(np.max(ev_for_rho.rho)) if ev_for_rho.rho.size else 0.0,
        bandwidth_usage=float(np.sum(bandwidth_hz)),
        solve_time_s=solve_time_s,
    )


def _try_discrete_rematch(
    scenario: Scenario,
    uav: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    bw: np.ndarray,
    current_eval,
    current_ok: bool,
    settings: SCASettings,
    iteration: int,
    history: list[SCAIterationLog],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, bool, bool, str]:
    """Re-match a,b at frozen q. Accept if newly feasible or true rate improves.

    Tries best-SE, and optionally process-cohesive a_ij, then keeps the
    candidate that passes the true-evaluate gate with the best score
    (newly-feasible first, then higher sum rate).
    """
    include_cohesive = bool(getattr(settings, "process_cohesive_candidate", False))
    last_reason = "association_unchanged"

    def _note_reject(reason: str) -> None:
        nonlocal last_reason
        if reason == "association_unchanged" and last_reason != "association_unchanged":
            return
        last_reason = reason

    best: (
        tuple[tuple[int, float], np.ndarray, np.ndarray, object, object, str] | None
    ) = None
    seen: list[np.ndarray] = []

    for proposal in iter_rematch_candidates(
        scenario,
        uav,
        a,
        b,
        include_process_cohesive=include_cohesive,
    ):
        if not proposal.ok:
            _note_reject(proposal.reason)
            continue
        a_new = proposal.association
        if any(_same_one_hot(a_new, prev) for prev in seen):
            continue
        seen.append(a_new)
        b_new = proposal.processing
        bw_res = solve_bandwidth_at_fixed_q(scenario, uav, a_new, b_new, settings)
        if bw_res.infeasible:
            _note_reject("rematch_bandwidth_infeasible")
            continue
        cand_eval = evaluate(scenario, uav, Allocation(a_new, b_new, bw_res.bandwidth_hz))
        cand_ok = true_gate_ok(cand_eval)
        newly_feasible = cand_ok and not current_ok
        improved = cand_ok and cand_eval.sum_rate_bit_per_s > (
            current_eval.sum_rate_bit_per_s + settings.improvement_tolerance
        )
        if not (newly_feasible or improved):
            _note_reject(
                "rematch_true_infeasible" if not cand_ok else "rematch_no_true_improvement"
            )
            continue
        key = (1 if newly_feasible else 0, float(cand_eval.sum_rate_bit_per_s))
        if best is None or key > best[0]:
            best = (key, a_new, b_new, bw_res, cand_eval, proposal.kind)

    if best is None:
        return a, b, bw, current_eval, current_ok, False, last_reason

    _key, a_new, b_new, bw_res, cand_eval, kind = best
    history.append(
        SCAIterationLog(
            iteration=iteration,
            current_true_objective=current_eval.sum_rate_bit_per_s,
            candidate_true_objective=cand_eval.sum_rate_bit_per_s,
            current_max_AoDT=_max_aodt(current_eval),
            candidate_max_AoDT=_max_aodt(cand_eval),
            current_min_separation=_min_sep(uav),
            candidate_min_separation=_min_sep(uav),
            step_size=_effective_step_size(settings),
            bandwidth_objective=bw_res.objective,
            true_objective=cand_eval.sum_rate_bit_per_s,
            accepted=True,
            rejection_reason=f"discrete_rematch:{kind}",
            solver_status=bw_res.status,
            qos_violations=0,
            aodt_violations=0,
            max_cpu_load=float(np.max(cand_eval.rho)) if cand_eval.rho.size else 0.0,
            bandwidth_usage=float(np.sum(bw_res.bandwidth_hz)),
            solve_time_s=bw_res.solve_time_s,
        )
    )
    return (
        a_new,
        b_new,
        bw_res.bandwidth_hz,
        cand_eval,
        True,
        True,
        f"discrete_rematch:{kind}",
    )


def solve_sca_joint(
    scenario: Scenario,
    seed: int,
    settings: SCASettings | None = None,
    uav_xyz_m: np.ndarray | None = None,
    allocation: Allocation | None = None,
) -> SCAResult:
    """Sequential SCA + discrete re-match. Scores always from evaluate()."""
    t_wall = perf_counter()
    settings = settings or SCASettings()
    cfg = scenario.cfg
    if _use_matlab(settings):
        raise RuntimeError(
            "sca_joint is CVXPY-only; use frozen SCA (--solver matlab) for MOSEK"
        )
    if uav_xyz_m is None or allocation is None:
        uav, alloc = initialize_sca(scenario, seed)
    else:
        uav = np.asarray(uav_xyz_m, dtype=float)
        alloc = allocation
    a = alloc.hard_association()
    b = alloc.hard_processing()
    a_init = a.copy()
    b_init = b.copy()

    rematch_accepted = 0
    rematch_rejected = 0
    rematch_attempts = 0
    rematch_kinds: list[str] = []
    include_cohesive = bool(getattr(settings, "process_cohesive_candidate", False))

    def rematch_now(iteration: int) -> bool:
        nonlocal a, b, bw, current_eval, current_ok, rematch_accepted, rematch_rejected, rematch_attempts
        rematch_attempts += 1
        a, b, bw, current_eval, current_ok, accepted, reason = _try_discrete_rematch(
            scenario,
            uav,
            a,
            b,
            bw,
            current_eval,
            current_ok,
            settings,
            iteration,
            history,
        )
        if reason == "association_unchanged":
            return False
        if accepted:
            rematch_accepted += 1
            rematch_kinds.append(reason)
        else:
            rematch_rejected += 1
        return accepted

    bw_init = solve_bandwidth_at_fixed_q(scenario, uav, a, b, settings)
    if bw_init.infeasible:
        current_eval = evaluate(scenario, uav, Allocation(a, b, alloc.bandwidth_hz))
        current_ok = true_gate_ok(current_eval)
        bw = np.asarray(alloc.bandwidth_hz, dtype=float).copy()
        history = [
            _history_row(
                iteration=0,
                current_eval=current_eval,
                cand_eval=None,
                uav=uav,
                uav_cand=None,
                step=_effective_step_size(settings),
                bandwidth_objective=None,
                accepted=False,
                rejection_reason="init_bandwidth_infeasible",
                solver_status=bw_init.status,
                solve_time_s=bw_init.solve_time_s,
                bandwidth_hz=bw,
            )
        ]
        rematch_now(0)
        if not current_ok:
            wall = perf_counter() - t_wall
            return SCAResult(
                uav_xyz_m=uav,
                allocation=Allocation(a, b, bw),
                surrogate_objective=float("nan"),
                true_objective=float(current_eval.sum_rate_bit_per_s),
                true_eval=current_eval,
                history=history,
                solver_status=bw_init.status,
                solver_name=bw_init.solver_name,
                n_iterations=0,
                diagnostics=_diag_base(
                    seed,
                    cfg,
                    settings,
                    {
                        "stop_reason": "init_bandwidth_infeasible",
                        "converged": False,
                        "accepted_steps": 0,
                        "rejected_steps": 1,
                        "step_size_reductions": 0,
                        "rematch_attempts": rematch_attempts,
                        "rematch_accepted": rematch_accepted,
                        "rematch_rejected": rematch_rejected,
                        "rematch_kinds": list(rematch_kinds),
                        "process_cohesive_candidate": include_cohesive,
                        "process_cohesive_a": association_process_cohesive(scenario, a),
                        "n_forwarding": n_forwarding(a, b),
                        "association_init_equals_final": _same_one_hot(a_init, a),
                        "processing_init_equals_final": _same_one_hot(b_init, b),
                        "wall_clock_s": wall,
                    },
                ),
            )
    else:
        bw = bw_init.bandwidth_hz
        current_eval = evaluate(scenario, uav, Allocation(a, b, bw))
        if not true_gate_ok(current_eval):
            raise RuntimeError(
                "sca_joint requires a true-feasible start after the bandwidth LP; "
                f"violations qos={current_eval.constraints.qos_violations} "
                f"aodt={current_eval.constraints.aodt_violations} "
                f"cpu={current_eval.constraints.cpu_unstable_count} "
                f"sep={current_eval.constraints.sep_violations} "
                f"bw_excess={current_eval.constraints.bw_excess_hz}"
            )
        current_ok = True
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
        rematch_now(0)

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
            obj_before = current_eval.sum_rate_bit_per_s
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
                    _history_row(
                        iteration=n,
                        current_eval=current_eval,
                        cand_eval=None,
                        uav=uav,
                        uav_cand=None,
                        step=step / settings.step_size_shrink,
                        bandwidth_objective=None,
                        accepted=False,
                        rejection_reason="convex_p_infeasible",
                        solver_status=joint.status,
                        solve_time_s=joint.solve_time_s,
                        bandwidth_hz=bw,
                    )
                )
                rematch_now(n)
                continue
            xy_cand = np.column_stack([joint.x_m, joint.y_m])
            move = float(np.max(np.abs(xy_cand - uav[:, :2])))
            if move < settings.min_step_size_m:
                history.append(
                    _history_row(
                        iteration=n,
                        current_eval=current_eval,
                        cand_eval=current_eval,
                        uav=uav,
                        uav_cand=uav,
                        step=step,
                        bandwidth_objective=joint.objective,
                        accepted=False,
                        rejection_reason="stationary_convex_p",
                        solver_status=joint.status,
                        solve_time_s=joint.solve_time_s,
                        bandwidth_hz=bw,
                    )
                )
                if rematch_now(n):
                    continue
                stop_reason = "CONVERGED"
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
                    _history_row(
                        iteration=n,
                        current_eval=current_eval,
                        cand_eval=None,
                        uav=uav,
                        uav_cand=uav_cand,
                        step=step / settings.step_size_shrink,
                        bandwidth_objective=None,
                        accepted=False,
                        rejection_reason="bandwidth_infeasible",
                        solver_status=bw_res.status,
                        solve_time_s=bw_res.solve_time_s,
                        bandwidth_hz=bw,
                    )
                )
                rematch_now(n)
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
                    _history_row(
                        iteration=n,
                        current_eval=current_eval,
                        cand_eval=cand_eval,
                        uav=uav,
                        uav_cand=uav_cand,
                        step=step / settings.step_size_shrink,
                        bandwidth_objective=bw_res.objective,
                        accepted=False,
                        rejection_reason=reason,
                        solver_status=bw_res.status,
                        solve_time_s=bw_res.solve_time_s,
                        bandwidth_hz=bw_res.bandwidth_hz,
                    )
                )
                rematch_now(n)
                if accepted_steps > 0 and step <= settings.min_step_size_m:
                    stop_reason = "CONVERGED"
                    break
                continue

            prev_aodt = _max_aodt(current_eval)
            prev_sep = _min_sep(uav)
            uav = uav_cand
            bw = bw_res.bandwidth_hz
            current_eval = cand_eval
            current_ok = True
            accepted_steps += 1
            history.append(
                SCAIterationLog(
                    iteration=n,
                    current_true_objective=obj_before,
                    candidate_true_objective=cand_eval.sum_rate_bit_per_s,
                    current_max_AoDT=prev_aodt,
                    candidate_max_AoDT=_max_aodt(cand_eval),
                    current_min_separation=prev_sep,
                    candidate_min_separation=_min_sep(uav),
                    step_size=step,
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
            rematch_now(n)
            delta = current_eval.sum_rate_bit_per_s - obj_before
            if abs(delta) <= settings.epsilon:
                stop_reason = "CONVERGED"
                break
        else:
            stop_reason = "MAX_ITERATIONS"

    true_final = evaluate(scenario, uav, Allocation(a, b, bw))
    if abs(true_final.sum_rate_bit_per_s - current_eval.sum_rate_bit_per_s) > 1e-6:
        current_eval = true_final

    n_accepts = accepted_steps + rematch_accepted
    stop_reason = classify_stop_reason(
        stop_reason,
        accepted_steps=n_accepts,
        step_size=step,
        min_step_size=float(settings.min_step_size_m),
    )
    wall = perf_counter() - t_wall
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
        diagnostics=_diag_base(
            seed,
            cfg,
            settings,
            {
                "stop_reason": stop_reason,
                "converged": stop_reason == "CONVERGED",
                "accepted_steps": accepted_steps,
                "final_step_m": step,
                "rejected_steps": rejected_steps,
                "step_size_reductions": n_reductions,
                "rematch_attempts": rematch_attempts,
                "rematch_accepted": rematch_accepted,
                "rematch_rejected": rematch_rejected,
                "rematch_kinds": list(rematch_kinds),
                "process_cohesive_candidate": include_cohesive,
                "process_cohesive_a": association_process_cohesive(scenario, a),
                "n_forwarding": n_forwarding(a, b),
                "association_init_equals_final": _same_one_hot(a_init, a),
                "processing_init_equals_final": _same_one_hot(b_init, b),
                "wall_clock_s": wall,
            },
        ),
    )
