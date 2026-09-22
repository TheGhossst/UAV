"""Block-coordinate a_ij / b_ij updates for SCA.

Not a change to Problem (P) or the convex (q, B) LP. Association and
processing stay out of CVXPY. Candidates are scored with the same
`evaluate()` / `true_gate_ok()` gate as the position step.

Used only when SCASettings.dynamic_assignment is True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from uavdt.computation import queue_unstable
from uavdt.evaluator import EvalResult, evaluate
from uavdt.models import Allocation, Scenario
from uavdt.resources import cpu_stable_processing, cpu_stable_processing_candidates
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
from uavdt.sca.linearize import spectral_efficiency
from uavdt.sca.settings import SCASettings

logger = logging.getLogger(__name__)


def _max_aodt(ev: EvalResult) -> float:
    finite = ev.aodt_s[np.isfinite(ev.aodt_s)]
    return float(np.max(finite)) if finite.size else float("inf")


@dataclass
class AssignmentEvent:
    iteration: int
    stage: str
    n_assoc_changed: int
    n_proc_changed: int
    objective_before: float
    objective_after: float | None
    feasible: bool | None
    accepted: bool
    reason: str
    solver_status: str = ""
    solve_time_s: float = 0.0
    qos_violations: int = 0
    aodt_violations: int = 0
    max_cpu_load: float = 0.0
    bandwidth_usage: float = 0.0
    bandwidth_objective: float | None = None
    max_aodt_before: float = float("inf")
    max_aodt_after: float | None = None

    def to_jsonable(self) -> dict:
        return {
            "iteration": self.iteration,
            "stage": self.stage,
            "n_assoc_changed": self.n_assoc_changed,
            "n_proc_changed": self.n_proc_changed,
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "feasible": self.feasible,
            "accepted": self.accepted,
            "reason": self.reason,
            "solver_status": self.solver_status,
            "solve_time_s": self.solve_time_s,
        }

    def log_line(self) -> str:
        after = (
            "n/a"
            if self.objective_after is None
            else f"{self.objective_after:.6g}"
        )
        feas = "n/a" if self.feasible is None else str(self.feasible)
        decision = "accepted" if self.accepted else "rejected"
        return (
            f"SCA it={self.iteration} {self.stage}: "
            f"n_assoc={self.n_assoc_changed} n_proc={self.n_proc_changed} "
            f"obj {self.objective_before:.6g} -> {after} "
            f"feasible={feas} {decision} ({self.reason})"
        )


@dataclass
class AssignmentAttempt:
    event: AssignmentEvent
    association: np.ndarray
    processing: np.ndarray
    bandwidth_hz: np.ndarray
    true_eval: EvalResult


def n_forwarding(association: np.ndarray, processing: np.ndarray) -> int:
    ja = np.argmax(association, axis=1)
    jb = np.argmax(processing, axis=1)
    return int(np.sum(ja != jb))


def n_row_changes(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.sum(np.argmax(left, axis=1) != np.argmax(right, axis=1)))


def same_one_hot(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left > 0.5, right > 0.5))


def best_se_association(scenario: Scenario, uav_xyz_m: np.ndarray) -> np.ndarray:
    """Each IoT to the UAV with highest SE at the current q. Holds (21)."""
    se = spectral_efficiency(scenario.iot_xyz_m, uav_xyz_m, scenario.cfg)
    a = np.zeros_like(se)
    a[np.arange(se.shape[0]), np.argmax(se, axis=1)] = 1.0
    return a


def exclusive_association(association: np.ndarray) -> bool:
    a = np.asarray(association, dtype=float) > 0.5
    if a.ndim != 2 or a.size == 0:
        return False
    return bool(np.all(a.sum(axis=1) == 1))


def processing_process_consistent(scenario: Scenario, processing: np.ndarray) -> bool:
    """Constraint (23): one processing UAV per process."""
    for proc in scenario.processes:
        members = proc.iot_indices
        if members.size <= 1:
            continue
        js = np.argmax(processing[members], axis=1)
        if np.unique(js).size > 1:
            return False
    return True


def assignment_legal(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
) -> tuple[bool, str]:
    """Hard constraints that do not need the bandwidth LP or channel."""
    a = np.asarray(association, dtype=float)
    b = np.asarray(processing, dtype=float)
    if not exclusive_association(a):
        return False, "not_exclusive"
    if not exclusive_association(b):
        return False, "processing_not_exclusive"
    if not processing_process_consistent(scenario, b):
        return False, "process_inconsistent"
    mu = float(scenario.cfg.service_rate_per_s)
    if np.any(queue_unstable(b, scenario.lambdas_per_s, mu)):
        return False, "cpu_unstable"
    return True, "ok"


def _event(
    *,
    iteration: int,
    stage: str,
    a: np.ndarray,
    b: np.ndarray,
    a_new: np.ndarray,
    b_new: np.ndarray,
    current_eval: EvalResult,
    accepted: bool,
    reason: str,
    cand_eval: EvalResult | None = None,
    solver_status: str = "",
    solve_time_s: float = 0.0,
    bandwidth_hz: np.ndarray | None = None,
    bandwidth_objective: float | None = None,
) -> AssignmentEvent:
    ev = cand_eval
    qos = 0 if ev is None else int(ev.constraints.qos_violations)
    aodt = 0 if ev is None else int(ev.constraints.aodt_violations)
    rho = current_eval.rho if ev is None else ev.rho
    bw_use = 0.0 if bandwidth_hz is None else float(np.sum(bandwidth_hz))
    return AssignmentEvent(
        iteration=iteration,
        stage=stage,
        n_assoc_changed=n_row_changes(a, a_new),
        n_proc_changed=n_row_changes(b, b_new),
        objective_before=float(current_eval.sum_rate_bit_per_s),
        objective_after=None if ev is None else float(ev.sum_rate_bit_per_s),
        feasible=None if ev is None else bool(ev.feasible),
        accepted=accepted,
        reason=reason,
        solver_status=solver_status,
        solve_time_s=solve_time_s,
        qos_violations=qos,
        aodt_violations=aodt,
        max_cpu_load=float(np.max(rho)) if rho.size else 0.0,
        bandwidth_usage=bw_use,
        bandwidth_objective=bandwidth_objective,
        max_aodt_before=_max_aodt(current_eval),
        max_aodt_after=None if ev is None else _max_aodt(ev),
    )


def gated_assignment_update(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    bandwidth_hz: np.ndarray,
    current_eval: EvalResult,
    a_new: np.ndarray,
    b_new: np.ndarray,
    settings: SCASettings | None = None,
    *,
    iteration: int,
    stage: str,
) -> AssignmentAttempt:
    """Re-solve B at fixed q, then accept only a true-feasible improvement.

    Same gate as the SCA position step: `true_gate_ok()` and a strict
    true-objective increase. If the current point is itself infeasible,
    a newly feasible candidate is accepted (init recovery).
    """
    from uavdt.sca.algorithm import true_gate_ok

    settings = settings or SCASettings()
    a = np.asarray(association, dtype=float)
    b = np.asarray(processing, dtype=float)
    a_new = np.asarray(a_new, dtype=float)
    b_new = np.asarray(b_new, dtype=float)
    bw = np.asarray(bandwidth_hz, dtype=float)
    current_ok = true_gate_ok(current_eval)

    def _reject(reason: str, **kwargs) -> AssignmentAttempt:
        event = _event(
            iteration=iteration,
            stage=stage,
            a=a,
            b=b,
            a_new=a_new,
            b_new=b_new,
            current_eval=current_eval,
            accepted=False,
            reason=reason,
            **kwargs,
        )
        return AssignmentAttempt(event, a, b, bw, current_eval)

    if same_one_hot(a, a_new) and same_one_hot(b, b_new):
        return _reject("unchanged")

    legal, why = assignment_legal(scenario, a_new, b_new)
    if not legal:
        return _reject(why)

    bw_res = solve_bandwidth_at_fixed_q(scenario, uav_xyz_m, a_new, b_new, settings)
    if bw_res.infeasible:
        return _reject(
            f"{stage}_bandwidth_infeasible",
            solver_status=bw_res.status,
            solve_time_s=bw_res.solve_time_s,
        )

    cand_eval = evaluate(scenario, uav_xyz_m, Allocation(a_new, b_new, bw_res.bandwidth_hz))
    cand_ok = true_gate_ok(cand_eval)
    newly_feasible = cand_ok and not current_ok
    improved = cand_ok and cand_eval.sum_rate_bit_per_s > (
        current_eval.sum_rate_bit_per_s + settings.improvement_tolerance
    )
    if not (newly_feasible or improved):
        reason = (
            f"{stage}_true_infeasible" if not cand_ok else f"{stage}_no_true_improvement"
        )
        return _reject(
            reason,
            cand_eval=cand_eval,
            solver_status=bw_res.status,
            solve_time_s=bw_res.solve_time_s,
            bandwidth_hz=bw_res.bandwidth_hz,
            bandwidth_objective=bw_res.objective,
        )

    reason = f"{stage}_update" if improved else f"{stage}_newly_feasible"
    event = _event(
        iteration=iteration,
        stage=stage,
        a=a,
        b=b,
        a_new=a_new,
        b_new=b_new,
        current_eval=current_eval,
        accepted=True,
        reason=reason,
        cand_eval=cand_eval,
        solver_status=bw_res.status,
        solve_time_s=bw_res.solve_time_s,
        bandwidth_hz=bw_res.bandwidth_hz,
        bandwidth_objective=bw_res.objective,
    )
    return AssignmentAttempt(
        event, a_new, b_new, bw_res.bandwidth_hz, cand_eval
    )


def _emit(event: AssignmentEvent, verbose: bool) -> None:
    line = event.log_line()
    logger.info(line)
    if verbose:
        print(line)


def _better_attempt(left: AssignmentAttempt, right: AssignmentAttempt) -> bool:
    """Prefer newly-feasible, then higher true sum rate."""
    l_new = left.event.reason.endswith("newly_feasible")
    r_new = right.event.reason.endswith("newly_feasible")
    if l_new != r_new:
        return l_new
    l_obj = left.event.objective_after
    r_obj = right.event.objective_after
    if l_obj is None:
        return False
    if r_obj is None:
        return True
    return l_obj > r_obj


def run_assignment_updates(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    bandwidth_hz: np.ndarray,
    current_eval: EvalResult,
    settings: SCASettings,
    *,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EvalResult, list[AssignmentEvent]]:
    """Association block, then processing block, each true-gated.

    Association: one greedy best-SE map, paired with cpu_stable_processing
    so (23)/(24) still hold. Processing: cpu_stable_processing at the
    current a, then the best extra CPU-stable map if it beats that
    proposal (capped, J^K small).
    """
    a = np.asarray(association, dtype=float).copy()
    b = np.asarray(processing, dtype=float).copy()
    bw = np.asarray(bandwidth_hz, dtype=float).copy()
    events: list[AssignmentEvent] = []
    verbose = bool(settings.verbose)

    a_hat = best_se_association(scenario, uav_xyz_m)
    if not same_one_hot(a, a_hat):
        b_pair = cpu_stable_processing(scenario, a_hat)
        attempt = gated_assignment_update(
            scenario,
            uav_xyz_m,
            a,
            b,
            bw,
            current_eval,
            a_hat,
            b_pair,
            settings,
            iteration=iteration,
            stage="association",
        )
        _emit(attempt.event, verbose)
        events.append(attempt.event)
        if attempt.event.accepted:
            a = attempt.association
            b = attempt.processing
            bw = attempt.bandwidth_hz
            current_eval = attempt.true_eval

    seen_b: list[np.ndarray] = [b]
    best_acc: AssignmentAttempt | None = None
    best_rej: AssignmentEvent | None = None
    current_fwd = n_forwarding(a, b)
    for cand in cpu_stable_processing_candidates(scenario, a, max_maps=4):
        if any(same_one_hot(cand, prev) for prev in seen_b):
            continue
        if n_forwarding(a, cand) > current_fwd:
            continue
        seen_b.append(cand)
        attempt = gated_assignment_update(
            scenario,
            uav_xyz_m,
            a,
            b,
            bw,
            current_eval,
            a,
            cand,
            settings,
            iteration=iteration,
            stage="processing",
        )
        if attempt.event.accepted:
            if best_acc is None or _better_attempt(attempt, best_acc):
                best_acc = attempt
        elif best_rej is None or (
            attempt.event.objective_after is not None
            and (
                best_rej.objective_after is None
                or attempt.event.objective_after > best_rej.objective_after
            )
        ):
            best_rej = attempt.event

    if best_acc is not None:
        _emit(best_acc.event, verbose)
        events.append(best_acc.event)
        a = best_acc.association
        b = best_acc.processing
        bw = best_acc.bandwidth_hz
        current_eval = best_acc.true_eval
    elif best_rej is not None:
        _emit(best_rej, verbose)
        events.append(best_rej)

    return a, b, bw, current_eval, events
