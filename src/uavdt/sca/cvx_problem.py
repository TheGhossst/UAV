"""Exact bandwidth subproblem at a fixed UAV geometry.

Positions are frozen; SE_ij(q) comes from the core channel.
The rate is exactly linear in B_ij, so the program is an LP.

AoDT S/(B*SE) <= slack is equivalent to B >= S/(SE*slack) for B>0.
That is an exact reformulation at frozen q, not a joint Taylor map.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from uavdt.aodt import queueing_term_s
from uavdt.computation import service_rate_per_s
from uavdt.models import Scenario
from uavdt.sca.linearize import linearize_separation, se_jacobian, spectral_efficiency
from uavdt.sca.settings import SCASettings


@dataclass
class BandwidthSolveResult:
    status: str
    solver_name: str
    bandwidth_hz: np.ndarray
    se: np.ndarray
    objective: float
    solve_time_s: float
    infeasible: bool
    message: str = ""


def _pick_lp_solver(requested: str | None) -> str:
    import cvxpy as cp

    installed = set(cp.installed_solvers())
    if requested is not None:
        name = requested.upper()
        if name not in installed:
            raise RuntimeError(
                f"requested solver {requested!r} not in {sorted(installed)}"
            )
        return name
    for name in ("MOSEK", "HIGHS", "CLARABEL", "ECOS", "SCS"):
        if name in installed:
            return name
    raise RuntimeError(f"no CVXPY LP solver; installed={sorted(installed)}")


def aodt_upload_slacks_s(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
) -> np.ndarray:
    """slack_i = T_k - Q_k - 1_fwd T_u2u for the process of IoT i.

    DERIVED from Eq. (17) with a, b fixed (Q_k constant).
    """
    cfg = scenario.cfg
    mu = service_rate_per_s(cfg)
    n_iot = association.shape[0]
    slacks = np.full(n_iot, np.nan)
    i_idx = np.arange(n_iot)
    j_assoc = np.argmax(association, axis=1)
    forwarded = processing[i_idx, j_assoc] < 0.5
    for proc in scenario.processes:
        members = proc.iot_indices
        q = queueing_term_s(scenario.lambdas_per_s[members], mu)
        for idx in members:
            slacks[idx] = cfg.aodt_threshold_s - q - (
                cfg.t_u2u_s if forwarded[idx] else 0.0
            )
    return slacks


def bandwidth_floors_hz(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    se: np.ndarray,
) -> np.ndarray:
    """B_ij floors from exact QoS (25) and AoDT (31) at frozen q."""
    cfg = scenario.cfg
    a = association > 0.5
    floors = np.zeros_like(se)
    slacks = aodt_upload_slacks_s(scenario, association, processing)
    j_assoc = np.argmax(association, axis=1)
    for i in range(association.shape[0]):
        if not np.any(a[i]):
            continue
        j = int(j_assoc[i])
        se_ij = float(se[i, j])
        if se_ij <= 1e-15 or not np.isfinite(slacks[i]) or slacks[i] <= 1e-12:
            floors[i, j] = np.inf
            continue
        qos = cfg.r_min_bit_per_s / se_ij
        aodt = cfg.task_size_bits / (se_ij * float(slacks[i]))
        floors[i, j] = max(qos, aodt)
    return floors


def rate_floors_bit_per_s(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
) -> np.ndarray:
    """Linearized (25) and (31): r_hat >= max(R_min, S/slack) on associated links."""
    cfg = scenario.cfg
    a = association > 0.5
    floors = np.full(association.shape, -1.0e12)
    slacks = aodt_upload_slacks_s(scenario, association, processing)
    j_assoc = np.argmax(association, axis=1)
    for i in range(association.shape[0]):
        if not np.any(a[i]):
            continue
        j = int(j_assoc[i])
        if not np.isfinite(slacks[i]) or slacks[i] <= 1e-12:
            floors[i, j] = np.inf
            continue
        floors[i, j] = max(cfg.r_min_bit_per_s, cfg.task_size_bits / float(slacks[i]))
    return floors


@dataclass
class JointSolveResult:
    status: str
    solver_name: str
    x_m: np.ndarray
    y_m: np.ndarray
    bandwidth_hz: np.ndarray
    objective: float
    solve_time_s: float
    infeasible: bool
    message: str = ""


def solve_joint_convex_step(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    bandwidth_hz: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    step_m: float,
    settings: SCASettings | None = None,
) -> JointSolveResult:
    """Algorithm 1 convexified (P): joint (q, B) LP at the current Taylor point."""
    import cvxpy as cp

    settings = settings or SCASettings()
    cfg = scenario.cfg
    a = association > 0.5
    se, gx, gy = se_jacobian(
        scenario.iot_xyz_m, uav_xyz_m, cfg, step_m=settings.fd_step_m
    )
    solver_name = _pick_lp_solver(
        None if settings.solver in {None, "matlab", "MATLAB", "mosek", "MOSEK"} else settings.solver
    )
    floors = rate_floors_bit_per_s(scenario, association, processing)
    if np.any(~np.isfinite(floors[a])):
        return JointSolveResult(
            status="aodt_slack_nonpositive",
            solver_name=solver_name,
            x_m=np.asarray(uav_xyz_m[:, 0], dtype=float).copy(),
            y_m=np.asarray(uav_xyz_m[:, 1], dtype=float).copy(),
            bandwidth_hz=np.zeros_like(se),
            objective=float("nan"),
            solve_time_s=0.0,
            infeasible=True,
            message="AoDT slack non-positive",
        )

    i_n, j_n = association.shape
    x = cp.Variable(j_n)
    y = cp.Variable(j_n)
    bw = cp.Variable((i_n, j_n), nonneg=True)
    dx = x - uav_xyz_m[:, 0]
    dy = y - uav_xyz_m[:, 1]
    pos = cp.multiply(gx, cp.reshape(dx, (1, j_n), order="C")) + cp.multiply(
        gy, cp.reshape(dy, (1, j_n), order="C")
    )
    rhat = cp.multiply(se, bw) + cp.multiply(bandwidth_hz, pos)
    a_f = a.astype(float)
    cons = [
        bw[~a] == 0,
        bw[a] <= cfg.link_bandwidth_cap_hz,
        cp.sum(bw) <= cfg.b_sys_hz,
        x >= 0,
        x <= cfg.area_x_m,
        y >= 0,
        y <= cfg.area_y_m,
        cp.abs(dx) <= step_m,
        cp.abs(dy) <= step_m,
        rhat[a] >= floors[a],
    ]
    for plane in linearize_separation(uav_xyz_m, cfg.uav_min_separation_m):
        cons.append(
            plane.u_x * (x[plane.j] - x[plane.l])
            + plane.u_y * (y[plane.j] - y[plane.l])
            >= plane.theta_m
        )
    problem = cp.Problem(
        cp.Maximize(
            cp.sum(cp.multiply(a_f, rhat))
            - 1e-6 * cp.sum(cp.abs(dx) + cp.abs(dy))
        ),
        cons,
    )
    t0 = perf_counter()
    try:
        problem.solve(solver=solver_name, verbose=settings.verbose)
    except Exception as exc:  # noqa: BLE001
        return JointSolveResult(
            status="solver_error",
            solver_name=solver_name,
            x_m=np.asarray(uav_xyz_m[:, 0], dtype=float).copy(),
            y_m=np.asarray(uav_xyz_m[:, 1], dtype=float).copy(),
            bandwidth_hz=np.zeros_like(se),
            objective=float("nan"),
            solve_time_s=perf_counter() - t0,
            infeasible=True,
            message=str(exc),
        )
    elapsed = perf_counter() - t0
    status = str(problem.status)
    bad = status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None or bw.value is None
    if bad:
        return JointSolveResult(
            status=status,
            solver_name=solver_name,
            x_m=np.asarray(uav_xyz_m[:, 0], dtype=float).copy(),
            y_m=np.asarray(uav_xyz_m[:, 1], dtype=float).copy(),
            bandwidth_hz=np.zeros_like(se),
            objective=float("nan"),
            solve_time_s=elapsed,
            infeasible=True,
            message=status,
        )
    x_val = np.clip(np.asarray(x.value, dtype=float).ravel(), 0.0, cfg.area_x_m)
    y_val = np.clip(np.asarray(y.value, dtype=float).ravel(), 0.0, cfg.area_y_m)
    bw_val = np.maximum(np.asarray(bw.value, dtype=float), 0.0)
    bw_val[~a] = 0.0
    bw_val = np.minimum(bw_val, cfg.link_bandwidth_cap_hz)
    return JointSolveResult(
        status=status,
        solver_name=solver_name,
        x_m=x_val,
        y_m=y_val,
        bandwidth_hz=bw_val,
        objective=float(problem.value) if problem.value is not None else float("nan"),
        solve_time_s=elapsed,
        infeasible=False,
        message=status,
    )


def solve_bandwidth_cvxpy(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    settings: SCASettings | None = None,
) -> BandwidthSolveResult:
    import cvxpy as cp

    settings = settings or SCASettings()
    cfg = scenario.cfg
    a = association > 0.5
    se = spectral_efficiency(scenario.iot_xyz_m, uav_xyz_m, cfg)
    solver_name = _pick_lp_solver(
        None if settings.solver in {None, "matlab", "MATLAB", "mosek", "MOSEK"} else settings.solver
    )
    floors = bandwidth_floors_hz(scenario, association, processing, se)
    if np.any(~np.isfinite(floors[a])):
        return BandwidthSolveResult(
            status="aodt_slack_nonpositive",
            solver_name=solver_name,
            bandwidth_hz=np.zeros_like(se),
            se=se,
            objective=float("nan"),
            solve_time_s=0.0,
            infeasible=True,
            message="AoDT slack T_k - Q_k - T_fwd <= 0 or SE=0",
        )

    i_n, j_n = association.shape
    bw = cp.Variable((i_n, j_n), nonneg=True)
    a_f = a.astype(float)
    cons = [
        bw[~a] == 0,
        bw[a] <= cfg.link_bandwidth_cap_hz,
        cp.sum(bw) <= cfg.b_sys_hz,
        bw >= floors,
    ]
    problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(a_f * se, bw))), cons)
    t0 = perf_counter()
    try:
        problem.solve(solver=solver_name, verbose=settings.verbose)
    except Exception as exc:  # noqa: BLE001
        return BandwidthSolveResult(
            status="solver_error",
            solver_name=solver_name,
            bandwidth_hz=np.zeros_like(se),
            se=se,
            objective=float("nan"),
            solve_time_s=perf_counter() - t0,
            infeasible=True,
            message=str(exc),
        )
    elapsed = perf_counter() - t0
    status = str(problem.status)
    infeas = status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or bw.value is None
    if infeas:
        return BandwidthSolveResult(
            status=status,
            solver_name=solver_name,
            bandwidth_hz=np.zeros_like(se),
            se=se,
            objective=float("nan"),
            solve_time_s=elapsed,
            infeasible=True,
            message=status,
        )
    bw_val = np.maximum(np.asarray(bw.value, dtype=float), 0.0)
    bw_val[~a] = 0.0
    bw_val = np.minimum(bw_val, cfg.link_bandwidth_cap_hz)
    if bw_val.sum() > cfg.b_sys_hz + 1e-6:
        bw_val *= cfg.b_sys_hz / bw_val.sum()
    return BandwidthSolveResult(
        status=status,
        solver_name=solver_name,
        bandwidth_hz=bw_val,
        se=se,
        objective=float(problem.value),
        solve_time_s=elapsed,
        infeasible=False,
        message=status,
    )


def solve_bandwidth_at_fixed_q(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    settings: SCASettings | None = None,
) -> BandwidthSolveResult:
    """Maximize sum a_ij SE_ij(q) B_ij at frozen q.

    If settings.solver is matlab/MOSEK, use MATLAB CVX+MOSEK when available.
    """
    settings = settings or SCASettings()
    want_matlab = settings.solver in {None, "matlab", "MATLAB", "mosek", "MOSEK"}
    # Default None stays on CVXPY so unit tests do not launch MATLAB.
    if settings.solver in {"matlab", "MATLAB", "mosek", "MOSEK"}:
        from uavdt.sca.matlab_bridge import matlab_available, solve_bandwidth_matlab

        if not matlab_available():
            raise RuntimeError("MATLAB CVX/MOSEK requested but not available")
        return solve_bandwidth_matlab(
            scenario, uav_xyz_m, association, processing, settings
        )
    _ = want_matlab
    return solve_bandwidth_cvxpy(
        scenario, uav_xyz_m, association, processing, settings
    )
