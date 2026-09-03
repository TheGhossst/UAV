"""Successive convex approximation of Problem (P), Algorithm 1.

IEEE TNSM 2026 §V (arXiv:2504.15967 §V): linearize the non-convex channel
terms (3)–(6) with a first-order Taylor expansion, form a convex program in
UAV positions and bandwidth, solve, update, repeat until the objective of
(P) stalls.

The paper solves that convex program with MATLAB CVX + MOSEK (primal-dual
interior point). After first-order linearization of (3)–(6), the QoS/AoDT
rate floors, constraint (28), and the inner approximation of the UAV
separation (29), the convexified (P) is a linear program. This file solves
the same LP with SciPy HiGHS. ``src.solvers.sca_cvx`` / ``--mode sca-cvx``
sends that identical LP to CVX + MOSEK.

Binaries a_ij, b_ij are held at the k-means / repair values during each
convex solve — Algorithm 1 only lists position and bandwidth updates. After
an accepted step, association is re-snapped to nearest-UAV if that feasible
integer point scores higher on the true evaluator.

Not an evaluator-gradient trust-region search, and not
fixed-association → bandwidth LP → finite-difference placement.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from src.comm import spectral_efficiency_grad
from src.config import PSO_PENALTY, SCA_MAX_ITER, SCA_TOL, SCA_TRUST
from src.evaluator import EvalResult, evaluate, fitness
from src.logutil import log, mbps
from src.repair import (
    allocate_constrained_bandwidth,
    associated_rate_floors,
    bandwidth_pools,
    clip_positions,
    complete_solution,
    enforce_separation,
    link_bandwidth_cap,
    process_consistent_processing,
)
from src.scenario import Scenario
from src.solvers.kmeans import kmeans

# Soft slack on linearized rate floors (bit/s). Keeps the LP feasible when
# T_k is so tight that the floors cannot be met; the true evaluator still
# scores the recovered point.
_SCA_SLACK_PENALTY = 1e3


def _allocate_bandwidth(
    scenario: Scenario,
    xy: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray | None = None,
) -> np.ndarray:
    """Exact B-slice of (P) at fixed positions: r_ij = SE_ij(q) B_ij."""
    if processing is None:
        processing = process_consistent_processing(scenario, association)
    return allocate_constrained_bandwidth(scenario, xy, association, processing)


def _eval_fixed(
    scenario: Scenario,
    xy: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    bandwidth: np.ndarray | None = None,
) -> tuple[float, EvalResult, np.ndarray, np.ndarray]:
    """Score a point of (P). Default B is the exact LP at this geometry."""
    xy = clip_positions(xy, scenario.cfg)
    bw = bandwidth if bandwidth is not None else _allocate_bandwidth(
        scenario, xy, association, processing
    )
    result = evaluate(scenario, xy, association, processing, bw)
    return fitness(result, PSO_PENALTY), result, xy, bw


def _associated_links(association: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(association > 0.5)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


@dataclass(frozen=True)
class ConvexifiedLP:
    """First-order convexification of (P) as a standard LP ``min c'x``."""

    c: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    bounds: list[tuple[float | None, float | None]]
    n_uav: int
    n_pos: int
    links: list[tuple[int, int]]


def _assemble_convexified_lp(
    scenario: Scenario,
    xy: np.ndarray,
    bw: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    trust: float,
) -> ConvexifiedLP | None:
    """Build the joint (q, B) LP. Same problem SciPy HiGHS and CVX+MOSEK solve."""
    cfg = scenario.cfg
    n_uav = xy.shape[0]
    links = _associated_links(association)
    n_links = len(links)
    if n_links == 0:
        return None

    se, g_x, g_y = spectral_efficiency_grad(scenario.iot_xy, xy, cfg)
    rate_need = associated_rate_floors(scenario, association, processing)

    n_slack = n_links
    n_pos = 2 * n_uav
    n_var = n_pos + n_links + n_slack
    ix = lambda u: u
    iy = lambda u: n_uav + u
    ib = lambda k: n_pos + k
    islack = lambda k: n_pos + n_links + k

    c = np.zeros(n_var)
    # max SE B + B^t ∇SE · q  →  min the negation
    for k, (i, u) in enumerate(links):
        c[ib(k)] = -se[i, u]
        c[ix(u)] += -bw[i, u] * g_x[i, u]
        c[iy(u)] += -bw[i, u] * g_y[i, u]
        c[islack(k)] = _SCA_SLACK_PENALTY

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    for mask, pool in bandwidth_pools(association, cfg):
        row = np.zeros(n_var)
        cap = link_bandwidth_cap(cfg, pool)
        for k, (i, u) in enumerate(links):
            if mask[i, u]:
                row[ib(k)] = 1.0
        a_ub.append(row)
        b_ub.append(float(pool))

        for k, (i, u) in enumerate(links):
            if not mask[i, u]:
                continue
            # Linearized rate + slack >= min(R_need, SE^t cap)
            need_hz = min(float(rate_need[i]) / max(float(se[i, u]), 1e-12), cap)
            r_floor = need_hz * float(se[i, u])
            bt = float(bw[i, u])
            gx, gy = float(g_x[i, u]), float(g_y[i, u])
            row = np.zeros(n_var)
            row[ib(k)] = -float(se[i, u])
            row[ix(u)] = -bt * gx
            row[iy(u)] = -bt * gy
            row[islack(k)] = -1.0
            rhs = -r_floor - bt * gx * float(xy[u, 0]) - bt * gy * float(xy[u, 1])
            a_ub.append(row)
            b_ub.append(rhs)

    for p in range(n_uav):
        for q in range(p + 1, n_uav):
            delta = xy[p] - xy[q]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9:
                delta = np.array([1.0, 0.0])
                dist = 1.0
            nrm = delta / dist
            # Inner approx of ||q_p - q_q|| >= θ: n^T (q_p - q_q) >= θ
            row = np.zeros(n_var)
            row[ix(p)] = -nrm[0]
            row[iy(p)] = -nrm[1]
            row[ix(q)] = nrm[0]
            row[iy(q)] = nrm[1]
            a_ub.append(row)
            b_ub.append(-float(cfg.uav_min_distance))

    bounds: list[tuple[float | None, float | None]] = []
    for u in range(n_uav):
        bounds.append(
            (
                max(0.0, float(xy[u, 0]) - trust),
                min(float(cfg.area_x), float(xy[u, 0]) + trust),
            )
        )
    for u in range(n_uav):
        bounds.append(
            (
                max(0.0, float(xy[u, 1]) - trust),
                min(float(cfg.area_y), float(xy[u, 1]) + trust),
            )
        )
    # Per-link cap: use the system pool's cap (same for every pool sharing a link).
    default_cap = link_bandwidth_cap(cfg, float(cfg.b_sys))
    for k, (i, u) in enumerate(links):
        cap_k = default_cap
        for mask, pool in bandwidth_pools(association, cfg):
            if mask[i, u]:
                cap_k = link_bandwidth_cap(cfg, pool)
                break
        bounds.append((0.0, float(cap_k)))
    for _ in range(n_slack):
        bounds.append((0.0, None))

    return ConvexifiedLP(
        c=c,
        a_ub=np.vstack(a_ub) if a_ub else np.zeros((0, n_var)),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        n_uav=n_uav,
        n_pos=n_pos,
        links=links,
    )


def _decode_convexified_solution(
    x: np.ndarray,
    lp: ConvexifiedLP,
    bw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xy_new = np.column_stack([x[: lp.n_uav], x[lp.n_uav : lp.n_pos]])
    bw_new = np.zeros_like(bw)
    for k, (i, u) in enumerate(lp.links):
        bw_new[i, u] = max(0.0, float(x[lp.n_pos + k]))
    return xy_new, bw_new


def _convexified_lp(
    scenario: Scenario,
    xy: np.ndarray,
    bw: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    trust: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve the first-order convexification of (P) jointly in (q, B) with HiGHS."""
    lp = _assemble_convexified_lp(scenario, xy, bw, association, processing, trust)
    if lp is None:
        return None
    result = linprog(
        lp.c,
        A_ub=lp.a_ub if lp.a_ub.size else None,
        b_ub=lp.b_ub if lp.b_ub.size else None,
        bounds=lp.bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        return None
    return _decode_convexified_solution(result.x, lp, bw)


LpSolver = Callable[
    [Scenario, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    tuple[np.ndarray, np.ndarray] | None,
]


def solve_sca(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    max_iter: int = SCA_MAX_ITER,
    trust: float = SCA_TRUST,
    *,
    lp_solver: LpSolver | None = None,
    backend: str = "highs",
) -> tuple[np.ndarray, EvalResult, float]:
    cfg = scenario.cfg
    j = n_uav if n_uav is not None else cfg.num_uav
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if lp_solver is None:
        lp_solver = _convexified_lp

    xy = kmeans(scenario.iot_xy, j, rng)
    xy, a, proc, bw = complete_solution(scenario, xy)
    xy = enforce_separation(clip_positions(xy, cfg), cfg, rng)
    # Start from a B-feasible point of the true (P) slice so the first Taylor
    # expansion of r = B SE(q) is taken at a bandwidth-optimal reference.
    bw = _allocate_bandwidth(scenario, xy, a, proc)
    fit, result, xy, bw = _eval_fixed(scenario, xy, a, proc, bw)
    prev = fit
    log.info(
        "sca  J=%d max_iter=%d trust=%g  backend=%s  (joint convex LP)",
        j,
        max_iter,
        trust,
        backend,
    )

    for it in range(max_iter):
        solved = lp_solver(scenario, xy, bw, a, proc, trust)
        if solved is None:
            trust *= 0.5
            if trust < 0.5:
                break
            continue

        step_xy, step_bw = solved
        step_xy = enforce_separation(clip_positions(step_xy, cfg), cfg, rng)

        cand_joint = _eval_fixed(scenario, step_xy, a, proc, step_bw)
        # Exact B-slice at the new geometry (r = SE(q) B is linear in B given q).
        cand_exact_b = _eval_fixed(scenario, step_xy, a, proc)
        if cand_exact_b[0] >= cand_joint[0]:
            cand, a_new, proc_new = cand_exact_b, a, proc
        else:
            cand, a_new, proc_new = cand_joint, a, proc

        refreshed_xy, a_r, proc_r, _ = complete_solution(scenario, step_xy)
        if not (np.array_equal(a_r, a) and np.array_equal(proc_r, proc)):
            cand_r = _eval_fixed(scenario, refreshed_xy, a_r, proc_r)
            if cand_r[0] > cand[0]:
                cand, a_new, proc_new = cand_r, a_r, proc_r

        new_fit, new_result, new_xy, new_bw = cand
        if new_fit + SCA_TOL < fit:
            trust *= 0.5
            if trust < 0.5:
                break
            continue
        xy, bw, result, fit = new_xy, new_bw, new_result, new_fit
        a, proc = a_new, proc_new
        if abs(fit - prev) <= SCA_TOL:
            break
        prev = fit
        log.debug("  sca iter %d  %s  trust=%g", it + 1, mbps(result.sum_rate), trust)

    result = evaluate(scenario, xy, a, proc, bw)
    return xy, result, time.perf_counter() - t0
