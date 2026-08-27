"""Successive convex approximation reconstruction (Python).

Algorithm 1 is a sketch. This implementation:
1. Fixes binaries via k-means + repair (association / processing).
2. Allocates bandwidth as the exact LP optimum for the linear-in-B rate model.
3. Moves UAV positions in a trust region along the numerical gradient of
   the true evaluator (first-order / successive linearization).

Known limitation: association and processing stay frozen after the initial
repair. Placement PSO re-associates every evaluation, so it can keep QoS
feasible after UAVs move; SCA cannot. A ~50% feasible-seed split on the
20-run Table II geometry is that ceiling, not a bandwidth-allocator artifact.
Refreshing ``a``/``proc`` at accepted steps would be the lever to close the
gap vs PSO; it is intentionally not done here.

Not bit-exact with MATLAB CVX+MOSEK.
"""

from __future__ import annotations

import time

import numpy as np

from src.comm import link_metrics
from src.config import PSO_PENALTY, SCA_MAX_ITER, SCA_STEP, SCA_TOL, SCA_TRUST
from src.evaluator import EvalResult, evaluate, fitness
from src.repair import clip_positions, complete_solution, enforce_separation, equal_bandwidth
from src.scenario import Scenario
from src.solvers.kmeans import kmeans


def _allocate_bandwidth(scenario: Scenario, xy: np.ndarray, association: np.ndarray) -> np.ndarray:
    """Exact LP for r_ij = c_ij B_ij at fixed SNR / association.

    max  sum c_ij B_ij
    s.t. sum B_ij = B_sys,  B_ij = 0 if not associated,
         B_ij >= R_min / c_ij when those floors fit in B_sys.

    If all floors fit, leftover bandwidth goes to the highest-c link (a vertex
    of the simplex). If they do not, fund as many full floors as possible
    (cheapest first, leaving a positive remainder for everyone else), then
    split the rest equally across still-unfunded associated links so every
    associated link has B > 0.
    """
    cfg = scenario.cfg
    dummy = equal_bandwidth(association, cfg)
    se = np.maximum(link_metrics(scenario.iot_xy, xy, np.ones_like(dummy), cfg)["rates"], 1e-12)
    mask = association > 0.5
    b = np.zeros_like(dummy)
    if not np.any(mask):
        return dummy

    se_a = se[mask]
    need = cfg.r_min / se_a
    n = int(need.size)
    remaining = float(cfg.b_sys)
    if float(need.sum()) <= remaining:
        alloc = need.copy()
        alloc[int(np.argmax(se_a))] += remaining - float(need.sum())
        b[mask] = alloc
        return b

    alloc = np.zeros_like(need)
    funded = np.zeros(n, dtype=bool)
    for k in np.argsort(need):
        n_other_unfunded = int(n - funded.sum() - 1)
        if remaining >= float(need[k]) and (n_other_unfunded == 0 or remaining > float(need[k])):
            alloc[k] = need[k]
            remaining -= float(need[k])
            funded[k] = True
        else:
            break
    unfunded = ~funded
    n_u = int(unfunded.sum())
    if n_u:
        alloc[unfunded] = remaining / n_u
    elif remaining > 0:
        alloc[int(np.argmax(se_a))] += remaining
    b[mask] = alloc
    return b


def _eval_fixed(
    scenario: Scenario,
    xy: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
) -> tuple[float, EvalResult, np.ndarray, np.ndarray]:
    """Score placement with binaries held fixed (no complete_solution)."""
    xy = clip_positions(xy, scenario.cfg)
    bw = _allocate_bandwidth(scenario, xy, association)
    result = evaluate(scenario, xy, association, processing, bw)
    return fitness(result, PSO_PENALTY), result, xy, bw


def _partial_axis(
    scenario: Scenario,
    xy: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    fit: float,
    u: int,
    ax: int,
    eps: float,
) -> float:
    """Forward, backward, or central difference; backward if +eps clips."""
    cfg = scenario.cfg
    limit = cfg.area_x if ax == 0 else cfg.area_y
    x0 = xy[u, ax]

    plus = xy.copy()
    plus[u, ax] = np.clip(x0 + eps, 0.0, limit)
    d_plus = plus[u, ax] - x0

    minus = xy.copy()
    minus[u, ax] = np.clip(x0 - eps, 0.0, limit)
    d_minus = x0 - minus[u, ax]

    if d_plus > 1e-12 and d_minus > 1e-12:
        f_plus, *_ = _eval_fixed(scenario, plus, association, processing)
        f_minus, *_ = _eval_fixed(scenario, minus, association, processing)
        return (f_plus - f_minus) / (plus[u, ax] - minus[u, ax])
    if d_plus > 1e-12:
        f_plus, *_ = _eval_fixed(scenario, plus, association, processing)
        return (f_plus - fit) / d_plus
    if d_minus > 1e-12:
        f_minus, *_ = _eval_fixed(scenario, minus, association, processing)
        return (fit - f_minus) / d_minus
    return 0.0


def solve_sca(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    max_iter: int = SCA_MAX_ITER,
    trust: float = SCA_TRUST,
) -> tuple[np.ndarray, EvalResult, float]:
    cfg = scenario.cfg
    j = n_uav if n_uav is not None else cfg.num_uav
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    xy = kmeans(scenario.iot_xy, j, rng)

    xy, a, proc, _ = complete_solution(scenario, xy)
    xy = enforce_separation(clip_positions(xy, cfg), cfg, rng)
    fit, result, xy, bw = _eval_fixed(scenario, xy, a, proc)
    prev = fit
    eps = 1.0  # finite-difference step (m)

    for _ in range(max_iter):
        grad = np.zeros_like(xy)
        for u in range(j):
            for ax in range(2):
                grad[u, ax] = _partial_axis(scenario, xy, a, proc, fit, u, ax, eps)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        step = xy + SCA_STEP * (trust / max(gnorm, 1e-9)) * grad
        step[:, 0] = np.clip(step[:, 0], 0.0, cfg.area_x)
        step[:, 1] = np.clip(step[:, 1], 0.0, cfg.area_y)
        delta = step - xy
        dnorm = np.linalg.norm(delta)
        if dnorm > trust:
            delta *= trust / dnorm
            step = xy + delta
        step = enforce_separation(clip_positions(step, cfg), cfg, rng)
        new_fit, new_result, new_xy, new_bw = _eval_fixed(scenario, step, a, proc)
        if new_fit + SCA_TOL < fit:
            trust *= 0.5
            if trust < 0.5:
                break
            continue
        xy, bw, result, fit = new_xy, new_bw, new_result, new_fit
        if abs(fit - prev) <= SCA_TOL:
            break
        prev = fit

    result = evaluate(scenario, xy, a, proc, bw)
    return xy, result, time.perf_counter() - t0
