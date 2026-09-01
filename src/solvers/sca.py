"""Successive convex approximation reconstruction (Python).

Algorithm 1 is a sketch. This implementation:
1. Fixes binaries via k-means + repair (association / processing).
2. Allocates bandwidth as the exact LP optimum for the linear-in-B rate model,
   raising floors for AoDT when the compute model is on so each sweep's T_k / λ / μ
   actually change the allocation.
3. Moves UAV positions in a trust region along the numerical gradient of
   the true evaluator (first-order / successive linearization).

Binaries are held fixed while the gradient is probed (so the finite differences
measure placement only), but each candidate step is also scored with binaries
refreshed for the new geometry, and the better of the two is taken. Without
that refresh SCA is stuck with the association k-means chose at iteration 0,
which makes it lose to the baselines that re-associate on every evaluation.

Not bit-exact with MATLAB CVX+MOSEK.
"""

from __future__ import annotations

import time

import numpy as np

from src.config import PSO_PENALTY, SCA_MAX_ITER, SCA_STEP, SCA_TOL, SCA_TRUST
from src.evaluator import EvalResult, evaluate, fitness
from src.logutil import log, mbps
from src.repair import (
    allocate_constrained_bandwidth,
    clip_positions,
    complete_solution,
    enforce_separation,
    process_consistent_processing,
)
from src.scenario import Scenario
from src.solvers.kmeans import kmeans


def _allocate_bandwidth(
    scenario: Scenario,
    xy: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the bandwidth LP independently on every pool of constraint (27).

    When compute is on, floors include AoDT delay requirements so SCA re-solves
    under the active T_k / λ / μ of this scenario.
    """
    if processing is None:
        processing = process_consistent_processing(scenario, association)
    return allocate_constrained_bandwidth(scenario, xy, association, processing)


def _eval_fixed(
    scenario: Scenario,
    xy: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
) -> tuple[float, EvalResult, np.ndarray, np.ndarray]:
    """Score placement with binaries held fixed (no complete_solution)."""
    xy = clip_positions(xy, scenario.cfg)
    bw = _allocate_bandwidth(scenario, xy, association, processing)
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
    log.info("sca  J=%d max_iter=%d trust=%g", j, max_iter, trust)

    for it in range(max_iter):
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

        cand = _eval_fixed(scenario, step, a, proc)
        a_new, proc_new = a, proc
        refreshed_xy, a_r, proc_r, _ = complete_solution(scenario, step)
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
