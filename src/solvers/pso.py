"""PSO solvers: placement-only and joint encoding."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.config import (
    PSO_C1,
    PSO_C2,
    PSO_N_ITER,
    PSO_N_PARTICLES,
    PSO_PENALTY,
    PSO_V_MAX_FRAC,
    PSO_W,
)
from src.evaluator import EvalResult, evaluate, fitness
from src.logutil import log, mbps
from src.repair import (
    clip_positions,
    complete_solution,
    enforce_separation,
    process_consistent_processing,
    softmax_association,
)
from src.scenario import Scenario


@dataclass
class PSOHistory:
    best_fitness: list[float]
    best_sum_rate: list[float]


def _eval_placement(scenario: Scenario, xy: np.ndarray) -> EvalResult:
    xy, a, b, bw = complete_solution(scenario, xy)
    return evaluate(scenario, xy, a, b, bw)


def solve_pso_placement(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    n_particles: int = PSO_N_PARTICLES,
    n_iter: int = PSO_N_ITER,
    w: float = PSO_W,
    c1: float = PSO_C1,
    c2: float = PSO_C2,
) -> tuple[np.ndarray, EvalResult, float, PSOHistory]:
    cfg = scenario.cfg
    j = n_uav if n_uav is not None else cfg.num_uav
    rng = np.random.default_rng(seed)
    dim = j * 2
    vmax = PSO_V_MAX_FRAC * max(cfg.area_x, cfg.area_y)

    pos = rng.uniform(0.0, 1.0, size=(n_particles, dim))
    pos[:, 0::2] *= cfg.area_x
    pos[:, 1::2] *= cfg.area_y
    vel = rng.uniform(-vmax, vmax, size=(n_particles, dim))

    pbest = pos.copy()
    pbest_fit = np.full(n_particles, -np.inf)
    pbest_rate = np.zeros(n_particles)
    gbest = pos[0].copy()
    gbest_fit = -np.inf
    gbest_rate = 0.0
    history = PSOHistory(best_fitness=[], best_sum_rate=[])

    t0 = time.perf_counter()
    log.info("pso placement  J=%d particles=%d iters=%d", j, n_particles, n_iter)
    for it in range(n_iter):
        for n in range(n_particles):
            xy = enforce_separation(clip_positions(pos[n].reshape(j, 2), cfg), cfg, rng)
            pos[n] = xy.reshape(-1)
            result = _eval_placement(scenario, xy)
            fit = fitness(result, PSO_PENALTY)
            if fit > pbest_fit[n]:
                pbest_fit[n] = fit
                pbest_rate[n] = result.sum_rate
                pbest[n] = pos[n].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_rate = result.sum_rate
                gbest = pos[n].copy()
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = pos + vel
        history.best_fitness.append(float(gbest_fit))
        history.best_sum_rate.append(float(gbest_rate))
        if it == 0 or (it + 1) % 25 == 0 or it + 1 == n_iter:
            log.debug("  pso iter %d/%d  best %s", it + 1, n_iter, mbps(gbest_rate))

    xy = enforce_separation(clip_positions(gbest.reshape(j, 2), cfg), cfg, rng)
    result = _eval_placement(scenario, xy)
    return xy, result, time.perf_counter() - t0, history


def _decode_joint(scenario: Scenario, vec: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = scenario.cfg
    i = cfg.num_iot
    n_xy = j * 2
    n_a = i * j
    n_p = i * j
    n_b = i * j
    xy = vec[:n_xy].reshape(j, 2)
    assoc_logits = vec[n_xy : n_xy + n_a].reshape(i, j)
    proc_logits = vec[n_xy + n_a : n_xy + n_a + n_p].reshape(i, j)
    bw_raw = vec[n_xy + n_a + n_p : n_xy + n_a + n_p + n_b].reshape(i, j)
    a = softmax_association(assoc_logits)
    b = process_consistent_processing(scenario, a, proc_logits)
    return complete_solution(scenario, xy, association=a, processing=b, bandwidth=np.abs(bw_raw))


def solve_pso_joint(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    n_particles: int = PSO_N_PARTICLES,
    n_iter: int = PSO_N_ITER,
) -> tuple[np.ndarray, EvalResult, float, PSOHistory]:
    cfg = scenario.cfg
    j = n_uav if n_uav is not None else cfg.num_uav
    i = cfg.num_iot
    rng = np.random.default_rng(seed)
    dim = j * 2 + 3 * i * j
    vmax = PSO_V_MAX_FRAC * max(cfg.area_x, cfg.area_y)

    from src.solvers.kmeans import kmeans

    pos = rng.normal(0.0, 0.5, size=(n_particles, dim))
    # Seed positions around k-means so joint search is not pure noise.
    base_xy = kmeans(scenario.iot_xy, j, rng)
    for n in range(n_particles):
        xy_n = base_xy + rng.normal(0.0, 30.0, size=base_xy.shape)
        xy_n[:, 0] = np.clip(xy_n[:, 0], 0.0, cfg.area_x)
        xy_n[:, 1] = np.clip(xy_n[:, 1], 0.0, cfg.area_y)
        pos[n, : j * 2] = xy_n.reshape(-1)
        d = np.linalg.norm(scenario.iot_xy[:, None, :] - xy_n[None, :, :], axis=-1)
        pos[n, j * 2 : j * 2 + i * j] = (-d).reshape(-1)
        pos[n, j * 2 + i * j : j * 2 + 2 * i * j] = (-d).reshape(-1)
        pos[n, j * 2 + 2 * i * j :] = cfg.b_sys / max(i, 1)
    vel = rng.uniform(-vmax, vmax, size=(n_particles, dim))

    pbest = pos.copy()
    pbest_fit = np.full(n_particles, -np.inf)
    gbest = pos[0].copy()
    gbest_fit = -np.inf
    gbest_rate = 0.0
    history = PSOHistory(best_fitness=[], best_sum_rate=[])

    t0 = time.perf_counter()
    log.info("pso joint  J=%d particles=%d iters=%d", j, n_particles, n_iter)
    for it in range(n_iter):
        for n in range(n_particles):
            xy, a, b, bw = _decode_joint(scenario, pos[n], j)
            pos[n, : j * 2] = xy.reshape(-1)
            result = evaluate(scenario, xy, a, b, bw)
            fit = fitness(result, PSO_PENALTY)
            if fit > pbest_fit[n]:
                pbest_fit[n] = fit
                pbest[n] = pos[n].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_rate = result.sum_rate
                gbest = pos[n].copy()
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = PSO_W * vel + PSO_C1 * r1 * (pbest - pos) + PSO_C2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = pos + vel
        history.best_fitness.append(float(gbest_fit))
        history.best_sum_rate.append(float(gbest_rate))
        if it == 0 or (it + 1) % 25 == 0 or it + 1 == n_iter:
            log.debug("  pso joint iter %d/%d  best %s", it + 1, n_iter, mbps(gbest_rate))

    xy, a, b, bw = _decode_joint(scenario, gbest, j)
    result = evaluate(scenario, xy, a, b, bw)
    return xy, result, time.perf_counter() - t0, history
