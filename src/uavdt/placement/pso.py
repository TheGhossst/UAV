"""PSO UAV placement. EXTERNAL baseline, not in the paper.

Paper §VII compares SCA, TD3, k-means, and random. PSO is this
reproduction's extra optimizer for the experimental campaign.
Hyperparameters are algorithm knobs, not Table II.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.config import SimConfig
from uavdt.evaluator import evaluate
from uavdt.models import Scenario
from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.random import place_random
from uavdt.scenario import make_uav_xyz_m


@dataclass(frozen=True)
class PSOSettings:
    """EXTERNAL PARAMETER. Not paper Table II / Problem (P)."""

    n_particles: int = 12
    n_iterations: int = 20
    inertia: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    v_max_frac: float = 0.2


def _repair_xy(
    xy: np.ndarray, cfg: SimConfig, rng: np.random.Generator
) -> np.ndarray:
    out = np.asarray(xy, dtype=float).copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, cfg.area_x_m)
    out[:, 1] = np.clip(out[:, 1], 0.0, cfg.area_y_m)
    theta = cfg.uav_min_separation_m
    for j in range(1, out.shape[0]):
        for _ in range(4_000):
            d = np.linalg.norm(out[:j] - out[j][None, :], axis=1)
            if np.all(d >= theta - 1e-9):
                break
            out[j] = np.array(
                [
                    rng.uniform(0.0, cfg.area_x_m),
                    rng.uniform(0.0, cfg.area_y_m),
                ]
            )
    return out


def _fitness_equal_share(
    scenario: Scenario, xy: np.ndarray, rng: np.random.Generator
) -> float:
    """Fast inner fitness: nearest-a, process-b, equal B_ij, true evaluate().

    IMPLEMENTATION CHOICE: the PSO search uses equal-share bandwidth so
    each particle is cheap. The campaign then re-scores the returned
    geometry with the exact frozen-q bandwidth LP (same LP as SCA's B
    step). Not a paper PSO.
    """
    cfg = scenario.cfg
    uav = make_uav_xyz_m(_repair_xy(xy, cfg, rng), cfg.uav_height_m)
    ev = evaluate(scenario, uav)
    if ev.feasible:
        return float(ev.sum_rate_bit_per_s)
    c = ev.constraints
    penalty = (
        1e6
        * (
            c.qos_violations
            + c.aodt_violations
            + c.sep_violations
            + c.cpu_unstable_count
            + (0 if c.bandwidth_budget_ok else 1)
        )
    )
    return float(ev.sum_rate_bit_per_s) - penalty


def place_pso(
    scenario: Scenario,
    seed: int,
    settings: PSOSettings | None = None,
) -> np.ndarray:
    """Swarm search over UAV (x, y); altitude H is fixed."""
    settings = settings or PSOSettings()
    cfg = scenario.cfg
    j = cfg.num_uav
    rng = np.random.default_rng(seed)
    vmax = settings.v_max_frac * max(cfg.area_x_m, cfg.area_y_m)

    pos = np.zeros((settings.n_particles, j, 2), dtype=float)
    vel = rng.uniform(-vmax, vmax, size=pos.shape)
    pos[0] = place_kmeans(scenario, seed)[:, :2]
    for p in range(1, settings.n_particles):
        pos[p] = place_random(j, seed=seed * 10_000 + p, cfg=cfg)[:, :2]
        pos[p] = _repair_xy(pos[p], cfg, rng)

    best_pos = pos.copy()
    best_fit = np.full(settings.n_particles, -np.inf)
    for p in range(settings.n_particles):
        best_fit[p] = _fitness_equal_share(scenario, pos[p], rng)
        best_pos[p] = pos[p]
    g = int(np.argmax(best_fit))
    gbest = best_pos[g].copy()
    gfit = float(best_fit[g])

    for _ in range(settings.n_iterations):
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vel = (
            settings.inertia * vel
            + settings.c1 * r1 * (best_pos - pos)
            + settings.c2 * r2 * (gbest[None, :, :] - pos)
        )
        vel = np.clip(vel, -vmax, vmax)
        pos = pos + vel
        for p in range(settings.n_particles):
            pos[p] = _repair_xy(pos[p], cfg, rng)
            fit = _fitness_equal_share(scenario, pos[p], rng)
            if fit > best_fit[p]:
                best_fit[p] = fit
                best_pos[p] = pos[p]
            if fit > gfit:
                gfit = fit
                gbest = pos[p].copy()
    return make_uav_xyz_m(_repair_xy(gbest, cfg, rng), cfg.uav_height_m)
