"""Numpy CMA-ES residual search (non-RL control)."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig
from uavdt.placement.cmaes import apply_residual_xy, cmaes_maximize, residual_bounds
from uavdt.placement.kmeans import place_kmeans
from uavdt.scenario import generate_scenario


def test_cmaes_maximizes_negative_sphere():
    def fn(x: np.ndarray) -> float:
        return -float(np.sum((x - 1.0) ** 2))

    result = cmaes_maximize(
        fn,
        np.zeros(2),
        sigma0=0.8,
        max_evals=250,
        seed=0,
        lower=np.array([-5.0, -5.0]),
        upper=np.array([5.0, 5.0]),
    )
    assert result.n_evals <= 250
    np.testing.assert_allclose(result.x, np.array([1.0, 1.0]), atol=0.15)
    assert result.value > -0.05


def test_residual_bounds_keep_uavs_in_field():
    cfg = SimConfig()
    sc = generate_scenario(1, cfg)
    origin = place_kmeans(sc, 1)
    lo, hi = residual_bounds(origin, cfg)
    j = cfg.num_uav
    delta = np.empty(2 * j, dtype=float)
    delta[0::2] = hi[0::2]
    delta[1::2] = lo[1::2]
    uav = apply_residual_xy(origin, delta, cfg)
    assert np.all(uav[:, 0] >= 0.0) and np.all(uav[:, 0] <= cfg.area_x_m)
    assert np.all(uav[:, 1] >= 0.0) and np.all(uav[:, 1] <= cfg.area_y_m)
    np.testing.assert_allclose(uav[:, 2], cfg.uav_height_m)
