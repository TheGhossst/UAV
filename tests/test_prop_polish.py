"""Local polish under proportional noise. Frozen SCA is not involved."""

import numpy as np
import pytest

from uavdt.analysis.prop_polish import polish_uav_positions
from uavdt.config import headline_sim_config
from uavdt.constraints import pairwise_uav_distance_m
from uavdt.placement.kmeans import place_kmeans
from uavdt.scenario import generate_scenario

cvxpy = pytest.importorskip("cvxpy")
scipy = pytest.importorskip("scipy")


def test_polish_stays_in_the_field_and_does_not_lose_the_start():
    cfg = headline_sim_config(area_m=500.0)
    sc = generate_scenario(1, cfg)
    start = place_kmeans(sc, 1)
    polished = polish_uav_positions(sc, start, max_evals=18, step_m=15.0)
    uav = polished.uav_xyz_m
    assert uav.shape == start.shape
    assert np.all(uav[:, 0] >= -1e-6) and np.all(uav[:, 0] <= cfg.area_x_m + 1e-6)
    assert np.all(uav[:, 1] >= -1e-6) and np.all(uav[:, 1] <= cfg.area_y_m + 1e-6)
    assert np.allclose(uav[:, 2], cfg.uav_height_m)
    sep = pairwise_uav_distance_m(uav)
    assert np.min(sep) + 1e-6 >= cfg.uav_min_separation_m
    assert polished.n_evals <= 18
    if polished.start_feasible and polished.feasible:
        assert polished.rate_mbps + 1e-3 >= polished.start_rate_mbps
