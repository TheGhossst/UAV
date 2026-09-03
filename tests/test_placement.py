"""Random (θ) and k-means placement."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig
from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.random import place_random
from uavdt.scenario import generate_scenario


def test_random_respects_field_height_and_separation():
    cfg = SimConfig()
    uav = place_random(3, seed=11, cfg=cfg)
    assert uav.shape == (3, 3)
    np.testing.assert_allclose(uav[:, 2], 100.0)
    assert np.all(uav[:, 0] >= 0.0) and np.all(uav[:, 0] <= 100.0)
    assert np.all(uav[:, 1] >= 0.0) and np.all(uav[:, 1] <= 100.0)
    for p in range(3):
        for q in range(p + 1, 3):
            d = np.linalg.norm(uav[p] - uav[q])
            assert d >= cfg.uav_min_separation_m - 1e-9


def test_random_seed_reproducible():
    cfg = SimConfig()
    a = place_random(3, seed=4, cfg=cfg)
    b = place_random(3, seed=4, cfg=cfg)
    np.testing.assert_allclose(a, b)


def test_kmeans_centroids_in_field_with_height(frozen_scenario):
    uav = place_kmeans(frozen_scenario, seed=2)
    assert uav.shape == (3, 3)
    np.testing.assert_allclose(uav[:, 2], 100.0)
    assert np.all(uav[:, 0] >= 0.0) and np.all(uav[:, 0] <= 100.0)
    for p in range(3):
        for q in range(p + 1, 3):
            assert np.linalg.norm(uav[p] - uav[q]) >= 10.0 - 1e-9


def test_kmeans_two_well_separated_clusters():
    cfg = SimConfig()
    iot = np.zeros((10, 3))
    iot[:5, 0] = 10.0
    iot[:5, 1] = np.linspace(10, 20, 5)
    iot[5:, 0] = 90.0
    iot[5:, 1] = np.linspace(80, 90, 5)
    sc = generate_scenario(seed=0, cfg=cfg, iot_xyz_m=iot)
    uav = place_kmeans(sc, seed=0, num_uav=2)
    xs = np.sort(uav[:, 0])
    assert xs[0] < 40.0
    assert xs[1] > 60.0
