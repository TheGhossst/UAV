"""Taylor / supporting-hyperplane maps (no solver required)."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig
from uavdt.sca.linearize import (
    linearize_rate_product,
    linearize_separation,
    se_jacobian,
    spectral_efficiency,
)
from uavdt.scenario import make_uav_xyz_m


def test_se_is_eq6_at_unit_bandwidth():
    from uavdt.channel import link_metrics

    cfg = SimConfig()
    iot = np.array([[10.0, 20.0, 0.0], [80.0, 15.0, 0.0]])
    uav = make_uav_xyz_m([[30.0, 40.0], [70.0, 60.0]], 100.0)
    se = spectral_efficiency(iot, uav, cfg)
    ones = np.ones((2, 2))
    rates = link_metrics(iot, uav, ones, cfg)["rates_bit_per_s"]
    np.testing.assert_allclose(se, rates)


def test_se_jacobian_matches_small_step():
    cfg = SimConfig()
    iot = np.array([[10.0, 20.0, 0.0]])
    uav0 = make_uav_xyz_m([[40.0, 40.0]], 100.0)
    se0, gx, gy = se_jacobian(iot, uav0, cfg, step_m=1e-4)
    dx, dy = 1e-4, -1.5e-4
    uav1 = uav0.copy()
    uav1[0, 0] += dx
    uav1[0, 1] += dy
    se1 = spectral_efficiency(iot, uav1, cfg)
    pred = se0 + gx * dx + gy * dy
    np.testing.assert_allclose(se1, pred, rtol=0, atol=5e-10)


def test_rate_product_taylor_at_linearization_point_equals_true_product():
    cfg = SimConfig()
    iot = np.array([[5.0, 5.0, 0.0], [90.0, 10.0, 0.0]])
    uav = make_uav_xyz_m([[20.0, 30.0], [60.0, 70.0]], 100.0)
    bw = np.array([[1000.0, 0.0], [0.0, 2500.0]])
    lin = linearize_rate_product(iot, uav, bw, cfg, step_m=1e-3)
    pred = lin.value_numpy(bw, uav[:, 0], uav[:, 1])
    se = spectral_efficiency(iot, uav, cfg)
    np.testing.assert_allclose(pred, se * bw)


def test_separation_hyperplane_is_conservative():
    theta = 10.0
    uav0 = np.array(
        [
            [0.0, 0.0, 100.0],
            [20.0, 0.0, 100.0],
            [10.0, 30.0, 100.0],
        ]
    )
    planes = linearize_separation(uav0, theta)
    rng = np.random.default_rng(0)
    for _ in range(200):
        xy = rng.uniform(0.0, 100.0, size=(3, 2))
        uav = np.column_stack([xy, np.full(3, 100.0)])
        if all(p.residual_numpy(uav) >= -1e-9 for p in planes):
            for p in range(3):
                for q in range(p + 1, 3):
                    d = np.linalg.norm(uav[p] - uav[q])
                    assert d >= theta - 1e-8


def test_separation_is_not_squared_distance_shortcut():
    # The implemented constraint is u·(p_j-p_l) >= θ, not ||Δ||^2 >= θ^2.
    uav0 = np.array([[0.0, 0.0, 100.0], [20.0, 0.0, 100.0]])
    planes = linearize_separation(uav0, 10.0)
    assert len(planes) == 1
    p = planes[0]
    # Point with large orthogonal offset: original distance holds, hyperplane may not.
    uav_off = np.array([[0.0, 0.0, 100.0], [5.0, 50.0, 100.0]])
    d = np.linalg.norm(uav_off[0] - uav_off[1])
    assert d > 10.0
    assert p.residual_numpy(uav_off) < 0.0
