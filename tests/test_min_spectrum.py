"""Min-spectrum B_sys (E1). Frozen-q search, not leftover-dump Mbps."""

from __future__ import annotations

import math

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.experiments.methods import run_method
from uavdt.min_spectrum import (
    associated_floors_hz,
    frozen_q_feasible,
    min_bsys_frozen_q,
    min_bsys_lower_bound_hz,
    scenario_with_radio,
)
from uavdt.placement.kmeans import place_kmeans
from uavdt.resources import cpu_stable_processing, nearest_association
from uavdt.scenario import generate_scenario, make_uav_xyz_m

cvxpy = pytest.importorskip("cvxpy")


def _kmeans_alloc(seed: int = 1, area_m: float = 100.0, share: float = 0.25):
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=share).with_square_area_m(area_m)
    sc = generate_scenario(seed, cfg)
    run = run_method(sc, "kmeans", seed)
    return sc, run


def test_scenario_with_radio_does_not_mutate():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    sc2 = scenario_with_radio(sc, b_sys_hz=1.0e6, max_bw_share=0.25)
    assert sc.cfg.b_sys_hz == 8.8e6
    assert sc2.cfg.b_sys_hz == 1.0e6
    np.testing.assert_allclose(sc.iot_xyz_m, sc2.iot_xyz_m)


def test_bound_is_max_of_sum_floors_and_cap():
    floors = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 200.0, 0.0],
            [0.0, 0.0, 50.0],
        ]
    )
    a = (floors > 0).astype(float)
    no_cap = min_bsys_lower_bound_hz(floors, a, None)
    assert no_cap["binding"] == "sum_floors"
    assert no_cap["bound_hz"] == pytest.approx(350.0)
    cap25 = min_bsys_lower_bound_hz(floors, a, 0.25)
    assert cap25["binding"] == "per_link_cap"
    assert cap25["bound_hz"] == pytest.approx(200.0 / 0.25)
    cap90 = min_bsys_lower_bound_hz(floors, a, 0.90)
    assert cap90["binding"] == "sum_floors"
    assert cap90["bound_hz"] == pytest.approx(350.0)


def test_nonpositive_aodt_slack_is_infeasible_at_any_pool():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25, aodt_threshold_s=0.01)
    sc = generate_scenario(1, cfg)
    uav = place_kmeans(sc, 1)
    a = nearest_association(sc.iot_xyz_m, uav)
    b = cpu_stable_processing(sc, a)
    floors, _se = associated_floors_hz(sc, uav, a, b)
    info = min_bsys_lower_bound_hz(floors, a, 0.25)
    assert info["binding"] == "infeasible_floors"
    assert not math.isfinite(info["bound_hz"])


def test_search_matches_bound_on_kmeans_100m():
    sc, run = _kmeans_alloc(seed=1, area_m=100.0)
    result = min_bsys_frozen_q(
        sc,
        run.uav_xyz_m,
        run.allocation,
        hi_hz=8.8e6,
        abs_tol_hz=1.0e3,
        place_eval=run.true_eval,
    )
    assert result.min_b_sys_hz is not None
    assert result.feasible_at_min
    assert result.bound_hz is not None
    assert result.min_b_sys_hz < 8.8e6
    assert abs(result.min_b_sys_hz - result.bound_hz) <= 2.0e3
    ok_lo, _, _ = frozen_q_feasible(
        sc,
        run.uav_xyz_m,
        run.allocation.hard_association(),
        run.allocation.hard_processing(),
        max(1.0, 0.5 * result.bound_hz),
        0.25,
    )
    assert not ok_lo


def test_zenith_needs_no_more_hertz_than_a_far_corner():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25).with_square_area_m(500.0)
    sc = generate_scenario(1, cfg)
    xy_z = sc.iot_xyz_m[:3, :2].copy()
    for j in range(1, 3):
        if np.linalg.norm(xy_z[j] - xy_z[0]) < 10.0:
            xy_z[j] = xy_z[0] + np.array([12.0, 0.0])
    uav_z = make_uav_xyz_m(xy_z, cfg.uav_height_m)
    uav_c = make_uav_xyz_m(
        np.array([[0.0, 0.0], [15.0, 0.0], [0.0, 15.0]]),
        cfg.uav_height_m,
    )
    run_z = run_method(sc, "random", 1, uav_xyz_m=uav_z)
    run_c = run_method(sc, "random", 1, uav_xyz_m=uav_c)
    z = min_bsys_frozen_q(sc, run_z.uav_xyz_m, run_z.allocation, hi_hz=8.8e6)
    c = min_bsys_frozen_q(sc, run_c.uav_xyz_m, run_c.allocation, hi_hz=8.8e6)
    assert z.min_b_sys_hz is not None
    assert c.min_b_sys_hz is not None
    assert z.min_b_sys_hz <= c.min_b_sys_hz + 1.0e3
    assert z.min_assoc_se >= c.min_assoc_se - 1e-9
