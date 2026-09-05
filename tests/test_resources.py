"""a_ij, b_ij, B_sys vs B_ij."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig
from uavdt.resources import (
    allocation_from_positions,
    equal_share_bandwidth_hz,
    nearest_association,
    process_consistent_processing,
)


def test_association_is_exclusive(frozen_scenario, three_uavs):
    a = nearest_association(frozen_scenario.iot_xyz_m, three_uavs)
    assert a.shape == (10, 3)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    assert set(np.unique(a).tolist()) <= {0.0, 1.0}


def test_processing_is_process_consistent(frozen_scenario, three_uavs):
    a = nearest_association(frozen_scenario.iot_xyz_m, three_uavs)
    b = process_consistent_processing(frozen_scenario, a)
    for proc in frozen_scenario.processes:
        js = np.argmax(b[proc.iot_indices], axis=1)
        assert np.unique(js).size == 1


def test_equal_share_uses_system_budget_not_full_band_per_iot(frozen_scenario, three_uavs):
    cfg = frozen_scenario.cfg
    a = nearest_association(frozen_scenario.iot_xyz_m, three_uavs)
    bw = equal_share_bandwidth_hz(a, cfg)
    np.testing.assert_allclose(bw.sum(), cfg.b_sys_hz)
    assert np.all(bw[a < 0.5] == 0.0)
    # No IoT receives the entire system band unless I=1.
    assert np.all(bw[a > 0.5] < cfg.b_sys_hz - 1.0)


def test_allocation_shapes(frozen_scenario, three_uavs):
    alloc = allocation_from_positions(frozen_scenario, three_uavs)
    assert alloc.association.shape == (10, 3)
    assert alloc.processing.shape == (10, 3)
    assert alloc.bandwidth_hz.shape == (10, 3)


def test_cpu_stable_keeps_majority_when_one_uav_can_take_both_groups(frozen_scenario):
    from uavdt.computation import queue_unstable
    from uavdt.resources import cpu_stable_processing

    a = np.zeros((10, 3), dtype=float)
    a[:, 0] = 1.0
    majority = process_consistent_processing(frozen_scenario, a)
    repaired = cpu_stable_processing(frozen_scenario, a)
    np.testing.assert_array_equal(repaired, majority)
    mu = frozen_scenario.cfg.service_rate_per_s
    assert not np.any(queue_unstable(majority, frozen_scenario.lambdas_per_s, mu))


def test_cpu_stable_splits_processes_when_majority_violates_24():
    from uavdt.computation import queue_unstable
    from uavdt.experiments.grids import config_for_counts
    from uavdt.resources import cpu_stable_processing
    from uavdt.scenario import generate_scenario

    cfg = config_for_counts(32, 3, SimConfig())
    sc = generate_scenario(1, cfg)
    a = np.zeros((32, 3), dtype=float)
    a[:, 0] = 1.0
    majority = process_consistent_processing(sc, a)
    mu = sc.cfg.service_rate_per_s
    assert np.any(queue_unstable(majority, sc.lambdas_per_s, mu))
    repaired = cpu_stable_processing(sc, a)
    assert not np.any(queue_unstable(repaired, sc.lambdas_per_s, mu))
    assigned = []
    for proc in sc.processes:
        js = np.argmax(repaired[proc.iot_indices], axis=1)
        assert np.unique(js).size == 1
        assigned.append(int(js[0]))
    assert assigned[0] != assigned[1]


def test_cpu_stable_keeps_majority_at_i24_both_on_one_uav():
    """I=24: 48/s < μ≈53.3/s, so sharing a UAV is legal — do not rematch."""
    from uavdt.computation import queue_unstable
    from uavdt.experiments.grids import config_for_counts
    from uavdt.resources import cpu_stable_processing
    from uavdt.scenario import generate_scenario

    cfg = config_for_counts(24, 3, SimConfig())
    sc = generate_scenario(1, cfg)
    a = np.zeros((24, 3), dtype=float)
    a[:, 0] = 1.0
    majority = process_consistent_processing(sc, a)
    mu = sc.cfg.service_rate_per_s
    assert not np.any(queue_unstable(majority, sc.lambdas_per_s, mu))
    repaired = cpu_stable_processing(sc, a)
    np.testing.assert_array_equal(repaired, majority)


def test_cpu_stable_splits_low_cpu_default_i10():
    from uavdt.computation import queue_unstable
    from uavdt.resources import cpu_stable_processing
    from uavdt.scenario import generate_scenario

    cfg = SimConfig(uav_cpu_cycles_per_s=0.5e8)
    sc = generate_scenario(1, cfg)
    a = np.zeros((10, 3), dtype=float)
    a[:, 0] = 1.0
    majority = process_consistent_processing(sc, a)
    mu = sc.cfg.service_rate_per_s
    assert np.any(queue_unstable(majority, sc.lambdas_per_s, mu))
    repaired = cpu_stable_processing(sc, a)
    assert not np.any(queue_unstable(repaired, sc.lambdas_per_s, mu))


def test_process_uav_map_enumeration_stays_small_at_campaign_k2():
    from uavdt.resources import _MAX_PROCESS_UAV_MAPS, _iter_process_uav_maps

    maps = list(_iter_process_uav_maps(k=2, j=3))
    assert len(maps) == 9
    assert len(set(maps)) == 9
    huge = list(_iter_process_uav_maps(k=8, j=8))
    assert len(huge) == _MAX_PROCESS_UAV_MAPS


def test_tk08_negative_slack_is_forwarding_not_cpu():
    """T_k=0.8 fails under nearest-a because T_u2u eats the slack, not (24)."""
    from dataclasses import replace

    from uavdt.experiments.grids import config_for_counts
    from uavdt.placement.kmeans import place_kmeans
    from uavdt.resources import cpu_stable_processing, nearest_association
    from uavdt.scenario import generate_scenario
    from uavdt.sca.cvx_problem import aodt_upload_slacks_s

    cfg = replace(
        config_for_counts(10, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)),
        aodt_threshold_s=0.8,
    )
    sc = generate_scenario(1, cfg)
    uav = place_kmeans(sc, 1)
    a_near = nearest_association(sc.iot_xyz_m, uav)
    b_near = cpu_stable_processing(sc, a_near)
    ja = np.argmax(a_near, axis=1)
    jb = np.argmax(b_near, axis=1)
    assert np.any(ja != jb)
    slacks_near = aodt_upload_slacks_s(sc, a_near, b_near)
    assert float(np.min(slacks_near[a_near.sum(axis=1) > 0.5])) < 0.0

    a_cohesive = np.zeros_like(a_near)
    for proc in sc.processes:
        members = proc.iot_indices
        centroid = sc.iot_xyz_m[members].mean(axis=0, keepdims=True)
        j_star = int(np.argmin(np.linalg.norm(centroid - uav, axis=1)))
        a_cohesive[members, j_star] = 1.0
    b_cohesive = process_consistent_processing(sc, a_cohesive)
    slacks_ok = aodt_upload_slacks_s(sc, a_cohesive, b_cohesive)
    assert float(np.min(slacks_ok[a_cohesive.sum(axis=1) > 0.5])) > 0.2

