"""Scenario: area, IoT altitude, explicit process groups."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig
from uavdt.scenario import generate_scenario, process_groups


def test_process_groups_are_explicit_and_disjoint():
    cfg = SimConfig()
    groups = process_groups(cfg)
    assert len(groups) == 2
    n1 = set(groups[0].iot_indices.tolist())
    n2 = set(groups[1].iot_indices.tolist())
    assert n1 == {0, 1, 2, 3, 4}
    assert n2 == {5, 6, 7, 8, 9}
    assert n1.isdisjoint(n2)


def test_iot_in_100x100_at_z0():
    sc = generate_scenario(seed=7, cfg=SimConfig())
    assert sc.iot_xyz_m.shape == (10, 3)
    assert np.all(sc.iot_xyz_m[:, 0] >= 0.0) and np.all(sc.iot_xyz_m[:, 0] <= 100.0)
    assert np.all(sc.iot_xyz_m[:, 1] >= 0.0) and np.all(sc.iot_xyz_m[:, 1] <= 100.0)
    assert np.allclose(sc.iot_xyz_m[:, 2], 0.0)
    assert sc.num_processes == 2
    assert sc.processes[0].process_id == 0
    np.testing.assert_array_equal(sc.delta_ik.sum(axis=1), 1.0)
    np.testing.assert_array_equal(sc.delta_ik.sum(axis=0), 5.0)


def test_same_seed_reproducible():
    a = generate_scenario(seed=3)
    b = generate_scenario(seed=3)
    np.testing.assert_allclose(a.iot_xyz_m, b.iot_xyz_m)


def test_different_seeds_differ():
    a = generate_scenario(seed=1)
    b = generate_scenario(seed=2)
    assert not np.allclose(a.iot_xyz_m, b.iot_xyz_m)


def test_per_iot_lambda_can_be_heterogeneous():
    lam = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 3.0, 3.0, 3.0, 3.0, 3.0])
    sc = generate_scenario(seed=1, lambdas_per_s=lam)
    np.testing.assert_allclose(sc.lambdas_per_s, lam)
