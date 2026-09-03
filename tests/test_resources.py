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
