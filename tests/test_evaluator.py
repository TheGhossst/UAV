"""Constraint flags and sum rate."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig
from uavdt.evaluator import evaluate
from uavdt.models import Allocation
from uavdt.placement.random import place_random
from uavdt.resources import allocation_from_positions, nearest_association


def test_sum_rate_is_associated_links_only(frozen_scenario, three_uavs):
    result = evaluate(frozen_scenario, three_uavs)
    a = result.extras["association"]
    expected = float((a * result.rates_bit_per_s).sum())
    np.testing.assert_allclose(result.sum_rate_bit_per_s, expected)


def test_uavs_have_altitude_100(frozen_scenario, three_uavs):
    assert three_uavs.shape == (3, 3)
    np.testing.assert_allclose(three_uavs[:, 2], 100.0)
    result = evaluate(frozen_scenario, three_uavs)
    assert result.constraints.uav_in_field_ok


def test_separation_violation_detected(frozen_scenario):
    uav = np.array(
        [
            [10.0, 10.0, 100.0],
            [12.0, 10.0, 100.0],  # 2 m < 10 m
            [80.0, 80.0, 100.0],
        ]
    )
    result = evaluate(frozen_scenario, uav)
    assert result.constraints.sep_violations >= 1
    assert not result.constraints.uav_separation_ok


def test_qos_uses_r_min():
    cfg = SimConfig(r_min_bit_per_s=1e12)  # impossible
    from uavdt.scenario import generate_scenario

    sc = generate_scenario(seed=1, cfg=cfg)
    uav = place_random(3, seed=1, cfg=cfg)
    result = evaluate(sc, uav)
    assert result.constraints.qos_violations == 10
    assert not result.constraints.qos_ok


def test_bandwidth_over_budget_detected(frozen_scenario, three_uavs):
    alloc = allocation_from_positions(frozen_scenario, three_uavs)
    bw = alloc.bandwidth_hz * 2.0
    bad = Allocation(alloc.association, alloc.processing, bw)
    result = evaluate(frozen_scenario, three_uavs, bad)
    assert not result.constraints.bandwidth_budget_ok


def test_aodt_threshold_flag(frozen_scenario, three_uavs):
    result = evaluate(frozen_scenario, three_uavs)
    for k, ok in enumerate(result.aodt_satisfied):
        if ok:
            assert result.aodt_s[k] <= frozen_scenario.cfg.aodt_threshold_s + 1e-9
        else:
            assert result.aodt_s[k] > frozen_scenario.cfg.aodt_threshold_s
    assert result.constraints.aodt_ok == bool(np.all(result.aodt_satisfied))
