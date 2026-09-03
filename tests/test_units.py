"""Display conversions only; internals stay SI."""

from __future__ import annotations

import numpy as np

from uavdt.config import BANDWIDTH_PRESETS, SimConfig
from uavdt.evaluator import evaluate
from uavdt.placement.random import place_random
from uavdt.scenario import generate_scenario


def test_bandwidth_presets_are_hz_not_khz():
    assert BANDWIDTH_PRESETS["20khz"] == 20_000.0
    assert BANDWIDTH_PRESETS["2.4mhz"] == 2_400_000.0
    assert BANDWIDTH_PRESETS["8.8mhz"] == 8_800_000.0


def test_three_bandwidths_change_sum_rate_proportionally():
    cfg0 = SimConfig(b_sys_hz=20_000.0)
    sc = generate_scenario(seed=1, cfg=cfg0)
    uav = place_random(3, seed=1, cfg=cfg0)
    r20 = evaluate(sc, uav).sum_rate_bit_per_s
    cfg24 = SimConfig(b_sys_hz=2_400_000.0)
    sc24 = generate_scenario(seed=1, cfg=cfg24, iot_xyz_m=sc.iot_xyz_m)
    r24 = evaluate(sc24, uav).sum_rate_bit_per_s
    cfg88 = SimConfig(b_sys_hz=8_800_000.0)
    sc88 = generate_scenario(seed=1, cfg=cfg88, iot_xyz_m=sc.iot_xyz_m)
    r88 = evaluate(sc88, uav).sum_rate_bit_per_s
    np.testing.assert_allclose(r24 / r20, 2_400_000.0 / 20_000.0)
    np.testing.assert_allclose(r88 / r20, 8_800_000.0 / 20_000.0)


def test_mbps_property():
    cfg = SimConfig()
    sc = generate_scenario(seed=1, cfg=cfg)
    uav = place_random(3, seed=1, cfg=cfg)
    result = evaluate(sc, uav)
    assert abs(result.sum_rate_mbps - result.sum_rate_bit_per_s / 1e6) < 1e-15
