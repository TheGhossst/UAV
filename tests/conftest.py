"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.scenario import generate_scenario, make_uav_xyz_m


@pytest.fixture
def cfg() -> SimConfig:
    return SimConfig()


@pytest.fixture
def frozen_scenario(cfg: SimConfig):
    """Deterministic 10 IoTs in the 100×100 field, two processes of five."""
    iot = np.zeros((10, 3), dtype=float)
    iot[:, 0] = np.linspace(5.0, 95.0, 10)
    iot[:, 1] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 15.0])
    return generate_scenario(seed=0, cfg=cfg, iot_xyz_m=iot)


@pytest.fixture
def three_uavs(cfg: SimConfig):
    xy = np.array([[20.0, 20.0], [50.0, 80.0], [80.0, 30.0]])
    return make_uav_xyz_m(xy, cfg.uav_height_m)
