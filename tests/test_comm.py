import numpy as np

from src.comm import average_path_loss, distances, los_probability, uplink_rate
from src.config import DEFAULT, FROZEN_UAV_XY, NOISE_POWER, SIGMA
from src.scenario import generate_scenario


def test_noise_is_sigma_squared():
    assert SIGMA == 0.01
    assert NOISE_POWER == SIGMA ** 2


def test_frozen_scenario_deterministic():
    cfg = DEFAULT
    s1 = generate_scenario(100, cfg)
    s2 = generate_scenario(100, cfg)
    np.testing.assert_allclose(s1.iot_xy, s2.iot_xy)
    uav = np.array([FROZEN_UAV_XY])
    d1 = distances(s1.iot_xy, uav, cfg.uav_height)
    d2 = distances(s2.iot_xy, uav, cfg.uav_height)
    np.testing.assert_allclose(d1, d2)
    assert d1.shape == (10, 1)
    assert np.all(d1 >= cfg.uav_height)


def test_rate_increases_with_bandwidth():
    cfg = DEFAULT
    s = generate_scenario(100, cfg)
    uav = np.array([[250.0, 250.0]])
    d = distances(s.iot_xy, uav, cfg.uav_height)
    lavg = average_path_loss(d, cfg)
    r1 = uplink_rate(np.full_like(d, 1000.0), lavg, cfg)
    r2 = uplink_rate(np.full_like(d, 2000.0), lavg, cfg)
    np.testing.assert_allclose(r2, 2 * r1)


def test_plos_in_unit_interval():
    s = generate_scenario(100, DEFAULT)
    uav = np.array([[250.0, 250.0]])
    d = distances(s.iot_xy, uav, DEFAULT.uav_height)
    p = los_probability(d, DEFAULT.uav_height, DEFAULT)
    assert np.all(p >= 0) and np.all(p <= 1)
