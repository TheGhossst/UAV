import numpy as np

from src.comm import average_path_loss, distances, los_probability, uplink_rate
from src.config import DEFAULT, RADIO_PROFILES, FROZEN_UAV_XY, NOISE_POWER, SIGMA, TABLE_II
from src.scenario import generate_scenario


def test_radio_profiles_disagree_only_on_the_documented_knobs():
    assert SIGMA == 0.01
    # table2 squares sigma; calibrated takes Table II's "noise power" as a power.
    assert RADIO_PROFILES["table2"].noise_power == SIGMA**2
    assert RADIO_PROFILES["calibrated"].noise_power == SIGMA
    assert TABLE_II.b_sys == 20_000.0
    assert TABLE_II.max_bw_share is None
    assert DEFAULT.noise_power == NOISE_POWER
    for field in ("f_c", "p_i", "eta_los", "eta_nlos", "env_a", "env_b", "r_min", "uav_height"):
        assert getattr(DEFAULT, field) == getattr(TABLE_II, field)


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
