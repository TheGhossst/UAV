"""Channel Eqs. (1)–(6), including σ vs σ² and 3D distance."""

from __future__ import annotations

import numpy as np

from uavdt.channel import (
    average_path_loss_db,
    distances_m,
    elevation_rad,
    free_space_offset_db,
    link_metrics,
    los_probability,
    path_loss_los_nlos_db,
    received_power_w,
    snr,
    uplink_rate_bit_per_s,
)
from uavdt.config import SimConfig
from uavdt.scenario import make_uav_xyz_m


def test_distance_under_uav_is_height():
    iot = np.array([[0.0, 0.0, 0.0]])
    uav = np.array([[0.0, 0.0, 100.0]])
    d = distances_m(iot, uav)
    np.testing.assert_allclose(d, [[100.0]])


def test_distance_is_3d_not_2d():
    iot = np.array([[30.0, 40.0, 0.0]])
    uav = np.array([[0.0, 0.0, 100.0]])
    d = distances_m(iot, uav)[0, 0]
    expected = np.sqrt(30.0**2 + 40.0**2 + 100.0**2)
    np.testing.assert_allclose(d, expected)


def test_jfs_log10():
    cfg = SimConfig()
    expected = 20.0 * np.log10(cfg.f_c_hz) + 20.0 * np.log10(4.0 * np.pi / cfg.c_light_m_per_s)
    np.testing.assert_allclose(free_space_offset_db(cfg), expected)


def test_elevation_zenith_is_pi_over_2():
    d = np.array([[100.0]])
    np.testing.assert_allclose(elevation_rad(d, 100.0), [[np.pi / 2]])


def test_los_probability_eq4_radians():
    cfg = SimConfig(los_angle_unit="rad")
    d = np.array([[100.0]])
    theta = np.arcsin(1.0)
    expected = 1.0 / (1.0 + cfg.env_a * np.exp(-cfg.env_b * (theta - cfg.env_a)))
    np.testing.assert_allclose(los_probability(d, 100.0, cfg), [[expected]])


def test_path_loss_adds_eta():
    cfg = SimConfig()
    d = np.array([[100.0]])
    l_los, l_nlos = path_loss_los_nlos_db(d, cfg)
    jfs = free_space_offset_db(cfg)
    logd = 20.0 * np.log10(100.0)
    np.testing.assert_allclose(l_los, jfs + logd + cfg.eta_los)
    np.testing.assert_allclose(l_nlos, jfs + logd + cfg.eta_nlos)


def test_average_path_loss_is_mixture():
    cfg = SimConfig()
    d = np.array([[120.0]])
    p = los_probability(d, 100.0, cfg)
    l_los, l_nlos = path_loss_los_nlos_db(d, cfg)
    expected = p * l_los + (1.0 - p) * l_nlos
    np.testing.assert_allclose(average_path_loss_db(d, 100.0, cfg), expected)


def test_noise_power_is_sigma_squared_not_sigma():
    cfg = SimConfig()
    assert cfg.sigma == 0.01
    np.testing.assert_allclose(cfg.noise_power_w, 1e-4)
    p_rx = np.array([[0.01]])
    np.testing.assert_allclose(snr(p_rx, cfg), [[0.01 / 1e-4]])


def test_received_power_and_rate_chain():
    cfg = SimConfig()
    iot = np.array([[0.0, 0.0, 0.0]])
    uav = make_uav_xyz_m([[0.0, 0.0]], cfg.uav_height_m)
    bw = np.array([[20_000.0]])
    m = link_metrics(iot, uav, bw, cfg)
    p_rx = received_power_w(m["l_avg_db"], cfg)
    np.testing.assert_allclose(m["received_power_w"], p_rx)
    np.testing.assert_allclose(m["snr"], p_rx / cfg.noise_power_w)
    expected_r = bw * np.log2(1.0 + m["snr"])
    np.testing.assert_allclose(m["rates_bit_per_s"], expected_r)
    np.testing.assert_allclose(m["sinr"], m["snr"])


def test_rate_scales_linearly_with_bandwidth():
    cfg = SimConfig()
    iot = np.array([[10.0, 10.0, 0.0]])
    uav = make_uav_xyz_m([[10.0, 10.0]], 100.0)
    m1 = link_metrics(iot, uav, np.array([[20_000.0]]), cfg)
    m2 = link_metrics(iot, uav, np.array([[2_400_000.0]]), cfg)
    np.testing.assert_allclose(m2["rates_bit_per_s"], m1["rates_bit_per_s"] * (2.4e6 / 2e4))


def test_p_los_in_unit_interval():
    cfg = SimConfig()
    iot = np.random.default_rng(0).uniform(0, 100, size=(10, 3))
    iot[:, 2] = 0.0
    uav = make_uav_xyz_m([[10.0, 20.0], [70.0, 80.0], [40.0, 40.0]], 100.0)
    d = distances_m(iot, uav)
    p = los_probability(d, 100.0, cfg)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
