"""Ground-to-air channel: paper Eqs. (1)–(6)."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig, DEFAULT


def free_space_offset_db(cfg: SimConfig = DEFAULT) -> float:
    """J_FS = 20 log10(f_c) + 20 log10(4 π / c). Paper writes `log` in dB."""
    return 20.0 * np.log10(cfg.f_c_hz) + 20.0 * np.log10(4.0 * np.pi / cfg.c_light_m_per_s)


def distances_m(iot_xyz_m: np.ndarray, uav_xyz_m: np.ndarray) -> np.ndarray:
    """Eq. (3). Shape (I, J). Uses full 3D coordinates."""
    delta = iot_xyz_m[:, None, :] - uav_xyz_m[None, :, :]
    return np.sqrt(np.sum(delta ** 2, axis=-1))


def elevation_rad(dist_m: np.ndarray, height_m: float) -> np.ndarray:
    """arcsin(H / d_ij), radians. Argument is clipped to [0, 1]."""
    arg = np.clip(height_m / np.maximum(dist_m, 1e-15), 0.0, 1.0)
    return np.arcsin(arg)


def los_probability(
    dist_m: np.ndarray,
    height_m: float,
    cfg: SimConfig = DEFAULT,
) -> np.ndarray:
    """Eq. (4). Default angle unit is radians, as written."""
    theta = elevation_rad(dist_m, height_m)
    if cfg.los_angle_unit == "deg":
        theta = np.degrees(theta)
    expo = -cfg.env_b * (theta - cfg.env_a)
    expo = np.clip(expo, -80.0, 80.0)
    return 1.0 / (1.0 + cfg.env_a * np.exp(expo))


def path_loss_los_nlos_db(
    dist_m: np.ndarray,
    cfg: SimConfig = DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Eqs. (1)–(2)."""
    jfs = free_space_offset_db(cfg)
    logd = 20.0 * np.log10(np.maximum(dist_m, 1e-15))
    l_los = jfs + logd + cfg.eta_los
    l_nlos = jfs + logd + cfg.eta_nlos
    return l_los, l_nlos


def average_path_loss_db(
    dist_m: np.ndarray,
    height_m: float,
    cfg: SimConfig = DEFAULT,
) -> np.ndarray:
    """Eq. (5)."""
    p_los = los_probability(dist_m, height_m, cfg)
    l_los, l_nlos = path_loss_los_nlos_db(dist_m, cfg)
    return p_los * l_los + (1.0 - p_los) * l_nlos


def received_power_w(l_avg_db: np.ndarray, cfg: SimConfig = DEFAULT) -> np.ndarray:
    """p_i * 10^(-L_avg/10), watts."""
    return cfg.p_i_w * (10.0 ** (-l_avg_db / 10.0))


def snr(received_w: np.ndarray, cfg: SimConfig = DEFAULT) -> np.ndarray:
    """Eq. (6) SNR: p_rx / σ². Orthogonal channels ⇒ no interference term."""
    return received_w / cfg.noise_power_w


def uplink_rate_bit_per_s(
    bandwidth_hz: np.ndarray,
    snr_lin: np.ndarray,
) -> np.ndarray:
    """Eq. (6): r = B log2(1 + SNR)."""
    return bandwidth_hz * np.log2(1.0 + np.maximum(snr_lin, 0.0))


def link_metrics(
    iot_xyz_m: np.ndarray,
    uav_xyz_m: np.ndarray,
    bandwidth_hz: np.ndarray,
    cfg: SimConfig = DEFAULT,
) -> dict[str, np.ndarray]:
    """All channel intermediates for one (IoT, UAV) geometry."""
    height_m = float(uav_xyz_m[0, 2]) if uav_xyz_m.size else cfg.uav_height_m
    dist = distances_m(iot_xyz_m, uav_xyz_m)
    elev = elevation_rad(dist, height_m)
    p_los = los_probability(dist, height_m, cfg)
    l_los, l_nlos = path_loss_los_nlos_db(dist, cfg)
    l_avg = p_los * l_los + (1.0 - p_los) * l_nlos
    p_rx = received_power_w(l_avg, cfg)
    snr_lin = snr(p_rx, cfg)
    rates = uplink_rate_bit_per_s(bandwidth_hz, snr_lin)
    return {
        "distance_m": dist,
        "elevation_rad": elev,
        "p_los": p_los,
        "l_los_db": l_los,
        "l_nlos_db": l_nlos,
        "l_avg_db": l_avg,
        "received_power_w": p_rx,
        "snr": snr_lin,
        "sinr": snr_lin,  # I = 0 under the paper's orthogonal-channel model
        "rates_bit_per_s": rates,
    }
