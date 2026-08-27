"""Ground-to-air channel: Eqs. (1)–(6)."""

from __future__ import annotations

import numpy as np

from src.config import SimConfig, DEFAULT


def free_space_offset(cfg: SimConfig = DEFAULT) -> float:
    """J_FS = 20 log10(f_c) + 20 log10(4 pi / c)."""
    return 20.0 * np.log10(cfg.f_c) + 20.0 * np.log10(4.0 * np.pi / cfg.c_light)


def distances(iot_xy: np.ndarray, uav_xy: np.ndarray, height: float) -> np.ndarray:
    """Euclidean distance d_ij, shape (I, J). Eq. (3)."""
    delta = iot_xy[:, None, :] - uav_xy[None, :, :]
    horiz = np.sqrt(np.sum(delta**2, axis=-1))
    return np.sqrt(horiz**2 + height**2)


def los_probability(
    dist: np.ndarray,
    height: float,
    cfg: SimConfig = DEFAULT,
) -> np.ndarray:
    """Eq. (4). Angle unit is a config option (paper does not state it)."""
    arg = np.clip(height / np.maximum(dist, 1e-12), 0.0, 1.0)
    theta = np.arcsin(arg)
    if cfg.los_angle_unit == "deg":
        theta = np.degrees(theta)
    expo = -cfg.env_b * (theta - cfg.env_a)
    expo = np.clip(expo, -60.0, 60.0)
    return 1.0 / (1.0 + cfg.env_a * np.exp(expo))


def path_loss_los_nlos(dist: np.ndarray, cfg: SimConfig = DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    jfs = free_space_offset(cfg)
    logd = 20.0 * np.log10(np.maximum(dist, 1e-12))
    los = jfs + logd + cfg.eta_los
    nlos = jfs + logd + cfg.eta_nlos
    return los, nlos


def average_path_loss(dist: np.ndarray, cfg: SimConfig = DEFAULT) -> np.ndarray:
    """Eq. (5)."""
    p_los = los_probability(dist, cfg.uav_height, cfg)
    l_los, l_nlos = path_loss_los_nlos(dist, cfg)
    return p_los * l_los + (1.0 - p_los) * l_nlos


def uplink_rate(bandwidth: np.ndarray, l_avg: np.ndarray, cfg: SimConfig = DEFAULT) -> np.ndarray:
    """Eq. (6): r = B log2(1 + p * 10^(-Lavg/10) / sigma^2)."""
    snr = cfg.p_i * (10.0 ** (-l_avg / 10.0)) / cfg.noise_power
    snr = np.maximum(snr, 0.0)
    return bandwidth * np.log2(1.0 + snr)


def link_metrics(
    iot_xy: np.ndarray,
    uav_xy: np.ndarray,
    bandwidth: np.ndarray,
    cfg: SimConfig = DEFAULT,
) -> dict[str, np.ndarray]:
    dist = distances(iot_xy, uav_xy, cfg.uav_height)
    p_los = los_probability(dist, cfg.uav_height, cfg)
    l_los, l_nlos = path_loss_los_nlos(dist, cfg)
    l_avg = p_los * l_los + (1.0 - p_los) * l_nlos
    rates = uplink_rate(bandwidth, l_avg, cfg)
    return {
        "distance": dist,
        "p_los": p_los,
        "l_los": l_los,
        "l_nlos": l_nlos,
        "l_avg": l_avg,
        "rates": rates,
    }
