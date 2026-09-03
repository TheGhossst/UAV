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


def spectral_efficiency(
    iot_xy: np.ndarray,
    uav_xy: np.ndarray,
    cfg: SimConfig = DEFAULT,
) -> np.ndarray:
    """Eq. (6) at B_ij = 1: SE_ij = log2(1 + SNR(q_j)), shape (I, J)."""
    i, j = iot_xy.shape[0], uav_xy.shape[0]
    return link_metrics(iot_xy, uav_xy, np.ones((i, j)), cfg)["rates"]


def spectral_efficiency_grad(
    iot_xy: np.ndarray,
    uav_xy: np.ndarray,
    cfg: SimConfig = DEFAULT,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-order Taylor data for Eqs. (3)–(6): SE and ∂SE_ij/∂(x_j, y_j).

    Finite differences of the channel only. This is not a gradient of
    ``evaluate()`` (no association, bandwidth LP, or penalty).
    """
    se0 = spectral_efficiency(iot_xy, uav_xy, cfg)
    n_uav = uav_xy.shape[0]
    g_x = np.zeros_like(se0)
    g_y = np.zeros_like(se0)
    for u in range(n_uav):
        for ax, out in ((0, g_x), (1, g_y)):
            limit = cfg.area_x if ax == 0 else cfg.area_y
            x0 = uav_xy[u, ax]
            plus = uav_xy.copy()
            plus[u, ax] = np.clip(x0 + eps, 0.0, limit)
            minus = uav_xy.copy()
            minus[u, ax] = np.clip(x0 - eps, 0.0, limit)
            d_plus = plus[u, ax] - x0
            d_minus = x0 - minus[u, ax]
            if d_plus > 1e-14 and d_minus > 1e-14:
                se_p = spectral_efficiency(iot_xy, plus, cfg)
                se_m = spectral_efficiency(iot_xy, minus, cfg)
                out[:, u] = (se_p[:, u] - se_m[:, u]) / (plus[u, ax] - minus[u, ax])
            elif d_plus > 1e-14:
                se_p = spectral_efficiency(iot_xy, plus, cfg)
                out[:, u] = (se_p[:, u] - se0[:, u]) / d_plus
            elif d_minus > 1e-14:
                se_m = spectral_efficiency(iot_xy, minus, cfg)
                out[:, u] = (se0[:, u] - se_m[:, u]) / d_minus
    return se0, g_x, g_y


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
