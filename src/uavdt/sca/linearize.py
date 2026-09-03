"""First-order Taylor maps used by SCA.

Each function states the original expression and the approximation type.
None of these replace the core channel/AoDT equations for final scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.channel import link_metrics
from uavdt.config import SimConfig


def spectral_efficiency(
    iot_xyz_m: np.ndarray,
    uav_xyz_m: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """SE_ij = log2(1 + SNR_ij) = Eq. (6) at B_ij = 1 Hz.

    Calls the core channel. Not a new radio model.
    """
    i, j = iot_xyz_m.shape[0], uav_xyz_m.shape[0]
    ones = np.ones((i, j), dtype=float)
    return link_metrics(iot_xyz_m, uav_xyz_m, ones, cfg)["rates_bit_per_s"]


def se_jacobian(
    iot_xyz_m: np.ndarray,
    uav_xyz_m: np.ndarray,
    cfg: SimConfig,
    step_m: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerical Jacobian of SE_ij w.r.t. UAV (x_j, y_j).

    Original: SE_ij(q_j) from Eqs. (3)–(6).
    Approximation type: none (derivative of the true SE).
    Central differences where the field box allows; one-sided at the boundary.
    """
    se0 = spectral_efficiency(iot_xyz_m, uav_xyz_m, cfg)
    n_uav = uav_xyz_m.shape[0]
    g_x = np.zeros_like(se0)
    g_y = np.zeros_like(se0)
    limits = (cfg.area_x_m, cfg.area_y_m)
    for u in range(n_uav):
        for ax, out, limit in ((0, g_x, limits[0]), (1, g_y, limits[1])):
            x0 = float(uav_xyz_m[u, ax])
            plus = uav_xyz_m.copy()
            minus = uav_xyz_m.copy()
            plus[u, ax] = min(x0 + step_m, limit)
            minus[u, ax] = max(x0 - step_m, 0.0)
            d_plus = float(plus[u, ax] - x0)
            d_minus = float(x0 - minus[u, ax])
            if d_plus > 1e-16 and d_minus > 1e-16:
                se_p = spectral_efficiency(iot_xyz_m, plus, cfg)
                se_m = spectral_efficiency(iot_xyz_m, minus, cfg)
                out[:, u] = (se_p[:, u] - se_m[:, u]) / (plus[u, ax] - minus[u, ax])
            elif d_plus > 1e-16:
                se_p = spectral_efficiency(iot_xyz_m, plus, cfg)
                out[:, u] = (se_p[:, u] - se0[:, u]) / d_plus
            elif d_minus > 1e-16:
                se_m = spectral_efficiency(iot_xyz_m, minus, cfg)
                out[:, u] = (se0[:, u] - se_m[:, u]) / d_minus
    return se0, g_x, g_y


@dataclass
class RateProductLinearization:
    """First-order Taylor of r_ij = B_ij * SE_ij(q_j) at (B0, q0).

    Original: r_ij = B_ij log2(1 + SNR_ij(q_j))  [Eq. (6)]
    Type: first-order Taylor of the product B * SE(q).

    Affine surrogate:
        r_lin = SE0 * B + B0 * (g_x (x - x0) + g_y (y - y0))
    """

    se0: np.ndarray
    g_x: np.ndarray
    g_y: np.ndarray
    bandwidth0_hz: np.ndarray
    x0_m: np.ndarray
    y0_m: np.ndarray

    def affine_numpy(
        self,
        i: int,
        j: int,
        bandwidth_hz: float,
        x_j: float,
        y_j: float,
    ) -> float:
        se = float(self.se0[i, j])
        gx = float(self.g_x[i, j])
        gy = float(self.g_y[i, j])
        b0 = float(self.bandwidth0_hz[i, j])
        return se * bandwidth_hz + b0 * (
            gx * (x_j - float(self.x0_m[j])) + gy * (y_j - float(self.y0_m[j]))
        )

    def value_numpy(self, bandwidth_hz: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        dx = x_m - self.x0_m
        dy = y_m - self.y0_m
        return self.se0 * bandwidth_hz + self.bandwidth0_hz * (
            self.g_x * dx[None, :] + self.g_y * dy[None, :]
        )


def linearize_rate_product(
    iot_xyz_m: np.ndarray,
    uav_xyz_m: np.ndarray,
    bandwidth_hz: np.ndarray,
    cfg: SimConfig,
    step_m: float = 1e-3,
) -> RateProductLinearization:
    se0, g_x, g_y = se_jacobian(iot_xyz_m, uav_xyz_m, cfg, step_m=step_m)
    return RateProductLinearization(
        se0=se0,
        g_x=g_x,
        g_y=g_y,
        bandwidth0_hz=np.asarray(bandwidth_hz, dtype=float),
        x0_m=np.asarray(uav_xyz_m[:, 0], dtype=float),
        y0_m=np.asarray(uav_xyz_m[:, 1], dtype=float),
    )


@dataclass
class SeparationHyperplane:
    """Supporting halfspace for ||q_j - q_l|| >= θ.

    Original: ||q_j - q_l||_2 >= θ  [constraint (28)]
    Type: first-order Taylor of the convex map ||.|| (inner / conservative).
    With equal altitude:
        u · (p_j - p_l) >= θ
    where u = (p_j0 - p_l0) / ||p_j0 - p_l0|| and p = (x, y).

    This is not squared_distance >= θ^2, which would remain nonconvex.
    """

    j: int
    l: int
    u_x: float
    u_y: float
    theta_m: float
    linearization_point_j: np.ndarray
    linearization_point_l: np.ndarray

    def residual_numpy(self, uav_xyz_m: np.ndarray) -> float:
        z = uav_xyz_m[self.j, :2] - uav_xyz_m[self.l, :2]
        return float(self.u_x * z[0] + self.u_y * z[1] - self.theta_m)


def linearize_separation(
    uav_xyz_m: np.ndarray,
    theta_m: float,
) -> list[SeparationHyperplane]:
    planes: list[SeparationHyperplane] = []
    jn = uav_xyz_m.shape[0]
    for j in range(jn):
        for l in range(j + 1, jn):
            z0 = uav_xyz_m[j, :2] - uav_xyz_m[l, :2]
            nrm = float(np.linalg.norm(z0))
            if nrm < 1e-12:
                raise ValueError(
                    f"UAVs {j} and {l} coincide; cannot linearize (28)"
                )
            u = z0 / nrm
            planes.append(
                SeparationHyperplane(
                    j=j,
                    l=l,
                    u_x=float(u[0]),
                    u_y=float(u[1]),
                    theta_m=float(theta_m),
                    linearization_point_j=uav_xyz_m[j].copy(),
                    linearization_point_l=uav_xyz_m[l].copy(),
                )
            )
    return planes
