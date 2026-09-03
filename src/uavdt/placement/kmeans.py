"""K-means UAV placement: IoT (x, y) clusters, UAVs at centroids, z = H."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig, DEFAULT
from uavdt.models import Scenario
from uavdt.scenario import make_uav_xyz_m


def _kmeans_xy(
    points_xy: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 80,
) -> np.ndarray:
    n = points_xy.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if n == 0:
        return np.zeros((k, 2), dtype=float)
    k_use = min(k, n)
    idx = rng.choice(n, size=k_use, replace=False)
    centers = points_xy[idx].astype(float).copy()
    if k_use < k:
        extra = np.column_stack(
            [
                rng.uniform(points_xy[:, 0].min(), points_xy[:, 0].max(), k - k_use),
                rng.uniform(points_xy[:, 1].min(), points_xy[:, 1].max(), k - k_use),
            ]
        )
        centers = np.vstack([centers, extra])
    for _ in range(max_iter):
        d = np.linalg.norm(points_xy[:, None, :] - centers[None, :, :], axis=-1)
        labels = d.argmin(axis=1)
        new = centers.copy()
        for j in range(centers.shape[0]):
            members = points_xy[labels == j]
            if members.size:
                new[j] = members.mean(axis=0)
        if np.allclose(new, centers):
            break
        centers = new
    return centers


def _enforce_min_separation(
    xy: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
    max_tries: int = 5_000,
) -> np.ndarray:
    """Jitter later UAVs if centroids violate θ. Not a paper algorithm."""
    out = xy.copy()
    theta = cfg.uav_min_separation_m
    for j in range(1, out.shape[0]):
        for _ in range(max_tries):
            d = np.linalg.norm(out[:j] - out[j][None, :], axis=1)
            if np.all(d >= theta):
                break
            out[j] = np.array(
                [
                    rng.uniform(0.0, cfg.area_x_m),
                    rng.uniform(0.0, cfg.area_y_m),
                ]
            )
        else:
            raise RuntimeError("k-means centroids could not be separated by θ")
    out[:, 0] = np.clip(out[:, 0], 0.0, cfg.area_x_m)
    out[:, 1] = np.clip(out[:, 1], 0.0, cfg.area_y_m)
    return out


def place_kmeans(
    scenario: Scenario,
    seed: int,
    num_uav: int | None = None,
    cfg: SimConfig | None = None,
) -> np.ndarray:
    """Paper §VII: UAV at cluster centroids of the IoT ground positions."""
    cfg = scenario.cfg if cfg is None else cfg
    j = cfg.num_uav if num_uav is None else num_uav
    rng = np.random.default_rng(seed)
    xy = _kmeans_xy(scenario.iot_xyz_m[:, :2], j, rng)
    xy = _enforce_min_separation(xy, cfg, rng)
    return make_uav_xyz_m(xy, cfg.uav_height_m)
