"""K-means UAV placement baseline (cluster IoT (x, y); UAVs at centroids)."""

from __future__ import annotations

import time

import numpy as np

from src.evaluator import EvalResult, evaluate
from src.repair import complete_solution
from src.scenario import Scenario


def kmeans(points: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 50) -> np.ndarray:
    n = points.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if n == 0:
        return np.zeros((k, 2))
    k_use = min(k, n)
    idx = rng.choice(n, size=k_use, replace=False)
    centers = points[idx].astype(float).copy()
    if k_use < k:
        extra = rng.uniform(points.min(axis=0), points.max(axis=0), size=(k - k_use, 2))
        centers = np.vstack([centers, extra])
    for _ in range(max_iter):
        d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
        labels = d.argmin(axis=1)
        new = centers.copy()
        for j in range(centers.shape[0]):
            members = points[labels == j]
            if members.size:
                new[j] = members.mean(axis=0)
        if np.allclose(new, centers):
            break
        centers = new
    return centers


def solve_kmeans(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    *,
    equal_split: bool = True,
) -> tuple[np.ndarray, EvalResult, float]:
    """K-means UAV placement. Bandwidth is an equal split unless ``equal_split=False``."""
    j = n_uav if n_uav is not None else scenario.cfg.num_uav
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    uav_xy = kmeans(scenario.iot_xy, j, rng)
    xy, a, b, bw = complete_solution(scenario, uav_xy, equal_split=equal_split)
    result = evaluate(scenario, xy, a, b, bw)
    return xy, result, time.perf_counter() - t0
