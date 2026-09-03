"""Random UAV placement with altitude H and minimum separation θ."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig, DEFAULT
from uavdt.scenario import make_uav_xyz_m


def place_random(
    num_uav: int,
    seed: int,
    cfg: SimConfig = DEFAULT,
    max_tries: int = 10_000,
) -> np.ndarray:
    """Uniform (x, y) in the field, z = H, pairwise 3D distance >= θ."""
    rng = np.random.default_rng(seed)
    xy = np.zeros((num_uav, 2), dtype=float)
    for j in range(num_uav):
        placed = False
        for _ in range(max_tries):
            cand = np.array(
                [
                    rng.uniform(0.0, cfg.area_x_m),
                    rng.uniform(0.0, cfg.area_y_m),
                ]
            )
            if j == 0:
                xy[0] = cand
                placed = True
                break
            d = np.linalg.norm(xy[:j] - cand[None, :], axis=1)
            # Horizontal distance equals 3D distance at equal altitude.
            if np.all(d >= cfg.uav_min_separation_m):
                xy[j] = cand
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"could not place UAV {j} with separation "
                f"{cfg.uav_min_separation_m} m in {cfg.area_x_m}×{cfg.area_y_m} m"
            )
    return make_uav_xyz_m(xy, cfg.uav_height_m)
