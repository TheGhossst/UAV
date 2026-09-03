"""IoT placement and explicit process groups N_k."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig, DEFAULT
from uavdt.models import PhysicalProcess, Scenario


def process_groups(cfg: SimConfig) -> tuple[PhysicalProcess, ...]:
    """N_1 = IoTs 0..4, N_2 = IoTs 5..9 (IoT_1–5 and IoT_6–10)."""
    groups = []
    for k in range(cfg.num_processes):
        start = k * cfg.iots_per_process
        stop = start + cfg.iots_per_process
        groups.append(
            PhysicalProcess(process_id=k, iot_indices=np.arange(start, stop))
        )
    return tuple(groups)


def generate_scenario(
    seed: int,
    cfg: SimConfig = DEFAULT,
    lambdas_per_s: np.ndarray | None = None,
    iot_xyz_m: np.ndarray | None = None,
) -> Scenario:
    """Random IoTs in [0, AREA]² at z=0, unless coordinates are supplied."""
    rng = np.random.default_rng(seed)
    i = cfg.num_iot
    if iot_xyz_m is None:
        xy = rng.uniform(0.0, 1.0, size=(i, 2))
        xy[:, 0] *= cfg.area_x_m
        xy[:, 1] *= cfg.area_y_m
        z = np.zeros((i, 1), dtype=float)
        iot_xyz_m = np.hstack([xy, z])
    else:
        iot_xyz_m = np.asarray(iot_xyz_m, dtype=float)
        if iot_xyz_m.shape != (i, 3):
            raise ValueError(f"iot_xyz_m must have shape ({i}, 3)")
        if np.any(np.abs(iot_xyz_m[:, 2]) > 1e-12):
            raise ValueError("IoT altitude must be z_i = 0")

    processes = process_groups(cfg)
    process_id_of_iot = np.empty(i, dtype=int)
    for proc in processes:
        process_id_of_iot[proc.iot_indices] = proc.process_id

    if lambdas_per_s is None:
        lambdas_per_s = np.full(i, cfg.lambda_i_per_s, dtype=float)
    else:
        lambdas_per_s = np.asarray(lambdas_per_s, dtype=float)
        if lambdas_per_s.shape != (i,):
            raise ValueError("lambdas_per_s must have length num_iot")

    return Scenario(
        iot_xyz_m=iot_xyz_m,
        process_id_of_iot=process_id_of_iot,
        processes=processes,
        lambdas_per_s=lambdas_per_s,
        seed=seed,
        cfg=cfg,
    )


def make_uav_xyz_m(xy_m: np.ndarray, height_m: float) -> np.ndarray:
    """q_j = (x_j, y_j, H)."""
    xy_m = np.asarray(xy_m, dtype=float).reshape(-1, 2)
    z = np.full((xy_m.shape[0], 1), float(height_m))
    return np.hstack([xy_m, z])
