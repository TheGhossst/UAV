"""IoT deployments and process groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import SimConfig, DEFAULT


@dataclass
class Scenario:
    iot_xy: np.ndarray  # (I, 2)
    process_of_iot: np.ndarray  # (I,) int in [0, K)
    groups: list[np.ndarray]  # K arrays of IoT indices
    lambdas: np.ndarray  # (I,)
    seed: int
    cfg: SimConfig


def generate_scenario(
    seed: int,
    cfg: SimConfig = DEFAULT,
    lambdas: np.ndarray | None = None,
) -> Scenario:
    rng = np.random.default_rng(seed)
    iot_xy = np.column_stack(
        [
            rng.uniform(0.0, cfg.area_x, cfg.num_iot),
            rng.uniform(0.0, cfg.area_y, cfg.num_iot),
        ]
    )
    process_of_iot = np.repeat(
        np.arange(cfg.num_processes), cfg.iots_per_process
    )[: cfg.num_iot]
    if process_of_iot.size < cfg.num_iot:
        extra = np.arange(cfg.num_iot - process_of_iot.size) % cfg.num_processes
        process_of_iot = np.concatenate([process_of_iot, extra])
    groups = [
        np.where(process_of_iot == k)[0] for k in range(cfg.num_processes)
    ]
    if lambdas is None:
        lambdas = np.full(cfg.num_iot, cfg.lambda_i, dtype=float)
    else:
        lambdas = np.asarray(lambdas, dtype=float)
        if lambdas.size != cfg.num_iot:
            raise ValueError("lambdas must have length num_iot")
    return Scenario(
        iot_xy=iot_xy,
        process_of_iot=process_of_iot,
        groups=groups,
        lambdas=lambdas,
        seed=seed,
        cfg=cfg,
    )
