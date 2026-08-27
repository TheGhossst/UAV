"""M/M/1 UAV queues: Eqs. (7)–(9)."""

from __future__ import annotations

import numpy as np

from src.config import SimConfig


def service_rate(cfg: SimConfig) -> float | None:
    """mu_j = f_j / L. None if L is unspecified."""
    if cfg.task_cycles is None:
        return None
    return cfg.uav_cpu / cfg.task_cycles


def offered_load(processing: np.ndarray, lambdas: np.ndarray, mu: float) -> np.ndarray:
    """rho_j = (sum_i b_ij lambda_i) / mu. processing is (I, J) binary/float."""
    lam_j = processing.T @ lambdas  # (J,)
    return lam_j / mu


def cpu_unstable(processing: np.ndarray, lambdas: np.ndarray, mu: float, eps: float = 1e-12) -> np.ndarray:
    """True where arrival rate is not strictly below mu (constraint 24)."""
    lam_j = processing.T @ lambdas
    return lam_j >= mu - eps
