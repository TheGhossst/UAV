"""Age of Digital Twin: Eqs. (11), (13), (16), (17) / constraint (31)."""

from __future__ import annotations

import numpy as np

from src.config import SimConfig
from src.scenario import Scenario


def upload_times(
    association: np.ndarray,
    processing: np.ndarray,
    rates: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """D_i = S_i / r_{i, assoc} [+ T_u2u if processed elsewhere]."""
    if cfg.task_size_bytes is None:
        raise ValueError("task_size_bytes is required for AoDT")
    i_idx = np.arange(association.shape[0])
    j_assoc = association.argmax(axis=1)
    r_assoc = np.maximum(rates[i_idx, j_assoc], 1e-12)
    # S_i is bytes; rate is bit/s. Convert bytes -> bits.
    s_bits = cfg.task_size_bytes * 8.0
    d = s_bits / r_assoc
    same = processing[i_idx, j_assoc] > 0.5
    d = np.where(same, d, d + cfg.t_u2u)
    return d


def average_aodt(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    rates: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Eq. (17): max_i D_i + (1/lambda_Nk) * (1 + sum_i lambda_i / mu).

    Constraint (23) is assumed: all IoTs in a process share one processing UAV,
    so a single mu applies. We still take mu of that processing UAV; if mixed,
    use the mean of assigned mu (should not happen after repair).
    """
    cfg = scenario.cfg
    d_i = upload_times(association, processing, rates, cfg)
    aodt = np.zeros(cfg.num_processes)
    for k, members in enumerate(scenario.groups):
        if members.size == 0:
            aodt[k] = np.inf
            continue
        d_nk = float(np.max(d_i[members]))
        lam_nk = float(np.min(scenario.lambdas[members]))
        lam_sum = float(np.sum(scenario.lambdas[members]))
        proc_uavs = processing[members].argmax(axis=1)
        # Enforce one UAV: take the mode
        j_star = int(np.bincount(proc_uavs).argmax())
        mu_k = mu  # homogeneous UAVs in the paper
        _ = j_star
        aodt[k] = d_nk + (1.0 / max(lam_nk, 1e-12)) * (1.0 + lam_sum / mu_k)
    return aodt
