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


def queueing_term(scenario: Scenario, members: np.ndarray, mu: float) -> float:
    """(1/λ_Nk) * (1 + Σ λ_i / μ) for one process group. Independent of rate."""
    if members.size == 0:
        return np.inf
    lam_nk = float(np.min(scenario.lambdas[members]))
    lam_sum = float(np.sum(scenario.lambdas[members]))
    return (1.0 / max(lam_nk, 1e-12)) * (1.0 + lam_sum / mu)


def delay_rate_floors(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Minimum associated rate (bit/s) so AoDT_k ≤ T_k given the queueing term.

    From Eq. (17): AoDT_k = D_Nk + Q_k, so D_i ≤ T_k - Q_k for every member.
    Does not enter the Shannon formula; it only sets a rate floor that bandwidth
    allocation must fund. Returns 0 when the compute model is off. A huge floor
    means T_k cannot be met by bandwidth alone (negative slack).
    """
    cfg = scenario.cfg
    i = association.shape[0]
    floors = np.zeros(i)
    if cfg.task_size_bytes is None:
        return floors
    s_bits = cfg.task_size_bytes * 8.0
    i_idx = np.arange(i)
    j_assoc = association.argmax(axis=1)
    forwarded = processing[i_idx, j_assoc] < 0.5
    for members in scenario.groups:
        if members.size == 0:
            continue
        slack = cfg.aodt_threshold - queueing_term(scenario, members, mu)
        for idx in members:
            budget = slack - (cfg.t_u2u if forwarded[idx] else 0.0)
            if budget <= 1e-12:
                floors[idx] = 1e18
            else:
                floors[idx] = s_bits / budget
    return floors


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
        proc_uavs = processing[members].argmax(axis=1)
        _ = int(np.bincount(proc_uavs).argmax())  # mode; homogeneous μ in the paper
        aodt[k] = d_nk + queueing_term(scenario, members, mu)
    return aodt
