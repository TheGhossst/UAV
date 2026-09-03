"""Age of Digital Twin: paper Eqs. (11), (13), (16), (17)."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig
from uavdt.models import Scenario


def upload_times_s(
    association: np.ndarray,
    processing: np.ndarray,
    rates_bit_per_s: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Eq. (11): D_i = S_i / r_{i,assoc} [+ T_u2u if forwarded].

    `task_size_bits` is used so that D has units of seconds when rate is
    bit/s. The paper labels S_i as bytes; that conversion is not applied
    here unless the caller already stored bits.
    """
    i_idx = np.arange(association.shape[0])
    j_assoc = np.argmax(association, axis=1)
    r_assoc = np.maximum(rates_bit_per_s[i_idx, j_assoc], 1e-30)
    d = cfg.task_size_bits / r_assoc
    processed_locally = processing[i_idx, j_assoc] > 0.5
    return np.where(processed_locally, d, d + cfg.t_u2u_s)


def process_min_rate(lambdas_per_s: np.ndarray) -> float:
    """Eq. (13): λ_{N_k} = min_{i in N_k} λ_i."""
    if lambdas_per_s.size == 0:
        return np.inf
    return float(np.min(lambdas_per_s))


def queueing_term_s(lambdas_per_s: np.ndarray, mu_per_s: float) -> float:
    """(1/λ_{N_k}) * (1 + Σ λ_i / μ) from Eq. (17)."""
    if lambdas_per_s.size == 0 or not np.isfinite(mu_per_s) or mu_per_s <= 0.0:
        return np.inf
    lam_nk = process_min_rate(lambdas_per_s)
    lam_sum = float(np.sum(lambdas_per_s))
    return (1.0 / lam_nk) * (1.0 + lam_sum / mu_per_s)


def average_aodt_s(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    rates_bit_per_s: np.ndarray,
    mu_per_uav: np.ndarray,
    uav_stable: np.ndarray | None = None,
) -> np.ndarray:
    """Eq. (17) per process. Not a mean of per-IoT ages.

    μ is the service rate of the unique processing UAV of the process
    (constraint (23)). If that UAV is unstable, AoDT is +∞.
    """
    cfg = scenario.cfg
    d_i = upload_times_s(association, processing, rates_bit_per_s, cfg)
    aodt = np.zeros(cfg.num_processes, dtype=float)
    for proc in scenario.processes:
        members = proc.iot_indices
        k = proc.process_id
        if members.size == 0:
            aodt[k] = np.inf
            continue
        proc_uavs = np.argmax(processing[members], axis=1)
        j_star = int(proc_uavs[0])
        if uav_stable is not None and not bool(uav_stable[j_star]):
            aodt[k] = np.inf
            continue
        d_nk = float(np.max(d_i[members]))
        mu = float(mu_per_uav[j_star])
        aodt[k] = d_nk + queueing_term_s(scenario.lambdas_per_s[members], mu)
    return aodt
