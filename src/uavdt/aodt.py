"""Age of Digital Twin: paper Eqs. (10)–(17) and FCFS comparison.

Problem (P) and `evaluate()` use **Eq. (17)** as the process AoDT.
Eqs. (14)–(15) are the LCFS-S intermediate forms. FCFS / FCFS-P are
the queueing alternatives the paper discusses but does not use in (P).
Eq. (12) download is neglected in the paper (Z = 0); the term is kept
so a non-zero download can be added without changing the default.
"""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig
from uavdt.models import Scenario


def instantaneous_age_s(
    t_s: np.ndarray | float,
    last_update_gen_s: np.ndarray | float,
) -> np.ndarray | float:
    """Eq. (10): ζ_i(t) = t − u_i(t).

    `last_update_gen_s` is the generation time of the last update from
    that source that has already been received at the destination.
    """
    return np.asarray(t_s, dtype=float) - np.asarray(last_update_gen_s, dtype=float)


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


def download_time_s(cfg: SimConfig) -> float:
    """Eq. (12): Z_l. Paper neglects UAV→BS download; default is 0."""
    return float(cfg.download_time_s)


def process_min_rate(lambdas_per_s: np.ndarray) -> float:
    """Eq. (13): λ_{N_k} = min_{i in N_k} λ_i."""
    if lambdas_per_s.size == 0:
        return np.inf
    return float(np.min(lambdas_per_s))


def process_max_upload_s(upload_s: np.ndarray) -> float:
    """Eq. (16): D_{N_k} = max_{i in N_k} D_i."""
    if upload_s.size == 0:
        return np.inf
    return float(np.max(upload_s))


def lcfs_s_aaio_s(lam_per_s: float, mu_per_s: float) -> float:
    """Eq. (14): LCFS-S / M/M/1/1 replacement AAoI (1/λ)(1 + λ/μ)."""
    if lam_per_s <= 0.0 or not np.isfinite(mu_per_s) or mu_per_s <= 0.0:
        return np.inf
    return (1.0 / lam_per_s) * (1.0 + lam_per_s / mu_per_s)


def fcfs_mm1_aaio_s(lam_per_s: float, mu_per_s: float) -> float:
    """M/M/1 FCFS average AoI (Kaul–Yates–Gruteser).

    Δ = 1/λ + 1/μ + λ / (μ(μ − λ)). Infinite if ρ ≥ 1.
    The paper prefers LCFS-S because this FCFS age is larger.
    """
    if lam_per_s <= 0.0 or not np.isfinite(mu_per_s) or mu_per_s <= 0.0:
        return np.inf
    if lam_per_s >= mu_per_s - 1e-15:
        return np.inf
    return (
        1.0 / lam_per_s
        + 1.0 / mu_per_s
        + lam_per_s / (mu_per_s * (mu_per_s - lam_per_s))
    )


def queueing_term_eq15_s(lambdas_per_s: np.ndarray, mu_per_s: float) -> float:
    """Queue term of Eq. (15): (1/λ_{N_k})(1 + λ_{N_k}/μ)."""
    if lambdas_per_s.size == 0 or not np.isfinite(mu_per_s) or mu_per_s <= 0.0:
        return np.inf
    lam_nk = process_min_rate(lambdas_per_s)
    return lcfs_s_aaio_s(lam_nk, mu_per_s)


def queueing_term_s(lambdas_per_s: np.ndarray, mu_per_s: float) -> float:
    """(1/λ_{N_k}) * (1 + Σ λ_i / μ) from Eq. (17).

    This is the Problem (P) queueing term. It is not the FCFS sojourn
    and it is not Eq. (15) when |N_k| > 1.
    """
    if lambdas_per_s.size == 0 or not np.isfinite(mu_per_s) or mu_per_s <= 0.0:
        return np.inf
    lam_nk = process_min_rate(lambdas_per_s)
    lam_sum = float(np.sum(lambdas_per_s))
    return (1.0 / lam_nk) * (1.0 + lam_sum / mu_per_s)


def _process_delay_parts(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    rates_bit_per_s: np.ndarray,
    mu_per_uav: np.ndarray,
    uav_stable: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-process D_Nk, μ, and stability mask."""
    cfg = scenario.cfg
    d_i = upload_times_s(association, processing, rates_bit_per_s, cfg)
    n_k = cfg.num_processes
    d_nk = np.full(n_k, np.inf, dtype=float)
    mu_k = np.full(n_k, np.nan, dtype=float)
    stable = np.ones(n_k, dtype=bool)
    for proc in scenario.processes:
        members = proc.iot_indices
        k = proc.process_id
        if members.size == 0:
            stable[k] = False
            continue
        proc_uavs = np.argmax(processing[members], axis=1)
        j_star = int(proc_uavs[0])
        if uav_stable is not None and not bool(uav_stable[j_star]):
            stable[k] = False
            continue
        d_nk[k] = process_max_upload_s(d_i[members])
        mu_k[k] = float(mu_per_uav[j_star])
    return d_nk, mu_k, stable, d_i


def average_aodt_eq15_s(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    rates_bit_per_s: np.ndarray,
    mu_per_uav: np.ndarray,
    uav_stable: np.ndarray | None = None,
) -> np.ndarray:
    """Eq. (15): Δ_k = D_{N_k} + (1/λ_{N_k})(1 + λ_{N_k}/μ) [+ Z]."""
    d_nk, mu_k, stable, _ = _process_delay_parts(
        scenario, association, processing, rates_bit_per_s, mu_per_uav, uav_stable
    )
    z = download_time_s(scenario.cfg)
    out = np.full(scenario.cfg.num_processes, np.inf, dtype=float)
    for proc in scenario.processes:
        k = proc.process_id
        if not stable[k]:
            continue
        members = proc.iot_indices
        out[k] = d_nk[k] + queueing_term_eq15_s(
            scenario.lambdas_per_s[members], float(mu_k[k])
        ) + z
    return out


def average_aodt_fcfs_closed_s(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    rates_bit_per_s: np.ndarray,
    mu_per_uav: np.ndarray,
    uav_stable: np.ndarray | None = None,
) -> np.ndarray:
    """FCFS closed form using the paper's slowest-source reduction.

    Δ_k^{FCFS} = D_{N_k} + Δ_{M/M/1 FCFS}(λ_{N_k}, μ) [+ Z].
    Load from other members of N_k is not in this single-source formula;
    the event simulator is the multi-source FCFS ground truth.
    """
    d_nk, mu_k, stable, _ = _process_delay_parts(
        scenario, association, processing, rates_bit_per_s, mu_per_uav, uav_stable
    )
    z = download_time_s(scenario.cfg)
    out = np.full(scenario.cfg.num_processes, np.inf, dtype=float)
    for proc in scenario.processes:
        k = proc.process_id
        if not stable[k]:
            continue
        lam_nk = process_min_rate(scenario.lambdas_per_s[proc.iot_indices])
        out[k] = d_nk[k] + fcfs_mm1_aaio_s(lam_nk, float(mu_k[k])) + z
    return out


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

    Eq. (12) download Z is added; the paper sets Z = 0.
    """
    d_nk, mu_k, stable, _ = _process_delay_parts(
        scenario, association, processing, rates_bit_per_s, mu_per_uav, uav_stable
    )
    z = download_time_s(scenario.cfg)
    aodt = np.full(scenario.cfg.num_processes, np.inf, dtype=float)
    for proc in scenario.processes:
        k = proc.process_id
        if not stable[k]:
            continue
        aodt[k] = (
            d_nk[k]
            + queueing_term_s(scenario.lambdas_per_s[proc.iot_indices], float(mu_k[k]))
            + z
        )
    return aodt
