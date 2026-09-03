"""Task arrivals (Poisson) and UAV M/M/1 computation: Eqs. (7)–(9)."""

from __future__ import annotations

import numpy as np

from uavdt.config import SimConfig


def service_rate_per_s(cfg: SimConfig) -> float:
    """Eq. (7): μ_j = f_j / L."""
    return cfg.service_rate_per_s


def mean_service_time_s(cfg: SimConfig) -> float:
    """1 / μ_j."""
    return 1.0 / cfg.service_rate_per_s


def arrival_rate_per_uav(
    processing: np.ndarray,
    lambdas_per_s: np.ndarray,
) -> np.ndarray:
    """λ_total,j = Σ_i b_ij λ_i. processing is (I, J) 0/1."""
    return processing.T @ lambdas_per_s


def offered_load(
    processing: np.ndarray,
    lambdas_per_s: np.ndarray,
    mu_per_s: float,
) -> np.ndarray:
    """Eq. (9): ρ_total,j = λ_total,j / μ_j."""
    return arrival_rate_per_uav(processing, lambdas_per_s) / mu_per_s


def queue_unstable(
    processing: np.ndarray,
    lambdas_per_s: np.ndarray,
    mu_per_s: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """True where constraint (24) fails: λ_total,j >= μ_j."""
    lam_j = arrival_rate_per_uav(processing, lambdas_per_s)
    return lam_j >= mu_per_s - eps


def sample_poisson_count(
    rate_per_s: float,
    duration_s: float,
    rng: np.random.Generator,
) -> int:
    """Number of events of a Poisson process of rate λ in an interval T."""
    return int(rng.poisson(rate_per_s * duration_s))


def sample_interarrival_times_s(
    rate_per_s: float,
    duration_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exponential inter-arrivals until the clock exceeds duration_s."""
    if rate_per_s <= 0.0 or duration_s <= 0.0:
        return np.zeros(0, dtype=float)
    times = []
    t = 0.0
    mean_gap = 1.0 / rate_per_s
    while True:
        t += float(rng.exponential(mean_gap))
        if t > duration_s:
            break
        times.append(t)
    return np.asarray(times, dtype=float)
