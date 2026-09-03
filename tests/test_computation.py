"""Poisson arrivals and M/M/1 load."""

from __future__ import annotations

import numpy as np

from uavdt.computation import (
    arrival_rate_per_uav,
    mean_service_time_s,
    offered_load,
    queue_unstable,
    sample_interarrival_times_s,
    sample_poisson_count,
    service_rate_per_s,
)
from uavdt.config import SimConfig


def test_mu_is_f_over_L():
    cfg = SimConfig(task_cycles=1.0e6)
    np.testing.assert_allclose(service_rate_per_s(cfg), 2.0e8 / 1.0e6)
    np.testing.assert_allclose(mean_service_time_s(cfg), 1.0 / 200.0)


def test_offered_load_and_stability():
    processing = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    lam = np.array([2.0, 2.0, 2.0])
    mu = 10.0
    np.testing.assert_allclose(arrival_rate_per_uav(processing, lam), [4.0, 2.0, 0.0])
    np.testing.assert_allclose(offered_load(processing, lam, mu), [0.4, 0.2, 0.0])
    assert not np.any(queue_unstable(processing, lam, mu))
    assert np.any(queue_unstable(processing, lam, mu_per_s=3.0))


def test_poisson_count_mean():
    rng = np.random.default_rng(0)
    counts = [sample_poisson_count(2.0, 5.0, rng) for _ in range(4000)]
    np.testing.assert_allclose(np.mean(counts), 10.0, atol=0.3)


def test_exponential_interarrivals_are_positive_and_within_horizon():
    rng = np.random.default_rng(1)
    times = sample_interarrival_times_s(2.0, 10.0, rng)
    assert times.size > 0
    assert np.all(times > 0.0)
    assert times[-1] <= 10.0
    assert np.all(np.diff(times) > 0.0)
