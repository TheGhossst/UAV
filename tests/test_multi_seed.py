"""Mean / std over seeds (paper uses 20 runs per plotted point)."""

from __future__ import annotations

from uavdt.config import SimConfig
from uavdt.experiments import run_seeds


def test_multi_seed_reports_mean_and_std():
    cfg = SimConfig()
    summary = run_seeds((1, 2, 3), cfg, placement="random")
    assert summary.seeds == (1, 2, 3)
    assert summary.sum_rate_bit_per_s.shape == (3,)
    assert summary.aodt_s.shape == (3, 2)
    assert summary.mean_sum_rate_bit_per_s > 0.0
    assert summary.std_sum_rate_bit_per_s >= 0.0
    assert 0.0 <= float(summary.feasible.mean()) <= 1.0
