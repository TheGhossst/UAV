"""Eq. (6)+(27) ceiling and 20 kHz at paper area."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.channel import sum_rate_ceiling_bit_per_s
from uavdt.config import PAPER_B_SYS_HZ
from uavdt.experiments.bsys_check import (
    GENEROUS_SNR_MAX,
    model_free_ceilings_bps,
    reading_b_magnitude,
    run_area_control,
)

cvxpy = pytest.importorskip("cvxpy")


def test_generous_ceiling_is_about_one_mbps_not_seven():
    payload = model_free_ceilings_bps()
    cap = payload["generous_ceiling_Mbps"]
    assert cap == pytest.approx(
        sum_rate_ceiling_bit_per_s(PAPER_B_SYS_HZ, GENEROUS_SNR_MAX) / 1.0e6
    )
    assert cap < 1.01
    assert cap > 0.99
    assert payload["area_independent"]
    assert payload["channel_constants_independent"]


def test_20khz_zero_feasible_at_paper_500m_field():
    row = run_area_control(area_m=500.0, n_runs=1, methods=("kmeans",))
    stats = row["by_method"]["kmeans"]
    assert stats["feasible_fraction"] == 0.0
    assert stats["mean_sum_rate_Mbps"] < 1.0
    assert row["model_snr_ceiling_Mbps"] < 1.0
    assert row["observed_max_snr"] < GENEROUS_SNR_MAX


def test_20khz_zero_feasible_at_100m_field():
    row = run_area_control(area_m=100.0, n_runs=1, methods=("kmeans",))
    assert row["by_method"]["kmeans"]["feasible_fraction"] == 0.0
    np.testing.assert_allclose(row["area_m"], [100.0, 100.0])
    realized = row["by_method"]["kmeans"]["mean_sum_rate_Mbps"]
    np.testing.assert_allclose(realized, row["equal_share_pred_mean_Mbps"], rtol=0.15)
    assert row["best_link_dump_mean_Mbps"] >= realized - 1e-9


def test_reading_b_written_channel_is_tens_of_times_20khz():
    mag = reading_b_magnitude(1.03)
    fig6 = next(r for r in mag["anchors"] if r["fig"] == "6")
    assert fig6["vs_stated_20kHz"] == pytest.approx(fig6["per_link_hz"] / 20_000.0)
    assert 30.0 < fig6["vs_stated_20kHz"] < 50.0
    generous = reading_b_magnitude(GENEROUS_SNR_MAX)
    fig6g = next(r for r in generous["anchors"] if r["fig"] == "6")
    assert fig6g["vs_stated_20kHz"] < 1.0
