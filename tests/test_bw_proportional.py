"""Bandwidth-proportional inner allocation. Frozen SCA is not involved."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.analysis.bw_proportional import (
    allocate_proportional,
    calibrated_noise_density_w_per_hz,
    constant_noise_snr,
    equal_share_reference_hz,
    proportional_rate_bit_per_s,
    proportional_snr,
    solve_bandwidth_proportional,
)
from uavdt.config import headline_sim_config
from uavdt.resources import cpu_stable_processing, equal_share_bandwidth_hz, nearest_association
from uavdt.scenario import generate_scenario, make_uav_xyz_m

cvxpy = pytest.importorskip("cvxpy")


def test_n0_matches_sigma2_at_equal_share():
    cfg = headline_sim_config()
    b_eq = equal_share_reference_hz(cfg)
    n0 = calibrated_noise_density_w_per_hz(cfg)
    assert b_eq == pytest.approx(cfg.b_sys_hz / cfg.num_iot)
    assert n0 * b_eq == pytest.approx(cfg.noise_power_w)

    sc = generate_scenario(1, cfg)
    uav = make_uav_xyz_m(sc.iot_xyz_m[: cfg.num_uav, :2], cfg.uav_height_m)
    snr = constant_noise_snr(sc.iot_xyz_m, uav, cfg)
    matched = proportional_snr(np.full_like(snr, b_eq), snr, b_eq)
    assert matched == pytest.approx(snr)

    rates = proportional_rate_bit_per_s(np.full_like(snr, b_eq), snr, b_eq)
    assert rates == pytest.approx(b_eq * np.log2(1.0 + snr))


def test_solver_spreads_bandwidth_and_matches_grid():
    """Two unequal links, no cap bind: both get Hertz, stronger gets more.

    The frozen linear objective would park almost all leftover Hertz on
    the stronger link. The concave rate does not.
    """
    snr = np.array([1.0, 0.4])
    floors = np.array([1.0e4, 1.0e4])
    b_eq = 1.0e6
    b_sys = 2.0e6
    cap = 2.0e6
    solved = allocate_proportional(
        snr,
        floors,
        b_eq_hz=b_eq,
        b_sys_hz=b_sys,
        cap_hz=cap,
    )
    assert not solved.infeasible
    bw = solved.bandwidth_hz
    assert bw.sum() == pytest.approx(b_sys, rel=1e-4)
    assert bw[0] > bw[1]
    assert bw[1] > 0.15 * b_sys

    # Brute force on the sum-cap line. Floors are far below either share.
    best = -1.0
    for share in np.linspace(0.05, 0.95, 181):
        trial = np.array([share * b_sys, (1.0 - share) * b_sys])
        best = max(best, float(proportional_rate_bit_per_s(trial, snr, b_eq).sum()))
    assert solved.objective_bit_per_s == pytest.approx(best, rel=1e-3)


def test_headline_geometry_beats_equal_share():
    cfg = headline_sim_config()
    sc = generate_scenario(2, cfg)
    uav = make_uav_xyz_m(sc.iot_xyz_m[: cfg.num_uav, :2], cfg.uav_height_m)
    a = nearest_association(sc.iot_xyz_m, uav)
    b = cpu_stable_processing(sc, a)
    solved = solve_bandwidth_proportional(sc, uav, a, b)
    assert not solved.infeasible
    assert solved.n0_w_per_hz * solved.b_eq_hz == pytest.approx(cfg.noise_power_w)
    assert solved.bandwidth_hz.sum() == pytest.approx(cfg.b_sys_hz, rel=1e-4)
    assert np.all(solved.bandwidth_hz <= cfg.link_bandwidth_cap_hz + 1.0)

    b_eq_bw = equal_share_bandwidth_hz(a, cfg)
    snr = constant_noise_snr(sc.iot_xyz_m, uav, cfg)
    equal_rate = float(
        (a * proportional_rate_bit_per_s(b_eq_bw, snr, solved.b_eq_hz)).sum()
    )
    assert solved.objective_bit_per_s + 1.0 >= equal_rate


def test_larger_b_eq_raises_the_optimal_rate():
    cfg = headline_sim_config()
    sc = generate_scenario(4, cfg)
    uav = make_uav_xyz_m(sc.iot_xyz_m[: cfg.num_uav, :2], cfg.uav_height_m)
    a = nearest_association(sc.iot_xyz_m, uav)
    b = cpu_stable_processing(sc, a)
    b_eq = equal_share_reference_hz(cfg)
    base = solve_bandwidth_proportional(sc, uav, a, b, b_eq_hz=b_eq)
    wider = solve_bandwidth_proportional(sc, uav, a, b, b_eq_hz=2.0 * b_eq)
    assert not base.infeasible and not wider.infeasible
    assert wider.n0_w_per_hz * wider.b_eq_hz == pytest.approx(cfg.noise_power_w)
    assert wider.objective_bit_per_s >= base.objective_bit_per_s - 1.0
