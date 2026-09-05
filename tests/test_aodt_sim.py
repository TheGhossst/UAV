"""Event-driven FCFS / FCFS-P / LCFS-S vs closed forms."""

from __future__ import annotations

import numpy as np

from uavdt.aodt import fcfs_mm1_aaio_s, instantaneous_age_s, lcfs_s_aaio_s
from uavdt.aodt_sim import simulate_process_queue
from uavdt.config import SimConfig
from uavdt.experiments.fig11 import lambdas_for_pattern, sensibility_checks
from uavdt.scenario import generate_scenario


def _mean_source_age(
    discipline: str,
    lam: float,
    mu: float,
    *,
    delay: float = 0.0,
    n_rep: int = 5,
    horizon: float = 180.0,
    warmup: float = 36.0,
    seed: int = 1,
    download: float = 0.0,
) -> float:
    ages = []
    for r in range(n_rep):
        res = simulate_process_queue(
            np.array([lam]),
            np.array([delay]),
            mu,
            discipline=discipline,
            horizon_s=horizon,
            warmup_s=warmup,
            download_s=download,
            seed=seed + 17 * r,
        )
        ages.append(float(res.mean_source_age_s[0]))
    return float(np.mean(ages))


def test_lcfs_s_matches_eq14_single_source():
    lam, mu = 6.0, 10.0
    sim = _mean_source_age("lcfs_s", lam, mu)
    np.testing.assert_allclose(sim, lcfs_s_aaio_s(lam, mu), rtol=0.12, atol=0.04)


def test_fcfs_matches_kaul_single_source():
    lam, mu = 6.0, 10.0
    sim = _mean_source_age("fcfs", lam, mu)
    np.testing.assert_allclose(sim, fcfs_mm1_aaio_s(lam, mu), rtol=0.12, atol=0.04)


def test_discipline_order_lcfs_s_le_fcfs_p_le_fcfs():
    lam, mu = 6.0, 10.0
    lcfs = _mean_source_age("lcfs_s", lam, mu)
    fcfs_p = _mean_source_age("fcfs_p", lam, mu)
    fcfs = _mean_source_age("fcfs", lam, mu)
    assert lcfs <= fcfs_p + 0.04
    assert fcfs_p <= fcfs + 0.04
    assert lcfs < fcfs - 0.04


def test_constant_delay_adds_to_lcfs_s_age():
    lam, mu, d = 5.0, 20.0, 0.4
    sim = _mean_source_age("lcfs_s", lam, mu, delay=d, n_rep=4, horizon=120.0, warmup=24.0)
    np.testing.assert_allclose(sim, d + lcfs_s_aaio_s(lam, mu), rtol=0.12, atol=0.05)


def test_eq12_download_adds_z():
    lam, mu, z = 5.0, 20.0, 0.3
    base = _mean_source_age("lcfs_s", lam, mu, n_rep=4, horizon=100.0, warmup=20.0)
    plus = _mean_source_age(
        "lcfs_s", lam, mu, n_rep=4, horizon=100.0, warmup=20.0, download=z
    )
    np.testing.assert_allclose(plus - base, z, rtol=0.15, atol=0.06)


def test_eq10_unit_slope_between_deliveries():
    res = simulate_process_queue(
        np.array([4.0]),
        np.array([0.0]),
        12.0,
        discipline="lcfs_s",
        horizon_s=8.0,
        warmup_s=0.0,
        seed=2,
        record_eq10=True,
    )
    assert res.eq10_t_s is not None and res.eq10_t_s.size >= 4
    t = res.eq10_t_s
    age = res.eq10_age_s[:, 0]
    # Between events that are not resets, ζ grows as Δt (Eq. 10).
    dt = np.diff(t)
    dage = np.diff(age)
    slope_ok = np.abs(dage - dt) < 1e-9
    reset = dage < -1e-9
    assert np.any(reset)
    assert np.any(slope_ok)
    np.testing.assert_allclose(age[slope_ok.nonzero()[0] + 1], instantaneous_age_s(
        t[slope_ok.nonzero()[0] + 1], t[slope_ok.nonzero()[0] + 1] - age[slope_ok.nonzero()[0] + 1]
    ))


def test_lcfs_s_drops_stale_packets_fcfs_does_not():
    lcfs = simulate_process_queue(
        np.array([8.0]), np.array([0.0]), 5.0,
        discipline="lcfs_s", horizon_s=40.0, warmup_s=5.0, seed=3,
    )
    fcfs = simulate_process_queue(
        np.array([8.0]), np.array([0.0]), 5.0,
        discipline="fcfs", horizon_s=40.0, warmup_s=5.0, seed=3,
    )
    assert lcfs.n_dropped > 0
    assert fcfs.n_dropped == 0
    assert fcfs.mean_source_age_s[0] > lcfs.mean_source_age_s[0]


def test_fig11_lambda_patterns_match_paper_caption():
    cfg = SimConfig()
    fast = lambdas_for_pattern(cfg, "uniform_fast")
    slow = lambdas_for_pattern(cfg, "uniform_slow")
    hetero = lambdas_for_pattern(cfg, "heterogeneous")
    np.testing.assert_allclose(fast[:5], 2.0)
    np.testing.assert_allclose(fast[5:], 3.0)
    np.testing.assert_allclose(slow[:5], 0.8)
    np.testing.assert_allclose(slow[5:], 1.0)
    assert hetero.min() == 0.8
    assert hetero.max() == 3.0
    assert process_min_rate_alias(hetero[:5]) == 0.8


def process_min_rate_alias(x):
    from uavdt.aodt import process_min_rate

    return process_min_rate(x)


def test_fig11_eq17_slowest_limited_on_one_seed():
    from uavdt.aodt import average_aodt_s
    from uavdt.evaluator import evaluate
    from uavdt.placement.kmeans import place_kmeans

    cfg = SimConfig()
    rows = {}
    for pattern in ("uniform_fast", "uniform_slow", "heterogeneous"):
        sc = generate_scenario(1, cfg, lambdas_per_s=lambdas_for_pattern(cfg, pattern))
        uav = place_kmeans(sc, 1)
        ev = evaluate(sc, uav)
        rows[pattern] = float(np.max(ev.aodt_s))
        a = ev.extras["association"]
        b = ev.extras["processing"]
        mu_vec = np.full(uav.shape[0], ev.mu_per_s)
        np.testing.assert_allclose(
            ev.aodt_s,
            average_aodt_s(sc, a, b, ev.rates_bit_per_s, mu_vec),
        )
    assert rows["uniform_fast"] < rows["heterogeneous"]
    assert rows["uniform_fast"] < rows["uniform_slow"]
    assert abs(rows["heterogeneous"] - rows["uniform_slow"]) < abs(
        rows["heterogeneous"] - rows["uniform_fast"]
    )


def test_fig11_smoke_one_seed():
    from uavdt.experiments.fig11 import run_fig11, sensibility_checks

    payload = run_fig11(
        SimConfig(b_sys_hz=8_800_000.0, max_bw_share=0.25),
        n_runs=1,
        uav_counts=(3,),
        horizon_s=20.0,
        warmup_s=4.0,
    )
    bp = payload["points"][0]["by_pattern"]
    assert set(bp) == {"uniform_fast", "uniform_slow", "heterogeneous"}
    report = sensibility_checks(payload)
    assert report["all_ok"], [c for c in report["checks"] if not c["ok"]]


def test_fig11_sensibility_helper_on_tiny_payload():
    from uavdt.experiments.fig11 import sensibility_checks

    payload = {
        "points": [
            {
                "num_uav": 3,
                "by_pattern": {
                    "uniform_fast": {
                        "mean_eq17_max_s": 0.55,
                        "mean_eq15_max_s": 0.53,
                        "mean_fcfs_closed_max_s": 0.56,
                        "sim_mean_max_process_age_s": {
                            "lcfs_s": 0.9,
                            "fcfs_p": 0.85,
                            "fcfs": 0.8,
                        },
                        "sim_mean_source_age_s": {
                            "lcfs_s": 0.60,
                            "fcfs_p": 0.61,
                            "fcfs": 0.62,
                        },
                    },
                    "uniform_slow": {
                        "mean_eq17_max_s": 1.30,
                        "sim_mean_max_process_age_s": {"fcfs": 2.5},
                        "sim_mean_source_age_s": {"fcfs": 1.20},
                    },
                    "heterogeneous": {
                        "mean_eq17_max_s": 1.32,
                        "sim_mean_max_process_age_s": {"fcfs": 1.6},
                        "sim_mean_source_age_s": {"fcfs": 0.85},
                    },
                },
            }
        ]
    }
    report = sensibility_checks(payload)
    assert report["all_ok"]
