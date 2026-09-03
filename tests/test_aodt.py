"""AoDT Eq. (17): process-level, not a per-IoT average."""

from __future__ import annotations

import numpy as np

from uavdt.aodt import average_aodt_s, process_min_rate, queueing_term_s, upload_times_s
from uavdt.config import SimConfig
from uavdt.scenario import generate_scenario


def test_upload_local_vs_forwarded():
    cfg = SimConfig(task_size_bits=10_000.0, t_u2u_s=0.3)
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b_local = np.array([[1.0, 0.0], [1.0, 0.0]])
    b_fwd = np.array([[0.0, 1.0], [0.0, 1.0]])
    rates = np.array([[20_000.0, 1.0], [10_000.0, 1.0]])
    d_local = upload_times_s(a, b_local, rates, cfg)
    d_fwd = upload_times_s(a, b_fwd, rates, cfg)
    np.testing.assert_allclose(d_local, [0.5, 1.0])
    np.testing.assert_allclose(d_fwd, [0.8, 1.3])


def test_eq17_hand_calculation():
    # N_k = two IoTs, λ=2, D = 0.5 and 0.8, μ=100
    # Δ = 0.8 + (1/2)*(1 + 4/100) = 1.32
    lam = np.array([2.0, 2.0])
    assert process_min_rate(lam) == 2.0
    q = queueing_term_s(lam, mu_per_s=100.0)
    np.testing.assert_allclose(q, 0.5 * (1.0 + 4.0 / 100.0))
    np.testing.assert_allclose(0.8 + q, 1.32)


def test_aodt_uses_max_D_not_mean(frozen_scenario, three_uavs):
    from uavdt.evaluator import evaluate
    from uavdt.resources import allocation_from_positions

    alloc = allocation_from_positions(frozen_scenario, three_uavs)
    result = evaluate(frozen_scenario, three_uavs, alloc)
    # Per-process age must equal max member upload + shared queue term,
    # not the mean of fictional per-IoT ages.
    from uavdt.aodt import queueing_term_s, upload_times_s

    d_i = upload_times_s(
        alloc.hard_association(),
        alloc.hard_processing(),
        result.rates_bit_per_s,
        frozen_scenario.cfg,
    )
    for proc in frozen_scenario.processes:
        members = proc.iot_indices
        j_star = int(np.argmax(alloc.hard_processing()[members[0]]))
        q = queueing_term_s(
            frozen_scenario.lambdas_per_s[members],
            result.mu_per_s,
        )
        expected = float(np.max(d_i[members])) + q
        mean_fake = float(np.mean(d_i[members])) + q
        np.testing.assert_allclose(result.aodt_s[proc.process_id], expected)
        if not np.allclose(expected, mean_fake):
            assert abs(result.aodt_s[proc.process_id] - mean_fake) > 1e-12


def test_aodt_inf_when_queue_unstable():
    cfg = SimConfig(task_cycles=2.0e8)  # μ = 1 /s; 5 IoTs at λ=2 overload
    sc = generate_scenario(seed=1, cfg=cfg)
    from uavdt.evaluator import evaluate
    from uavdt.placement.random import place_random

    uav = place_random(cfg.num_uav, seed=1, cfg=cfg)
    result = evaluate(sc, uav)
    assert np.any(np.isinf(result.aodt_s)) or result.constraints.cpu_unstable_count > 0
