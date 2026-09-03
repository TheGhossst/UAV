"""Phase 1: λ / T_k / f_j must change allocation and feasibility, not Shannon."""

from dataclasses import replace as dc_replace
from dataclasses import replace

import numpy as np

from src.aodt import delay_rate_floors
from src.comm import uplink_rate
from src.compute import service_rate
from src.config import AODT_THRESHOLD, DEFAULT, LAMBDA_I, UAV_CPU
from src.evaluator import constraint_activity, evaluate
from src.repair import (
    allocate_constrained_bandwidth,
    associated_rate_floors,
    complete_solution,
    equal_bandwidth,
    nearest_association,
    process_consistent_processing,
)
from src.scenario import generate_scenario
from src.solvers.sca import solve_sca
from src.solvers.td3 import UAVAoDTEnv


XY = np.array([[120.0, 130.0], [380.0, 200.0], [250.0, 400.0]])


def _scenario(seed=100, **cfg_kw):
    cfg = DEFAULT.with_compute()
    if cfg_kw:
        cfg = replace(cfg, **cfg_kw)
    return generate_scenario(seed, cfg)


def _freeze(s):
    a = nearest_association(s.iot_xy, XY)
    proc = process_consistent_processing(s, a)
    return a, proc


def test_shannon_rate_ignores_lambda_cpu_and_tk():
    cfg_a = DEFAULT.with_compute()
    cfg_b = replace(cfg_a, lambda_i=3.5, aodt_threshold=0.8, uav_cpu=1e8)
    bw = np.full((2, 2), 1e5)
    l_avg = np.full((2, 2), 80.0)
    np.testing.assert_allclose(uplink_rate(bw, l_avg, cfg_a), uplink_rate(bw, l_avg, cfg_b))


def test_no_compute_complete_solution_keeps_equal_split():
    s = generate_scenario(100, DEFAULT)
    _xy, a, _proc, bw = complete_solution(s, XY)
    np.testing.assert_allclose(bw, equal_bandwidth(a, s.cfg))


def test_compute_complete_solution_uses_constrained_lp():
    s = _scenario()
    xy, a, proc, bw = complete_solution(s, XY)
    expected = allocate_constrained_bandwidth(s, xy, a, proc)
    np.testing.assert_allclose(bw, expected)
    assert not np.allclose(bw, equal_bandwidth(a, s.cfg))


def test_tight_aodt_raises_delay_floors_above_r_min():
    s_tight = _scenario(aodt_threshold=0.8)
    s_loose = _scenario(aodt_threshold=3.0)
    a, proc = _freeze(s_tight)
    mu = service_rate(s_tight.cfg)
    tight = delay_rate_floors(s_tight, a, proc, mu)
    loose = delay_rate_floors(s_loose, a, proc, mu)
    assert np.all(tight >= s_tight.cfg.r_min - 1e-9)
    assert np.max(tight) > np.max(loose)
    assert np.max(tight) > s_tight.cfg.r_min


def test_frozen_geometry_sum_rate_rises_when_tk_slackens():
    s_tight = _scenario(aodt_threshold=0.8)
    s_loose = _scenario(aodt_threshold=3.0)
    a, proc = _freeze(s_tight)
    bw_t = allocate_constrained_bandwidth(s_tight, XY, a, proc)
    bw_l = allocate_constrained_bandwidth(s_loose, XY, a, proc)
    r_t = evaluate(s_tight, XY, a, proc, bw_t)
    r_l = evaluate(s_loose, XY, a, proc, bw_l)
    assert r_l.sum_rate > r_t.sum_rate
    # Tight T_k spends leftover on delay floors instead of the best links.
    assert float(np.max(bw_t) - np.min(bw_t[a > 0.5])) < float(np.max(bw_l) - np.min(bw_l[a > 0.5])) + 1.0


def test_frozen_geometry_lambda_changes_delay_floors():
    kw = dict(task_size_bytes=2000.0, task_cycles=2e6)
    s_lo = _scenario(lambda_i=1.0, aodt_threshold=1.2, **kw)
    s_hi = _scenario(lambda_i=3.5, aodt_threshold=1.2, **kw)
    a, proc = _freeze(s_lo)
    floors_lo = associated_rate_floors(s_lo, a, proc)
    floors_hi = associated_rate_floors(s_hi, a, proc)
    # Higher λ shrinks 1/λ_Nk, so the delay floor drops toward R_min.
    assert float(floors_lo.mean()) >= float(floors_hi.mean()) - 1e-9
    bw_lo = allocate_constrained_bandwidth(s_lo, XY, a, proc)
    bw_hi = allocate_constrained_bandwidth(s_hi, XY, a, proc)
    r_lo = evaluate(s_lo, XY, a, proc, bw_lo)
    r_hi = evaluate(s_hi, XY, a, proc, bw_hi)
    assert r_hi.sum_rate >= r_lo.sum_rate - 1.0


def test_frozen_geometry_cpu_changes_delay_floors_when_tk_binds():
    kw = dict(task_size_bytes=2000.0, task_cycles=2e6)
    s_slow = _scenario(uav_cpu=1e8, aodt_threshold=1.2, **kw)
    s_fast = _scenario(uav_cpu=2.5e8, aodt_threshold=1.2, **kw)
    a, proc = _freeze(s_slow)
    floors_slow = associated_rate_floors(s_slow, a, proc)
    floors_fast = associated_rate_floors(s_fast, a, proc)
    assert float(floors_slow.mean()) >= float(floors_fast.mean()) - 1e-9
    r_slow = evaluate(s_slow, XY, a, proc, allocate_constrained_bandwidth(s_slow, XY, a, proc))
    r_fast = evaluate(s_fast, XY, a, proc, allocate_constrained_bandwidth(s_fast, XY, a, proc))
    assert r_fast.sum_rate >= r_slow.sum_rate - 1.0


def test_paper_sweep_points_report_constraint_activity():
    """Document which constraints actually bind at the paper's sweep values."""
    a, proc = _freeze(_scenario())
    rows = []
    for key, values in (
        ("lambda_i", (1.0, 2.0, 3.5)),
        ("aodt_threshold", (0.8, 2.0, 2.8, 3.0)),
        ("uav_cpu", (1e8, 2e8, 2.5e8)),
    ):
        for v in values:
            s = _scenario(**{key: v})
            bw = allocate_constrained_bandwidth(s, XY, a, proc)
            act = constraint_activity(evaluate(s, XY, a, proc, bw))
            rows.append((key, v, act))
    # T_k = 0.8 must be the binding end of the AoDT sweep.
    tight = next(act for key, v, act in rows if key == "aodt_threshold" and v == 0.8)
    loose = next(act for key, v, act in rows if key == "aodt_threshold" and v == 3.0)
    assert tight["aodt"] >= loose["aodt"]
    assert loose["feasible"] == 1


def test_sca_reoptimizes_under_active_aodt_threshold():
    s_tight = _scenario(aodt_threshold=0.8)
    s_loose = _scenario(aodt_threshold=3.0)
    _xy_t, r_t, _ = solve_sca(s_tight, seed=0, max_iter=8)
    _xy_l, r_l, _ = solve_sca(s_loose, seed=0, max_iter=8)
    assert r_l.sum_rate > r_t.sum_rate
    assert r_l.aodt_excess <= r_t.aodt_excess + 1e-9


def test_td3_state_sees_absolute_lambda_tk_and_cpu():
    s_a = _scenario(lambda_i=1.0, aodt_threshold=0.8, uav_cpu=1e8)
    s_b = _scenario(lambda_i=3.5, aodt_threshold=3.0, uav_cpu=2.5e8)
    obs_a = UAVAoDTEnv(s_a, n_uav=3, seed=0).reset()
    obs_b = UAVAoDTEnv(s_b, n_uav=3, seed=0).reset()
    assert obs_a.shape == obs_b.shape
    assert not np.allclose(obs_a, obs_b)
    # λ_i / LAMBDA_I occupies the per-IoT λ block; last two entries are T_k, f_j.
    assert obs_a[-2] == s_a.cfg.aodt_threshold / AODT_THRESHOLD
    assert obs_b[-2] == s_b.cfg.aodt_threshold / AODT_THRESHOLD
    assert obs_a[-1] == s_a.cfg.uav_cpu / UAV_CPU
    assert obs_b[-1] == s_b.cfg.uav_cpu / UAV_CPU
    i, j = s_a.cfg.num_iot, 3
    # layout: 2j + 2i + K + 1 + j + i + i*j + 2
    lam_start = 2 * j + 2 * i + s_a.cfg.num_processes + 1 + j
    np.testing.assert_allclose(obs_a[lam_start : lam_start + i], 1.0 / LAMBDA_I)
    np.testing.assert_allclose(obs_b[lam_start : lam_start + i], 3.5 / LAMBDA_I)


def test_td3_reward_penalizes_aodt_excess_when_compute_on():
    s = _scenario(aodt_threshold=0.8)
    env = UAVAoDTEnv(s, n_uav=3, seed=0)
    env.reset()
    assert env.last_result is not None
    r_ok = env._reward(env.last_result)
    bad = dc_replace(env.last_result, aodt_excess=2.0, aodt_violations=1, feasible=False)
    r_bad = env._reward(bad)
    assert r_bad < r_ok


def test_sweep_summary_separates_feasible_and_infeasible():
    from src.experiments.sweeps import _row, _summarize
    from src.solvers.random import solve_random

    s = _scenario(aodt_threshold=0.8)
    rows = []
    for seed in (100, 101):
        sc = generate_scenario(seed, s.cfg)
        xy, result, rt = solve_random(sc, seed=seed)
        rows.append(_row(seed, "random", result, rt, xy, sc.cfg.uav_height, aodt_threshold=0.8))
    summary = _summarize(rows, "aodt_threshold")
    assert len(summary) == 1
    rec = summary[0]
    assert "feasible_n" in rec
    assert "feasible_frac" in rec
    assert "feasible_sum_rate_mean" in rec
    assert rec["n"] == 2
    assert 0.0 <= rec["feasible_frac"] <= 1.0
