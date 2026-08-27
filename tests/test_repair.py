from dataclasses import replace

import numpy as np

from src.config import DEFAULT
from src.evaluator import evaluate
from src.repair import (
    bandwidth_from_weights,
    bandwidth_pools,
    complete_solution,
    equal_bandwidth,
    link_bandwidth_cap,
    nearest_association,
    process_consistent_processing,
)
from src.scenario import generate_scenario

XY = np.array([[120.0, 130.0], [380.0, 200.0], [250.0, 400.0]])


def test_system_scope_is_one_pool_and_per_uav_is_one_per_uav():
    s = generate_scenario(100, DEFAULT)
    a = nearest_association(s.iot_xy, XY)

    system = bandwidth_pools(a, replace(DEFAULT, bandwidth_scope="system"))
    assert len(system) == 1
    assert int(system[0][0].sum()) == DEFAULT.num_iot

    per_uav = bandwidth_pools(a, replace(DEFAULT, bandwidth_scope="per_uav"))
    assert len(per_uav) == XY.shape[0]
    assert sum(int(m.sum()) for m, _ in per_uav) == DEFAULT.num_iot
    # Each UAV gets its own B_sys, so the total spectrum grows with J.
    assert sum(pool for _, pool in per_uav) == XY.shape[0] * DEFAULT.b_sys


def test_per_uav_scope_gives_each_pool_its_own_budget():
    cfg = replace(DEFAULT, bandwidth_scope="per_uav")
    s = generate_scenario(100, cfg)
    a = nearest_association(s.iot_xy, XY)
    b = equal_bandwidth(a, cfg)
    for u in range(XY.shape[0]):
        assert b[:, u].sum() <= cfg.b_sys + 1e-6


def test_equal_bandwidth_respects_the_link_cap():
    cfg = replace(DEFAULT, num_iot=10, max_bw_share=0.05)
    s = generate_scenario(100, cfg)
    a = nearest_association(s.iot_xy, XY)
    b = equal_bandwidth(a, cfg)
    cap = link_bandwidth_cap(cfg, cfg.b_sys)
    assert np.all(b <= cap + 1e-9)
    assert np.all(b[a > 0.5] > 0.0)
    assert np.all(b[a < 0.5] == 0.0)
    assert b.sum() <= cfg.b_sys + 1e-6
    # 10 * 0.05 = 0.5 of the pool: every link is at the cap, leftover is unused.
    assert np.allclose(b[a > 0.5], cap)
    assert b.sum() < cfg.b_sys - 1.0


def test_equal_bandwidth_redistributes_leftover_when_links_have_room():
    """When n * cap >= pool, the equal split must spend the whole pool."""
    cfg = replace(DEFAULT, max_bw_share=0.25)
    s = generate_scenario(100, cfg)
    a = nearest_association(s.iot_xy, XY)
    b = equal_bandwidth(a, cfg)
    cap = link_bandwidth_cap(cfg, cfg.b_sys)
    n = int((a > 0.5).sum())
    assert n * cap >= cfg.b_sys
    assert np.all(b <= cap + 1e-9)
    assert np.all(b[a < 0.5] == 0.0)
    assert np.isclose(b.sum(), cfg.b_sys)


def test_equal_bandwidth_uncapped_is_a_plain_equal_split():
    cfg = replace(DEFAULT, max_bw_share=None)
    s = generate_scenario(100, cfg)
    a = nearest_association(s.iot_xy, XY)
    b = equal_bandwidth(a, cfg)
    n = int((a > 0.5).sum())
    assert np.allclose(b[a > 0.5], cfg.b_sys / n)
    assert np.all(b[a < 0.5] == 0.0)
    assert np.isclose(b.sum(), cfg.b_sys)


def test_weights_only_spend_the_surplus_left_after_the_r_min_floors():
    s = generate_scenario(100, DEFAULT)
    cfg = s.cfg
    a = nearest_association(s.iot_xy, XY)
    proc = process_consistent_processing(s, a)

    # A request that asks for everything on one link must still fund the floors.
    weights = np.zeros_like(a)
    weights[0, a[0].argmax()] = 1.0
    b = bandwidth_from_weights(s, XY, a, weights)

    assert b.sum() <= cfg.b_sys + 1e-6
    assert np.all(b[a < 0.5] == 0.0)
    assert np.all(b <= link_bandwidth_cap(cfg, cfg.b_sys) + 1e-6)
    result = evaluate(s, XY, a, proc, b)
    assert result.qos_violations == 0
    assert result.qos_shortfall == 0.0


def test_unstructured_request_would_starve_links_without_the_floors():
    """The failure mode bandwidth_from_weights exists to prevent."""
    s = generate_scenario(100, DEFAULT)
    a = nearest_association(s.iot_xy, XY)
    proc = process_consistent_processing(s, a)
    rng = np.random.default_rng(0)

    starved = 0
    for _ in range(20):
        raw = rng.uniform(0.0, 1.0, size=a.shape) * a
        naive = raw / raw.sum() * s.cfg.b_sys
        if evaluate(s, XY, a, proc, naive).qos_violations > 0:
            starved += 1
        repaired = bandwidth_from_weights(s, XY, a, raw)
        assert evaluate(s, XY, a, proc, repaired).qos_violations == 0
    assert starved > 0


def test_placement_only_baselines_keep_the_naive_equal_split():
    s = generate_scenario(100, DEFAULT)
    _xy, a, _proc, bw = complete_solution(s, XY)
    np.testing.assert_allclose(bw, equal_bandwidth(a, s.cfg))


def test_evaluator_flags_bandwidth_over_a_pool_and_over_the_link_cap():
    s = generate_scenario(100, DEFAULT)
    cfg = s.cfg
    a = nearest_association(s.iot_xy, XY)
    proc = process_consistent_processing(s, a)

    ok = equal_bandwidth(a, cfg)
    assert evaluate(s, XY, a, proc, ok).bw_excess == 0.0

    over_pool = ok * 2.0
    assert evaluate(s, XY, a, proc, over_pool).bw_excess > 0.0

    over_cap = np.zeros_like(ok)
    i0, j0 = 0, int(a[0].argmax())
    over_cap[i0, j0] = link_bandwidth_cap(cfg, cfg.b_sys) * 1.5
    assert evaluate(s, XY, a, proc, over_cap).bw_excess > 0.0

    orphan = ok.copy()
    orphan[a < 0.5] = 1.0
    assert evaluate(s, XY, a, proc, orphan).bw_excess > 0.0


def test_bandwidth_from_weights_processes_per_uav_pools_independently():
    """A pool that cannot fund its floors must not wipe a pool that can.

    Under per_uav scope the old early-return called project_bandwidth for the
    whole matrix as soon as one UAV's floors overflowed, dropping the R_min
    floors already assigned to the other UAVs.
    """
    from src.comm import link_metrics

    cfg = replace(DEFAULT, bandwidth_scope="per_uav", max_bw_share=None, b_sys=30_000.0)
    s = generate_scenario(100, cfg)
    s.iot_xy = np.zeros((cfg.num_iot, 2))
    s.iot_xy[0] = [250.0, 250.0]
    s.iot_xy[1] = [260.0, 250.0]
    s.iot_xy[2:] = [20.0, 20.0]
    xy = np.array([[255.0, 250.0], [0.0, 0.0], [500.0, 500.0]])
    a = np.zeros((cfg.num_iot, 3))
    a[0, 0] = 1.0
    a[1, 0] = 1.0
    a[2:, 1] = 1.0

    se = np.maximum(link_metrics(s.iot_xy, xy, np.ones_like(a), cfg)["rates"], 1e-12)
    floors0 = cfg.r_min / se[a[:, 0] > 0.5, 0]
    floors1 = cfg.r_min / se[a[:, 1] > 0.5, 1]
    assert float(floors0.sum()) <= cfg.b_sys
    assert float(floors1.sum()) > cfg.b_sys

    weights = np.zeros_like(a)
    weights[0, 0] = 1.0
    b = bandwidth_from_weights(s, xy, a, weights)

    assert np.all(b[a < 0.5] == 0.0)
    assert b[:, 0].sum() <= cfg.b_sys + 1e-6
    assert b[:, 1].sum() <= cfg.b_sys + 1e-6
    # UAV 0 still funds both floors; surplus follows the weight on IoT 0.
    assert b[1, 0] >= floors0[1] - 1e-6
    assert b[0, 0] > b[1, 0]
    assert np.isclose(b[:, 0].sum(), cfg.b_sys)
    # UAV 1 is infeasible for floors but still gets a projected allocation.
    assert b[:, 1].sum() > 0.0


def test_qos_shortfall_is_continuous_where_the_count_saturates():
    s = generate_scenario(100, DEFAULT)
    cfg = s.cfg
    a = nearest_association(s.iot_xy, XY)
    proc = process_consistent_processing(s, a)
    bw = equal_bandwidth(a, cfg)

    near = evaluate(s, XY + 400.0, a, proc, bw)
    far = evaluate(s, XY + 800.0, a, proc, bw)
    assert near.qos_violations == far.qos_violations == cfg.num_iot
    assert far.qos_shortfall > near.qos_shortfall
