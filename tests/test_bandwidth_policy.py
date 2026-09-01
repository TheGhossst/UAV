from dataclasses import replace

import numpy as np

from src.comm import link_metrics
from src.config import DEFAULT
from src.evaluator import evaluate
from src.repair import (
    allocate_constrained_bandwidth,
    complete_solution,
    equal_bandwidth,
    nearest_association,
)
from src.scenario import generate_scenario
from src.solvers.kmeans import kmeans, solve_kmeans
from src.solvers.random import solve_random


XY = np.array([[120.0, 130.0], [380.0, 200.0], [250.0, 400.0]])


def test_complete_solution_default_still_uses_lp_when_compute_on():
    s = generate_scenario(100, DEFAULT.with_compute())
    xy, a, proc, bw = complete_solution(s, XY)
    expected = allocate_constrained_bandwidth(s, xy, a, proc)
    np.testing.assert_allclose(bw, expected)
    assert not np.allclose(bw, equal_bandwidth(a, s.cfg))


def test_kmeans_and_random_equal_split_even_with_compute():
    s = generate_scenario(100, DEFAULT.with_compute())
    rng = np.random.default_rng(100)
    uav_xy = kmeans(s.iot_xy, s.cfg.num_uav, rng)
    xy, a, proc, bw_eq = complete_solution(s, uav_xy, equal_split=True)
    _xy, _a, _p, bw_lp = complete_solution(s, uav_xy, equal_split=False)
    np.testing.assert_allclose(bw_eq, equal_bandwidth(a, s.cfg))
    assert not np.allclose(bw_eq, bw_lp)

    xy_k, res_k, _ = solve_kmeans(s, seed=100)
    np.testing.assert_allclose(xy_k, xy)
    assert np.isclose(res_k.sum_rate, evaluate(s, xy, a, proc, bw_eq).sum_rate)

    rng_r = np.random.default_rng(100)
    drawn = np.column_stack(
        [
            rng_r.uniform(0.0, s.cfg.area_x, s.cfg.num_uav),
            rng_r.uniform(0.0, s.cfg.area_y, s.cfg.num_uav),
        ]
    )
    xy_d, a_d, proc_d, bw_d = complete_solution(s, drawn, equal_split=True)
    xy_r, res_r, _ = solve_random(s, seed=100)
    np.testing.assert_allclose(xy_r, xy_d)
    np.testing.assert_allclose(bw_d, equal_bandwidth(a_d, s.cfg))
    assert np.isclose(res_r.sum_rate, evaluate(s, xy_d, a_d, proc_d, bw_d).sum_rate)


def test_aodt_surplus_helps_the_worst_device_before_best_se():
    cfg = replace(
        DEFAULT.with_compute(),
        num_iot=2,
        num_uav=1,
        num_processes=1,
        iots_per_process=2,
        aodt_threshold=3.0,
        max_bw_share=None,
    )
    s = generate_scenario(0, cfg)
    s.iot_xy = np.array([[250.0, 250.0], [20.0, 20.0]])
    xy = np.array([[250.0, 250.0]])
    a = np.ones((2, 1))
    proc = np.ones((2, 1))
    se = link_metrics(s.iot_xy, xy, np.ones((2, 1)), cfg)["rates"]
    assert se[0, 0] > se[1, 0]

    bw_se = allocate_constrained_bandwidth(s, xy, a, proc, aodt_first=False)
    bw_af = allocate_constrained_bandwidth(s, xy, a, proc, aodt_first=True)
    assert bw_se[0, 0] > bw_se[1, 0]
    assert bw_af[1, 0] > bw_se[1, 0]

    r_se = evaluate(s, xy, a, proc, bw_se)
    r_af = evaluate(s, xy, a, proc, bw_af)
    assert float(np.max(r_af.aodt)) <= float(np.max(r_se.aodt)) + 1e-9


def test_dropping_the_link_cap_concentrates_sca_leftover():
    s_cap = generate_scenario(100, DEFAULT)
    s_free = generate_scenario(100, replace(DEFAULT, max_bw_share=None))
    a = nearest_association(s_cap.iot_xy, XY)
    from src.solvers.sca import _allocate_bandwidth

    b_cap = _allocate_bandwidth(s_cap, XY, a)
    b_free = _allocate_bandwidth(s_free, XY, a)
    cap = s_cap.cfg.max_bw_share * s_cap.cfg.b_sys
    assert int(np.sum(b_cap[a > 0.5] >= cap - 1e-6)) >= 2
    extra = b_free - (s_free.cfg.r_min / np.maximum(
        link_metrics(s_free.iot_xy, XY, np.ones_like(b_free), s_free.cfg)["rates"], 1e-12
    ))
    extra = np.where(a > 0.5, extra, 0.0)
    assert int(np.count_nonzero(extra > 1e-6)) == 1
