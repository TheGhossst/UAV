from dataclasses import replace

import numpy as np

from src.config import DEFAULT
from src.evaluator import evaluate
from src.repair import (
    complete_solution,
    link_bandwidth_cap,
    nearest_association,
    process_consistent_processing,
)
from src.scenario import generate_scenario
from src.solvers.sca import _allocate_bandwidth, _convexified_lp, _eval_fixed, solve_sca


def test_bandwidth_residual_fills_best_links_up_to_the_cap():
    s = generate_scenario(100, DEFAULT)
    cfg = s.cfg
    xy = np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]])
    a = nearest_association(s.iot_xy, xy)
    b = _allocate_bandwidth(s, xy, a)
    assert np.isclose(b.sum(), cfg.b_sys)
    assert np.all(b[a < 0.5] == 0.0)

    from src.comm import link_metrics

    cap = link_bandwidth_cap(cfg, cfg.b_sys)
    assert np.all(b <= cap + 1e-6)

    se = link_metrics(s.iot_xy, xy, np.ones_like(b), cfg)["rates"]
    mask = a > 0.5
    need = np.zeros_like(b)
    need[mask] = cfg.r_min / np.maximum(se[mask], 1e-12)
    leftover = cfg.b_sys - float(need[mask].sum())
    assert leftover >= 0
    # Residual is poured into the highest-SE links first, each up to the cap, so
    # in descending-SE order the allocation is cap... cap, one partial, floors.
    order = np.argsort(-se[mask])
    b_a, need_a = b[mask][order], need[mask][order]
    filled = b_a >= cap - 1e-6
    at_floor = b_a <= need_a + 1e-6
    n_partial = int(np.sum(~filled & ~at_floor))
    assert n_partial <= 1
    assert np.all(np.diff(filled.astype(int)) <= 0)  # capped links form a prefix
    assert np.all(np.diff(at_floor.astype(int)) >= 0)  # floored links form a suffix


def test_bandwidth_cap_is_the_only_thing_stopping_a_single_link_vertex():
    cfg = replace(DEFAULT, max_bw_share=None)
    s = generate_scenario(100, cfg)
    xy = np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]])
    a = nearest_association(s.iot_xy, xy)
    b = _allocate_bandwidth(s, xy, a)
    from src.comm import link_metrics

    se = link_metrics(s.iot_xy, xy, np.ones_like(b), cfg)["rates"]
    mask = a > 0.5
    need = np.zeros_like(b)
    need[mask] = cfg.r_min / np.maximum(se[mask], 1e-12)
    extra = np.where(mask, b - need, 0.0)
    assert np.count_nonzero(extra > 1e-6) == 1
    best = np.unravel_index(np.argmax(np.where(mask, se, -np.inf)), se.shape)
    assert extra[best] > 0.0


def test_infeasible_floors_keep_positive_associated_bandwidth():
    cfg = replace(DEFAULT, b_sys=1.0, r_min=10_000.0)
    s = generate_scenario(100, cfg)
    xy = np.array([[50.0, 50.0], [450.0, 50.0], [250.0, 450.0]])
    a = nearest_association(s.iot_xy, xy)
    proc = process_consistent_processing(s, a)
    b = _allocate_bandwidth(s, xy, a)
    mask = a > 0.5
    assert np.isclose(b.sum(), cfg.b_sys)
    assert np.all(b[mask] > 0.0)
    assert np.all(b[~mask] == 0.0)
    result = evaluate(s, xy, a, proc, b)
    assert result.min_assoc_rate > 0.0


def test_eval_fixed_keeps_association_across_probes():
    s = generate_scenario(100, DEFAULT)
    xy, a, proc, _ = complete_solution(s, np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]]))
    a0 = a.copy()
    probes = [
        xy,
        xy + np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        xy + np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        np.clip(xy + 5.0, 0.0, s.cfg.area_x),
    ]
    for probe in probes:
        _, _, _, bw = _eval_fixed(s, probe, a, proc)
        np.testing.assert_array_equal(a, a0)
        assert np.all(bw[a0 < 0.5] == 0.0)
        assert np.all(bw[a0 > 0.5] > 0.0)


def test_sca_preserves_zero_separation_violations():
    s = generate_scenario(100, DEFAULT)
    xy0, a, proc, bw0 = complete_solution(
        s, np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]])
    )
    r0 = evaluate(s, xy0, a, proc, bw0)
    assert r0.sep_violations == 0
    xy, result, _ = solve_sca(s, seed=0, max_iter=8)
    assert result.sep_violations == 0
    for p in range(xy.shape[0]):
        for q in range(p + 1, xy.shape[0]):
            assert np.linalg.norm(xy[p] - xy[q]) >= s.cfg.uav_min_distance - 1e-6


def test_sca_runs_with_frozen_binaries():
    s = generate_scenario(100, DEFAULT)
    xy, result, _ = solve_sca(s, seed=0, max_iter=5)
    assert xy.shape == (3, 2)
    assert result.sum_rate > 0


def test_zero_trust_convex_lp_reduces_to_the_bandwidth_slice():
    """Pin q → the joint LP is exactly the linear-in-B slice of (P)."""
    s = generate_scenario(100, DEFAULT)
    xy, a, proc, _ = complete_solution(
        s, np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]])
    )
    bw0 = _allocate_bandwidth(s, xy, a, proc)
    solved = _convexified_lp(s, xy, bw0, a, proc, trust=1e-8)
    assert solved is not None
    xy1, bw1 = solved
    np.testing.assert_allclose(xy1, xy, atol=1e-6)
    np.testing.assert_allclose(bw1, bw0, rtol=1e-5, atol=1.0)
    assert np.isclose(bw1.sum(), s.cfg.b_sys, rtol=1e-5)


def test_convex_step_jointly_moves_positions_and_bandwidth():
    """Algorithm 1 step 4: one LP, both q and B. Not FD placement then a B LP."""
    s = generate_scenario(100, DEFAULT)
    xy, a, proc, _ = complete_solution(
        s, np.array([[120.0, 130.0], [380.0, 200.0], [250.0, 400.0]])
    )
    bw0 = _allocate_bandwidth(s, xy, a, proc)
    solved = _convexified_lp(s, xy, bw0, a, proc, trust=25.0)
    assert solved is not None
    xy1, bw1 = solved
    assert xy1.shape == xy.shape
    assert bw1.shape == bw0.shape
    assert np.all(bw1[a < 0.5] == 0.0)
    assert np.all(xy1[:, 0] >= -1e-9) and np.all(xy1[:, 0] <= s.cfg.area_x + 1e-9)
    assert np.all(xy1[:, 1] >= -1e-9) and np.all(xy1[:, 1] <= s.cfg.area_y + 1e-9)
    import src.solvers.sca as sca_mod

    assert not hasattr(sca_mod, "_partial_axis")
    bw_at_old = _allocate_bandwidth(s, xy, a, proc)
    np.testing.assert_allclose(bw_at_old, bw0)
    if np.linalg.norm(xy1 - xy) > 1e-3:
        bw_at_new = _allocate_bandwidth(s, xy1, a, proc)
        assert not np.allclose(bw1, bw_at_new) or not np.allclose(bw1, bw0)


def test_taylor_rate_matches_true_rate_at_expansion_point():
    from src.comm import average_path_loss, distances, spectral_efficiency_grad, uplink_rate

    s = generate_scenario(100, DEFAULT)
    xy = np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]])
    a = nearest_association(s.iot_xy, xy)
    bw = _allocate_bandwidth(s, xy, a)
    se, g_x, g_y = spectral_efficiency_grad(s.iot_xy, xy, s.cfg)
    r_lin = se * bw
    dist = distances(s.iot_xy, xy, s.cfg.uav_height)
    r_true = uplink_rate(bw, average_path_loss(dist, s.cfg), s.cfg)
    np.testing.assert_allclose(r_lin, r_true, rtol=1e-10)
    xy_shift = xy.copy()
    xy_shift[0, 0] += 1.0
    se_shift, _, _ = spectral_efficiency_grad(s.iot_xy, xy_shift, s.cfg)
    pred = se[:, 0] + g_x[:, 0] * 1.0
    np.testing.assert_allclose(pred, se_shift[:, 0], rtol=0.05, atol=1e-4)
