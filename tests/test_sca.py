"""Sequential SCA-style solver tests. Uses the shared core evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uavdt.config import DEFAULT, SimConfig
from uavdt.evaluator import evaluate
from uavdt.models import Allocation
from uavdt.sca import SCASettings, solve_sca
from uavdt.sca.algorithm import true_gate_ok
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
from uavdt.sca.debug import run_sca_seq_debug
from uavdt.sca.initialize import initialize_sca
from uavdt.scenario import generate_scenario, make_uav_xyz_m

cvxpy = pytest.importorskip("cvxpy")


def test_fixed_q_bandwidth_is_dcp_and_meets_true_qos():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(seed=1, cfg=cfg)
    uav, alloc = initialize_sca(sc, 1)
    a = alloc.hard_association()
    b = alloc.hard_processing()
    res = solve_bandwidth_at_fixed_q(sc, uav, a, b)
    assert not res.infeasible
    ev = evaluate(sc, uav, Allocation(a, b, res.bandwidth_hz))
    assert true_gate_ok(ev)
    np.testing.assert_allclose(ev.sum_rate_bit_per_s, (a * ev.rates_bit_per_s).sum())


def test_sca_tiny_instance_moves_toward_the_iot_cluster():
    cfg = SimConfig(
        num_iot=2,
        num_processes=1,
        iots_per_process=2,
        num_uav=1,
        b_sys_hz=2.4e6,
    )
    iot = np.array([[50.0, 50.0, 0.0], [50.0, 50.0, 0.0]])
    sc = generate_scenario(seed=0, cfg=cfg, iot_xyz_m=iot)
    uav0 = make_uav_xyz_m([[10.0, 10.0]], cfg.uav_height_m)
    assoc = np.array([[1.0], [1.0]])
    proc = assoc.copy()
    bw = np.array([[cfg.b_sys_hz * 0.5], [cfg.b_sys_hz * 0.5]])
    alloc0 = Allocation(assoc, proc, bw)
    result = solve_sca(
        sc,
        seed=0,
        settings=SCASettings(max_iterations=15, epsilon=1e-3, step_size_m=2.0),
        uav_xyz_m=uav0,
        allocation=alloc0,
    )
    xy = result.uav_xyz_m[0, :2]
    assert np.linalg.norm(xy - np.array([50.0, 50.0])) < np.linalg.norm(
        uav0[0, :2] - np.array([50.0, 50.0])
    )
    assert result.true_objective > result.history[0].true_objective
    assert result.diagnostics.get("accepted_steps", 0) >= 1
    np.testing.assert_allclose(result.true_objective, result.true_eval.sum_rate_bit_per_s)
    assert true_gate_ok(result.true_eval)
    for row in result.history:
        if row.accepted:
            assert row.qos_violations == 0
            assert row.aodt_violations == 0


def test_initialize_sca_i32_does_not_fail_cpu_stability():
    from uavdt.computation import queue_unstable
    from uavdt.experiments.grids import config_for_counts

    cfg = config_for_counts(32, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25))
    sc = generate_scenario(seed=1, cfg=cfg)
    uav, alloc = initialize_sca(sc, 1)
    b = alloc.hard_processing()
    assigned = [int(np.argmax(b[p.iot_indices][0])) for p in sc.processes]
    assert assigned[0] != assigned[1]
    assert not np.any(queue_unstable(b, sc.lambdas_per_s, sc.cfg.service_rate_per_s))


def test_tk08_bandwidth_lp_feasible_with_process_cohesive_association():
    from dataclasses import replace

    from uavdt.experiments.grids import config_for_counts
    from uavdt.models import Allocation
    from uavdt.placement.kmeans import place_kmeans
    from uavdt.resources import nearest_association, process_consistent_processing

    cfg = replace(
        config_for_counts(10, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)),
        aodt_threshold_s=0.8,
    )
    sc = generate_scenario(seed=1, cfg=cfg)
    uav = place_kmeans(sc, 1)
    a = np.zeros((10, 3), dtype=float)
    for proc in sc.processes:
        members = proc.iot_indices
        centroid = sc.iot_xyz_m[members].mean(axis=0, keepdims=True)
        j_star = int(np.argmin(np.linalg.norm(centroid - uav, axis=1)))
        a[members, j_star] = 1.0
    b = process_consistent_processing(sc, a)
    res = solve_bandwidth_at_fixed_q(sc, uav, a, b)
    assert not res.infeasible
    ev = evaluate(sc, uav, Allocation(a, b, res.bandwidth_hz))
    assert ev.feasible
    near = nearest_association(sc.iot_xyz_m, uav)
    res_near = solve_bandwidth_at_fixed_q(sc, uav, near, process_consistent_processing(sc, near))
    assert res_near.infeasible


def test_sca_output_constraints_on_main_scenario():
    cfg = SimConfig(b_sys_hz=2.4e6)
    snapshot = (
        DEFAULT.area_x_m,
        DEFAULT.b_sys_hz,
        DEFAULT.sigma,
        DEFAULT.task_size_bits,
        DEFAULT.task_cycles,
    )
    sc = generate_scenario(seed=1, cfg=cfg)
    result = solve_sca(
        sc, seed=1, settings=SCASettings(max_iterations=8, epsilon=1.0, step_size_m=1.0)
    )
    uav = result.uav_xyz_m
    alloc = result.allocation
    a = alloc.hard_association()
    b = alloc.hard_processing()
    bw = alloc.bandwidth_hz

    assert uav.shape[0] == 3
    assert np.all(uav[:, 0] >= -1e-6) and np.all(uav[:, 0] <= 100.0 + 1e-6)
    assert np.all(uav[:, 1] >= -1e-6) and np.all(uav[:, 1] <= 100.0 + 1e-6)
    np.testing.assert_allclose(uav[:, 2], 100.0)
    assert np.all(bw >= -1e-6)
    assert float(bw.sum()) <= cfg.b_sys_hz + 1e-3
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    np.testing.assert_allclose(b.sum(axis=1), 1.0)
    for proc in sc.processes:
        js = np.argmax(b[proc.iot_indices], axis=1)
        assert np.unique(js).size == 1

    true = evaluate(sc, uav, alloc)
    np.testing.assert_allclose(result.true_objective, true.sum_rate_bit_per_s)
    assert true.constraints.queue_stable
    assert true.constraints.association_ok
    assert true.constraints.processing_ok
    assert true.constraints.process_consistent
    assert true_gate_ok(true)

    assert snapshot == (
        DEFAULT.area_x_m,
        DEFAULT.b_sys_hz,
        DEFAULT.sigma,
        DEFAULT.task_size_bits,
        DEFAULT.task_cycles,
    )


def test_sca_bandwidth_presets_are_independent():
    sc20 = generate_scenario(seed=1, cfg=SimConfig(b_sys_hz=20_000.0))
    sc24 = generate_scenario(seed=1, cfg=SimConfig(b_sys_hz=2_400_000.0), iot_xyz_m=sc20.iot_xyz_m)
    settings = SCASettings(max_iterations=3, epsilon=1e-2, step_size_m=1.0)
    r20 = solve_sca(sc20, seed=1, settings=settings)
    r24 = solve_sca(sc24, seed=1, settings=settings)
    assert r20.allocation.bandwidth_hz.sum() <= 20_000.0 + 1.0
    assert r24.allocation.bandwidth_hz.sum() <= 2_400_000.0 + 1.0
    assert r24.true_objective > r20.true_objective
    assert sc20.cfg.b_sys_hz == 20_000.0
    assert sc24.cfg.b_sys_hz == 2_400_000.0


def test_frozen_positions_qos_floor_on_true_rates():
    """step_size=0: only B is optimized; true (25) should hold at q0."""
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(seed=1, cfg=cfg)
    result = solve_sca(
        sc,
        seed=1,
        settings=SCASettings(max_iterations=2, epsilon=1e-9, trust_region_m=0.0),
    )
    assert result.true_eval.constraints.qos_violations == 0
    assert result.true_eval.constraints.aodt_violations == 0
    assert true_gate_ok(result.true_eval)
    assert float(result.allocation.bandwidth_hz.sum()) <= cfg.b_sys_hz + 1.0
    assert result.diagnostics.get("accepted_steps") == 0
    assert result.diagnostics.get("stop_reason") == "STEP_SIZE_LIMIT"


def test_initialize_is_physically_evaluable():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(seed=1, cfg=cfg)
    uav, alloc = initialize_sca(sc, seed=1)
    ev = evaluate(sc, uav, alloc)
    assert np.all(np.isfinite(ev.assoc_rates_bit_per_s))
    assert ev.constraints.uav_in_field_ok
    assert ev.constraints.uav_separation_ok
    assert ev.constraints.queue_stable
    assert float(alloc.bandwidth_hz.sum()) <= cfg.b_sys_hz + 1e-6


def test_max_bw_share_caps_each_link():
    cfg = SimConfig(b_sys_hz=2.4e6, max_bw_share=0.25)
    sc = generate_scenario(seed=1, cfg=cfg)
    uav, alloc = initialize_sca(sc, seed=1)
    cap = cfg.link_bandwidth_cap_hz
    assert float(alloc.bandwidth_hz.max()) <= cap + 1e-3
    assert abs(float(alloc.bandwidth_hz.sum()) - cfg.b_sys_hz) < 1.0
    a = alloc.hard_association()
    b = alloc.hard_processing()
    res = solve_bandwidth_at_fixed_q(sc, uav, a, b)
    assert not res.infeasible
    assert float(res.bandwidth_hz.max()) <= cap + 1.0
    assert abs(float(res.bandwidth_hz.sum()) - cfg.b_sys_hz) < 1.0
    n_at_cap = int(np.sum(res.bandwidth_hz > 0.9 * cap))
    assert n_at_cap >= 2


def test_never_linearizes_around_true_infeasible(tmp_path: Path):
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(seed=1, cfg=cfg)
    result = solve_sca(
        sc, seed=1, settings=SCASettings(max_iterations=12, step_size_m=1.0)
    )
    assert true_gate_ok(result.true_eval)
    np.testing.assert_allclose(
        result.true_objective, result.true_eval.sum_rate_bit_per_s, rtol=0, atol=1e-6
    )
    for row in result.history:
        if row.accepted:
            assert row.qos_violations == 0
            assert row.aodt_violations == 0
    payload = run_sca_seq_debug(
        1,
        cfg,
        max_iterations=4,
        out_json=str(tmp_path / "sca_seq_debug.json"),
    )
    data = json.loads(Path(payload["_json"]).read_text(encoding="utf-8"))
    assert data["true_gate_ok"]
    assert data["reported_equals_evaluator"]
    assert Path(payload["_csv"]).exists()
    a0 = np.array(data["association"])
    # frozen binaries
    np.testing.assert_allclose(a0.sum(axis=1), 1.0)


def test_stop_reason_is_step_size_limit_when_no_accepted_moves():
    from uavdt.sca.algorithm import classify_stop_reason

    assert (
        classify_stop_reason(
            "CONVERGED",
            accepted_steps=0,
            step_size=9.765625e-4,
            min_step_size=1e-3,
        )
        == "STEP_SIZE_LIMIT"
    )
    assert (
        classify_stop_reason(
            "CONVERGED",
            accepted_steps=0,
            step_size=1.0,
            min_step_size=1e-3,
        )
        == "STEP_SIZE_LIMIT"
    )
    assert (
        classify_stop_reason(
            "CONVERGED",
            accepted_steps=3,
            step_size=9.765625e-4,
            min_step_size=1e-3,
        )
        == "CONVERGED"
    )
    assert (
        classify_stop_reason(
            "MAX_ITERATIONS",
            accepted_steps=3,
            step_size=5e-4,
            min_step_size=1e-3,
        )
        == "CONVERGED"
    )
    assert (
        classify_stop_reason(
            "MAX_ITERATIONS",
            accepted_steps=0,
            step_size=0.5,
            min_step_size=1e-3,
        )
        == "MAX_ITERATIONS"
    )
    assert (
        classify_stop_reason(
            "init_bandwidth_infeasible",
            accepted_steps=0,
            step_size=0.0,
            min_step_size=1e-3,
        )
        == "init_bandwidth_infeasible"
    )


def test_gap_vs_uncapped_is_difference_not_rescaled():
    from uavdt.sca.reporting import gap_vs_uncapped

    gap, pct = gap_vs_uncapped(10.0, 9.0)
    assert gap == 1.0
    assert abs(pct - 10.0) < 1e-12
    gap_bps, pct_bps = gap_vs_uncapped(2.4e6, 1.8e6)
    assert gap_bps == 0.6e6
    assert abs(pct_bps - 25.0) < 1e-12
