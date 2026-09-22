"""Dynamic a_ij / b_ij updates on Algorithm 1 SCA."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.evaluator import evaluate
from uavdt.models import Allocation
from uavdt.sca import SCASettings, solve_sca
from uavdt.sca.algorithm import true_gate_ok
from uavdt.sca.assignment import (
    assignment_legal,
    best_se_association,
    gated_assignment_update,
    n_row_changes,
)
from uavdt.scenario import generate_scenario, make_uav_xyz_m

cvxpy = pytest.importorskip("cvxpy")


def _two_uav_toy(*, aodt_threshold_s: float = 2.8) -> tuple:
    cfg = SimConfig(
        num_iot=2,
        num_processes=1,
        iots_per_process=2,
        num_uav=2,
        b_sys_hz=8.8e6,
        max_bw_share=0.25,
        aodt_threshold_s=aodt_threshold_s,
    )
    iot = np.array([[50.0, 50.0, 0.0], [51.0, 50.0, 0.0]])
    sc = generate_scenario(seed=0, cfg=cfg, iot_xyz_m=iot)
    uav = make_uav_xyz_m([[10.0, 50.0], [50.0, 50.0]], cfg.uav_height_m)
    return sc, uav


def test_frozen_mode_matches_explicit_false_and_keeps_init_assignment():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(seed=1, cfg=cfg)
    settings = SCASettings(max_iterations=4, step_size_m=1.0)
    frozen_default = solve_sca(sc, seed=1, settings=settings)
    frozen_flag = solve_sca(
        sc, seed=1, settings=SCASettings(max_iterations=4, step_size_m=1.0, dynamic_assignment=False)
    )
    np.testing.assert_allclose(frozen_default.uav_xyz_m, frozen_flag.uav_xyz_m)
    np.testing.assert_allclose(
        frozen_default.allocation.association, frozen_flag.allocation.association
    )
    np.testing.assert_allclose(
        frozen_default.allocation.processing, frozen_flag.allocation.processing
    )
    np.testing.assert_allclose(frozen_default.true_objective, frozen_flag.true_objective)
    assert frozen_default.diagnostics.get("dynamic_assignment") is False
    assert frozen_default.diagnostics.get("association_init_equals_final") is True
    assert frozen_default.diagnostics.get("processing_init_equals_final") is True
    assert frozen_default.diagnostics.get("method") != "sca_joint"
    assert true_gate_ok(frozen_default.true_eval)


def test_dynamic_assignment_switches_wrong_association_when_beneficial():
    sc, uav = _two_uav_toy()
    a_wrong = np.array([[1.0, 0.0], [1.0, 0.0]])
    b_local_far = a_wrong.copy()
    bw = np.array(
        [[sc.cfg.b_sys_hz * 0.5, 0.0], [sc.cfg.b_sys_hz * 0.5, 0.0]]
    )
    alloc = Allocation(a_wrong, b_local_far, bw)
    greedy = best_se_association(sc, uav)
    assert int(np.argmax(greedy[0])) == 1
    frozen = solve_sca(
        sc,
        seed=0,
        settings=SCASettings(
            max_iterations=2,
            trust_region_m=0.0,
            dynamic_assignment=False,
        ),
        uav_xyz_m=uav,
        allocation=alloc,
    )
    result = solve_sca(
        sc,
        seed=0,
        settings=SCASettings(
            max_iterations=2,
            trust_region_m=0.0,
            dynamic_assignment=True,
        ),
        uav_xyz_m=uav,
        allocation=alloc,
    )
    a_final = result.allocation.hard_association()
    assert int(np.argmax(a_final[0])) == 1
    assert int(np.argmax(a_final[1])) == 1
    assert np.argmax(frozen.allocation.hard_association(), axis=1).tolist() == [0, 0]
    assert result.diagnostics.get("dynamic_assignment") is True
    assert result.diagnostics.get("association_init_equals_final") is False
    assert result.diagnostics.get("assignment_accepted", 0) >= 1
    assert result.true_objective > frozen.true_objective
    assert true_gate_ok(result.true_eval)
    assoc_rows = [
        row
        for row in result.history
        if row.assignment_stage == "association" and row.accepted
    ]
    assert assoc_rows
    assert assoc_rows[0].n_assoc_changed == 2
    assert assoc_rows[0].rejection_reason.endswith("_update") or assoc_rows[
        0
    ].rejection_reason.endswith("_newly_feasible")


def test_dynamic_assignment_updates_processing_when_beneficial():
    sc, uav = _two_uav_toy(aodt_threshold_s=0.8)
    a_close = np.array([[0.0, 1.0], [0.0, 1.0]])
    b_far = np.array([[1.0, 0.0], [1.0, 0.0]])
    bw = np.array([[0.0, sc.cfg.b_sys_hz * 0.5], [0.0, sc.cfg.b_sys_hz * 0.5]])
    alloc = Allocation(a_close, b_far, bw)
    frozen = solve_sca(
        sc,
        seed=0,
        settings=SCASettings(max_iterations=2, trust_region_m=0.0, dynamic_assignment=False),
        uav_xyz_m=uav,
        allocation=alloc,
    )
    assert frozen.true_eval.feasible is False
    result = solve_sca(
        sc,
        seed=0,
        settings=SCASettings(
            max_iterations=2,
            trust_region_m=0.0,
            dynamic_assignment=True,
        ),
        uav_xyz_m=uav,
        allocation=alloc,
    )
    b_final = result.allocation.hard_processing()
    assert int(np.argmax(b_final[0])) == 1
    assert int(np.argmax(b_final[1])) == 1
    assert result.diagnostics.get("processing_init_equals_final") is False
    assert true_gate_ok(result.true_eval)
    proc_rows = [
        row
        for row in result.history
        if row.assignment_stage == "processing" and row.accepted
    ]
    assert proc_rows
    assert proc_rows[0].n_proc_changed == 2


def test_gated_update_rejects_illegal_processing():
    sc, uav = _two_uav_toy()
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = a.copy()
    bw = np.array([[0.0, sc.cfg.b_sys_hz * 0.5], [0.0, sc.cfg.b_sys_hz * 0.5]])
    current = evaluate(sc, uav, Allocation(a, b, bw))
    assert true_gate_ok(current)
    b_split = np.array([[1.0, 0.0], [0.0, 1.0]])
    ok, reason = assignment_legal(sc, a, b_split)
    assert not ok
    assert reason == "process_inconsistent"
    attempt = gated_assignment_update(
        sc,
        uav,
        a,
        b,
        bw,
        current,
        a,
        b_split,
        SCASettings(dynamic_assignment=True),
        iteration=1,
        stage="processing",
    )
    assert attempt.event.accepted is False
    assert attempt.event.reason == "process_inconsistent"
    np.testing.assert_array_equal(attempt.processing, b)
    np.testing.assert_array_equal(attempt.association, a)


def test_gated_update_rejects_cpu_unstable_processing():
    from uavdt.experiments.grids import config_for_counts
    from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
    from uavdt.sca.initialize import initialize_sca

    cfg = config_for_counts(32, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25))
    sc = generate_scenario(seed=1, cfg=cfg)
    uav, alloc = initialize_sca(sc, 1)
    a = alloc.hard_association()
    b = alloc.hard_processing()
    bw_res = solve_bandwidth_at_fixed_q(sc, uav, a, b)
    assert not bw_res.infeasible
    current = evaluate(sc, uav, Allocation(a, b, bw_res.bandwidth_hz))
    assert true_gate_ok(current)
    b_all = np.zeros_like(b)
    b_all[:, 0] = 1.0
    attempt = gated_assignment_update(
        sc,
        uav,
        a,
        b,
        bw_res.bandwidth_hz,
        current,
        a,
        b_all,
        SCASettings(dynamic_assignment=True),
        iteration=1,
        stage="processing",
    )
    assert attempt.event.accepted is False
    assert attempt.event.reason == "cpu_unstable"
    assert n_row_changes(attempt.processing, b) == 0


def test_dynamic_final_solution_passes_true_gate():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(seed=1, cfg=cfg)
    result = solve_sca(
        sc,
        seed=1,
        settings=SCASettings(
            max_iterations=6,
            step_size_m=1.0,
            dynamic_assignment=True,
        ),
    )
    assert true_gate_ok(result.true_eval)
    np.testing.assert_allclose(
        result.true_objective, result.true_eval.sum_rate_bit_per_s, rtol=0, atol=1e-6
    )
    a = result.allocation.hard_association()
    b = result.allocation.hard_processing()
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    np.testing.assert_allclose(b.sum(axis=1), 1.0)
    for proc in sc.processes:
        js = np.argmax(b[proc.iot_indices], axis=1)
        assert np.unique(js).size == 1
    for row in result.history:
        if row.accepted and row.assignment_stage:
            assert row.qos_violations == 0
            assert row.aodt_violations == 0
            log = result.diagnostics.get("assignment_log") or []
            assert any(
                item["iteration"] == row.iteration and item["accepted"] for item in log
            )


def test_run_method_sca_is_dynamic_and_frozen_sca_is_fixed():
    from uavdt.experiments.methods import run_method

    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(seed=1, cfg=cfg)
    dyn = run_method(
        sc,
        "sca",
        1,
        sca_settings=SCASettings(solver=None, max_iterations=2, step_size_m=1.0),
    )
    frz = run_method(
        sc,
        "frozen_sca",
        1,
        sca_settings=SCASettings(solver=None, max_iterations=2, step_size_m=1.0),
    )
    assert dyn.method == "sca"
    assert frz.method == "frozen_sca"
    assert dyn.diagnostics.get("dynamic_assignment") is True
    assert frz.diagnostics.get("dynamic_assignment") is False


def test_cli_exposes_dynamic_assignment_flag():
    from uavdt.experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["sca", "--dynamic-assignment", "--bandwidth-preset", "2.4mhz", "--solver", "cvxpy"]
    )
    assert args.dynamic_assignment is True
    args_off = parser.parse_args(["sca", "--bandwidth-preset", "2.4mhz"])
    assert args_off.dynamic_assignment is False
