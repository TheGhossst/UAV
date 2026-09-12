"""Zenith-anchor SCA. Frozen Algorithm 1 is unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.experiments.cli import _parse_methods
from uavdt.experiments.grids import config_for_counts
from uavdt.experiments.methods import KNOWN_METHODS, METHODS, run_method
from uavdt.sca import SCASettings, solve_sca
from uavdt.sca_anchor import (
    AnchorSettings,
    anchors_separated,
    leftover_dump_upper_bound,
    n_anchor_combos,
    repair_anchor_xy,
    score_anchor_combos,
    solve_sca_anchor,
    uav_for_combo,
)
from uavdt.scenario import generate_scenario

cvxpy = pytest.importorskip("cvxpy")


def test_known_methods_keep_anchor_opt_in():
    assert "sca_anchor" not in METHODS
    assert "sca_anchor" in KNOWN_METHODS
    assert METHODS == ("random", "kmeans", "pso", "sca")


def test_campaign_cli_accepts_sca_anchor():
    assert _parse_methods("sca_anchor") == ("sca_anchor",)
    assert _parse_methods("random,sca,sca_anchor")[-1] == "sca_anchor"


def test_n_combos_and_separation_guard():
    assert n_anchor_combos(10, 3) == 120
    assert n_anchor_combos(32, 3) == 4960
    assert n_anchor_combos(10, 11) == 0
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 50.0]])
    assert not anchors_separated(xy[:2], 10.0)
    assert anchors_separated(xy[1:], 10.0)


def test_default_i10_j3_uses_full_enum():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    info = score_anchor_combos(
        sc,
        seed=1,
        settings=AnchorSettings(top_k=2, max_enumerate=1000),
        sca_settings=SCASettings(solver=None, max_iterations=1),
    )
    assert info["mode"] == "full"
    assert info["n_combos"] == 120
    assert info["n_lp"] <= 120
    assert 1 <= len(info["top"]) <= 2


def test_i32_uses_beam_not_full_enum():
    cfg = config_for_counts(32, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25))
    sc = generate_scenario(1, cfg)
    info = score_anchor_combos(
        sc,
        seed=1,
        settings=AnchorSettings(top_k=2, max_enumerate=50, beam_width=4),
        sca_settings=SCASettings(solver=None, max_iterations=1),
    )
    assert info["mode"] == "beam"
    assert info["n_combos"] == 4960
    assert info["n_lp"] < 2000
    assert len(info["top"]) <= 2


def test_keep_best_never_worse_than_frozen():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    settings = SCASettings(solver=None, max_iterations=2)
    frozen = solve_sca(sc, 1, settings=settings)
    anchored = solve_sca_anchor(
        sc,
        1,
        settings=settings,
        anchor=AnchorSettings(top_k=1, include_frozen=True),
    )
    assert anchored.true_eval.sum_rate_mbps + 1e-9 >= frozen.true_eval.sum_rate_mbps
    assert anchored.diagnostics["method"] == "sca_anchor"
    assert anchored.diagnostics["delta_vs_frozen_Mbps"] >= -1e-9


def test_process_cohesive_off_by_default():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    run = solve_sca_anchor(
        sc,
        1,
        settings=SCASettings(solver=None, max_iterations=1),
        anchor=AnchorSettings(top_k=1, include_frozen=True),
    )
    assert run.diagnostics.get("process_cohesive_candidate") is False


def test_process_cohesive_scores_tiny_tk08():
    from dataclasses import replace

    cfg = replace(
        config_for_counts(4, 2, SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)),
        aodt_threshold_s=0.8,
    )
    sc = generate_scenario(1, cfg)
    info = score_anchor_combos(
        sc,
        seed=1,
        settings=AnchorSettings(
            top_k=2,
            process_cohesive_candidate=True,
        ),
        sca_settings=SCASettings(solver=None, max_iterations=1),
    )
    assert info["n_lp"] >= info["n_combos"]
    kinds = {row.get("assoc_kind") for row in info["top"]}
    assert kinds <= {"nearest", "process_cohesive"}


def test_run_method_tags_sca_anchor():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    run = run_method(
        sc,
        "sca_anchor",
        seed=1,
        sca_settings=SCASettings(solver=None, max_iterations=1),
        anchor_settings=AnchorSettings(top_k=1, include_frozen=True),
    )
    assert run.method == "sca_anchor"
    assert run.diagnostics.get("method") == "sca_anchor"
    assert run.feasible
    assert run.sum_rate_mbps > 0.0


def test_sep_jitter_keeps_close_hosts():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    iot = np.array(
        [
            [10.0, 10.0, 0.0],
            [11.0, 10.0, 0.0],
            [50.0, 50.0, 0.0],
            [80.0, 20.0, 0.0],
            [20.0, 80.0, 0.0],
            [30.0, 30.0, 0.0],
            [70.0, 70.0, 0.0],
            [15.0, 60.0, 0.0],
            [60.0, 15.0, 0.0],
            [40.0, 80.0, 0.0],
        ]
    )
    sc = generate_scenario(1, cfg, iot_xyz_m=iot)
    repaired, n_jit = repair_anchor_xy(iot[:2, :2], 10.0, cfg)
    assert repaired is not None
    assert n_jit >= 1
    assert float(np.linalg.norm(repaired[0] - repaired[1])) + 1e-9 >= 10.0
    uav = uav_for_combo(sc, (0, 1, 2), seed=1)
    assert uav is not None
    assert float(np.linalg.norm(uav[0, :2] - uav[1, :2])) + 1e-9 >= 10.0


def test_max_enumerate_zero_uses_beam():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    info = score_anchor_combos(
        sc,
        seed=1,
        settings=AnchorSettings(top_k=2, max_enumerate=0, beam_width=4),
        sca_settings=SCASettings(solver=None, max_iterations=1),
    )
    assert info["mode"] == "beam"
    assert len(info["top"]) <= 2


def test_random_selection_scores_one_subset():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    info = score_anchor_combos(
        sc,
        seed=1,
        settings=AnchorSettings(selection="random", n_random=1, top_k=1),
        sca_settings=SCASettings(solver=None, max_iterations=1),
    )
    assert info["mode"] == "random"
    assert info["n_lp"] <= 1
    assert len(info["top"]) <= 1


def test_leftover_bound_at_least_lp_best():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25).with_square_area_m(500.0)
    sc = generate_scenario(1, cfg)
    info = score_anchor_combos(
        sc,
        seed=1,
        settings=AnchorSettings(top_k=1, max_enumerate=1000),
        sca_settings=SCASettings(solver=None, max_iterations=1),
    )
    bound = leftover_dump_upper_bound(sc)
    assert bound["bound_Mbps"] > 0.0
    if info["top"]:
        assert bound["bound_Mbps"] + 1e-4 >= float(info["top"][0]["sum_rate_Mbps"])


def test_polish_and_bound_diagnostics():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25).with_square_area_m(500.0)
    sc = generate_scenario(1, cfg)
    run = solve_sca_anchor(
        sc,
        1,
        settings=SCASettings(solver=None, max_iterations=2),
        anchor=AnchorSettings(top_k=1, include_frozen=True),
    )
    d = run.diagnostics
    assert d.get("bound_Mbps") is not None
    assert "n_jittered_sep" in d
    if d.get("winner_kind") == "anchor":
        assert d.get("winner_polish_mean_disp_m") is not None
