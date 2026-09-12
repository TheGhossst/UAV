"""SCA-joint methodology probe tests. Frozen SCA stays the headline solver."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.experiments.methods import KNOWN_METHODS, METHODS, run_method
from uavdt.models import Allocation
from uavdt.resources import nearest_association
from uavdt.sca import SCASettings, solve_sca
from uavdt.scenario import generate_scenario, make_uav_xyz_m
from uavdt.sca_joint import solve_sca_joint
from uavdt.sca_joint.rematch import (
    association_process_cohesive,
    best_se_association,
    centroid_cohesive_association,
    iter_rematch_candidates,
    propose_process_cohesive,
    propose_rematch,
)

cvxpy = pytest.importorskip("cvxpy")


def test_known_methods_keep_sca_joint_opt_in():
    assert "sca_joint" not in METHODS
    assert "sca_joint" in KNOWN_METHODS
    assert "sca_multistart" in KNOWN_METHODS
    assert METHODS == ("random", "kmeans", "pso", "sca")


def test_best_se_association_holds_exclusive(frozen_scenario, three_uavs):
    a = best_se_association(frozen_scenario, three_uavs)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    assert set(np.unique(a).tolist()) <= {0.0, 1.0}


def test_rematch_switches_from_wrong_uav():
    cfg = SimConfig(
        num_iot=2,
        num_processes=1,
        iots_per_process=2,
        num_uav=2,
        b_sys_hz=8.8e6,
        max_bw_share=0.25,
    )
    iot = np.array([[50.0, 50.0, 0.0], [51.0, 50.0, 0.0]])
    sc = generate_scenario(seed=0, cfg=cfg, iot_xyz_m=iot)
    uav = make_uav_xyz_m([[10.0, 50.0], [50.0, 50.0]], cfg.uav_height_m)
    a_wrong = np.array([[1.0, 0.0], [1.0, 0.0]])
    b_wrong = a_wrong.copy()
    prop = propose_rematch(sc, uav, a_wrong, b_wrong)
    assert prop.ok
    assert int(np.argmax(prop.association[0])) == 1
    assert association_process_cohesive(sc, prop.association)


def test_rematch_rejects_cpu_unstable_switch():
    from uavdt.placement.kmeans import place_kmeans

    cfg = SimConfig(
        num_iot=10,
        num_uav=3,
        b_sys_hz=8.8e6,
        uav_cpu_cycles_per_s=1.0,
        max_bw_share=0.25,
    )
    sc = generate_scenario(seed=1, cfg=cfg)
    uav = place_kmeans(sc, 1)
    greedy = best_se_association(sc, uav)
    a_wrong = np.roll(greedy, 1, axis=1)
    assert not np.array_equal(greedy > 0.5, a_wrong > 0.5)
    prop = propose_rematch(sc, uav, a_wrong, a_wrong)
    assert not prop.ok
    assert prop.reason == "cpu_unstable"


def test_run_method_tags_sca_joint():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    run = run_method(
        sc, "sca_joint", seed=1, sca_settings=SCASettings(solver=None, max_iterations=2)
    )
    assert run.method == "sca_joint"
    assert run.diagnostics.get("method") == "sca_joint"


def test_frozen_sca_still_freezes_association():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(1, cfg)
    result = solve_sca(
        sc, seed=1, settings=SCASettings(max_iterations=4, step_size_m=1.0)
    )
    assert result.diagnostics.get("association_init_equals_final") is True
    assert result.diagnostics.get("method") != "sca_joint"


def test_sca_joint_tiny_instance_still_moves():
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
    alloc0 = Allocation(assoc, assoc.copy(), np.array([[cfg.b_sys_hz * 0.5], [cfg.b_sys_hz * 0.5]]))
    result = solve_sca_joint(
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
    assert result.diagnostics.get("method") == "sca_joint"
    assert result.true_eval.feasible


def test_campaign_cli_accepts_sca_joint():
    from uavdt.experiments.cli import _parse_methods

    assert _parse_methods("sca_joint") == ("sca_joint",)
    assert _parse_methods("random,kmeans,pso,sca,sca_joint")[-1] == "sca_joint"


def test_centroid_construction_is_cohesive():
    from uavdt.experiments.grids import config_for_counts
    from uavdt.placement.kmeans import place_kmeans

    cfg = config_for_counts(10, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25))
    sc = generate_scenario(1, cfg)
    uav = place_kmeans(sc, 1)
    a = centroid_cohesive_association(sc, uav)
    assert association_process_cohesive(sc, a)
    near = nearest_association(sc.iot_xyz_m, uav)
    # Hand construction differs from nearest on this seed (split N_k).
    assert not np.array_equal(a > 0.5, near > 0.5)


def _tk08_scenario(seed: int = 1):
    from dataclasses import replace

    from uavdt.config import PRIMARY_MAX_BW_SHARE
    from uavdt.experiments.grids import config_for_counts

    cfg = replace(
        config_for_counts(10, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)),
        aodt_threshold_s=0.8,
    )
    return generate_scenario(seed, cfg)


def test_iter_rematch_candidates_best_se_only_by_default():
    from uavdt.placement.kmeans import place_kmeans

    sc = _tk08_scenario(1)
    uav = place_kmeans(sc, 1)
    a = nearest_association(sc.iot_xyz_m, uav)
    cands = iter_rematch_candidates(sc, uav, a, a, include_process_cohesive=False)
    assert len(cands) == 1
    assert cands[0].kind == "best_se"


def test_iter_rematch_candidates_yields_cohesive_beside_best_se():
    from uavdt.placement.kmeans import place_kmeans

    sc = _tk08_scenario(1)
    uav = place_kmeans(sc, 1)
    a = nearest_association(sc.iot_xyz_m, uav)
    cands = iter_rematch_candidates(sc, uav, a, a, include_process_cohesive=True)
    assert [c.kind for c in cands] == ["best_se", "process_cohesive"]
    assert not cands[0].ok
    assert cands[0].reason == "association_unchanged"
    assert cands[1].ok
    assert association_process_cohesive(sc, cands[1].association)
    assert not np.array_equal(cands[1].association > 0.5, a > 0.5)


def test_propose_process_cohesive_differs_from_nearest_at_tk08():
    from uavdt.placement.kmeans import place_kmeans

    sc = _tk08_scenario(1)
    uav = place_kmeans(sc, 1)
    a = nearest_association(sc.iot_xyz_m, uav)
    prop = propose_process_cohesive(sc, uav, a)
    assert prop.ok
    assert prop.kind == "process_cohesive"
    greedy = best_se_association(sc, uav)
    assert np.array_equal(greedy > 0.5, a > 0.5)


def test_sca_joint_default_stays_infeasible_at_tk08_seed1():
    sc = _tk08_scenario(1)
    result = solve_sca_joint(
        sc, seed=1, settings=SCASettings(solver=None, max_iterations=2)
    )
    assert result.diagnostics.get("process_cohesive_candidate") is False
    assert result.true_eval.feasible is False
    assert result.n_iterations == 0


def test_sca_joint_cohesive_candidate_recovers_tk08_seed1():
    """Canary for the mechanistic story: if this fails, do not force it."""
    sc = _tk08_scenario(1)
    result = solve_sca_joint(
        sc,
        seed=1,
        settings=SCASettings(
            solver=None,
            max_iterations=8,
            process_cohesive_candidate=True,
        ),
    )
    assert result.diagnostics.get("process_cohesive_candidate") is True
    assert result.true_eval.feasible
    assert result.diagnostics.get("process_cohesive_a") is True
    kinds = result.diagnostics.get("rematch_kinds") or []
    assert any(str(k).endswith("process_cohesive") for k in kinds)


def test_frozen_sca_ignores_process_cohesive_candidate():
    sc = _tk08_scenario(1)
    off = solve_sca(sc, seed=1, settings=SCASettings(solver=None, max_iterations=2))
    on = solve_sca(
        sc,
        seed=1,
        settings=SCASettings(
            solver=None,
            max_iterations=2,
            process_cohesive_candidate=True,
        ),
    )
    assert off.true_eval.feasible is False
    assert on.true_eval.feasible is False
    np.testing.assert_allclose(off.uav_xyz_m, on.uav_xyz_m)
    np.testing.assert_allclose(off.allocation.association, on.allocation.association)


def test_sca_joint_cli_exposes_cohesive_flag():
    from uavdt.experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["sca-joint", "--process-cohesive-candidate", "--bandwidth-preset", "8.8mhz"]
    )
    assert args.process_cohesive_candidate is True
    args_off = parser.parse_args(["sca-joint", "--bandwidth-preset", "8.8mhz"])
    assert args_off.process_cohesive_candidate is False
