"""Keep-best multi-start SCA. Frozen Algorithm 1 is unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.experiments.cli import _parse_methods
from uavdt.experiments.methods import KNOWN_METHODS, METHODS, run_method
from uavdt.sca import SCASettings, solve_sca
from uavdt.sca_multistart import (
    MultiStartSettings,
    extra_starts,
    solve_sca_multistart,
)
from uavdt.scenario import generate_scenario

cvxpy = pytest.importorskip("cvxpy")


def test_known_methods_keep_multistart_opt_in():
    assert "sca_multistart" not in METHODS
    assert "sca_multistart" in KNOWN_METHODS
    assert METHODS == ("random", "kmeans", "pso", "sca")


def test_extra_starts_match_experiment_a():
    assert extra_starts(7) == (
        ("random", 7001),
        ("random", 7002),
        ("kmeans", 7001),
        ("kmeans", 7002),
    )
    assert extra_starts(1, MultiStartSettings(n_random=0, n_kmeans=1)) == (
        ("kmeans", 1001),
    )


def test_campaign_cli_accepts_sca_multistart():
    assert _parse_methods("sca_multistart") == ("sca_multistart",)
    assert _parse_methods("random,sca,sca_multistart")[-1] == "sca_multistart"


def test_frozen_only_matches_one_shot_sca():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    settings = SCASettings(solver=None, max_iterations=2)
    frozen = solve_sca(sc, 1, settings=settings)
    multi = solve_sca_multistart(
        sc,
        1,
        settings=settings,
        multistart=MultiStartSettings(n_random=0, n_kmeans=0, include_frozen=True),
    )
    np.testing.assert_allclose(multi.uav_xyz_m, frozen.uav_xyz_m)
    np.testing.assert_allclose(
        multi.true_eval.sum_rate_bit_per_s, frozen.true_eval.sum_rate_bit_per_s
    )
    assert multi.diagnostics["method"] == "sca_multistart"
    assert multi.diagnostics["winner_kind"] == "frozen_kmeans"
    assert multi.diagnostics["delta_vs_frozen_Mbps"] == 0.0


def test_keep_best_never_worse_than_frozen():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    settings = SCASettings(solver=None, max_iterations=2)
    frozen = solve_sca(sc, 1, settings=settings)
    multi = solve_sca_multistart(
        sc,
        1,
        settings=settings,
        multistart=MultiStartSettings(n_random=1, n_kmeans=0, include_frozen=True),
    )
    assert multi.true_eval.sum_rate_mbps + 1e-9 >= frozen.true_eval.sum_rate_mbps
    assert multi.diagnostics["n_starts"] == 2
    assert len(multi.diagnostics["starts"]) == 2


def test_run_method_tags_sca_multistart():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    run = run_method(
        sc,
        "sca_multistart",
        seed=1,
        sca_settings=SCASettings(solver=None, max_iterations=2),
        multistart_settings=MultiStartSettings(
            n_random=0, n_kmeans=0, include_frozen=True
        ),
    )
    assert run.method == "sca_multistart"
    assert run.diagnostics.get("method") == "sca_multistart"
    assert run.feasible
    assert run.sum_rate_mbps > 0.0
