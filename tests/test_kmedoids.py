"""P-median / k-medoids covering subsets."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.experiments.cli import _parse_methods
from uavdt.experiments.methods import KNOWN_METHODS, METHODS, run_method
from uavdt.placement.kmedoids import (
    covering_combos,
    exact_pmedian_combos,
    kmedoids_combo,
    place_kmedoids,
)
from uavdt.sca_anchor import AnchorSettings, score_anchor_combos
from uavdt.scenario import generate_scenario

cvxpy = pytest.importorskip("cvxpy")


def test_medoid_is_opt_in():
    assert "sca_medoid" not in METHODS
    assert "sca_medoid" in KNOWN_METHODS
    assert _parse_methods("sca_medoid") == ("sca_medoid",)


def test_pmedian_splits_two_clusters():
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [50.0, 0.0],
            [51.0, 0.0],
            [52.0, 0.0],
        ]
    )
    best, ties, cost = exact_pmedian_combos(xy, 2)
    left = set(best) & {0, 1, 2}
    right = set(best) & {3, 4, 5}
    assert len(left) == 1 and len(right) == 1
    assert cost > 0.0
    assert best in ties


def test_kmedoids_returns_j_indices():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 100.0, size=(10, 2))
    combo = kmedoids_combo(xy, 3, rng)
    assert len(combo) == 3
    assert len(set(combo)) == 3
    assert all(0 <= i < 10 for i in combo)


def test_covering_combos_include_pmedian():
    rng = np.random.default_rng(1)
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    combos = covering_combos(sc.iot_xyz_m[:, :2], 3, rng, n_pam_inits=3)
    best, _ties, _ = exact_pmedian_combos(sc.iot_xyz_m[:, :2], 3)
    assert best in combos
    assert all(len(c) == 3 for c in combos)


def test_place_kmedoids_in_field():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    uav = place_kmedoids(sc, 1)
    assert uav.shape == (3, 3)
    np.testing.assert_allclose(uav[:, 2], 100.0)
    assert np.all(uav[:, 0] >= 0.0) and np.all(uav[:, 0] <= 100.0)


def test_medoid_selection_scores_few_combos():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    info = score_anchor_combos(
        sc,
        seed=1,
        settings=AnchorSettings(selection="medoid", top_k=2, n_medoid_inits=3),
    )
    assert info["mode"] == "medoid"
    assert 1 <= len(info["top"]) <= 2
    assert info["n_lp"] <= 20


def test_sca_medoid_keep_best_not_worse_than_sca():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(1, cfg)
    sca = run_method(sc, "sca", 1)
    med = run_method(sc, "sca_medoid", 1)
    assert med.diagnostics.get("selection") == "medoid"
    if sca.feasible and med.feasible:
        assert med.sum_rate_mbps + 1e-9 >= sca.sum_rate_mbps
