"""PSO placement baseline and campaign smoke tests. SCA is only called."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.experiments.campaign import CampaignSettings, run_campaign
from uavdt.experiments.grids import config_for_counts, iter_axis
from uavdt.experiments.methods import run_method
from uavdt.placement.pso import PSOSettings, place_pso
from uavdt.scenario import generate_scenario


def test_config_for_counts_keeps_two_equal_processes():
    cfg = config_for_counts(32, 3, SimConfig(b_sys_hz=2.4e6))
    assert cfg.num_iot == 32
    assert cfg.num_uav == 3
    assert cfg.num_processes == 2
    assert cfg.iots_per_process == 16
    sc = generate_scenario(1, cfg)
    assert sc.iot_xyz_m.shape == (32, 3)
    assert len(sc.processes[0].iot_indices) == 16
    assert len(sc.processes[1].iot_indices) == 16


def test_pso_respects_field_and_separation():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(1, cfg)
    uav = place_pso(sc, seed=1, settings=PSOSettings(n_particles=4, n_iterations=3))
    assert uav.shape == (3, 3)
    np.testing.assert_allclose(uav[:, 2], 100.0)
    assert np.all(uav[:, 0] >= 0.0) and np.all(uav[:, 0] <= 100.0)
    for p in range(3):
        for q in range(p + 1, 3):
            assert np.linalg.norm(uav[p] - uav[q]) >= cfg.uav_min_separation_m - 1e-9


cvxpy = pytest.importorskip("cvxpy")


def test_run_method_kmeans_uses_true_evaluator():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(1, cfg)
    run = run_method(sc, "kmeans", seed=1)
    assert run.method == "kmeans"
    assert run.true_eval.sum_rate_bit_per_s > 0.0
    np.testing.assert_allclose(
        run.allocation.bandwidth_hz.sum(), cfg.b_sys_hz, atol=1.0
    )


def test_campaign_uavs_smoke(tmp_path):
    cfg = SimConfig(b_sys_hz=2.4e6)
    from uavdt.experiments.grids import UAV_COUNTS
    from uavdt.sca.settings import SCASettings

    settings = CampaignSettings(
        n_runs=1,
        methods=("kmeans",),
        sca_settings=SCASettings(solver=None, max_iterations=1),
    )
    # Restrict to one J by calling run_point via a tiny axis loop.
    points = [p for p in iter_axis("uavs", cfg) if p.x_value == 3]
    assert len(points) == 1
    from uavdt.experiments.campaign import run_point

    row = run_point(points[0], settings)
    assert "kmeans" in row["by_method"]
    assert row["by_method"]["kmeans"]["n"] == 1
    _ = UAV_COUNTS
    _ = tmp_path
    _ = run_campaign


def test_replace_points_swaps_matching_rows_only():
    from uavdt.experiments.campaign import replace_points

    payload = {
        "points": [
            {"axis": "iots", "x": 24.0, "v": "old24"},
            {"axis": "iots", "x": 28.0, "v": "old28"},
        ]
    }
    merged = replace_points(
        payload, [{"axis": "iots", "x": 28.0, "v": "new28"}]
    )
    assert merged["points"][0]["v"] == "old24"
    assert merged["points"][1]["v"] == "new28"
