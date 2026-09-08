"""100-scenario bank: save layouts, replay, average eval."""

from __future__ import annotations

import json

import numpy as np
import pytest

from uavdt.config import SimConfig
from uavdt.experiments.methods import run_method
from uavdt.experiments.n100 import evaluate_bank, write_eval
from uavdt.experiments.scenario_bank import (
    generate_bank,
    load_bank,
    scenario_from_record,
    uav_from_record,
    write_bank,
)
from uavdt.placement.random import place_random


def test_generate_bank_roundtrip(tmp_path):
    cfg = SimConfig(b_sys_hz=2.4e6)
    bank = generate_bank(4, cfg, seed_start=11, num_iot=10, num_uav=3)
    path = write_bank(bank, tmp_path / "bank.json")
    loaded = load_bank(path)
    assert loaded["n_scenarios"] == 4
    assert loaded["geometry"]["num_iot"] == 10
    assert loaded["geometry"]["num_uav"] == 3
    rec = loaded["scenarios"][0]
    assert rec["seed"] == 11
    iot = np.asarray(rec["iot_xyz_m"])
    uav = np.asarray(rec["uav_xyz_m"])
    assert iot.shape == (10, 3)
    assert uav.shape == (3, 3)
    assert np.allclose(iot[:, 2], 0.0)
    assert np.allclose(uav[:, 2], cfg.uav_height_m)
    sc = scenario_from_record(rec, cfg)
    np.testing.assert_allclose(sc.iot_xyz_m, iot)
    replay = place_random(cfg.num_uav, rec["seed"], cfg)
    np.testing.assert_allclose(uav_from_record(rec), replay)


def test_bank_layouts_differ_across_seeds():
    cfg = SimConfig()
    bank = generate_bank(3, cfg, seed_start=1)
    a = np.asarray(bank["scenarios"][0]["iot_xyz_m"])
    b = np.asarray(bank["scenarios"][1]["iot_xyz_m"])
    assert not np.allclose(a, b)


def test_random_method_replays_saved_uav():
    cfg = SimConfig(b_sys_hz=2.4e6)
    bank = generate_bank(1, cfg, seed_start=5)
    rec = bank["scenarios"][0]
    sc = scenario_from_record(rec, cfg)
    saved = uav_from_record(rec)
    run = run_method(sc, "random", rec["seed"], uav_xyz_m=saved)
    np.testing.assert_allclose(run.uav_xyz_m, saved)


def test_evaluate_bank_checkpoint_and_means(tmp_path):
    pytest.importorskip("cvxpy")
    cfg = SimConfig(b_sys_hz=2.4e6, max_bw_share=0.25)
    bank = generate_bank(2, cfg, seed_start=1)
    ckpt = tmp_path / "ckpt.json"
    payload = evaluate_bank(
        bank,
        cfg,
        ("random", "kmeans"),
        checkpoint_path=ckpt,
        resume=True,
        bank_path=tmp_path / "bank.json",
    )
    assert payload["n_scenarios"] == 2
    assert payload["by_method"]["kmeans"]["n"] == 2
    assert payload["by_method"]["random"]["mean_sum_rate_Mbps"] > 0.0
    out = write_eval(payload, tmp_path / "eval.json")
    assert out.exists()
    assert out.with_suffix(".csv").exists()
    assert json.loads(ckpt.read_text(encoding="utf-8"))["runs"]

    payload2 = evaluate_bank(
        bank,
        cfg,
        ("random", "kmeans"),
        checkpoint_path=ckpt,
        resume=True,
    )
    np.testing.assert_allclose(
        payload["by_method"]["kmeans"]["per_seed_Mbps"],
        payload2["by_method"]["kmeans"]["per_seed_Mbps"],
    )


def test_n100_plot_smoke(tmp_path):
    pytest.importorskip("cvxpy")
    matplotlib = pytest.importorskip("matplotlib")
    _ = matplotlib
    from uavdt.experiments.n100_plot import plot_n100_figures

    cfg = SimConfig(b_sys_hz=2.4e6, max_bw_share=0.25)
    bank = generate_bank(3, cfg, seed_start=1)
    payload = evaluate_bank(bank, cfg, ("random", "kmeans"))
    paths = plot_n100_figures(payload, bank, tmp_path / "figs")
    names = {p.name for p in paths}
    assert "n100_mean_sum_rate.png" in names
    assert "n100_scenario_maps.png" in names
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
