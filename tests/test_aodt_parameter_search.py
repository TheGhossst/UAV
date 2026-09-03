"""S_i / L search: config overrides, grids, scoring, and a tiny end-to-end run."""

from argparse import Namespace

import numpy as np
import pytest
from dataclasses import replace

from src.aodt import upload_times
from src.config import (
    DEFAULT,
    EXPERIMENTAL_TASK_CYCLES,
    EXPERIMENTAL_TASK_SIZE_BYTES,
)
from src.experiments.aodt_parameter_search import (
    COARSE_TASK_CYCLES,
    COARSE_TASK_SIZES,
    PAPER_FIG9_MBPS,
    SearchSpec,
    coarse_pairs,
    cpu_metrics,
    pair_key,
    rank_pairs,
    refine_around,
    refine_pairs,
    run_aodt_parameter_search,
    score_pair,
)
from src.main import _cfg_from_args, build_parser


def test_experimental_defaults_unchanged():
    assert EXPERIMENTAL_TASK_SIZE_BYTES == 12000.0
    assert EXPERIMENTAL_TASK_CYCLES == 3.75e6
    assert DEFAULT.task_size_bytes is None
    assert DEFAULT.task_cycles is None
    assert DEFAULT.use_compute_model is False
    cfg = DEFAULT.with_compute()
    assert cfg.task_size_bytes == 12000.0
    assert cfg.task_cycles == 3.75e6
    assert cfg.b_sys == 8.8e6


def test_with_compute_override_does_not_mutate_defaults():
    cfg = DEFAULT.with_compute(task_size_bytes=8000.0, task_cycles=1e6)
    assert cfg.task_size_bytes == 8000.0
    assert cfg.task_cycles == 1e6
    assert cfg.use_compute_model is True
    assert EXPERIMENTAL_TASK_SIZE_BYTES == 12000.0
    assert EXPERIMENTAL_TASK_CYCLES == 3.75e6
    assert DEFAULT.with_compute().task_size_bytes == 12000.0


def test_with_compute_keeps_replace_then_enable():
    cfg = replace(DEFAULT, task_size_bytes=16000.0, task_cycles=5e6).with_compute()
    assert cfg.task_size_bytes == 16000.0
    assert cfg.task_cycles == 5e6


def test_cli_task_size_and_cycles_enable_compute():
    args = build_parser().parse_args(["--task-size-bytes", "64000", "--task-cycles", "1e7"])
    cfg = _cfg_from_args(args)
    assert cfg.use_compute_model is True
    assert cfg.task_size_bytes == 64000.0
    assert cfg.task_cycles == 1e7
    assert cfg.b_sys == 8.8e6


def test_cli_compute_without_overrides_keeps_experimental_defaults():
    args = build_parser().parse_args(["--compute"])
    cfg = _cfg_from_args(args)
    assert cfg.task_size_bytes == EXPERIMENTAL_TASK_SIZE_BYTES
    assert cfg.task_cycles == EXPERIMENTAL_TASK_CYCLES


def test_upload_time_still_converts_bytes_to_bits():
    cfg = DEFAULT.with_compute(task_size_bytes=1000.0)
    association = np.array([[1.0]])
    processing = np.array([[1.0]])
    rates = np.array([[8000.0]])
    d = upload_times(association, processing, rates, cfg)
    # 1000 bytes → 8000 bits; r = 8000 bit/s → D = 1 s. Not S_i/r = 0.125 s.
    np.testing.assert_allclose(d, [1.0])


def test_coarse_grid_is_the_requested_product():
    pairs = coarse_pairs()
    assert len(pairs) == len(COARSE_TASK_SIZES) * len(COARSE_TASK_CYCLES)
    assert (2000.0, 2e6) in pairs
    assert (256000.0, 1e5) in pairs
    assert len(set(pair_key(s, L) for s, L in pairs)) == len(pairs)


def test_refine_grid_is_finer_and_includes_centre():
    local = refine_around(8000.0, 1e6)
    keys = {pair_key(s, L) for s, L in local}
    assert pair_key(8000.0, 1e6) in keys
    sizes = sorted({s for s, _ in local})
    gaps = np.diff(sizes)
    assert np.max(gaps) < 4000.0  # coarse step around 8k is 4000
    assert min(sizes) < 8000.0 < max(sizes)


def test_refine_pairs_from_ranked_top_n():
    ranked = [
        {"task_size_bytes": 8000.0, "task_cycles": 1e6, "score": 9.0},
        {"task_size_bytes": 16000.0, "task_cycles": 2e6, "score": 8.0},
        {"task_size_bytes": 500.0, "task_cycles": 1e5, "score": 1.0},
    ]
    got = refine_pairs(ranked, n=2)
    keys = {pair_key(s, L) for s, L in got}
    assert pair_key(8000.0, 1e6) in keys
    assert pair_key(16000.0, 2e6) in keys
    assert pair_key(500.0, 1e5) not in keys


def test_cpu_metrics_mu_and_queue_term():
    cfg = DEFAULT.with_compute(task_size_bytes=2000.0, task_cycles=2e6)
    m = cpu_metrics(cfg)
    assert m["mu"] == pytest.approx(100.0)
    # Q = (1/2) * (1 + 5*2/100) = 0.5 * 1.1 = 0.55
    assert m["queue_term"] == pytest.approx(0.55)
    assert m["rho_one_process"] == pytest.approx(0.10)


def _curve_rows(si, L, sca_mbps, kmeans_mbps, random_mbps, feas_08=1.0):
    tks = (0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0)
    rows = []
    for i, tk in enumerate(tks):
        for method, curve in (("sca", sca_mbps), ("kmeans", kmeans_mbps), ("random", random_mbps)):
            rows.append(
                {
                    "task_size_bytes": si,
                    "task_cycles": L,
                    "aodt_threshold": tk,
                    "seed": 100,
                    "method": method,
                    "sum_rate": curve[i] * 1e6,
                    "feasible": feas_08 if tk == 0.8 else 1.0,
                    "qos": 0,
                    "aodt_violations": 2.0 if tk == 0.8 else 0.0,
                    "aodt_mean": 0.7 + 0.1 * i,
                    "aodt_max": 0.8 + 0.1 * i,
                    "cpu_unstable": 0,
                    "rho_mean": 0.2,
                    "rho_max": 0.3,
                    "mu": 2e8 / L,
                    "queue_term": 0.55,
                    "runtime": 0.01,
                }
            )
    return rows


def test_score_prefers_fig9_shape_over_flat_high_rate():
    paperish = _curve_rows(
        64000.0,
        1e6,
        sca_mbps=(4.0, 4.7, 5.5, 6.3, 6.8, 7.2, 7.4),
        kmeans_mbps=(1.8, 2.3, 2.8, 3.3, 3.8, 4.2, 4.4),
        random_mbps=(0.8, 1.2, 1.7, 2.2, 2.7, 3.0, 3.2),
    )
    flat = _curve_rows(
        2000.0,
        2e6,
        sca_mbps=(4.9, 6.6, 6.9, 7.0, 7.14, 7.14, 7.14),
        kmeans_mbps=(4.6, 6.1, 6.3, 6.4, 6.43, 6.43, 6.43),
        random_mbps=(2.2, 2.6, 3.2, 3.6, 3.8, 3.81, 3.81),
        feas_08=0.0,
    )
    ranked = rank_pairs(paperish + flat)
    assert ranked[0]["task_size_bytes"] == 64000.0
    assert ranked[0]["late_frac"] > ranked[1]["late_frac"]
    assert ranked[1]["flatten_early"] == 1.0


def test_score_does_not_reward_sumrate_alone():
    high_flat = score_pair(
        _curve_rows(
            2000.0,
            2e6,
            sca_mbps=(7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1),
            kmeans_mbps=(6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4),
            random_mbps=(3.8, 3.8, 3.8, 3.8, 3.8, 3.8, 3.8),
        )
    )
    shaped = score_pair(
        _curve_rows(
            32000.0,
            1e6,
            sca_mbps=(3.8, 4.6, 5.4, 6.1, 6.7, 7.1, 7.3),
            kmeans_mbps=(1.7, 2.2, 2.7, 3.2, 3.7, 4.1, 4.3),
            random_mbps=(0.7, 1.1, 1.6, 2.1, 2.6, 2.9, 3.1),
        )
    )
    assert shaped["score"] > high_flat["score"]
    assert high_flat["flatten_early"] == 1.0
    assert PAPER_FIG9_MBPS[3.0]["td3"] == 6.0


def test_tiny_search_writes_outputs(tmp_path):
    spec = SearchSpec(
        task_sizes=(2000.0, 4000.0),
        task_cycles=(2e6,),
        thresholds=(2.8, 3.0),
        seeds=(100,),
        methods=("random", "kmeans"),
        refine_count=1,
        shortlist_count=1,
        shortlist_seeds=(100, 101),
        skip_td3=True,
    )
    args = Namespace(
        out=str(tmp_path),
        particles=20,
        iters=100,
        td3_steps=10,
        skip_td3=True,
        aodt_search_stage="all",
        resume=False,
        with_td3=False,
    )
    payload = run_aodt_parameter_search(DEFAULT, args, spec=spec)
    assert (tmp_path / "coarse_raw.csv").exists()
    assert (tmp_path / "coarse_pairs.csv").exists()
    assert (tmp_path / "table_top10.md").exists()
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "meta.json").exists()
    assert payload["coarse"]
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "S_i" in report
    assert EXPERIMENTAL_TASK_SIZE_BYTES == 12000.0
    assert EXPERIMENTAL_TASK_CYCLES == 3.75e6


def test_search_refuses_non_calibrated_radio():
    args = Namespace(
        out="results/aodt_parameter_search",
        skip_td3=True,
        aodt_search_stage="coarse",
        resume=False,
    )
    with pytest.raises(SystemExit):
        run_aodt_parameter_search(
            DEFAULT.with_radio_profile("table2"), args, spec=SearchSpec(skip_td3=True)
        )


def test_td3_shortlist_unpacks_four_tuple_solve(monkeypatch):
    from src.experiments import aodt_parameter_search as mod
    from src.solvers import td3 as td3_mod

    class FakeResult:
        sum_rate = 1.0e6
        feasible = True
        qos_violations = 0
        aodt_violations = 0
        aodt = np.array([1.0, 1.2])
        compute_available = True
        cpu_unstable = 0
        rho = np.array([0.1, 0.2, 0.1])

    monkeypatch.setattr(
        td3_mod,
        "train_td3_across_scenarios",
        lambda *a, **k: (object(), object(), object()),
    )
    monkeypatch.setattr(
        td3_mod,
        "solve_td3",
        lambda *a, **k: (np.zeros((3, 2)), FakeResult(), 0.01, None),
    )
    spec = SearchSpec(shortlist_seeds=(100,), td3_thresholds=(3.0,), skip_td3=False)
    args = Namespace(td3_steps=1)
    rows = mod.run_td3_shortlist(DEFAULT, spec, [(12000.0, 3.75e6)], args)
    assert len(rows) == 1
    assert rows[0]["method"] == "td3"
    assert rows[0]["sum_rate"] == 1.0e6
