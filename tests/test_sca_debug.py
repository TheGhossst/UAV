"""Sequential SCA-style debug command."""

from __future__ import annotations

from pathlib import Path

import pytest

from uavdt.config import SimConfig
from uavdt.sca.debug import run_sca_seq_debug
from uavdt.sca.initialize import initialize_sca
from uavdt.scenario import generate_scenario

cvxpy = pytest.importorskip("cvxpy")


def test_seq_debug_writes_machine_readable_logs(tmp_path: Path):
    cfg = SimConfig(b_sys_hz=2.4e6)
    payload = run_sca_seq_debug(
        1,
        cfg,
        max_iterations=3,
        step_size_m=1.0,
        out_json=str(tmp_path / "sca_seq_debug.json"),
    )
    assert payload["true_gate_ok"]
    assert payload["reported_equals_evaluator"]
    assert payload["association"]
    assert payload["processing"]
    assert Path(payload["_csv"]).exists()


def test_init_matches_initialize_helper():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(1, cfg)
    uav, alloc = initialize_sca(sc, 1)
    assert uav.shape == (3, 3)
    assert alloc.hard_association().shape == (10, 3)
