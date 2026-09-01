"""Algebra and restore tests for the T_k=0.8 diagnostic. No production changes."""

import numpy as np
import pytest

from scripts.diagnose_aodt_08 import (
    L_WINNER,
    colocated_association,
    l_max_for_q_below,
    l_max_for_q_plus_tu2u_below,
    q_plus_tu2u,
    queue_term,
)
from src.config import DEFAULT, T_U2U
from src.scenario import generate_scenario
import src.repair as repair_mod


def test_no_l_makes_q_plus_tu2u_strictly_below_0_8():
    assert l_max_for_q_plus_tu2u_below(0.8) is None
    assert q_plus_tu2u(1e-9) == pytest.approx(0.8, abs=1e-6)
    assert q_plus_tu2u(L_WINNER) > 0.8
    assert q_plus_tu2u(1e7) > q_plus_tu2u(1e5)


def test_local_processing_allows_all_coarse_L():
    cap = l_max_for_q_below(0.8)
    assert cap is not None
    assert cap == pytest.approx(1.2e7)
    for L in (1e5, 2e6, 3.75e6, 1e7):
        assert queue_term(L) < 0.8


def test_queue_term_winner_matches_closed_form():
    # Q = 0.5 + L / 4e7
    assert queue_term(L_WINNER) == pytest.approx(0.5 + L_WINNER / 4e7)
    assert queue_term(L_WINNER) + T_U2U == pytest.approx(q_plus_tu2u(L_WINNER))


def test_colocated_patch_restores_and_clears_forwarding():
    cfg = DEFAULT.with_compute(task_size_bytes=12000.0, task_cycles=L_WINNER)
    s = generate_scenario(100, cfg)
    xy = np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]])
    _xy, a0, b0, _bw = repair_mod.complete_solution(s, xy)
    with colocated_association():
        _xy, a1, b1, _bw = repair_mod.complete_solution(s, xy)
        np.testing.assert_array_equal(a1, b1)
    _xy, a2, b2, _bw = repair_mod.complete_solution(s, xy)
    np.testing.assert_array_equal(a2, a0)
    np.testing.assert_array_equal(b2, b0)
