"""Paired-difference / Wilcoxon helpers used by scripts/paired_winrate.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paired_winrate import (
    align_by_seed,
    analyze_campaign,
    betainc_reg,
    compare_pair,
    is_repeated_default,
    quote_line,
    student_t_sf_two_sided,
    wilcoxon_exact_p,
    wilcoxon_signed_rank,
    bh_qvalues,
)


def test_wilcoxon_all_positive_n5():
    # All ranks 1..5 positive => W+=15, two-sided p = 2/32 = 0.0625
    wx = wilcoxon_signed_rank(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert wx["method"] == "exact"
    assert wx["W_plus"] == 15.0
    assert wx["p_two_sided"] == pytest.approx(2.0 / 32.0)
    assert wx["p_greater"] == pytest.approx(1.0 / 32.0)


def test_wilcoxon_exact_symmetric():
    assert wilcoxon_exact_p(0, 4, "two-sided") == pytest.approx(2.0 / 16.0)
    assert wilcoxon_exact_p(10, 4, "two-sided") == pytest.approx(2.0 / 16.0)


def test_student_t_two_sided_known():
    # P(|T_inf| > 1.96) ~ 0.05; df=1e6 is close
    p = student_t_sf_two_sided(1.95996398454, 1.0e6)
    assert p == pytest.approx(0.05, abs=1e-3)
    # I_0.5(a,a) = 0.5
    assert betainc_reg(2.0, 2.0, 0.5) == pytest.approx(0.5, abs=1e-10)


def test_align_and_compare_pair():
    champ = {
        "seeds": [1, 2, 3, 4],
        "per_seed_Mbps": [10.0, 10.0, 10.0, 10.0],
        "per_seed_feasible": [True, True, True, False],
    }
    base = {
        "seeds": [4, 3, 2, 1],
        "per_seed_Mbps": [9.0, 9.5, 10.0, 9.8],
        "per_seed_feasible": [True, True, True, True],
    }
    seeds, c, b, joint = align_by_seed(champ, base)
    assert list(seeds) == [1, 2, 3, 4]
    np.testing.assert_allclose(c, [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(b, [9.8, 10.0, 9.5, 9.0])
    assert list(joint) == [True, True, True, False]

    stats = compare_pair(champ, base)
    assert stats["n_seeds"] == 4
    assert stats["wins"] == 3
    assert stats["ties"] == 1
    assert stats["losses"] == 0
    assert stats["mean_delta_Mbps"] == pytest.approx(0.425)

    feas = compare_pair(champ, base, jointly_feasible_only=True)
    assert feas["n_seeds"] == 3
    assert feas["wins"] == 2


def test_quote_and_repeated_default_flag():
    stats = {
        "mean_delta_Mbps": 0.014,
        "std_delta_Mbps": 0.018,
        "n_seeds": 20,
        "wins": 16,
        "wilcoxon_p_two_sided": 0.002,
    }
    q = quote_line("sca", "random", stats)
    assert q.startswith("SCA wins by 0.014+/-0.018 Mbps vs random")
    assert "16/20 seeds" in q
    assert is_repeated_default({"axis": "lambda", "x": 2.0})
    assert not is_repeated_default({"axis": "uavs", "x": 3.0})


def test_analyze_campaign_tiny():
    payload = {
        "n_runs": 2,
        "b_sys_hz": 8.8e6,
        "max_bw_share": 0.25,
        "methods": ["random", "sca"],
        "points": [
            {
                "axis": "uavs",
                "x_name": "num_uav",
                "x": 3.0,
                "label": "J=3",
                "by_method": {
                    "sca": {
                        "seeds": [1, 2],
                        "per_seed_Mbps": [9.0, 9.1],
                        "per_seed_feasible": [True, True],
                    },
                    "random": {
                        "seeds": [1, 2],
                        "per_seed_Mbps": [8.8, 9.0],
                        "per_seed_feasible": [True, True],
                    },
                },
            }
        ],
    }
    out = analyze_campaign(payload)
    assert len(out["rows"]) == 1
    assert out["rows"][0]["wins"] == 2
    assert not out["rows"][0]["repeated_default"]
    assert math.isfinite(out["rows"][0]["wilcoxon_p_two_sided"])


def _tiny_method(rates):
    n = len(rates)
    return {
        "seeds": list(range(1, n + 1)),
        "per_seed_Mbps": list(rates),
        "per_seed_feasible": [True] * n,
    }


def test_marks_identical_lambda_as_duplicate_of_j3():
    payload = {
        "n_runs": 2,
        "b_sys_hz": 8.8e6,
        "max_bw_share": 0.25,
        "methods": ["random", "sca"],
        "points": [
            {
                "axis": "uavs",
                "x_name": "num_uav",
                "x": 3.0,
                "label": "J=3",
                "by_method": {
                    "sca": _tiny_method([9.0, 9.1]),
                    "random": _tiny_method([8.8, 9.0]),
                },
            },
            {
                "axis": "lambda",
                "x_name": "lambda_i_per_s",
                "x": 1.0,
                "label": "lam=1",
                "by_method": {
                    "sca": _tiny_method([9.0, 9.1]),
                    "random": _tiny_method([8.8, 9.0]),
                },
            },
        ],
    }
    rows = analyze_campaign(payload)["rows"]
    lam = next(r for r in rows if r["axis"] == "lambda")
    uav = next(r for r in rows if r["axis"] == "uavs")
    assert lam["identical_to_uavs_j3"]
    assert lam["repeated_default"]
    assert not uav["identical_to_uavs_j3"]


def test_bh_qvalues_monotone_and_bounded():
    q = bh_qvalues([0.001, 0.04, 0.5])
    assert q[0] <= q[1] <= q[2]
    assert all(0.0 <= x <= 1.0 for x in q)
    assert q[0] == pytest.approx(0.003)
    # two tiny p-values in a family of 2: q = 2*p/rank
    q2 = bh_qvalues([0.01, 0.01])
    assert q2[0] == pytest.approx(0.01)
    assert q2[1] == pytest.approx(0.01)
