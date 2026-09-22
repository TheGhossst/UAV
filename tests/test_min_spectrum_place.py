"""E1b keep-best Hertz placement."""

from __future__ import annotations

import pytest

from uavdt.config import SimConfig
from uavdt.min_spectrum import hertz_bound_at_layout
from uavdt.min_spectrum_place import HertzPlaceSettings, solve_min_hertz_place
from uavdt.placement.kmeans import place_kmeans
from uavdt.scenario import generate_scenario

cvxpy = pytest.importorskip("cvxpy")


def test_hertz_place_never_worse_than_kmeans_when_included():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25).with_square_area_m(500.0)
    sc = generate_scenario(1, cfg)
    km = place_kmeans(sc, 1)
    km_hz = float(hertz_bound_at_layout(sc, km)["bound_hz"])
    out = solve_min_hertz_place(
        sc,
        1,
        HertzPlaceSettings(
            n_kmeans=2,
            n_random=1,
            n_medoid_inits=2,
            enum_zenith_max=0,
            polish=False,
            verify_search=False,
        ),
    )
    assert out["kmeans_bound_hz"] == pytest.approx(km_hz, rel=0, abs=1.0)
    assert out["min_b_sys_hz"] <= km_hz + 1.0


def test_hertz_bound_matches_e1_closed_form():
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    sc = generate_scenario(2, cfg)
    uav = place_kmeans(sc, 2)
    info = hertz_bound_at_layout(sc, uav)
    assert info["binding"] in {"sum_floors", "per_link_cap"}
    assert info["bound_hz"] < 8.8e6
    assert info["bound_hz"] > 1e4
