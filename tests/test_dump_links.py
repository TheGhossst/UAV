"""Dump-link SE audit (mechanism probes)."""

import numpy as np

from uavdt.analysis.dump_links import dump_link_se_audit, zenith_se_max_bit_per_hz
from uavdt.config import SimConfig
from uavdt.resources import nearest_association
from uavdt.scenario import generate_scenario, make_uav_xyz_m


def test_zenith_placement_has_all_good_dump_links_at_25pct_cap():
    cfg = SimConfig(b_sys_hz=10_000_000.0, max_bw_share=0.25, area_x_m=500, area_y_m=500)
    sc = generate_scenario(1, cfg)
    iot = sc.iot_xyz_m
    # Park each UAV on a distinct IoT (zenith hosts).
    xy = iot[: cfg.num_uav, :2]
    uav = make_uav_xyz_m(xy, cfg.uav_height_m)
    a = nearest_association(iot, uav)
    audit = dump_link_se_audit(sc, uav, a, se_quality_frac=0.98)
    assert audit.k_dump == 4
    assert audit.n_good_dump_links == audit.k_dump


def test_se_max_matches_zenith_hover():
    cfg = SimConfig()
    sc = generate_scenario(3, cfg)
    se_max = zenith_se_max_bit_per_hz(sc)
    assert se_max > 0.5
