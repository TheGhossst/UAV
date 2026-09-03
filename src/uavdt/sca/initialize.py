"""Feasible-ish SCA starting point. Not a paper algorithm."""

from __future__ import annotations

import numpy as np

from uavdt.computation import offered_load, queue_unstable, service_rate_per_s
from uavdt.config import SimConfig
from uavdt.models import Allocation, Scenario
from uavdt.placement.kmeans import place_kmeans
from uavdt.resources import nearest_association, process_consistent_processing
from uavdt.sca.linearize import spectral_efficiency


def qos_floor_bandwidth(
    association: np.ndarray,
    se: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Initial B_ij: R_min/SE floors, leftover to highest-SE associated links.

    Leftover is poured onto associated links in decreasing-SE order, each
    up to link_bandwidth_cap_hz. IMPLEMENTATION CHOICE for a physically
    evaluable start. Not used inside the convex program.
    """
    a = association > 0.5
    b = np.zeros_like(se)
    if not np.any(a):
        return b
    se_safe = np.maximum(se, 1e-15)
    floors = np.zeros_like(se)
    floors[a] = cfg.r_min_bit_per_s / se_safe[a]
    cap = cfg.link_bandwidth_cap_hz
    total_floor = float(floors.sum())
    if total_floor <= cfg.b_sys_hz:
        b[:, :] = floors
        leftover = cfg.b_sys_hz - total_floor
        se_masked = np.where(a, se, -np.inf)
        order = np.argsort(-se_masked, axis=None)
        for flat in order:
            i_star, j_star = np.unravel_index(int(flat), se.shape)
            if not a[i_star, j_star]:
                continue
            room = cap - float(b[i_star, j_star])
            if room <= 1e-12:
                continue
            take = min(leftover, room)
            b[i_star, j_star] += take
            leftover -= take
            if leftover <= 1e-9:
                break
    else:
        b[:, :] = floors * (cfg.b_sys_hz / total_floor)
    b[~a] = 0.0
    return b


def initialize_sca(scenario: Scenario, seed: int) -> tuple[np.ndarray, Allocation]:
    """K-means UAVs (seeded), nearest association, process-consistent processing.

    IMPLEMENTATION CHOICE: Algorithm 1 does not specify the initial geometry.
    K-means is the paper's named placement initialization baseline (§VII).
    """
    cfg = scenario.cfg
    uav = place_kmeans(scenario, seed)
    assoc = nearest_association(scenario.iot_xyz_m, uav)
    proc = process_consistent_processing(scenario, assoc)
    mu = service_rate_per_s(cfg)
    if np.any(queue_unstable(proc, scenario.lambdas_per_s, mu)):
        rho = offered_load(proc, scenario.lambdas_per_s, mu)
        raise RuntimeError(
            "initial processing assignment violates CPU stability (24): "
            f"rho={rho}, mu={mu}"
        )
    se = spectral_efficiency(scenario.iot_xyz_m, uav, cfg)
    bw = qos_floor_bandwidth(assoc, se, cfg)
    alloc = Allocation(association=assoc, processing=proc, bandwidth_hz=bw)
    return uav, alloc
