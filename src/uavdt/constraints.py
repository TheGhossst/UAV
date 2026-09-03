"""Problem (P) constraint checks that do not require a solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.config import SimConfig
from uavdt.models import Scenario


@dataclass
class ConstraintReport:
    association_ok: bool
    processing_ok: bool
    process_consistent: bool
    qos_ok: bool
    bandwidth_support_ok: bool
    bandwidth_budget_ok: bool
    uav_separation_ok: bool
    uav_in_field_ok: bool
    queue_stable: bool
    aodt_ok: bool
    qos_violations: int
    aodt_violations: int
    sep_violations: int
    cpu_unstable_count: int
    bw_excess_hz: float

    @property
    def feasible(self) -> bool:
        return (
            self.association_ok
            and self.processing_ok
            and self.process_consistent
            and self.qos_ok
            and self.bandwidth_support_ok
            and self.bandwidth_budget_ok
            and self.uav_separation_ok
            and self.uav_in_field_ok
            and self.queue_stable
            and self.aodt_ok
        )


def pairwise_uav_distance_m(uav_xyz_m: np.ndarray) -> np.ndarray:
    j = uav_xyz_m.shape[0]
    d = np.full((j, j), np.inf)
    for p in range(j):
        for q in range(p + 1, j):
            dist = float(np.linalg.norm(uav_xyz_m[p] - uav_xyz_m[q]))
            d[p, q] = dist
            d[q, p] = dist
    return d


def check_constraints(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    bandwidth_hz: np.ndarray,
    assoc_rates_bit_per_s: np.ndarray,
    rho: np.ndarray,
    aodt_s: np.ndarray,
) -> ConstraintReport:
    cfg: SimConfig = scenario.cfg
    a = association > 0.5
    b = processing > 0.5

    assoc_ok = bool(np.all(np.abs(a.sum(axis=1) - 1.0) < 1e-9))
    proc_ok = bool(np.all(np.abs(b.sum(axis=1) - 1.0) < 1e-9))

    consistent = True
    for proc in scenario.processes:
        members = proc.iot_indices
        if members.size <= 1:
            continue
        js = np.argmax(processing[members], axis=1)
        if np.unique(js).size > 1:
            consistent = False

    qos_viol = int(np.sum(assoc_rates_bit_per_s < cfg.r_min_bit_per_s - 1e-9))
    bw_on_unassoc = float(np.sum(bandwidth_hz * (~a)))
    bw_excess = max(0.0, float(np.sum(bandwidth_hz)) - cfg.b_sys_hz)
    bw_support_ok = bw_on_unassoc <= 1e-9
    bw_budget_ok = bw_excess <= 1e-3

    sep_viol = 0
    dists = pairwise_uav_distance_m(uav_xyz_m)
    j = uav_xyz_m.shape[0]
    for p in range(j):
        for q in range(p + 1, j):
            if dists[p, q] < cfg.uav_min_separation_m - 1e-9:
                sep_viol += 1

    in_field = bool(
        np.all(uav_xyz_m[:, 0] >= -1e-9)
        and np.all(uav_xyz_m[:, 0] <= cfg.area_x_m + 1e-9)
        and np.all(uav_xyz_m[:, 1] >= -1e-9)
        and np.all(uav_xyz_m[:, 1] <= cfg.area_y_m + 1e-9)
        and np.allclose(uav_xyz_m[:, 2], cfg.uav_height_m)
    )

    cpu_bad = int(np.sum(rho >= 1.0 - 1e-12))
    aodt_viol = int(np.sum(aodt_s > cfg.aodt_threshold_s + 1e-9))

    return ConstraintReport(
        association_ok=assoc_ok,
        processing_ok=proc_ok,
        process_consistent=consistent,
        qos_ok=qos_viol == 0,
        bandwidth_support_ok=bw_support_ok,
        bandwidth_budget_ok=bw_budget_ok,
        uav_separation_ok=sep_viol == 0,
        uav_in_field_ok=in_field,
        queue_stable=cpu_bad == 0,
        aodt_ok=aodt_viol == 0,
        qos_violations=qos_viol,
        aodt_violations=aodt_viol,
        sep_violations=sep_viol,
        cpu_unstable_count=cpu_bad,
        bw_excess_hz=bw_excess + bw_on_unassoc,
    )
