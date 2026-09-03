"""Single evaluation of a deployment against the paper model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uavdt.aodt import average_aodt_s, upload_times_s
from uavdt.channel import link_metrics
from uavdt.computation import (
    arrival_rate_per_uav,
    mean_service_time_s,
    offered_load,
    queue_unstable,
    service_rate_per_s,
)
from uavdt.constraints import ConstraintReport, check_constraints
from uavdt.models import Allocation, Scenario
from uavdt.resources import allocation_from_positions


@dataclass
class EvalResult:
    sum_rate_bit_per_s: float
    assoc_rates_bit_per_s: np.ndarray
    rates_bit_per_s: np.ndarray
    distances_m: np.ndarray
    p_los: np.ndarray
    l_avg_db: np.ndarray
    received_power_w: np.ndarray
    snr: np.ndarray
    upload_times_s: np.ndarray
    mu_per_s: float
    mean_service_time_s: float
    lambda_total_per_s: np.ndarray
    rho: np.ndarray
    aodt_s: np.ndarray
    aodt_satisfied: np.ndarray
    constraints: ConstraintReport
    extras: dict = field(default_factory=dict)

    @property
    def sum_rate_mbps(self) -> float:
        return self.sum_rate_bit_per_s / 1.0e6

    @property
    def feasible(self) -> bool:
        return self.constraints.feasible


def evaluate(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    allocation: Allocation | None = None,
) -> EvalResult:
    """Compute channel, rates, queues, AoDT, and constraints.

    If `allocation` is omitted, nearest-UAV association, process-consistent
    processing, and equal bandwidth share are used (evaluation convention).
    """
    cfg = scenario.cfg
    uav_xyz_m = np.asarray(uav_xyz_m, dtype=float)
    if uav_xyz_m.ndim != 2 or uav_xyz_m.shape[1] != 3:
        raise ValueError("uav_xyz_m must have shape (J, 3)")
    if allocation is None:
        allocation = allocation_from_positions(scenario, uav_xyz_m)

    a = allocation.hard_association()
    b = allocation.hard_processing()
    bw = np.asarray(allocation.bandwidth_hz, dtype=float)

    i, j = cfg.num_iot, uav_xyz_m.shape[0]
    if a.shape != (i, j) or b.shape != (i, j) or bw.shape != (i, j):
        raise ValueError("allocation matrices must be (I, J)")

    metrics = link_metrics(scenario.iot_xyz_m, uav_xyz_m, bw, cfg)
    rates = metrics["rates_bit_per_s"]
    assoc_rates = (a * rates).sum(axis=1)
    sum_rate = float((a * rates).sum())

    mu = service_rate_per_s(cfg)
    mu_vec = np.full(j, mu, dtype=float)
    lam_j = arrival_rate_per_uav(b, scenario.lambdas_per_s)
    rho = offered_load(b, scenario.lambdas_per_s, mu)
    stable = ~queue_unstable(b, scenario.lambdas_per_s, mu)

    d_i = upload_times_s(a, b, rates, cfg)
    aodt = average_aodt_s(scenario, a, b, rates, mu_vec, uav_stable=stable)
    aodt_ok = aodt <= cfg.aodt_threshold_s + 1e-9

    report = check_constraints(
        scenario,
        uav_xyz_m,
        a,
        b,
        bw,
        assoc_rates,
        rho,
        aodt,
    )

    return EvalResult(
        sum_rate_bit_per_s=sum_rate,
        assoc_rates_bit_per_s=assoc_rates,
        rates_bit_per_s=rates,
        distances_m=metrics["distance_m"],
        p_los=metrics["p_los"],
        l_avg_db=metrics["l_avg_db"],
        received_power_w=metrics["received_power_w"],
        snr=metrics["snr"],
        upload_times_s=d_i,
        mu_per_s=mu,
        mean_service_time_s=mean_service_time_s(cfg),
        lambda_total_per_s=lam_j,
        rho=rho,
        aodt_s=aodt,
        aodt_satisfied=aodt_ok,
        constraints=report,
        extras={
            "elevation_rad": metrics["elevation_rad"],
            "l_los_db": metrics["l_los_db"],
            "l_nlos_db": metrics["l_nlos_db"],
            "sinr": metrics["sinr"],
            "association": a,
            "processing": b,
            "bandwidth_hz": bw,
        },
    )
