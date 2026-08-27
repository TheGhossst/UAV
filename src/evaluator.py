"""Shared Problem (P) evaluator. All solvers must call this."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.aodt import average_aodt
from src.comm import link_metrics
from src.compute import cpu_unstable, offered_load, service_rate
from src.scenario import Scenario


@dataclass
class EvalResult:
    sum_rate: float
    rates: np.ndarray
    aodt: np.ndarray
    rho: np.ndarray
    qos_violations: int
    bw_excess: float
    sep_violations: int
    cpu_unstable: int
    aodt_violations: int
    association_violations: int
    processing_violations: int
    process_consistency_violations: int
    feasible: bool
    min_assoc_rate: float
    compute_available: bool
    extras: dict = field(default_factory=dict)

    @property
    def violation_count(self) -> int:
        n = (
            self.qos_violations
            + int(self.bw_excess > 1e-6)
            + self.sep_violations
            + self.cpu_unstable
            + self.aodt_violations
            + self.association_violations
            + self.processing_violations
            + self.process_consistency_violations
        )
        return n


def _hard_association(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    out[np.arange(a.shape[0]), a.argmax(axis=1)] = 1.0
    return out


def evaluate(
    scenario: Scenario,
    uav_xy: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    bandwidth: np.ndarray,
) -> EvalResult:
    cfg = scenario.cfg
    uav_xy = np.asarray(uav_xy, dtype=float).reshape(-1, 2)
    association = np.asarray(association, dtype=float)
    processing = np.asarray(processing, dtype=float)
    bandwidth = np.asarray(bandwidth, dtype=float)

    i, j = cfg.num_iot, uav_xy.shape[0]
    if association.shape != (i, j) or processing.shape != (i, j) or bandwidth.shape != (i, j):
        raise ValueError("association, processing, bandwidth must be (I, J)")

    a_hard = _hard_association(association)
    b_hard = _hard_association(processing)

    metrics = link_metrics(scenario.iot_xy, uav_xy, bandwidth, cfg)
    rates = metrics["rates"]
    assoc_rates = (a_hard * rates).sum(axis=1)
    sum_rate = float((a_hard * rates).sum())
    min_assoc_rate = float(assoc_rates.min()) if i else 0.0

    # (21) one association
    assoc_viol = int(np.any(np.abs(a_hard.sum(axis=1) - 1.0) > 1e-6))
    # (22) one processing UAV
    proc_viol = int(np.any(np.abs(b_hard.sum(axis=1) - 1.0) > 1e-6))
    # (23) same process -> same processing UAV
    cons_viol = 0
    for members in scenario.groups:
        if members.size <= 1:
            continue
        js = b_hard[members].argmax(axis=1)
        if np.unique(js).size > 1:
            cons_viol += 1

    qos_viol = int(np.sum(assoc_rates < cfg.r_min - 1e-9))
    bw_used = float(bandwidth.sum())
    bw_excess = max(0.0, bw_used - cfg.b_sys)
    # bandwidth only if associated (26)
    bw_orphan = float(np.sum(bandwidth * (a_hard < 0.5)))
    bw_excess += bw_orphan

    sep_viol = 0
    for p in range(j):
        for q in range(p + 1, j):
            d = np.linalg.norm(uav_xy[p] - uav_xy[q])
            if d < cfg.uav_min_distance - 1e-9:
                sep_viol += 1

    compute_available = bool(
        cfg.use_compute_model
        and cfg.task_cycles is not None
        and cfg.task_size_bytes is not None
    )
    mu = service_rate(cfg) if compute_available else None
    if mu is None:
        rho = np.zeros(j)
        cpu_bad = 0
        aodt = np.full(cfg.num_processes, np.nan)
        aodt_viol = 0
    else:
        rho = offered_load(b_hard, scenario.lambdas, mu)
        cpu_bad = int(np.sum(cpu_unstable(b_hard, scenario.lambdas, mu)))
        aodt = average_aodt(scenario, a_hard, b_hard, rates, mu)
        aodt_viol = int(np.sum(aodt > cfg.aodt_threshold + 1e-9))

    feasible = (
        assoc_viol == 0
        and proc_viol == 0
        and cons_viol == 0
        and qos_viol == 0
        and bw_excess <= 1e-3
        and sep_viol == 0
        and cpu_bad == 0
        and aodt_viol == 0
    )

    return EvalResult(
        sum_rate=sum_rate,
        rates=rates,
        aodt=aodt,
        rho=rho,
        qos_violations=qos_viol,
        bw_excess=bw_excess,
        sep_violations=sep_viol,
        cpu_unstable=cpu_bad,
        aodt_violations=aodt_viol,
        association_violations=assoc_viol,
        processing_violations=proc_viol,
        process_consistency_violations=cons_viol,
        feasible=feasible,
        min_assoc_rate=min_assoc_rate,
        compute_available=compute_available,
        extras={"p_los": metrics["p_los"], "distance": metrics["distance"], "l_avg": metrics["l_avg"]},
    )


def fitness(result: EvalResult, penalty: float) -> float:
    return result.sum_rate - penalty * result.violation_count
