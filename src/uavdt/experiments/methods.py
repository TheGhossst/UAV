"""Run one method on one scenario. All methods share evaluate().

SCA code is frozen: this module only *calls* solve_sca.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uavdt.evaluator import EvalResult, evaluate
from uavdt.models import Allocation, Scenario
from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.pso import PSOSettings, place_pso
from uavdt.placement.random import place_random
from uavdt.resources import nearest_association, process_consistent_processing
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
from uavdt.sca.settings import SCASettings


METHODS = ("random", "kmeans", "pso", "sca")


@dataclass
class MethodRun:
    method: str
    seed: int
    uav_xyz_m: np.ndarray
    allocation: Allocation
    true_eval: EvalResult
    diagnostics: dict = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return bool(self.true_eval.feasible)

    @property
    def sum_rate_bit_per_s(self) -> float:
        return float(self.true_eval.sum_rate_bit_per_s)

    @property
    def sum_rate_mbps(self) -> float:
        return float(self.true_eval.sum_rate_mbps)


def _alloc_at_positions(scenario: Scenario, uav: np.ndarray) -> Allocation:
    """Nearest a_ij, process-consistent b_ij, exact frozen-q bandwidth LP.

    IMPLEMENTATION CHOICE: placement baselines get the same B LP as SCA's
    bandwidth step so the comparison is positions, not a hidden B policy.
    """
    a = nearest_association(scenario.iot_xyz_m, uav)
    b = process_consistent_processing(scenario, a)
    res = solve_bandwidth_at_fixed_q(
        scenario, uav, a, b, SCASettings(solver=None)
    )
    if res.infeasible:
        from uavdt.resources import equal_share_bandwidth_hz

        bw = equal_share_bandwidth_hz(a, scenario.cfg)
        return Allocation(a, b, bw)
    return Allocation(a, b, res.bandwidth_hz)


def run_method(
    scenario: Scenario,
    method: str,
    seed: int,
    *,
    sca_settings: SCASettings | None = None,
    pso_settings: PSOSettings | None = None,
) -> MethodRun:
    name = method.lower().strip()
    if name not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    if name == "sca":
        from uavdt.sca.algorithm import solve_sca

        settings = sca_settings or SCASettings()
        result = solve_sca(scenario, seed, settings=settings)
        return MethodRun(
            method="sca",
            seed=seed,
            uav_xyz_m=result.uav_xyz_m,
            allocation=result.allocation,
            true_eval=result.true_eval,
            diagnostics={
                "stop_reason": result.diagnostics.get("stop_reason"),
                "accepted_steps": result.diagnostics.get("accepted_steps"),
                "solver_backend": result.diagnostics.get("solver_backend"),
                "solver_status": result.solver_status,
                "se_max_abs_diff": result.diagnostics.get("se_max_abs_diff"),
            },
        )

    if name == "random":
        uav = place_random(scenario.cfg.num_uav, seed, scenario.cfg)
    elif name == "kmeans":
        uav = place_kmeans(scenario, seed)
    else:
        uav = place_pso(scenario, seed, settings=pso_settings)

    alloc = _alloc_at_positions(scenario, uav)
    ev = evaluate(scenario, uav, alloc)
    return MethodRun(
        method=name,
        seed=seed,
        uav_xyz_m=uav,
        allocation=alloc,
        true_eval=ev,
        diagnostics={"bandwidth": "frozen_q_lp"},
    )
