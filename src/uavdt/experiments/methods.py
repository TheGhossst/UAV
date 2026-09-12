"""Run one method on one scenario. All methods share evaluate().

SCA code is frozen: this module only *calls* solve_sca.
SCA-joint is a methodology probe (method=\"sca_joint\").
Multi-start SCA is opt-in (method=\"sca_multistart\"): keep-best extra
inits of frozen SCA. TD3 is opt-in (method=\"td3\"); knobs are
TD3Settings, not SimConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from uavdt.evaluator import EvalResult, evaluate
from uavdt.models import Allocation, Scenario
from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.pso import PSOSettings, place_pso
from uavdt.placement.random import place_random
from uavdt.resources import cpu_stable_processing, nearest_association
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
from uavdt.sca.settings import SCASettings
from uavdt.sca_multistart import MultiStartSettings
from uavdt.td3.settings import TD3Settings


# Headline campaign methods. sca_joint, sca_multistart, and td3 are opt-in.
METHODS = ("random", "kmeans", "pso", "sca")
KNOWN_METHODS = METHODS + ("sca_joint", "sca_multistart", "td3")


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
    """Nearest a_ij, CPU-stable b_ij, exact frozen-q bandwidth LP.

    IMPLEMENTATION CHOICE: placement baselines get the same B LP as SCA's
    bandwidth step so the comparison is positions, not a hidden B policy.
    cpu_stable_processing is the same helper SCA uses at init, so I=28/32
    / low-CPU (24) collisions are repaired for random, k-means, and PSO
    too — not only SCA.
    """
    a = nearest_association(scenario.iot_xyz_m, uav)
    b = cpu_stable_processing(scenario, a)
    res = solve_bandwidth_at_fixed_q(
        scenario, uav, a, b, SCASettings(solver=None)
    )
    if res.infeasible:
        from uavdt.resources import equal_share_bandwidth_hz

        bw = equal_share_bandwidth_hz(a, scenario.cfg)
        return Allocation(a, b, bw)
    return Allocation(a, b, res.bandwidth_hz)


def _run_sca_family(
    scenario: Scenario,
    method: str,
    seed: int,
    sca_settings: SCASettings | None,
) -> MethodRun:
    settings = sca_settings or SCASettings()
    if method == "sca":
        from uavdt.sca.algorithm import solve_sca as solver
    else:
        from uavdt.sca_joint import solve_sca_joint as solver
    t0 = perf_counter()
    try:
        result = solver(scenario, seed, settings=settings)
    except RuntimeError as exc:
        # Settled L lowers mu; large I can violate (24) at k-means init.
        if "CPU stability" not in str(exc):
            raise
        uav = place_kmeans(scenario, seed)
        alloc = _alloc_at_positions(scenario, uav)
        ev = evaluate(scenario, uav, alloc)
        return MethodRun(
            method=method,
            seed=seed,
            uav_xyz_m=uav,
            allocation=alloc,
            true_eval=ev,
            diagnostics={
                "method": method,
                "stop_reason": "init_cpu_unstable",
                "accepted_steps": 0,
                "n_iterations": 0,
                "solver_backend": "skipped",
                "solver_status": "init_cpu_unstable",
                "wall_clock_s": perf_counter() - t0,
            },
        )
    elapsed = perf_counter() - t0
    diag = {
        "method": method,
        "stop_reason": result.diagnostics.get("stop_reason"),
        "accepted_steps": result.diagnostics.get("accepted_steps"),
        "n_iterations": result.n_iterations,
        "solver_backend": result.diagnostics.get("solver_backend"),
        "solver_status": result.solver_status,
        "se_max_abs_diff": result.diagnostics.get("se_max_abs_diff"),
        "wall_clock_s": elapsed,
        "association_init_equals_final": result.diagnostics.get(
            "association_init_equals_final"
        ),
        "processing_init_equals_final": result.diagnostics.get(
            "processing_init_equals_final"
        ),
        "rematch_attempts": result.diagnostics.get("rematch_attempts"),
        "rematch_accepted": result.diagnostics.get("rematch_accepted"),
        "rematch_rejected": result.diagnostics.get("rematch_rejected"),
        "rematch_kinds": result.diagnostics.get("rematch_kinds"),
        "process_cohesive_candidate": result.diagnostics.get(
            "process_cohesive_candidate"
        ),
        "process_cohesive_a": result.diagnostics.get("process_cohesive_a"),
        "n_forwarding": result.diagnostics.get("n_forwarding"),
    }
    return MethodRun(
        method=method,
        seed=seed,
        uav_xyz_m=result.uav_xyz_m,
        allocation=result.allocation,
        true_eval=result.true_eval,
        diagnostics=diag,
    )


def _run_sca_multistart(
    scenario: Scenario,
    seed: int,
    sca_settings: SCASettings | None,
    multistart_settings: MultiStartSettings | None,
) -> MethodRun:
    from uavdt.sca_multistart import solve_sca_multistart

    t0 = perf_counter()
    try:
        result = solve_sca_multistart(
            scenario,
            seed,
            settings=sca_settings,
            multistart=multistart_settings,
        )
    except RuntimeError as exc:
        if "CPU stability" not in str(exc) and "every start failed" not in str(exc):
            raise
        uav = place_kmeans(scenario, seed)
        alloc = _alloc_at_positions(scenario, uav)
        ev = evaluate(scenario, uav, alloc)
        return MethodRun(
            method="sca_multistart",
            seed=seed,
            uav_xyz_m=uav,
            allocation=alloc,
            true_eval=ev,
            diagnostics={
                "method": "sca_multistart",
                "stop_reason": "init_cpu_unstable",
                "accepted_steps": 0,
                "n_iterations": 0,
                "solver_backend": "skipped",
                "solver_status": "init_cpu_unstable",
                "wall_clock_s": perf_counter() - t0,
            },
        )
    elapsed = perf_counter() - t0
    diag = dict(result.diagnostics)
    diag.setdefault("wall_clock_s", elapsed)
    diag["method"] = "sca_multistart"
    return MethodRun(
        method="sca_multistart",
        seed=seed,
        uav_xyz_m=result.uav_xyz_m,
        allocation=result.allocation,
        true_eval=result.true_eval,
        diagnostics=diag,
    )


def _run_td3(
    scenario: Scenario,
    seed: int,
    td3_settings: TD3Settings | None,
) -> MethodRun:
    from uavdt.td3.solve import solve_td3

    t0 = perf_counter()
    result = solve_td3(scenario, seed, settings=td3_settings)
    diag = dict(result.diagnostics)
    diag.setdefault("wall_clock_s", perf_counter() - t0)
    return MethodRun(
        method="td3",
        seed=seed,
        uav_xyz_m=result.uav_xyz_m,
        allocation=result.allocation,
        true_eval=result.true_eval,
        diagnostics=diag,
    )


def run_method(
    scenario: Scenario,
    method: str,
    seed: int,
    *,
    sca_settings: SCASettings | None = None,
    pso_settings: PSOSettings | None = None,
    td3_settings: TD3Settings | None = None,
    multistart_settings: MultiStartSettings | None = None,
    uav_xyz_m: np.ndarray | None = None,
) -> MethodRun:
    name = method.lower().strip()
    if name not in KNOWN_METHODS:
        raise ValueError(f"method must be one of {KNOWN_METHODS}, got {method!r}")

    if name in {"sca", "sca_joint"}:
        return _run_sca_family(scenario, name, seed, sca_settings)

    if name == "sca_multistart":
        return _run_sca_multistart(
            scenario, seed, sca_settings, multistart_settings
        )

    if name == "td3":
        return _run_td3(scenario, seed, td3_settings)

    if name == "random":
        if uav_xyz_m is None:
            uav = place_random(scenario.cfg.num_uav, seed, scenario.cfg)
        else:
            uav = np.asarray(uav_xyz_m, dtype=float)
            expected = (scenario.cfg.num_uav, 3)
            if uav.shape != expected:
                raise ValueError(f"uav_xyz_m must have shape {expected}")
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
