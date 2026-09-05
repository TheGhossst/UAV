"""Association a_ij, processing b_ij, bandwidth B_ij.

Problem (P) treats these as decision variables. Helpers below are
evaluation conventions for placement-only runs, not paper algorithms.
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np

from uavdt.computation import offered_load, queue_unstable
from uavdt.config import SimConfig
from uavdt.models import Allocation, Scenario


def nearest_association(iot_xyz_m: np.ndarray, uav_xyz_m: np.ndarray) -> np.ndarray:
    """Each IoT associated with the nearest UAV in 3D. Shape (I, J)."""
    delta = iot_xyz_m[:, None, :] - uav_xyz_m[None, :, :]
    dist = np.sqrt(np.sum(delta ** 2, axis=-1))
    a = np.zeros_like(dist)
    a[np.arange(dist.shape[0]), np.argmin(dist, axis=1)] = 1.0
    return a


def process_consistent_processing(
    scenario: Scenario,
    association: np.ndarray,
) -> np.ndarray:
    """Constraint (23): one processing UAV per process, majority of a_ij."""
    i, j = association.shape
    b = np.zeros((i, j), dtype=float)
    for proc in scenario.processes:
        members = proc.iot_indices
        if members.size == 0:
            continue
        votes = association[members].sum(axis=0)
        j_star = int(np.argmax(votes))
        b[members, j_star] = 1.0
    return b


# Campaign K=2 is at most J^2 maps (25 at J=5). Full enumeration of J^K is
# exponential in K; stop after this many unique maps and keep the best
# feasible one seen (or majority if none). Fallback, not a completeness claim.
_MAX_PROCESS_UAV_MAPS = 4096


def cpu_stable_processing(
    scenario: Scenario,
    association: np.ndarray,
) -> np.ndarray:
    """Constraint (23) processing that also satisfies (24) when a map exists.

    Default is majority-of-association (same as process_consistent_processing).
    If that assignment is queue-unstable, try process→UAV maps (injections
    first when J >= K, then all maps if J^K is small) and keep the feasible
    one closest to majority vote. When majority is already stable, the
    assignment is unchanged so I=10 / default-μ campaign points do not move.
    """
    majority = process_consistent_processing(scenario, association)
    mu = float(scenario.cfg.service_rate_per_s)
    lam = scenario.lambdas_per_s
    if not np.any(queue_unstable(majority, lam, mu)):
        return majority
    repaired = _stable_processing_among_maps(scenario, association)
    if repaired is not None:
        return repaired
    return majority


def _iter_process_uav_maps(k: int, j: int):
    """Yield unique process→UAV assignments, injections first.

    Stops at _MAX_PROCESS_UAV_MAPS. If J^K is larger than that, only
    injections are attempted (and if J < K, nothing). Campaign K=2 never
    hits the cap.
    """
    seen: set[tuple[int, ...]] = set()
    n = 0
    if j >= k:
        for assign in permutations(range(j), k):
            if n >= _MAX_PROCESS_UAV_MAPS:
                return
            if assign in seen:
                continue
            seen.add(assign)
            n += 1
            yield assign
    n_all = j ** k
    if n_all > _MAX_PROCESS_UAV_MAPS:
        return
    for assign in product(range(j), repeat=k):
        if n >= _MAX_PROCESS_UAV_MAPS:
            return
        if assign in seen:
            continue
        seen.add(assign)
        n += 1
        yield assign


def _stable_processing_among_maps(
    scenario: Scenario,
    association: np.ndarray,
) -> np.ndarray | None:
    processes = scenario.processes
    k = len(processes)
    j = int(association.shape[1])
    if k == 0 or j == 0:
        return None
    votes = [association[p.iot_indices].sum(axis=0) for p in processes]
    mu = float(scenario.cfg.service_rate_per_s)
    lam = scenario.lambdas_per_s
    best: np.ndarray | None = None
    best_key: tuple[float, float] | None = None
    for assign in _iter_process_uav_maps(k, j):
        b = np.zeros(association.shape, dtype=float)
        for pk, j_star in enumerate(assign):
            members = processes[pk].iot_indices
            if members.size:
                b[members, int(j_star)] = 1.0
        if np.any(queue_unstable(b, lam, mu)):
            continue
        vote_score = float(sum(votes[pk][int(j_star)] for pk, j_star in enumerate(assign)))
        rho = offered_load(b, lam, mu)
        key = (vote_score, -float(np.max(rho)))
        if best_key is None or key > best_key:
            best_key = key
            best = b
    return best


def equal_share_bandwidth_hz(association: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Split B_sys equally across associated links.

    Not specified by the paper. Used when B_ij is not otherwise given.
    Unassociated links get 0 (constraint (26)).
    """
    a = association > 0.5
    n = int(a.sum())
    b = np.zeros(association.shape, dtype=float)
    if n == 0:
        return b
    b[a] = cfg.b_sys_hz / float(n)
    return b


def allocation_from_positions(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
) -> Allocation:
    """Nearest association, CPU-stable processing, equal B_ij."""
    a = nearest_association(scenario.iot_xyz_m, uav_xyz_m)
    b = cpu_stable_processing(scenario, a)
    bw = equal_share_bandwidth_hz(a, scenario.cfg)
    return Allocation(association=a, processing=b, bandwidth_hz=bw)
