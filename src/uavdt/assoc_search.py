"""Legal association search at frozen UAV geometry (Experiment C).

Does not edit `uavdt.sca`. Algorithm 1 still freezes a,b; this module
searches a at a fixed q, then the caller may `solve_sca` with the winner.

Legal set:
  (21) exclusive a_ij
  (23) one processing UAV per process
  (24) CPU stability
  cohesive N_k on every process with T_k - Q_k < T_u2u

Score is the same (P) number as residual-on-SCA: frozen-q B LP + evaluate().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uavdt.aodt import queueing_term_s
from uavdt.computation import queue_unstable
from uavdt.models import Scenario
from uavdt.placement.cmaes import feasible_lp_score
from uavdt.resources import cpu_stable_processing
from uavdt.sca_joint.rematch import (
    best_se_association,
    n_forwarding,
    processing_process_consistent,
)


def forwarding_slack_s(scenario: Scenario) -> np.ndarray:
    """s_k = T_k - Q_k - T_u2u. Negative ⇒ process k cannot forward."""
    cfg = scenario.cfg
    mu = float(cfg.service_rate_per_s)
    t_k = float(cfg.aodt_threshold_s)
    t_fwd = float(cfg.t_u2u_s)
    out = np.empty(cfg.num_processes, dtype=float)
    for proc in scenario.processes:
        q_k = queueing_term_s(scenario.lambdas_per_s[proc.iot_indices], mu)
        out[proc.process_id] = t_k - q_k - t_fwd
    return out


def must_group_mask(scenario: Scenario) -> np.ndarray:
    """True where grouping is required: T_k - Q_k < T_u2u."""
    cfg = scenario.cfg
    mu = float(cfg.service_rate_per_s)
    t_k = float(cfg.aodt_threshold_s)
    t_fwd = float(cfg.t_u2u_s)
    out = np.zeros(cfg.num_processes, dtype=bool)
    for proc in scenario.processes:
        q_k = queueing_term_s(scenario.lambdas_per_s[proc.iot_indices], mu)
        out[proc.process_id] = (t_k - q_k) < t_fwd
    return out


def _exclusive_association(association: np.ndarray) -> bool:
    a = np.asarray(association, dtype=float) > 0.5
    if a.ndim != 2 or a.size == 0:
        return False
    return bool(np.all(a.sum(axis=1) == 1))


def _assoc_key(association: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.argmax(np.asarray(association, dtype=float), axis=1))


def hamming_association(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.sum(np.argmax(left, axis=1) != np.argmax(right, axis=1)))


def cohesive_on_required(scenario: Scenario, association: np.ndarray) -> bool:
    mask = must_group_mask(scenario)
    if not np.any(mask):
        return True
    for proc in scenario.processes:
        if not mask[proc.process_id]:
            continue
        members = proc.iot_indices
        if members.size <= 1:
            continue
        hosts = np.unique(np.argmax(association[members], axis=1))
        if hosts.size > 1:
            return False
    return True


def is_legal_association(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray | None = None,
) -> tuple[bool, str]:
    a = np.asarray(association, dtype=float)
    if not _exclusive_association(a):
        return False, "not_exclusive"
    if not cohesive_on_required(scenario, a):
        return False, "must_group_split"
    b = (
        cpu_stable_processing(scenario, a)
        if processing is None
        else np.asarray(processing, dtype=float)
    )
    if not processing_process_consistent(scenario, b):
        return False, "process_inconsistent"
    mu = float(scenario.cfg.service_rate_per_s)
    if np.any(queue_unstable(b, scenario.lambdas_per_s, mu)):
        return False, "cpu_unstable"
    return True, "ok"


@dataclass
class ScoredMap:
    association: np.ndarray
    processing: np.ndarray
    reward: float
    sum_rate_Mbps: float
    feasible: bool
    n_forwarding: int
    hamming_from_start: int
    reason: str = "ok"


@dataclass
class AssociationSearchResult:
    incumbent: ScoredMap
    n_lp_evals: int
    n_accepted: int
    n_neighbors_scored: int
    n_random_scored: int
    n_illegal: int
    rounds: int
    start_key: tuple[int, ...]
    best_key: tuple[int, ...]
    cache_size: int
    history: list[float] = field(default_factory=list)


def score_association(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    *,
    start_association: np.ndarray | None = None,
) -> tuple[ScoredMap | None, str]:
    a = np.asarray(association, dtype=float)
    ok, reason = is_legal_association(scenario, a)
    if not ok:
        return None, reason
    b = cpu_stable_processing(scenario, a)
    reward, ev = feasible_lp_score(scenario, uav_xyz_m, a, b)
    start = start_association if start_association is not None else a
    return (
        ScoredMap(
            association=a.copy(),
            processing=np.asarray(b, dtype=float).copy(),
            reward=float(reward),
            sum_rate_Mbps=float(ev.sum_rate_mbps),
            feasible=bool(ev.feasible),
            n_forwarding=n_forwarding(a, b),
            hamming_from_start=hamming_association(a, start),
            reason=reason,
        ),
        reason,
    )


def _rank(scored: ScoredMap) -> tuple[int, float]:
    return (int(bool(scored.feasible)), float(scored.reward))


def iter_one_opt_neighbors(association: np.ndarray):
    """Move one IoT to a different UAV. I*(J-1) neighbours."""
    a0 = np.asarray(association, dtype=float)
    i_n, j_n = a0.shape
    hosts = np.argmax(a0, axis=1)
    for i in range(i_n):
        for j in range(j_n):
            if int(hosts[i]) == j:
                continue
            a = a0.copy()
            a[i, :] = 0.0
            a[i, j] = 1.0
            yield i, j, a


def search_legal_association(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    *,
    max_rounds: int = 50,
    n_random: int = 0,
    seed: int = 0,
    include_best_se: bool = True,
) -> AssociationSearchResult:
    """Best-improvement 1-opt on legal maps, optional random sample + best-SE."""
    start_a = np.asarray(association, dtype=float)
    cache: dict[tuple[int, ...], ScoredMap] = {}
    n_illegal = 0
    n_neighbors = 0
    n_random_scored = 0

    def _get(a: np.ndarray) -> ScoredMap | None:
        nonlocal n_illegal
        key = _assoc_key(a)
        if key in cache:
            return cache[key]
        scored, reason = score_association(
            scenario, uav_xyz_m, a, start_association=start_a
        )
        if scored is None:
            n_illegal += 1
            return None
        cache[key] = scored
        return scored

    start = _get(start_a)
    if start is None:
        b = cpu_stable_processing(scenario, start_a)
        reward, ev = feasible_lp_score(scenario, uav_xyz_m, start_a, b)
        start = ScoredMap(
            association=start_a.copy(),
            processing=np.asarray(b, dtype=float).copy(),
            reward=float(reward),
            sum_rate_Mbps=float(ev.sum_rate_mbps),
            feasible=bool(ev.feasible),
            n_forwarding=n_forwarding(start_a, b),
            hamming_from_start=0,
            reason="start_not_legal",
        )
        cache[_assoc_key(start_a)] = start

    incumbent = start
    history = [float(incumbent.reward)]
    n_accepted = 0
    rounds = 0

    if include_best_se:
        scored_se = _get(best_se_association(scenario, uav_xyz_m))
        if scored_se is not None and _rank(scored_se) > _rank(incumbent):
            incumbent = scored_se
            history.append(float(incumbent.reward))
            n_accepted += 1

    for _ in range(int(max_rounds)):
        rounds += 1
        best_nb: ScoredMap | None = None
        for _i, _j, a_nb in iter_one_opt_neighbors(incumbent.association):
            n_neighbors += 1
            scored = _get(a_nb)
            if scored is None:
                continue
            if best_nb is None or _rank(scored) > _rank(best_nb):
                best_nb = scored
        if best_nb is None or _rank(best_nb) <= _rank(incumbent):
            break
        incumbent = best_nb
        history.append(float(incumbent.reward))
        n_accepted += 1

    if int(n_random) > 0:
        rng = np.random.default_rng(int(seed))
        i_n, j_n = start_a.shape
        for _ in range(int(n_random)):
            hosts = rng.integers(0, j_n, size=i_n)
            a_r = np.zeros_like(start_a)
            a_r[np.arange(i_n), hosts] = 1.0
            n_random_scored += 1
            scored = _get(a_r)
            if scored is not None and _rank(scored) > _rank(incumbent):
                incumbent = scored
                history.append(float(incumbent.reward))
                n_accepted += 1

    return AssociationSearchResult(
        incumbent=incumbent,
        n_lp_evals=len(cache),
        n_accepted=n_accepted,
        n_neighbors_scored=n_neighbors,
        n_random_scored=n_random_scored,
        n_illegal=n_illegal,
        rounds=rounds,
        start_key=_assoc_key(start_a),
        best_key=_assoc_key(incumbent.association),
        cache_size=len(cache),
        history=history,
    )
