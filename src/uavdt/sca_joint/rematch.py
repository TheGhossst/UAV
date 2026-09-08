"""Discrete association/processing re-match used by SCA-joint.

Not a paper algorithm. Frozen SCA never calls this. Constraint numbers
follow Problem (P): (21) exclusive a_ij, (23) one processing UAV per
process, (24) CPU stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.computation import queue_unstable
from uavdt.models import Scenario
from uavdt.resources import cpu_stable_processing
from uavdt.sca.linearize import spectral_efficiency


def best_se_association(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
) -> np.ndarray:
    """Each IoT to the UAV with highest SE at the current q. Holds (21)."""
    se = spectral_efficiency(scenario.iot_xyz_m, uav_xyz_m, scenario.cfg)
    a = np.zeros_like(se)
    a[np.arange(se.shape[0]), np.argmax(se, axis=1)] = 1.0
    return a


def processing_process_consistent(scenario: Scenario, processing: np.ndarray) -> bool:
    """Constraint (23): one processing UAV per process."""
    for proc in scenario.processes:
        members = proc.iot_indices
        if members.size <= 1:
            continue
        js = np.argmax(processing[members], axis=1)
        if np.unique(js).size > 1:
            return False
    return True


def association_process_cohesive(scenario: Scenario, association: np.ndarray) -> bool:
    """All of N_k share one associated UAV (no split, so no T_u2u from a)."""
    for proc in scenario.processes:
        members = proc.iot_indices
        if members.size <= 1:
            continue
        js = np.argmax(association[members], axis=1)
        if np.unique(js).size > 1:
            return False
    return True


def centroid_cohesive_association(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
) -> np.ndarray:
    """Hand construction from docs/RESULTS.md §2.4: N_k → UAV nearest the centroid."""
    i, j = scenario.iot_xyz_m.shape[0], uav_xyz_m.shape[0]
    a = np.zeros((i, j), dtype=float)
    for proc in scenario.processes:
        members = proc.iot_indices
        if members.size == 0:
            continue
        centroid = scenario.iot_xyz_m[members].mean(axis=0, keepdims=True)
        j_star = int(np.argmin(np.linalg.norm(centroid - uav_xyz_m, axis=1)))
        a[members, j_star] = 1.0
    return a


def n_forwarding(association: np.ndarray, processing: np.ndarray) -> int:
    ja = np.argmax(association, axis=1)
    jb = np.argmax(processing, axis=1)
    return int(np.sum(ja != jb))


def _same_one_hot(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left > 0.5, right > 0.5))


@dataclass(frozen=True)
class RematchProposal:
    association: np.ndarray | None
    processing: np.ndarray | None
    reason: str
    association_changed: bool
    kind: str = "best_se"

    @property
    def ok(self) -> bool:
        return self.association is not None and self.processing is not None


def _proposal_from_association(
    scenario: Scenario,
    a_new: np.ndarray,
    a_cur: np.ndarray,
    kind: str,
) -> RematchProposal:
    if _same_one_hot(a_new, a_cur):
        return RematchProposal(
            association=None,
            processing=None,
            reason="association_unchanged",
            association_changed=False,
            kind=kind,
        )
    b_new = cpu_stable_processing(scenario, a_new)
    if not processing_process_consistent(scenario, b_new):
        return RematchProposal(
            association=None,
            processing=None,
            reason="process_inconsistent",
            association_changed=True,
            kind=kind,
        )
    mu = float(scenario.cfg.service_rate_per_s)
    if np.any(queue_unstable(b_new, scenario.lambdas_per_s, mu)):
        return RematchProposal(
            association=None,
            processing=None,
            reason="cpu_unstable",
            association_changed=True,
            kind=kind,
        )
    return RematchProposal(
        association=a_new,
        processing=b_new,
        reason="ok",
        association_changed=True,
        kind=kind,
    )


def propose_rematch(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
) -> RematchProposal:
    """Best-SE a_ij, then cpu_stable_processing for (23)/(24).

    Returns a no-op proposal when greedy a already matches the current
    association. Rejects the switch when the repaired b still violates
    (23) or (24).
    """
    _ = processing
    return _proposal_from_association(
        scenario,
        best_se_association(scenario, uav_xyz_m),
        association,
        "best_se",
    )


def propose_process_cohesive(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
) -> RematchProposal:
    """N_k → UAV nearest the process centroid, then cpu_stable_processing."""
    return _proposal_from_association(
        scenario,
        centroid_cohesive_association(scenario, uav_xyz_m),
        association,
        "process_cohesive",
    )


def iter_rematch_candidates(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    *,
    include_process_cohesive: bool,
) -> list[RematchProposal]:
    """Best-SE first; optional process-cohesive candidate beside it."""
    out = [propose_rematch(scenario, uav_xyz_m, association, processing)]
    if include_process_cohesive:
        out.append(propose_process_cohesive(scenario, uav_xyz_m, association))
    return out
