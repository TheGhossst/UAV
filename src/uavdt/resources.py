"""Association a_ij, processing b_ij, bandwidth B_ij.

Problem (P) treats these as decision variables. Helpers below are
evaluation conventions for placement-only runs, not paper algorithms.
"""

from __future__ import annotations

import numpy as np

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
    """Nearest association, process-consistent processing, equal B_ij."""
    a = nearest_association(scenario.iot_xyz_m, uav_xyz_m)
    b = process_consistent_processing(scenario, a)
    bw = equal_share_bandwidth_hz(a, scenario.cfg)
    return Allocation(association=a, processing=b, bandwidth_hz=bw)
