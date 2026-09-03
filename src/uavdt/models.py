"""Entities: IoT devices, UAVs, physical processes, allocations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.config import SimConfig


@dataclass(frozen=True)
class PhysicalProcess:
    """One physical process k monitored by an explicit IoT set N_k."""

    process_id: int
    iot_indices: np.ndarray  # 0-based indices into the IoT arrays

    def __post_init__(self) -> None:
        object.__setattr__(self, "iot_indices", np.asarray(self.iot_indices, dtype=int))


@dataclass
class Scenario:
    """Ground deployment and process membership.

    IoT coordinates are (x, y, 0). Process membership is stored both as
    `process_id_of_iot` and as explicit `processes[k].iot_indices` (N_k).
    """

    iot_xyz_m: np.ndarray  # (I, 3)
    process_id_of_iot: np.ndarray  # (I,) values in [0, K)
    processes: tuple[PhysicalProcess, ...]
    lambdas_per_s: np.ndarray  # (I,)
    seed: int
    cfg: SimConfig

    @property
    def num_iot(self) -> int:
        return int(self.iot_xyz_m.shape[0])

    @property
    def num_processes(self) -> int:
        return len(self.processes)

    @property
    def delta_ik(self) -> np.ndarray:
        """Membership matrix δ_ik, shape (I, K)."""
        i, k = self.num_iot, self.num_processes
        out = np.zeros((i, k), dtype=float)
        out[np.arange(i), self.process_id_of_iot] = 1.0
        return out


@dataclass
class Allocation:
    """Decision variables a_ij, b_ij, B_ij (Hz). Shapes (I, J)."""

    association: np.ndarray
    processing: np.ndarray
    bandwidth_hz: np.ndarray

    def hard_association(self) -> np.ndarray:
        return _one_hot_rows(self.association)

    def hard_processing(self) -> np.ndarray:
        return _one_hot_rows(self.processing)


def _one_hot_rows(mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mat, dtype=float)
    if mat.size == 0:
        return out
    out[np.arange(mat.shape[0]), np.argmax(mat, axis=1)] = 1.0
    return out
