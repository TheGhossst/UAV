"""Documented repair filling missing decision variables for placement-only solvers."""

from __future__ import annotations

import numpy as np

from src.compute import service_rate
from src.config import SimConfig
from src.scenario import Scenario


def equal_bandwidth(association: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Split B_sys equally across associated links (plan.md Phase 2)."""
    i, j = association.shape
    n_assoc = max(int(association.sum()), 1)
    b = np.zeros((i, j))
    share = cfg.b_sys / n_assoc
    b[association > 0.5] = share
    return b


def nearest_association(iot_xy: np.ndarray, uav_xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(iot_xy[:, None, :] - uav_xy[None, :, :], axis=-1)
    a = np.zeros_like(d)
    a[np.arange(d.shape[0]), d.argmin(axis=1)] = 1.0
    return a


def softmax_association(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    p = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
    a = np.zeros_like(p)
    a[np.arange(p.shape[0]), p.argmax(axis=1)] = 1.0
    return a


def process_consistent_processing(
    scenario: Scenario,
    association: np.ndarray,
    proc_logits: np.ndarray | None = None,
) -> np.ndarray:
    """Assign one processing UAV per process group (constraint 23)."""
    i, j = association.shape
    b = np.zeros((i, j))
    for k, members in enumerate(scenario.groups):
        if members.size == 0:
            continue
        if proc_logits is not None:
            scores = proc_logits[members].mean(axis=0)
            j_star = int(np.argmax(scores))
        else:
            votes = association[members].sum(axis=0)
            j_star = int(np.argmax(votes))
        b[members, j_star] = 1.0
    return b


def stabilize_processing(scenario: Scenario, processing: np.ndarray) -> np.ndarray:
    """If a UAV is overloaded, move whole process groups to the least-loaded UAV."""
    cfg = scenario.cfg
    mu = service_rate(cfg)
    if mu is None:
        return processing
    b = processing.copy()
    j = b.shape[1]
    for _ in range(j * cfg.num_processes + 1):
        lam_j = b.T @ scenario.lambdas
        if np.all(lam_j < mu):
            break
        busy = int(np.argmax(lam_j))
        groups_on_busy = []
        for k, members in enumerate(scenario.groups):
            if members.size and b[members[0], busy] > 0.5:
                groups_on_busy.append(k)
        if not groups_on_busy:
            break
        # move smallest group
        k_move = min(groups_on_busy, key=lambda k: scenario.groups[k].size)
        members = scenario.groups[k_move]
        dest = int(np.argmin(lam_j))
        if dest == busy:
            break
        b[members] = 0.0
        b[members, dest] = 1.0
    return b


def project_bandwidth(raw: np.ndarray, association: np.ndarray, cfg: SimConfig) -> np.ndarray:
    b = np.maximum(raw, 0.0) * association
    total = b.sum()
    if total <= 1e-12:
        return equal_bandwidth(association, cfg)
    if total > cfg.b_sys:
        b *= cfg.b_sys / total
    return b


def enforce_separation(uav_xy: np.ndarray, cfg: SimConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    xy = uav_xy.copy()
    j = xy.shape[0]
    rng = rng or np.random.default_rng()
    for _ in range(50):
        moved = False
        for p in range(j):
            for q in range(p + 1, j):
                d = np.linalg.norm(xy[p] - xy[q])
                if d < cfg.uav_min_distance:
                    direction = xy[q] - xy[p]
                    if np.linalg.norm(direction) < 1e-9:
                        direction = rng.normal(size=2)
                    direction = direction / (np.linalg.norm(direction) + 1e-12)
                    push = 0.5 * (cfg.uav_min_distance - d + 1e-3)
                    xy[p] -= push * direction
                    xy[q] += push * direction
                    moved = True
        xy[:, 0] = np.clip(xy[:, 0], 0.0, cfg.area_x)
        xy[:, 1] = np.clip(xy[:, 1], 0.0, cfg.area_y)
        if not moved:
            break
    return xy


def clip_positions(uav_xy: np.ndarray, cfg: SimConfig) -> np.ndarray:
    xy = np.asarray(uav_xy, dtype=float).reshape(-1, 2)
    xy[:, 0] = np.clip(xy[:, 0], 0.0, cfg.area_x)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, cfg.area_y)
    return xy


def complete_solution(
    scenario: Scenario,
    uav_xy: np.ndarray,
    association: np.ndarray | None = None,
    processing: np.ndarray | None = None,
    bandwidth: np.ndarray | None = None,
    assoc_logits: np.ndarray | None = None,
    proc_logits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill missing variables so evaluate() can run."""
    cfg = scenario.cfg
    xy = enforce_separation(clip_positions(uav_xy, cfg), cfg)
    if association is None:
        if assoc_logits is not None:
            association = softmax_association(assoc_logits)
        else:
            association = nearest_association(scenario.iot_xy, xy)
    else:
        association = _one_hot_rows(association)

    if processing is None:
        processing = process_consistent_processing(scenario, association, proc_logits)
    else:
        processing = _one_hot_rows(processing)
        processing = process_consistent_processing(scenario, association, processing)

    processing = stabilize_processing(scenario, processing)

    if bandwidth is None:
        bandwidth = equal_bandwidth(association, cfg)
    else:
        bandwidth = project_bandwidth(bandwidth, association, cfg)
    return xy, association, processing, bandwidth


def _one_hot_rows(m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(m, dtype=float)
    out[np.arange(m.shape[0]), np.asarray(m).argmax(axis=1)] = 1.0
    return out


def dummy_equal_split_for_single_uav(scenario: Scenario, uav_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    j = uav_xy.shape[0]
    a = np.ones((scenario.cfg.num_iot, j)) / j if j > 1 else np.ones((scenario.cfg.num_iot, 1))
    if j == 1:
        a = np.ones((scenario.cfg.num_iot, 1))
    else:
        a = nearest_association(scenario.iot_xy, uav_xy)
    return complete_solution(scenario, uav_xy, association=a)
