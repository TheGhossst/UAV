"""Parse a continuous TD3 action into UAV xy and a discrete Allocation.

Algorithm 2: movement, association logits, processing logits.
Bandwidth is not in the actor; inner steps use leftover-dump B_ij
(Algorithm 2 "update bandwidth") unless TD3Settings.inner_bandwidth says
otherwise. Export still uses the frozen-q LP.
Constraint (23) is enforced by decoding one processing UAV per process
(K×J logits), not independent I×J processing logits.

Move decode is an implementation choice on TD3Settings.move_mode:
- delta: Δx,Δy × 10 m (Alg. 2 letter; a constant actor cannot hover)
- setpoint: 10 m/step toward a tanh-mapped field target (can hover)
- absolute: teleport to the tanh-mapped field target
"""

from __future__ import annotations

import numpy as np

from uavdt.channel import distances_m
from uavdt.config import SimConfig
from uavdt.models import Allocation, Scenario
from uavdt.resources import (
    cpu_stable_processing,
    equal_share_bandwidth_hz,
    nearest_association,
)
from uavdt.scenario import make_uav_xyz_m
from uavdt.td3.settings import TD3Settings


def action_size(cfg: SimConfig) -> int:
    j = cfg.num_uav
    return 2 * j + cfg.num_iot * j + cfg.num_processes * j


def observation_size(cfg: SimConfig) -> int:
    i, j, k = cfg.num_iot, cfg.num_uav, cfg.num_processes
    return 2 * j + 2 * i + i + j + k + 3 * i * j + 1


def _slices(cfg: SimConfig) -> tuple[int, int, int]:
    j = cfg.num_uav
    n_move = 2 * j
    n_assoc = cfg.num_iot * j
    n_proc = cfg.num_processes * j
    return n_move, n_assoc, n_proc


def _target_xy(move: np.ndarray, cfg: SimConfig) -> np.ndarray:
    unit = 0.5 * (np.clip(move, -1.0, 1.0) + 1.0)
    target = np.empty_like(unit)
    target[:, 0] = unit[:, 0] * cfg.area_x_m
    target[:, 1] = unit[:, 1] * cfg.area_y_m
    return target


def _move_xy(
    raw_move: np.ndarray,
    uav_xyz_m: np.ndarray,
    cfg: SimConfig,
    settings: TD3Settings,
    origin_xyz_m: np.ndarray | None = None,
) -> np.ndarray:
    j = cfg.num_uav
    move = np.clip(np.asarray(raw_move, dtype=float), -1.0, 1.0).reshape(j, 2)
    cur = np.asarray(uav_xyz_m, dtype=float)[:, :2]
    if settings.move_mode == "delta":
        xy = cur + settings.move_scale_m * move
    elif settings.move_mode == "absolute":
        xy = _target_xy(move, cfg)
    elif settings.move_mode == "residual":
        base = cur if origin_xyz_m is None else np.asarray(origin_xyz_m, dtype=float)[:, :2]
        xy = base + settings.move_scale_m * move
    else:
        target = _target_xy(move, cfg)
        step = np.clip(target - cur, -settings.move_scale_m, settings.move_scale_m)
        xy = cur + step
    xy[:, 0] = np.clip(xy[:, 0], 0.0, cfg.area_x_m)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, cfg.area_y_m)
    return xy


def decode_action(
    action: np.ndarray,
    uav_xyz_m: np.ndarray,
    scenario: Scenario,
    settings: TD3Settings | None = None,
    *,
    origin_xyz_m: np.ndarray | None = None,
) -> tuple[np.ndarray, Allocation]:
    """Map actor output to clipped UAV positions and hard a_ij / b_ij."""
    settings = settings or TD3Settings()
    cfg = scenario.cfg
    n_move, n_assoc, n_proc = _slices(cfg)
    expected = n_move + n_assoc + n_proc
    raw = np.asarray(action, dtype=float).reshape(-1)
    if raw.size != expected:
        raise ValueError(f"action length {raw.size} != {expected}")

    j = cfg.num_uav
    i = cfg.num_iot
    xy = _move_xy(raw[:n_move], uav_xyz_m, cfg, settings, origin_xyz_m=origin_xyz_m)
    uav = make_uav_xyz_m(xy, cfg.uav_height_m)

    assoc_logits = raw[n_move : n_move + n_assoc].reshape(i, j)
    if settings.assoc_mode == "nearest":
        a = nearest_association(scenario.iot_xyz_m, uav)
    else:
        scores = assoc_logits
        if settings.assoc_mode == "logits_plus_dist":
            coef = float(settings.assoc_distance_coef)
            if coef > 0.0:
                dist = distances_m(scenario.iot_xyz_m, uav)
                scale = max(float(np.hypot(cfg.area_x_m, cfg.area_y_m)), 1e-12)
                scores = assoc_logits - coef * (dist / scale)
        a = np.zeros((i, j), dtype=float)
        a[np.arange(i), np.argmax(scores, axis=1)] = 1.0

    proc_logits = raw[n_move + n_assoc :].reshape(cfg.num_processes, j)
    if settings.process_mode == "cpu_stable":
        b = cpu_stable_processing(scenario, a)
    else:
        b = np.zeros((i, j), dtype=float)
        for proc in scenario.processes:
            members = proc.iot_indices
            if members.size == 0:
                continue
            j_star = int(np.argmax(proc_logits[proc.process_id]))
            b[members, j_star] = 1.0

    bw = equal_share_bandwidth_hz(a, cfg)
    return uav, Allocation(association=a, processing=b, bandwidth_hz=bw)
