"""Documented repair filling missing decision variables for placement-only solvers."""

from __future__ import annotations

import numpy as np

from src.compute import service_rate
from src.config import SimConfig
from src.scenario import Scenario


def bandwidth_pools(association: np.ndarray, cfg: SimConfig) -> list[tuple[np.ndarray, float]]:
    """Constraint (27) as (link mask, pool in Hz) pairs.

    ``system`` scope is one shared pool over all associated links (literal
    Table II reading). ``per_uav`` gives every UAV its own B_sys, so the total
    spectrum grows with J.
    """
    i, j = association.shape
    assoc = association > 0.5
    if cfg.bandwidth_scope == "per_uav":
        pools = []
        for u in range(j):
            mask = np.zeros((i, j), dtype=bool)
            mask[:, u] = assoc[:, u]
            pools.append((mask, float(cfg.b_sys)))
        return pools
    return [(assoc, float(cfg.b_sys))]


def link_bandwidth_cap(cfg: SimConfig, pool: float) -> float:
    """Per-link ceiling on B_ij.

    Sum rate is linear in B_ij, so an uncapped pool is always maximised at a
    vertex: one link takes everything left after the R_min floors. That makes
    the optimum insensitive to placement and to J. The cap keeps the allocation
    spread over the best few links instead.
    """
    if cfg.max_bw_share is None:
        return pool
    return float(cfg.max_bw_share) * pool


def equal_bandwidth(association: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Split each pool equally across the associated links it covers.

    Each associated link is capped at ``link_bandwidth_cap``. Leftover after
    the equal share is offered to links that still have room. If every link
    is already at the cap, unused pool is left unused rather than violating
    the cap or constraint (27). Unassociated links stay at 0. With no cap
    (``max_bw_share is None``) this is a plain equal split of the pool.
    """
    b = np.zeros(association.shape, dtype=float)
    for mask, pool in bandwidth_pools(association, cfg):
        n = int(mask.sum())
        if n == 0:
            continue
        cap = link_bandwidth_cap(cfg, pool)
        b[mask] = _spread_by_weight(np.ones(n, dtype=float), pool, np.full(n, cap, dtype=float))
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
    """Make a raw bandwidth request feasible for (26), (27) and the link cap."""
    b = np.maximum(np.asarray(raw, dtype=float), 0.0) * (association > 0.5)
    fallback = equal_bandwidth(association, cfg)
    for mask, pool in bandwidth_pools(association, cfg):
        if not np.any(mask):
            continue
        cap = link_bandwidth_cap(cfg, pool)
        vals = np.minimum(b[mask], cap)
        total = float(vals.sum())
        if total <= 1e-12:
            b[mask] = fallback[mask]
            continue
        if total > pool:
            # Scaling can only lower values, so the cap still holds afterwards.
            vals = vals * (pool / total)
        b[mask] = vals
    return b


def _spread_by_weight(weights: np.ndarray, amount: float, room: np.ndarray) -> np.ndarray:
    """Hand out ``amount`` in proportion to ``weights`` without exceeding ``room``."""
    out = np.zeros_like(room)
    left = float(amount)
    active = room > 1e-12
    for _ in range(int(room.size) + 1):
        w = np.where(active, np.maximum(weights, 0.0), 0.0)
        total = float(w.sum())
        if left <= 1e-9 or not np.any(active):
            break
        share = left * (w / total) if total > 1e-12 else left * active / max(int(active.sum()), 1)
        take = np.minimum(share, room - out)
        if float(take.sum()) <= 1e-12:
            break
        out += take
        left -= float(take.sum())
        active = active & (room - out > 1e-12)
    return out


def bandwidth_from_weights(
    scenario: Scenario,
    uav_xy: np.ndarray,
    association: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Turn a solver's raw bandwidth request into a QoS-aware allocation.

    Every associated link first gets its R_min floor ``R_min / c_ij``; only the
    surplus is distributed by ``weights``. Without the floors an unstructured
    request nearly always starves some link below R_min, so the solution is
    rejected as infeasible and the solver never gets credit for the allocation
    it chose -- which is what kept TD3 pinned to its restart states.

    If the floors do not fit in a pool, fall back to plain projection for
    that pool only: other pools are unchanged. Returning from the whole
    function would be wrong under ``bandwidth_scope="per_uav"``.
    """
    from src.comm import link_metrics  # local: comm has no repair dependency

    cfg = scenario.cfg
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    se = np.maximum(
        link_metrics(scenario.iot_xy, uav_xy, np.ones(association.shape), cfg)["rates"], 1e-12
    )
    b = np.zeros(association.shape, dtype=float)
    # Computed lazily if a pool cannot fund its floors; used only for that pool.
    projected: np.ndarray | None = None
    for mask, pool in bandwidth_pools(association, cfg):
        if not np.any(mask):
            continue
        cap = link_bandwidth_cap(cfg, pool)
        floors = np.minimum(cfg.r_min / se[mask], cap)
        if float(floors.sum()) > pool:
            if projected is None:
                projected = project_bandwidth(weights, association, cfg)
            b[mask] = projected[mask]
            continue
        b[mask] = floors + _spread_by_weight(w[mask], pool - float(floors.sum()), cap - floors)
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
        # Placement-only baselines get the naive equal split.
        bandwidth = equal_bandwidth(association, cfg)
    else:
        bandwidth = bandwidth_from_weights(scenario, xy, association, bandwidth)
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
