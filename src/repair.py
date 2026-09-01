"""Documented repair filling missing decision variables for placement-only solvers."""

from __future__ import annotations

import numpy as np

from src.aodt import average_aodt, delay_rate_floors, upload_times
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
    """Map dimensionless relative weights onto feasible B_ij in Hz.

    ``raw`` is not hertz. TD3 passes ``1 + tanh(a)`` in [0, 2]; other solvers
    pass the same kind of non-negative request. Treating those values as Hz
    under-fills the pool (2 Hz on a 20 kHz system) and associated links at
    weight 0 get B = 0, which makes AoDT upload delay S_i / 1e-12.

    Each pool is spent in proportion to the weights, subject to (26), (27)
    and the per-link cap — the cap-aware form of B = pool · w / Σw. All-zero
    weights on a pool fall back to an equal split. A mix of zeros and
    positives gets a uniform + relative mix (w ← w + 1) so every associated
    link stays at B > 0 without inverting the weight order.
    """
    w = np.maximum(np.asarray(raw, dtype=float), 0.0) * (association > 0.5)
    b = np.zeros_like(w, dtype=float)
    fallback = equal_bandwidth(association, cfg)
    for mask, pool in bandwidth_pools(association, cfg):
        if not np.any(mask):
            continue
        cap = link_bandwidth_cap(cfg, pool)
        n = int(mask.sum())
        room = np.full(n, cap, dtype=float)
        w_m = w[mask]
        if float(w_m.sum()) <= 1e-12:
            b[mask] = fallback[mask]
            continue
        if np.any(w_m <= 1e-12):
            # Associated zeros would otherwise receive 0 Hz. Adding 1 to every
            # weight on the pool is a uniform share plus the request, so the
            # mapping stays monotone (0 → 1, 2 → 3) and B > 0.
            w_m = w_m + 1.0
        b[mask] = _spread_by_weight(w_m, pool, room)
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


def associated_rate_floors(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
) -> np.ndarray:
    """Per-IoT associated-rate floor (bit/s): max(R_min, AoDT delay floor).

    Delay floors apply only when the compute model is on. They depend on λ, μ,
    and T_k through the queueing term, not through the Shannon equation.
    """
    cfg = scenario.cfg
    floors = np.full(association.shape[0], cfg.r_min, dtype=float)
    mu = service_rate(cfg)
    if mu is None or not cfg.use_compute_model:
        return floors
    delay = delay_rate_floors(scenario, association, processing, mu)
    return np.maximum(floors, delay)


def allocate_pool(se_a: np.ndarray, pool: float, cap: float, need: np.ndarray) -> np.ndarray:
    """Exact LP for r_i = c_i B_i on one bandwidth pool.

    max  sum c_i B_i
    s.t. sum B_i = pool,  0 <= B_i <= cap,  B_i >= need_i if the floors fit.

    ``need`` is the per-link bandwidth floor in Hz (already rate_floor / c_i).
    Leftover after floors is poured into the highest-c links up to ``cap``
    (callers that want an AoDT-first surplus use ``allocate_constrained_bandwidth``).
    If the floors do not fit, fund as many cheapest floors as possible and split
    the rest equally over the unfunded links so every associated link keeps B > 0.
    """
    n = int(se_a.size)
    need = np.asarray(need, dtype=float).reshape(n)
    remaining = float(pool)

    if float(need.sum()) <= remaining:
        alloc = np.minimum(need, cap)
        remaining -= float(alloc.sum())
        for k in np.argsort(-se_a):
            if remaining <= 1e-12:
                break
            room = cap - alloc[k]
            take = min(room, remaining)
            alloc[k] += take
            remaining -= take
        return alloc

    alloc = np.zeros_like(need)
    funded = np.zeros(n, dtype=bool)
    for k in np.argsort(need):
        n_other_unfunded = int(n - funded.sum() - 1)
        if remaining >= float(need[k]) and (n_other_unfunded == 0 or remaining > float(need[k])):
            alloc[k] = min(float(need[k]), cap)
            remaining -= alloc[k]
            funded[k] = True
        else:
            break
    unfunded = ~funded
    n_u = int(unfunded.sum())
    if n_u:
        alloc[unfunded] = remaining / n_u
    elif remaining > 0:
        for k in np.argsort(-se_a):
            if remaining <= 1e-12:
                break
            take = min(cap - alloc[k], remaining)
            alloc[k] += take
            remaining -= take
    return alloc


def _pour_highest_se(alloc: np.ndarray, se_a: np.ndarray, remaining: float, cap: float) -> np.ndarray:
    """Dump leftover hertz into the highest spectral-efficiency links up to ``cap``."""
    remaining = float(remaining)
    for k in np.argsort(-se_a):
        if remaining <= 1e-12:
            break
        take = min(cap - alloc[k], remaining)
        if take <= 0.0:
            continue
        alloc[k] += take
        remaining -= take
    return alloc


def _aodt_state(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    se: np.ndarray,
    b: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    rates = se * np.maximum(b, 0.0)
    d_i = upload_times(association, processing, rates, scenario.cfg)
    aodt = average_aodt(scenario, association, processing, rates, mu)
    return d_i, aodt


def _waterfill_equal_rate(
    b: np.ndarray,
    se: np.ndarray,
    links: list[tuple[int, int]],
    amount: float,
    cap: float,
) -> float:
    """Raise associated rates of ``links`` by the same dR, spending ``amount`` Hz."""
    remaining = float(amount)
    if remaining <= 1e-12 or not links:
        return remaining
    for _ in range(len(links) + 1):
        room = [(i, j, cap - float(b[i, j])) for i, j in links if cap - float(b[i, j]) > 1e-12]
        if remaining <= 1e-12 or not room:
            break
        inv_se = sum(1.0 / float(se[i, j]) for i, j, _r in room)
        if inv_se <= 1e-18:
            break
        d_r = remaining / inv_se
        spent = 0.0
        for i, j, r_hz in room:
            take = min(r_hz, d_r / float(se[i, j]))
            b[i, j] += take
            spent += take
        if spent <= 1e-12:
            break
        remaining -= spent
    return remaining


def _give_aodt_chunk(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    se: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    remaining: float,
    cap: float,
    pool: float,
) -> float:
    """Spend surplus to pull down max D_i of the current worst process.

    AoDT_k is max_i D_i + Q, so helping one tied member does not move the max.
    Leftover after floors is waterfilled across every member of the worst
    process that currently sits at that max (equal rate increase). If that
    process is uniquely worst, waterfill continues until it matches the
    second-worst process; leftover then goes to sum-rate. If processes are
    tied, one quarter-pool chunk is still spent on the bottleneck set.
    The chunk size is a fraction of the pool, not ``max_bw_share``, so
    dropping the per-link cap does not dump the whole surplus into AoDT.
    """
    remaining = float(remaining)
    if remaining <= 1e-12 or not scenario.cfg.use_compute_model:
        return remaining
    if scenario.cfg.task_size_bytes is None:
        return remaining
    mu = service_rate(scenario.cfg)
    if mu is None:
        return remaining

    j_assoc = association.argmax(axis=1)

    def bottlenecks(d_i: np.ndarray, k_star: int) -> list[tuple[int, int]]:
        members = scenario.groups[k_star]
        if members.size == 0:
            return []
        in_pool = [int(i) for i in members if mask[int(i), int(j_assoc[int(i)])]]
        if not in_pool:
            return []
        d_max = max(float(d_i[i]) for i in in_pool)
        out = []
        for i in in_pool:
            if float(d_i[i]) < d_max - 1e-9:
                continue
            j = int(j_assoc[i])
            if cap - float(b[i, j]) > 1e-12:
                out.append((i, j))
        return out

    for _ in range(64):
        if remaining <= 1e-12:
            break
        d_i, aodt = _aodt_state(scenario, association, processing, se, b, mu)
        score = np.where(np.isfinite(aodt), aodt, -np.inf)
        if not np.any(np.isfinite(score)):
            break
        k_star = int(np.argmax(score))
        others = [float(aodt[k]) for k in range(aodt.size) if k != k_star and np.isfinite(aodt[k])]
        unique = not others or float(aodt[k_star]) > max(others) + 1e-9
        links = bottlenecks(d_i, k_star)
        if not links:
            break
        chunk = 0.25 * float(pool)
        if unique:
            step = min(remaining, max(chunk * 0.4, remaining * 0.05))
        else:
            step = min(remaining, chunk)
        before = remaining
        leftover_after_step = _waterfill_equal_rate(b, se, links, step, cap)
        remaining = remaining - step + leftover_after_step
        if remaining >= before - 1e-12:
            break
        if unique:
            _d2, aodt2 = _aodt_state(scenario, association, processing, se, b, mu)
            others2 = [float(aodt2[k]) for k in range(aodt2.size) if k != k_star and np.isfinite(aodt2[k])]
            if others2 and float(aodt2[k_star]) <= max(others2) + 1e-9:
                break
    return remaining


def allocate_constrained_bandwidth(
    scenario: Scenario,
    uav_xy: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    *,
    aodt_first: bool = True,
) -> np.ndarray:
    """Rate-optimal bandwidth under QoS and (when enabled) AoDT rate floors.

    Same linear-in-B LP as SCA, with floors raised so D_Nk leaves room for the
    queueing term under T_k. λ, μ, and T_k never enter the Shannon formula.

    After the floors are funded, leftover waterfills the current worst
    process's bottleneck members (equal rate increase) until that process is
    no longer uniquely worst; anything left goes to the highest-SE links.
    Pass ``aodt_first=False`` to skip that AoDT pour (sum-rate-only surplus).
    """
    from src.comm import link_metrics  # local: comm has no repair dependency

    cfg = scenario.cfg
    dummy = equal_bandwidth(association, cfg)
    if not np.any(association > 0.5):
        return dummy
    se = np.maximum(
        link_metrics(scenario.iot_xy, uav_xy, np.ones_like(dummy), cfg)["rates"], 1e-12
    )
    rate_need = associated_rate_floors(scenario, association, processing)
    b = np.zeros_like(dummy)
    for mask, pool in bandwidth_pools(association, cfg):
        if not np.any(mask):
            continue
        cap = link_bandwidth_cap(cfg, pool)
        rows, _cols = np.where(mask)
        need_hz = np.minimum(rate_need[rows] / se[mask], cap)
        if float(need_hz.sum()) > pool:
            b[mask] = allocate_pool(se[mask], pool, cap, need_hz)
            continue
        alloc = np.asarray(need_hz, dtype=float).copy()
        remaining = float(pool) - float(alloc.sum())
        b[mask] = alloc
        if aodt_first:
            remaining = _give_aodt_chunk(
                scenario, association, processing, se, b, mask, remaining, cap, pool
            )
        b[mask] = _pour_highest_se(b[mask], se[mask], remaining, cap)
    return b


def bandwidth_from_weights(
    scenario: Scenario,
    uav_xy: np.ndarray,
    association: np.ndarray,
    weights: np.ndarray,
    processing: np.ndarray | None = None,
) -> np.ndarray:
    """Turn a solver's raw bandwidth request into a QoS/AoDT-aware allocation.

    Every associated link first gets max(R_min, delay_floor) / c_ij. The next
    surplus chunk goes to the current AoDT bottleneck; only what remains is
    distributed by ``weights``. Without the floors an unstructured request
    nearly always starves some link below R_min, so the solution is rejected
    as infeasible and the solver never gets credit for the allocation it
    chose -- which is what kept TD3 pinned to its restart states.

    If the floors do not fit in a pool, fall back to ``project_bandwidth``
    for that pool only: the request is treated as relative weights on the
    pool, not as hertz. Other pools are unchanged. Returning from the whole
    function would be wrong under ``bandwidth_scope="per_uav"``.
    """
    from src.comm import link_metrics  # local: comm has no repair dependency

    cfg = scenario.cfg
    if processing is None:
        processing = process_consistent_processing(scenario, association)
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    se = np.maximum(
        link_metrics(scenario.iot_xy, uav_xy, np.ones(association.shape), cfg)["rates"], 1e-12
    )
    rate_need = associated_rate_floors(scenario, association, processing)
    b = np.zeros(association.shape, dtype=float)
    # Computed lazily if a pool cannot fund its floors; used only for that pool.
    projected: np.ndarray | None = None
    for mask, pool in bandwidth_pools(association, cfg):
        if not np.any(mask):
            continue
        cap = link_bandwidth_cap(cfg, pool)
        rows, _cols = np.where(mask)
        floors = np.minimum(rate_need[rows] / se[mask], cap)
        if float(floors.sum()) > pool:
            if projected is None:
                projected = project_bandwidth(weights, association, cfg)
            b[mask] = projected[mask]
            continue
        b[mask] = floors
        remaining = float(pool) - float(floors.sum())
        remaining = _give_aodt_chunk(
            scenario, association, processing, se, b, mask, remaining, cap, pool
        )
        room = np.maximum(cap - b[mask], 0.0)
        b[mask] = b[mask] + _spread_by_weight(w[mask], remaining, room)
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
    *,
    equal_split: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill missing variables so evaluate() can run.

    ``equal_split=True`` divides each bandwidth pool equally across associated
    links (Random / K-means). The default still uses the constrained LP when
    the compute model is on (SCA / TD3 / PSO).
    """
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
        if equal_split or not cfg.use_compute_model:
            bandwidth = equal_bandwidth(association, cfg)
        else:
            bandwidth = allocate_constrained_bandwidth(scenario, xy, association, processing)
    else:
        bandwidth = bandwidth_from_weights(scenario, xy, association, bandwidth, processing)
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
