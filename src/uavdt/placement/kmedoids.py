"""Discrete p-median / k-medoids on IoT (x, y).

Facilities are required to sit at demand points. That is the covering
flavoured 'park on IoTs' rule, as opposed to leftover-dump subset enum
(``sca_anchor``) or continuous k-means centroids.

Exact p-median enumerates ``C(I, J)`` when that is small. PAM k-medoids
is the fallback.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from uavdt.config import SimConfig
from uavdt.models import Scenario
from uavdt.placement.kmeans import _enforce_min_separation
from uavdt.scenario import make_uav_xyz_m

# Match zenith-anchor's full-enum cutoff.
DEFAULT_MAX_ENUMERATE = 1000


def pairwise_dist_m(xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)


def pmedian_cost(dist: np.ndarray, combo: tuple[int, ...]) -> float:
    if not combo:
        return math.inf
    return float(np.min(dist[:, list(combo)], axis=1).sum())


def exact_pmedian_combos(
    xy: np.ndarray,
    k: int,
    *,
    cost_tie_m: float = 1e-9,
) -> tuple[tuple[int, ...], list[tuple[int, ...]], float]:
    """All min-cost J-subsets of the demand points. Euclidean p-median."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = int(pts.shape[0])
    k_use = int(k)
    if k_use <= 0 or n == 0:
        return tuple(), [], math.inf
    if k_use >= n:
        combo = tuple(range(n))
        return combo, [combo], 0.0
    dist = pairwise_dist_m(pts)
    best_cost = math.inf
    ties: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(n), k_use):
        cost = pmedian_cost(dist, combo)
        if cost + cost_tie_m < best_cost:
            best_cost = cost
            ties = [combo]
        elif abs(cost - best_cost) <= cost_tie_m:
            ties.append(combo)
    if not ties:
        return tuple(), [], math.inf
    return ties[0], ties, float(best_cost)


def kmedoids_combo(
    xy: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    max_iter: int = 50,
) -> tuple[int, ...]:
    """PAM-style: assign to nearest medoid, swap to min cluster sum-distance."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = int(pts.shape[0])
    k_use = min(int(k), n)
    if k_use <= 0 or n == 0:
        return tuple()
    dist = pairwise_dist_m(pts)
    medoids = np.sort(rng.choice(n, size=k_use, replace=False))
    for _ in range(max_iter):
        labels = dist[:, medoids].argmin(axis=1)
        used: set[int] = set()
        new = []
        for j in range(k_use):
            members = np.where(labels == j)[0]
            if members.size == 0:
                leftover = [i for i in range(n) if i not in used]
                if not leftover:
                    pick = int(medoids[j])
                else:
                    dmin = dist[np.ix_(leftover, list(used) or leftover)].min(axis=1)
                    pick = int(leftover[int(np.argmax(dmin))])
            else:
                costs = dist[np.ix_(members, members)].sum(axis=1)
                order = np.argsort(costs)
                pick = int(members[int(order[0])])
                t = 1
                while pick in used and t < order.size:
                    pick = int(members[int(order[t])])
                    t += 1
                if pick in used:
                    leftover = [i for i in range(n) if i not in used]
                    pick = int(leftover[0]) if leftover else pick
            used.add(pick)
            new.append(pick)
        new_arr = np.sort(np.asarray(new, dtype=int))
        if np.array_equal(new_arr, medoids):
            break
        medoids = new_arr
    return tuple(int(x) for x in medoids)


def covering_combos(
    xy: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    n_pam_inits: int = 5,
    max_enumerate: int = DEFAULT_MAX_ENUMERATE,
) -> list[tuple[int, ...]]:
    """Exact p-median when cheap, plus extra PAM inits. Unique J-subsets."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = int(pts.shape[0])
    k_use = int(k)
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    n_all = 0 if k_use < 0 or k_use > n else int(math.comb(n, k_use))

    def _add(combo: tuple[int, ...]) -> None:
        if not combo or combo in seen:
            return
        seen.add(combo)
        out.append(combo)

    if 0 < k_use <= n and n_all <= int(max_enumerate):
        _best, ties, _cost = exact_pmedian_combos(pts, k_use)
        for combo in ties:
            _add(combo)
            if len(out) >= 8:
                break
    for i in range(max(0, int(n_pam_inits))):
        sub = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + 17 * i)
        _add(kmedoids_combo(pts, k_use, sub))
    return out


def place_kmedoids(
    scenario: Scenario,
    seed: int,
    num_uav: int | None = None,
    cfg: SimConfig | None = None,
) -> np.ndarray:
    """UAV at zenith over a p-median / k-medoids J-subset of IoTs."""
    cfg = scenario.cfg if cfg is None else cfg
    j = cfg.num_uav if num_uav is None else num_uav
    rng = np.random.default_rng(int(seed) + 9001)
    combos = covering_combos(
        scenario.iot_xyz_m[:, :2], int(j), rng, n_pam_inits=4
    )
    combo = combos[0] if combos else tuple(range(min(int(j), scenario.num_iot)))
    xy = scenario.iot_xyz_m[list(combo), :2].astype(float, copy=True)
    if xy.shape[0] < int(j):
        extra = np.column_stack(
            [
                rng.uniform(0.0, cfg.area_x_m, int(j) - xy.shape[0]),
                rng.uniform(0.0, cfg.area_y_m, int(j) - xy.shape[0]),
            ]
        )
        xy = np.vstack([xy, extra])
    xy = _enforce_min_separation(xy[: int(j)], cfg, rng)
    return make_uav_xyz_m(xy, cfg.uav_height_m)
