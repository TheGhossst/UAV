"""Local UAV-position polish under bandwidth-proportional noise.

Each evaluation freezes the candidate (x, y), rebuilds nearest association
and CPU-stable processing, and calls ``solve_bandwidth_proportional``.
Height stays at ``cfg.uav_height_m``. Frozen Algorithm 1 is not called.

The search is Nelder–Mead on the 2J ground coordinates, about
``max_evals`` inner solves, with a simplex step of a few tens of metres.
The start point is always kept if the search does not beat it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.analysis.bw_proportional import (
    equal_share_reference_hz,
    solve_bandwidth_proportional,
)
from uavdt.aodt import average_aodt_s
from uavdt.computation import offered_load, queue_unstable, service_rate_per_s
from uavdt.constraints import check_constraints, pairwise_uav_distance_m
from uavdt.models import Scenario
from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.kmedoids import exact_pmedian_combos, place_kmedoids
from uavdt.resources import cpu_stable_processing, nearest_association
from uavdt.sca_anchor import uav_for_combo
from uavdt.scenario import make_uav_xyz_m


@dataclass
class LayoutScore:
    rate_mbps: float
    feasible: bool
    status: str
    max_link_share: float


@dataclass
class PolishResult:
    uav_xyz_m: np.ndarray
    rate_mbps: float
    feasible: bool
    n_evals: int
    start_rate_mbps: float
    start_feasible: bool
    moved_rms_m: float
    status: str


def score_layout(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray | None = None,
    processing: np.ndarray | None = None,
    *,
    b_eq_hz: float | None = None,
) -> LayoutScore:
    """Proportional inner solve plus the same constraint report as evaluate()."""
    uav = np.asarray(uav_xyz_m, dtype=float)
    if association is None or processing is None:
        association = nearest_association(scenario.iot_xyz_m, uav)
        processing = cpu_stable_processing(scenario, association)
    res = solve_bandwidth_proportional(
        scenario,
        uav,
        association,
        processing,
        b_eq_hz=b_eq_hz,
    )
    share = float("nan")
    if res.bandwidth_hz.size and scenario.cfg.b_sys_hz > 0.0:
        share = float(np.max(res.bandwidth_hz) / scenario.cfg.b_sys_hz)
    if res.infeasible or not np.isfinite(res.objective_bit_per_s):
        return LayoutScore(
            rate_mbps=float("nan"),
            feasible=False,
            status=res.status,
            max_link_share=share,
        )
    cfg = scenario.cfg
    a = np.asarray(association, dtype=float)
    b = np.asarray(processing, dtype=float)
    mu = service_rate_per_s(cfg)
    rho = offered_load(b, scenario.lambdas_per_s, mu)
    stable = ~queue_unstable(b, scenario.lambdas_per_s, mu)
    aodt = average_aodt_s(
        scenario,
        a,
        b,
        res.rates_bit_per_s,
        np.full(uav.shape[0], mu),
        uav_stable=stable,
    )
    assoc_rates = (a * res.rates_bit_per_s).sum(axis=1)
    report = check_constraints(
        scenario, uav, a, b, res.bandwidth_hz, assoc_rates, rho, aodt
    )
    cap_ok = bool(np.all(res.bandwidth_hz <= cfg.link_bandwidth_cap_hz + 1.0))
    return LayoutScore(
        rate_mbps=float(res.objective_mbps),
        feasible=bool(report.feasible) and cap_ok,
        status=res.status,
        max_link_share=share,
    )


def pmedian_start(
    scenario: Scenario,
    seed: int,
    *,
    b_eq_hz: float | None = None,
) -> np.ndarray:
    """Euclidean p-median on IoT xy. Ties broken by the proportional score."""
    cfg = scenario.cfg
    _primary, ties, _cost = exact_pmedian_combos(
        scenario.iot_xyz_m[:, :2], int(cfg.num_uav)
    )
    best_uav: np.ndarray | None = None
    best_key: tuple[int, float] | None = None
    for combo in ties:
        uav = uav_for_combo(scenario, combo, seed)
        if uav is None:
            continue
        scored = score_layout(scenario, uav, b_eq_hz=b_eq_hz)
        rate = scored.rate_mbps if np.isfinite(scored.rate_mbps) else -1.0
        key = (int(scored.feasible), float(rate))
        if best_key is None or key > best_key:
            best_key = key
            best_uav = uav
    if best_uav is None:
        return place_kmedoids(scenario, seed)
    return best_uav


def kmeans_start(scenario: Scenario, seed: int) -> np.ndarray:
    return place_kmeans(scenario, seed)


def _xy_of(uav: np.ndarray) -> np.ndarray:
    return np.asarray(uav, dtype=float)[:, :2].reshape(-1).copy()


def _uav_from_xy(xy: np.ndarray, height_m: float, n_uav: int) -> np.ndarray:
    return make_uav_xyz_m(np.asarray(xy, dtype=float).reshape(n_uav, 2), height_m)


def _clip_xy(xy: np.ndarray, area_x: float, area_y: float) -> np.ndarray:
    out = np.asarray(xy, dtype=float).copy()
    out[0::2] = np.clip(out[0::2], 0.0, area_x)
    out[1::2] = np.clip(out[1::2], 0.0, area_y)
    return out


def _min_sep_m(uav: np.ndarray) -> float:
    dist = pairwise_uav_distance_m(uav)
    if dist.size == 0:
        return float("inf")
    return float(np.min(dist))


def _initial_simplex(x0: np.ndarray, step_m: float, area_x: float, area_y: float) -> np.ndarray:
    n = int(x0.size)
    simplex = np.zeros((n + 1, n), dtype=float)
    simplex[0] = x0
    for i in range(n):
        limit = area_x if i % 2 == 0 else area_y
        direction = step_m if x0[i] + step_m <= limit else -step_m
        vertex = x0.copy()
        vertex[i] = float(np.clip(x0[i] + direction, 0.0, limit))
        if abs(vertex[i] - x0[i]) < 1e-6:
            vertex[i] = float(np.clip(x0[i] + 0.5 * direction, 0.0, limit))
        simplex[i + 1] = vertex
    return simplex


def polish_uav_positions(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    *,
    max_evals: int = 200,
    step_m: float = 20.0,
    b_eq_hz: float | None = None,
) -> PolishResult:
    """Nelder–Mead polish. Fitness is proportional Mbps at rebuilt (a, b)."""
    from scipy.optimize import minimize

    if max_evals < 1:
        raise ValueError("max_evals must be positive")
    cfg = scenario.cfg
    start = np.asarray(uav_xyz_m, dtype=float)
    start_score = score_layout(scenario, start, b_eq_hz=b_eq_hz)
    x0 = _clip_xy(_xy_of(start), cfg.area_x_m, cfg.area_y_m)
    n_uav = int(start.shape[0])
    theta = float(cfg.uav_min_separation_m)
    budget = int(max_evals)
    n_evals = 0
    best_x = x0.copy()
    best_cost = float("inf")
    best_score = start_score

    def _consider(xy: np.ndarray, cost: float, scored: LayoutScore | None) -> None:
        nonlocal best_x, best_cost, best_score
        if cost < best_cost:
            best_cost = float(cost)
            best_x = np.asarray(xy, dtype=float).copy()
            if scored is not None:
                best_score = scored

    def _cost(xy: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        clipped = _clip_xy(xy, cfg.area_x_m, cfg.area_y_m)
        uav = _uav_from_xy(clipped, cfg.uav_height_m, n_uav)
        gap = theta - _min_sep_m(uav)
        if gap > 1e-9:
            cost = 100.0 + 10.0 * gap
            _consider(clipped, cost, None)
            return cost
        scored = score_layout(scenario, uav, b_eq_hz=b_eq_hz)
        if not scored.feasible or not np.isfinite(scored.rate_mbps):
            cost = 10.0
            _consider(clipped, cost, scored)
            return cost
        cost = -float(scored.rate_mbps)
        _consider(clipped, cost, scored)
        return cost

    # The start is a candidate even if Nelder–Mead never returns to it.
    _cost(x0)
    if budget > 1:
        simplex = _initial_simplex(x0, float(step_m), cfg.area_x_m, cfg.area_y_m)
        remaining = max(budget - n_evals, n_uav + 2)

        def _limited(xy: np.ndarray) -> float:
            if n_evals >= budget:
                return float(best_cost)
            return _cost(xy)

        try:
            minimize(
                _limited,
                x0,
                method="Nelder-Mead",
                options={
                    "maxfev": int(max(remaining, budget)),
                    "xatol": 1.0,
                    "fatol": 1e-3,
                    "adaptive": True,
                    "initial_simplex": simplex,
                },
            )
        except Exception:
            # A failed simplex still leaves the best evaluated point.
            pass

    polished = _uav_from_xy(best_x, cfg.uav_height_m, n_uav)
    if _min_sep_m(polished) + 1e-9 < theta or not best_score.feasible:
        polished = start
        best_score = start_score
    delta = polished[:, :2] - start[:, :2]
    moved = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
    return PolishResult(
        uav_xyz_m=polished,
        rate_mbps=float(best_score.rate_mbps),
        feasible=bool(best_score.feasible),
        n_evals=int(n_evals),
        start_rate_mbps=float(start_score.rate_mbps),
        start_feasible=bool(start_score.feasible),
        moved_rms_m=moved,
        status=best_score.status,
    )


def nominal_b_eq_hz(scenario: Scenario, scale: float = 1.0) -> float:
    """``scale * B_sys / I``. ``scale`` 0.1 and 10 are the N0 sensitivity."""
    if scale <= 0.0:
        raise ValueError("b_eq scale must be positive")
    return float(scale) * equal_share_reference_hz(scenario.cfg)
