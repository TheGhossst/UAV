"""Min-spectrum B_sys at frozen (q, a, b). Experiment E1.

Paper mapping
-------------
Eq. (6): r_ij = B_ij * SE_ij(q).
(25): r_i >= R_min.
(31)/(17): D_i = S_i/r_i [+ T_u2u] and D_Nk + Q_k <= T_k.
At frozen q, a, b the AoDT/QoS floors are exact
``B_ij >= max(R_min, S/slack_i) / SE_ij`` (see ``bandwidth_floors_hz``).
(27): sum B_ij <= B_sys.
Optional EXTERNAL cap: B_ij <= c * B_sys.

Then the smallest feasible pool is
    B_sys >= max( sum floors,  max(floors)/c ).
Leftover-dump sum-rate is not the score. Hertz is.

Placement is *not* re-solved here. Callers freeze a method's geometry
(typically from the 8.8 MHz / 25% leftover-dump radio) and ask how much
spectrum that geometry still needs. Re-solving SCA at each candidate
B_sys is a different experiment (``min_bsys_resolve_method``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from uavdt.evaluator import evaluate
from uavdt.models import Allocation, Scenario
from uavdt.resources import cpu_stable_processing, nearest_association
from uavdt.sca.cvx_problem import bandwidth_floors_hz, solve_bandwidth_at_fixed_q
from uavdt.sca.linearize import spectral_efficiency
from uavdt.sca.settings import SCASettings


@dataclass(frozen=True)
class MinSpectrumResult:
    """Smallest B_sys that keeps frozen (q, a, b) feasible."""

    min_b_sys_hz: float | None
    bound_hz: float | None
    binding: str
    feasible_at_place_radio: bool
    feasible_at_min: bool
    n_lp: int
    sum_floors_hz: float
    max_floor_hz: float
    min_assoc_se: float
    n_forwarding: int
    sum_rate_Mbps_at_min: float | None
    place_sum_rate_Mbps: float
    wall_clock_s: float
    message: str = ""


def scenario_with_radio(
    scenario: Scenario,
    *,
    b_sys_hz: float,
    max_bw_share: float | None,
) -> Scenario:
    """Copy the geometry; only the (27) pool and optional per-link cap change."""
    return replace(
        scenario,
        cfg=replace(
            scenario.cfg,
            b_sys_hz=float(b_sys_hz),
            max_bw_share=max_bw_share,
        ),
    )


def associated_floors_hz(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-associated-link AoDT/QoS floors (Hz) and SE_ij."""
    se = spectral_efficiency(scenario.iot_xyz_m, uav_xyz_m, scenario.cfg)
    floors = bandwidth_floors_hz(scenario, association, processing, se)
    return floors, se


def min_bsys_lower_bound_hz(
    floors: np.ndarray,
    association: np.ndarray,
    max_bw_share: float | None,
) -> dict:
    """Closed-form B_sys lower bound. Floors do not depend on B_sys.

    Returns inf if any associated floor is non-finite (AoDT slack <= 0
    or SE = 0): no pool size can repair that geometry.
    """
    a = association > 0.5
    link = np.asarray(floors, dtype=float)[a]
    if link.size == 0 or np.any(~np.isfinite(link)):
        return {
            "bound_hz": math.inf,
            "binding": "infeasible_floors",
            "sum_floors_hz": float("nan"),
            "max_floor_hz": float("nan"),
        }
    sum_f = float(np.sum(link))
    max_f = float(np.max(link))
    if max_bw_share is None:
        return {
            "bound_hz": sum_f,
            "binding": "sum_floors",
            "sum_floors_hz": sum_f,
            "max_floor_hz": max_f,
        }
    cap_req = max_f / float(max_bw_share)
    if cap_req > sum_f + 1e-9:
        binding = "per_link_cap"
        bound = cap_req
    else:
        binding = "sum_floors"
        bound = sum_f
    return {
        "bound_hz": float(bound),
        "binding": binding,
        "sum_floors_hz": sum_f,
        "max_floor_hz": max_f,
    }


def hertz_bound_at_layout(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    *,
    association: np.ndarray | None = None,
    processing: np.ndarray | None = None,
) -> dict:
    """Closed-form min B_sys at this q with nearest a / CPU-stable b.

    Ranking oracle for E1b. Does not run the leftover-dump LP.
    """
    uav = np.asarray(uav_xyz_m, dtype=float)
    a = (
        nearest_association(scenario.iot_xyz_m, uav)
        if association is None
        else np.asarray(association, dtype=float)
    )
    b = (
        cpu_stable_processing(scenario, a)
        if processing is None
        else np.asarray(processing, dtype=float)
    )
    floors, se = associated_floors_hz(scenario, uav, a, b)
    info = min_bsys_lower_bound_hz(floors, a, scenario.cfg.max_bw_share)
    j_assoc = np.argmax(a, axis=1)
    se_assoc = np.array(
        [float(se[i, int(j_assoc[i])]) for i in range(a.shape[0])],
        dtype=float,
    )
    return {
        **info,
        "allocation": Allocation(a, b, np.zeros_like(a)),
        "min_assoc_se": float(np.min(se_assoc)) if se_assoc.size else float("nan"),
        "n_forwarding": int(np.sum(b[np.arange(a.shape[0]), j_assoc] < 0.5)),
    }


def frozen_q_feasible(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    b_sys_hz: float,
    max_bw_share: float | None,
    settings: SCASettings | None = None,
) -> tuple[bool, object, object]:
    """One leftover-dump LP + evaluate() at a candidate pool size."""
    sc = scenario_with_radio(
        scenario, b_sys_hz=b_sys_hz, max_bw_share=max_bw_share
    )
    res = solve_bandwidth_at_fixed_q(
        sc, uav_xyz_m, association, processing, settings
    )
    if res.infeasible:
        return False, None, res
    ev = evaluate(
        sc,
        uav_xyz_m,
        Allocation(association, processing, res.bandwidth_hz),
    )
    return bool(ev.feasible), ev, res


def min_bsys_frozen_q(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    allocation: Allocation,
    *,
    max_bw_share: float | None = None,
    hi_hz: float = 8.8e6,
    abs_tol_hz: float = 1.0e3,
    rel_tol: float = 1.0e-4,
    max_hi_hz: float = 1.0e8,
    settings: SCASettings | None = None,
    place_eval=None,
) -> MinSpectrumResult:
    """Binary-search the smallest feasible B_sys at frozen q, a, b.

    ``max_bw_share`` defaults to the scenario's cap (headline 0.25).
    The closed-form bound is the start; the search only absorbs LP
    numerics. Score is ``min_b_sys_hz`` (None = infeasible at any pool).
    """
    t0 = perf_counter()
    settings = settings or SCASettings(solver=None)
    share = (
        scenario.cfg.max_bw_share if max_bw_share is None else max_bw_share
    )
    a = allocation.hard_association()
    b = allocation.hard_processing()
    uav = np.asarray(uav_xyz_m, dtype=float)
    if place_eval is None:
        place_eval = evaluate(scenario, uav, allocation)
    floors, se = associated_floors_hz(scenario, uav, a, b)
    info = min_bsys_lower_bound_hz(floors, a, share)
    bound = float(info["bound_hz"])
    j_assoc = np.argmax(a, axis=1)
    se_assoc = np.array(
        [float(se[i, int(j_assoc[i])]) for i in range(a.shape[0])],
        dtype=float,
    )
    n_fwd = int(np.sum(b[np.arange(a.shape[0]), j_assoc] < 0.5))
    n_lp = 0

    def _probe(hz: float) -> tuple[bool, object]:
        nonlocal n_lp
        n_lp += 1
        ok, ev, _res = frozen_q_feasible(
            scenario, uav, a, b, hz, share, settings
        )
        return ok, ev

    if not np.isfinite(bound) or bound <= 0.0:
        return MinSpectrumResult(
            min_b_sys_hz=None,
            bound_hz=None if not np.isfinite(bound) else bound,
            binding=str(info["binding"]),
            feasible_at_place_radio=bool(place_eval.feasible),
            feasible_at_min=False,
            n_lp=n_lp,
            sum_floors_hz=float(info["sum_floors_hz"]),
            max_floor_hz=float(info["max_floor_hz"]),
            min_assoc_se=float(np.min(se_assoc)) if se_assoc.size else float("nan"),
            n_forwarding=n_fwd,
            sum_rate_Mbps_at_min=None,
            place_sum_rate_Mbps=float(place_eval.sum_rate_mbps),
            wall_clock_s=perf_counter() - t0,
            message="AoDT slack non-positive or empty association",
        )

    hi = max(float(hi_hz), bound)
    ok_hi, ev_hi = _probe(hi)
    while not ok_hi and hi < max_hi_hz:
        hi = min(max_hi_hz, hi * 2.0)
        ok_hi, ev_hi = _probe(hi)
    if not ok_hi:
        return MinSpectrumResult(
            min_b_sys_hz=None,
            bound_hz=bound,
            binding=str(info["binding"]),
            feasible_at_place_radio=bool(place_eval.feasible),
            feasible_at_min=False,
            n_lp=n_lp,
            sum_floors_hz=float(info["sum_floors_hz"]),
            max_floor_hz=float(info["max_floor_hz"]),
            min_assoc_se=float(np.min(se_assoc)),
            n_forwarding=n_fwd,
            sum_rate_Mbps_at_min=None,
            place_sum_rate_Mbps=float(place_eval.sum_rate_mbps),
            wall_clock_s=perf_counter() - t0,
            message=f"infeasible up to {hi:g} Hz",
        )

    lo = 0.0
    ev_star = ev_hi
    while (hi - lo) > max(float(abs_tol_hz), float(rel_tol) * hi):
        mid = 0.5 * (lo + hi)
        ok, ev = _probe(mid)
        if ok:
            hi = mid
            ev_star = ev
        else:
            lo = mid

    min_hz = float(math.ceil(hi))
    if min_hz > hi + 1e-9:
        ok_ceil, ev_ceil = _probe(min_hz)
        if ok_ceil:
            ev_star = ev_ceil
        else:
            min_hz = float(math.ceil(hi + abs_tol_hz))
            ok_pad, ev_pad = _probe(min_hz)
            if ok_pad:
                ev_star = ev_pad
    rate = None if ev_star is None else float(ev_star.sum_rate_mbps)
    return MinSpectrumResult(
        min_b_sys_hz=min_hz,
        bound_hz=bound,
        binding=str(info["binding"]),
        feasible_at_place_radio=bool(place_eval.feasible),
        feasible_at_min=True,
        n_lp=n_lp,
        sum_floors_hz=float(info["sum_floors_hz"]),
        max_floor_hz=float(info["max_floor_hz"]),
        min_assoc_se=float(np.min(se_assoc)),
        n_forwarding=n_fwd,
        sum_rate_Mbps_at_min=rate,
        place_sum_rate_Mbps=float(place_eval.sum_rate_mbps),
        wall_clock_s=perf_counter() - t0,
        message="ok",
    )


def min_bsys_resolve_method(
    scenario: Scenario,
    method: str,
    seed: int,
    *,
    max_bw_share: float | None = None,
    hi_hz: float = 8.8e6,
    abs_tol_hz: float = 1.0e3,
    rel_tol: float = 1.0e-4,
    max_hi_hz: float = 1.0e8,
    sca_settings: SCASettings | None = None,
    pso_settings=None,
    anchor_settings=None,
    multistart_settings=None,
) -> dict:
    """Re-run the method at each candidate B_sys. Slow; not the E1 core."""
    from uavdt.experiments.methods import run_method

    t0 = perf_counter()
    share = (
        scenario.cfg.max_bw_share if max_bw_share is None else max_bw_share
    )
    n_run = 0

    def _ok(hz: float) -> bool:
        nonlocal n_run
        n_run += 1
        sc = scenario_with_radio(scenario, b_sys_hz=hz, max_bw_share=share)
        run = run_method(
            sc,
            method,
            seed,
            sca_settings=sca_settings,
            pso_settings=pso_settings,
            anchor_settings=anchor_settings,
            multistart_settings=multistart_settings,
        )
        return bool(run.feasible)

    hi = float(hi_hz)
    if not _ok(hi):
        while hi < max_hi_hz:
            hi = min(max_hi_hz, hi * 2.0)
            if _ok(hi):
                break
        else:
            return {
                "min_b_sys_hz": None,
                "n_method_runs": n_run,
                "wall_clock_s": perf_counter() - t0,
                "message": f"infeasible up to {hi:g} Hz",
            }
    lo = 0.0
    while (hi - lo) > max(float(abs_tol_hz), float(rel_tol) * hi):
        mid = 0.5 * (lo + hi)
        if _ok(mid):
            hi = mid
        else:
            lo = mid
    return {
        "min_b_sys_hz": float(math.ceil(hi)),
        "n_method_runs": n_run,
        "wall_clock_s": perf_counter() - t0,
        "message": "ok",
    }
