"""Inner bandwidth allocation under bandwidth-proportional noise.

The frozen radio is Eq. (6): SNR = p g / sigma^2, so rate is linear in
B and the inner problem is an LP. This module does not change that path.

Alternative model, used only here:

    SNR(B) = p g / (N0 * B)
    r(B) = B log2(1 + p g / (N0 * B))

N0 is chosen so the noise power in an equal-share Hertz matches the
frozen sigma^2. With every IoT associated, equal share is

    B_eq = B_sys / I
    N0 = sigma^2 / B_eq

and therefore SNR(B_eq) = p g / sigma^2. At any other bandwidth the SNR
moves as B_eq / B.

r(B) is concave and increasing in B. CVXPY represents it with rel_entr
(natural log):

    B log(1 + c/B) = -rel_entr(B, B + c)
    c = p g / N0 = SNR_const * B_eq

Maximizing sum rate subject to the usual sum cap, per-link cap, and
the QoS / AoDT rate floors is a convex program. UAV positions and the
frozen SCA solver are not updated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from uavdt.channel import average_path_loss_db, distances_m, received_power_w
from uavdt.config import SimConfig
from uavdt.models import Scenario
from uavdt.sca.cvx_problem import rate_floors_bit_per_s

# rel_entr needs an exponential-cone solver. HiGHS is LP-only.
_EXP_SOLVERS = ("CLARABEL", "MOSEK", "SCS", "ECOS")


def equal_share_reference_hz(cfg: SimConfig) -> float:
    """B_sys / I. Matches equal-share when each IoT has one associated link."""
    n = int(cfg.num_iot)
    if n <= 0:
        raise ValueError("num_iot must be positive")
    return float(cfg.b_sys_hz) / float(n)


def calibrated_noise_density_w_per_hz(cfg: SimConfig) -> float:
    """N0 such that N0 * B_eq = sigma^2, with B_eq = B_sys / I.

    SNR at B_eq then equals the frozen p g / sigma^2.
    """
    b_eq = equal_share_reference_hz(cfg)
    if b_eq <= 0.0:
        raise ValueError("equal-share bandwidth must be positive")
    return float(cfg.noise_power_w) / b_eq


def constant_noise_snr(
    iot_xyz_m: np.ndarray,
    uav_xyz_m: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Frozen Eq. (6) SNR, p_rx / sigma^2. Independent of bandwidth."""
    iot = np.asarray(iot_xyz_m, dtype=float)
    uav = np.asarray(uav_xyz_m, dtype=float)
    height = float(uav[0, 2]) if uav.size else float(cfg.uav_height_m)
    dist = distances_m(iot, uav)
    p_rx = received_power_w(average_path_loss_db(dist, height, cfg), cfg)
    return p_rx / float(cfg.noise_power_w)


def proportional_snr(
    bandwidth_hz: np.ndarray,
    snr_const: np.ndarray,
    b_eq_hz: float,
) -> np.ndarray:
    """SNR(B) = snr_const * B_eq / B. Zero where B is zero."""
    b = np.asarray(bandwidth_hz, dtype=float)
    s = np.asarray(snr_const, dtype=float)
    out = np.zeros(np.broadcast(b, s).shape, dtype=float)
    mask = b > 0.0
    out[mask] = s[mask] * float(b_eq_hz) / b[mask]
    return out


def proportional_rate_bit_per_s(
    bandwidth_hz: np.ndarray,
    snr_const: np.ndarray,
    b_eq_hz: float,
) -> np.ndarray:
    """r = B log2(1 + snr_const * B_eq / B). Zero at B = 0."""
    b = np.asarray(bandwidth_hz, dtype=float)
    s = np.asarray(snr_const, dtype=float)
    out = np.zeros(np.broadcast(b, s).shape, dtype=float)
    mask = b > 0.0
    ratio = np.zeros(out.shape, dtype=float)
    ratio[mask] = s[mask] * float(b_eq_hz) / b[mask]
    out[mask] = b[mask] * np.log2(1.0 + ratio[mask])
    return out


def _asymptote_bit_per_s(snr_const: float, b_eq_hz: float) -> float:
    """lim_{B->inf} r(B) = (snr_const * B_eq) / ln(2)."""
    return float(snr_const) * float(b_eq_hz) / math.log(2.0)


def _min_bandwidth_hz(
    snr_const: float,
    floor_bit_per_s: float,
    b_eq_hz: float,
    cap_hz: float,
) -> float:
    """Smallest B in [0, cap] with r(B) >= floor. inf if none."""
    floor = float(floor_bit_per_s)
    if floor <= 0.0:
        return 0.0
    snr = float(snr_const)
    cap = float(cap_hz)
    if snr <= 0.0 or cap <= 0.0:
        return math.inf
    if floor >= _asymptote_bit_per_s(snr, b_eq_hz):
        return math.inf
    cap_rate = float(
        proportional_rate_bit_per_s(np.array([cap]), np.array([snr]), b_eq_hz)[0]
    )
    if cap_rate + 1e-6 < floor:
        return math.inf
    lo = 0.0
    hi = cap
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        rate = float(
            proportional_rate_bit_per_s(np.array([mid]), np.array([snr]), b_eq_hz)[0]
        )
        if rate >= floor:
            hi = mid
        else:
            lo = mid
    return float(hi)


def _installed_exp_solvers() -> list[str]:
    import cvxpy as cp

    installed = set(cp.installed_solvers())
    return [name for name in _EXP_SOLVERS if name in installed]


@dataclass
class ProportionalSolveResult:
    status: str
    solver_name: str
    bandwidth_hz: np.ndarray
    snr_const: np.ndarray
    n0_w_per_hz: float
    b_eq_hz: float
    rates_bit_per_s: np.ndarray
    objective_bit_per_s: float
    solve_time_s: float
    infeasible: bool
    message: str = ""

    @property
    def objective_mbps(self) -> float:
        if not np.isfinite(self.objective_bit_per_s):
            return float("nan")
        return float(self.objective_bit_per_s) / 1.0e6


def allocate_proportional(
    snr_const: np.ndarray,
    rate_floors_bit_per_s: np.ndarray,
    *,
    b_eq_hz: float,
    b_sys_hz: float,
    cap_hz: float,
    solver: str | None = None,
) -> ProportionalSolveResult:
    """Maximize sum_i B_i log2(1 + snr_i B_eq / B_i) on n links.

    ``snr_const`` is the frozen p g / sigma^2 of each link. Bandwidth is
    a vector (one entry per link), not an (I, J) matrix.
    """
    import cvxpy as cp

    snr = np.asarray(snr_const, dtype=float).ravel()
    floors = np.asarray(rate_floors_bit_per_s, dtype=float).ravel()
    n = int(snr.size)
    b_eq = float(b_eq_hz)
    b_sys = float(b_sys_hz)
    cap = float(cap_hz)
    empty = np.zeros(n, dtype=float)
    t0 = perf_counter()

    def _bad(status: str, message: str, solver_name: str = "") -> ProportionalSolveResult:
        return ProportionalSolveResult(
            status=status,
            solver_name=solver_name,
            bandwidth_hz=empty.copy(),
            snr_const=snr.copy(),
            n0_w_per_hz=float("nan"),
            b_eq_hz=b_eq,
            rates_bit_per_s=empty.copy(),
            objective_bit_per_s=float("nan"),
            solve_time_s=perf_counter() - t0,
            infeasible=True,
            message=message,
        )

    if n == 0:
        return _bad("no_links", "no associated links")
    if b_eq <= 0.0 or b_sys <= 0.0 or cap <= 0.0:
        return _bad("bad_budget", "bandwidth parameters must be positive")
    if floors.shape != snr.shape:
        raise ValueError("rate floors must match snr_const")
    if np.any(~np.isfinite(floors)) or np.any(floors < 0.0):
        return _bad("rate_floor_invalid", "rate floor is negative or non-finite")

    mins = np.array(
        [_min_bandwidth_hz(float(snr[i]), float(floors[i]), b_eq, cap) for i in range(n)]
    )
    if np.any(~np.isfinite(mins)) or float(mins.sum()) > b_sys + 1.0:
        return _bad(
            "rate_floor_infeasible",
            "QoS/AoDT floors do not fit in the cap and B_sys",
        )

    names = _installed_exp_solvers()
    if solver is not None:
        want = solver.upper()
        if want not in names and want not in set(cp.installed_solvers()):
            return _bad("solver_missing", f"{solver} is not installed")
        names = [want]
    if not names:
        return _bad("solver_missing", "no exponential-cone solver (CLARABEL/MOSEK/SCS/ECOS)")

    # u = B / B_eq is O(1). r = B_eq * (-rel_entr(u, u+snr) / ln 2).
    u = cp.Variable(n, nonneg=True)
    snr_c = np.maximum(snr, 0.0)
    ent = cp.rel_entr(u, u + snr_c)
    ln2 = float(np.log(2.0))
    u_cap = cap / b_eq
    u_budget = b_sys / b_eq
    # Stay off the rel_entr boundary. 1e-8 * B_eq is a fraction of a Hertz
    # at MHz budgets and is below any QoS floor that the pre-check accepted.
    u_lb = np.full(n, 1e-8)
    cons = [
        u >= u_lb,
        u <= u_cap,
        cp.sum(u) <= u_budget,
        ent <= (-floors * ln2 / b_eq),
    ]
    # Maximize sum -rel_entr(u, u+snr). Multiplying by B_eq/ln2 would
    # scale the objective to ~1e7 and makes cone solvers declare the
    # same point inaccurate. Reported bit/s use the closed form.
    problem = cp.Problem(cp.Maximize(cp.sum(-ent)), cons)

    last_message = "no solver attempted"
    solver_name = names[0]
    for solver_name in names:
        try:
            problem.solve(solver=solver_name, verbose=False)
        except Exception as exc:  # noqa: BLE001
            last_message = str(exc)
            continue
        status = str(problem.status)
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
            bw = np.maximum(np.asarray(u.value, dtype=float).ravel(), 0.0) * b_eq
            bw = np.minimum(bw, cap)
            if float(bw.sum()) > b_sys + 1e-3:
                bw *= b_sys / float(bw.sum())
            rates = proportional_rate_bit_per_s(bw, snr, b_eq)
            short = rates + 1.0 < floors
            over = float(bw.sum()) > b_sys + 1e-2
            if np.any(short) or over:
                return _bad(
                    status,
                    "solver point missed a rate floor or the sum cap",
                    solver_name,
                )
            return ProportionalSolveResult(
                status=status,
                solver_name=solver_name,
                bandwidth_hz=bw,
                snr_const=snr.copy(),
                n0_w_per_hz=float("nan"),
                b_eq_hz=b_eq,
                rates_bit_per_s=rates,
                objective_bit_per_s=float(np.sum(rates)),
                solve_time_s=perf_counter() - t0,
                infeasible=False,
                message=status,
            )
        last_message = status
    return _bad(str(problem.status), last_message, solver_name)


def solve_bandwidth_proportional(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    *,
    solver: str | None = None,
    b_eq_hz: float | None = None,
) -> ProportionalSolveResult:
    """Concave inner allocation at frozen (q, a, b).

    Rate floors are the same QoS / AoDT floors the frozen LP uses
    (``rate_floors_bit_per_s``). Only the map from B to rate changes.

    ``b_eq_hz`` overrides the calibration bandwidth. The default is
    ``B_sys / I``, which sets ``N0 * B_eq = sigma^2``. A scale of 0.1
    or 10 moves the matched-SNR point and changes how quickly extra
    Hertz saturates. The sum cap and the per-link cap stay on ``B_sys``.
    """
    cfg = scenario.cfg
    uav = np.asarray(uav_xyz_m, dtype=float)
    a = np.asarray(association, dtype=float) > 0.5
    if b_eq_hz is None:
        b_eq = equal_share_reference_hz(cfg)
    else:
        b_eq = float(b_eq_hz)
        if b_eq <= 0.0:
            raise ValueError("b_eq_hz must be positive")
    n0 = float(cfg.noise_power_w) / b_eq
    snr = constant_noise_snr(scenario.iot_xyz_m, uav, cfg)
    floors = rate_floors_bit_per_s(scenario, association, processing)
    t0 = perf_counter()

    def _bad(status: str, message: str) -> ProportionalSolveResult:
        z = np.zeros_like(snr)
        return ProportionalSolveResult(
            status=status,
            solver_name="",
            bandwidth_hz=z,
            snr_const=snr,
            n0_w_per_hz=n0,
            b_eq_hz=b_eq,
            rates_bit_per_s=z,
            objective_bit_per_s=float("nan"),
            solve_time_s=perf_counter() - t0,
            infeasible=True,
            message=message,
        )

    if np.any(~np.isfinite(floors[a])):
        return _bad("aodt_slack_nonpositive", "AoDT slack non-positive or SE = 0")
    if not np.any(a):
        return _bad("no_links", "no associated links")

    rows, cols = np.where(a)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    snr_links = np.array([float(snr[i, j]) for i, j in pairs])
    floor_links = np.array([float(floors[i, j]) for i, j in pairs])
    solved = allocate_proportional(
        snr_links,
        floor_links,
        b_eq_hz=b_eq,
        b_sys_hz=float(cfg.b_sys_hz),
        cap_hz=float(cfg.link_bandwidth_cap_hz),
        solver=solver,
    )
    bw = np.zeros_like(snr)
    rates = np.zeros_like(snr)
    for (i, j), b_ij, r_ij in zip(pairs, solved.bandwidth_hz, solved.rates_bit_per_s):
        bw[i, j] = float(b_ij)
        rates[i, j] = float(r_ij)
    return ProportionalSolveResult(
        status=solved.status,
        solver_name=solved.solver_name,
        bandwidth_hz=bw,
        snr_const=snr,
        n0_w_per_hz=n0,
        b_eq_hz=b_eq,
        rates_bit_per_s=rates,
        objective_bit_per_s=solved.objective_bit_per_s,
        solve_time_s=perf_counter() - t0,
        infeasible=solved.infeasible,
        message=solved.message,
    )
