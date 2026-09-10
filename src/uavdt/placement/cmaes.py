"""CMA-ES on a residual Δxy around an incumbent (SCA) placement.

Non-RL control for residual-on-SCA: same frozen a/b, frozen-q bandwidth LP,
feasible-Mbps fitness. Not a named proposed swarm.

Strategy parameters follow Hansen (2016) tutorial defaults. Searches the
residual in metres, then the caller may SCA-polish the best q.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uavdt.evaluator import EvalResult, evaluate
from uavdt.models import Allocation, Scenario
from uavdt.scenario import make_uav_xyz_m
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
from uavdt.sca.settings import SCASettings


@dataclass
class CMAResult:
    x: np.ndarray
    value: float
    n_evals: int
    history: list[float] = field(default_factory=list)


def residual_bounds(origin_xyz_m: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Δxy box so origin+Δ stays in the field."""
    origin = np.asarray(origin_xyz_m, dtype=float)[:, :2]
    j = origin.shape[0]
    lower = np.empty(2 * j, dtype=float)
    upper = np.empty(2 * j, dtype=float)
    lower[0::2] = -origin[:, 0]
    lower[1::2] = -origin[:, 1]
    upper[0::2] = cfg.area_x_m - origin[:, 0]
    upper[1::2] = cfg.area_y_m - origin[:, 1]
    return lower, upper


def apply_residual_xy(origin_xyz_m: np.ndarray, delta_xy: np.ndarray, cfg) -> np.ndarray:
    origin = np.asarray(origin_xyz_m, dtype=float)
    j = origin.shape[0]
    delta = np.asarray(delta_xy, dtype=float).reshape(j, 2)
    xy = origin[:, :2] + delta
    xy[:, 0] = np.clip(xy[:, 0], 0.0, cfg.area_x_m)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, cfg.area_y_m)
    return make_uav_xyz_m(xy, cfg.uav_height_m)


def feasible_lp_score(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    *,
    infeasible_reward: float = -100.0,
) -> tuple[float, EvalResult]:
    """Same (P) score as residual-on-SCA: LP B, evaluate(), feasible Mbps else penalty."""
    a = np.asarray(association, dtype=float)
    b = np.asarray(processing, dtype=float)
    res = solve_bandwidth_at_fixed_q(
        scenario, uav_xyz_m, a, b, SCASettings(solver=None)
    )
    if res.infeasible:
        bw = np.zeros_like(a, dtype=float)
    else:
        bw = res.bandwidth_hz
    ev = evaluate(scenario, uav_xyz_m, Allocation(a, b, bw))
    reward = float(ev.sum_rate_mbps) if ev.feasible else float(infeasible_reward)
    return reward, ev


def cmaes_maximize(
    objective,
    x0: np.ndarray,
    *,
    sigma0: float = 5.0,
    max_evals: int = 400,
    seed: int = 0,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> CMAResult:
    """Maximize `objective(x)` with CMA-ES. `x0` is the residual (often zeros)."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    n = int(x0.size)
    if n < 1:
        raise ValueError("x0 must be non-empty")
    lam = int(4 + np.floor(3.0 * np.log(n)))
    mu = max(lam // 2, 1)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
    weights = weights / float(np.sum(weights))
    mueff = 1.0 / float(np.sum(weights**2))
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = n**0.5 * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

    mean = x0.copy()
    sigma = float(sigma0)
    cov = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    b_mat = np.eye(n)
    d_vec = np.ones(n)
    eigeneval = 0
    counteval = 0
    rng = np.random.default_rng(int(seed))
    best_x = mean.copy()
    best_f = float("-inf")
    history: list[float] = []
    lo = None if lower is None else np.asarray(lower, dtype=float).reshape(-1)
    hi = None if upper is None else np.asarray(upper, dtype=float).reshape(-1)

    def _clip(vec: np.ndarray) -> np.ndarray:
        out = vec.copy()
        if lo is not None:
            out = np.maximum(out, lo)
        if hi is not None:
            out = np.minimum(out, hi)
        return out

    mean = _clip(mean)
    while counteval < max_evals:
        n_off = min(lam, max_evals - counteval)
        if n_off < 1:
            break
        arz = rng.standard_normal((n_off, n))
        ary = arz * d_vec[None, :]
        ary = ary @ b_mat.T
        arx = mean[None, :] + sigma * ary
        for k in range(n_off):
            arx[k] = _clip(arx[k])
        fits = np.empty(n_off, dtype=float)
        for k in range(n_off):
            fits[k] = float(objective(arx[k]))
            counteval += 1
            if fits[k] > best_f:
                best_f = float(fits[k])
                best_x = arx[k].copy()
        history.append(float(np.max(fits)))
        if n_off < mu:
            break
        order = np.argsort(-fits)
        arx = arx[order]
        ary = ary[order]
        mean_old = mean.copy()
        mean = _clip(weights @ arx[:mu])
        y_w = (mean - mean_old) / max(sigma, 1e-12)
        invsqrt = b_mat @ np.diag(1.0 / d_vec) @ b_mat.T
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (invsqrt @ y_w)
        denom = np.sqrt(max(1.0 - (1.0 - cs) ** (2.0 * counteval / lam), 1e-12))
        hsig = float(np.linalg.norm(ps) / denom / chi_n < 1.4 + 2.0 / (n + 1.0))
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w
        artmp = ary[:mu]
        cov = (
            (1.0 - c1 - cmu) * cov
            + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * cov)
            + cmu * (artmp.T * weights) @ artmp
        )
        sigma *= float(np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1.0)))
        sigma = min(max(sigma, 1e-12), 1.0e3)
        if counteval - eigeneval > lam / (c1 + cmu) / n / 10.0:
            eigeneval = counteval
            cov = np.triu(cov) + np.triu(cov, 1).T
            eig, b_mat = np.linalg.eigh(cov)
            d_vec = np.sqrt(np.maximum(eig, 1e-20))
    return CMAResult(x=best_x, value=float(best_f), n_evals=int(counteval), history=history)
