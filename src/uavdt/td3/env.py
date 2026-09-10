"""UAVAoDTEnv: static Problem (P) search wrapped as an MDP.

Physics stay in evaluate(). Inner bandwidth follows TD3Settings
(leftover by default; LP for residual-on-SCA). Best snapshots are
re-scored with the frozen-q LP at export unless the inner step already
used that LP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.channel import (
    average_path_loss_db,
    received_power_w,
    snr,
    sum_rate_ceiling_bit_per_s,
)
from uavdt.config import SimConfig
from uavdt.constraints import pairwise_uav_distance_m
from uavdt.evaluator import EvalResult, evaluate
from uavdt.models import Allocation, Scenario
from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.random import place_random
from uavdt.resources import allocation_from_positions, equal_share_bandwidth_hz
from uavdt.sca.initialize import qos_floor_bandwidth
from uavdt.sca.linearize import spectral_efficiency
from uavdt.td3.decode import action_size, decode_action, observation_size
from uavdt.td3.settings import TD3Settings


def zenith_r_max_bit_per_s(cfg: SimConfig) -> float:
    """R_max = B_sys log2(1 + SNR(d=H)). IMPLEMENTATION CHOICE."""
    dist = np.array([[cfg.uav_height_m]], dtype=float)
    l_avg = average_path_loss_db(dist, cfg.uav_height_m, cfg)
    p_rx = received_power_w(l_avg, cfg)
    snr_lin = float(snr(p_rx, cfg)[0, 0])
    ceiling = sum_rate_ceiling_bit_per_s(cfg.b_sys_hz, max(snr_lin, 0.0))
    return max(ceiling, 1.0)


def dist_penalty(uav_xyz_m: np.ndarray, theta_m: float) -> float:
    """P_dist = Σ_{j<l} max(0, θ − d_jl) / θ."""
    if theta_m <= 0.0:
        return 0.0
    dists = pairwise_uav_distance_m(uav_xyz_m)
    j = uav_xyz_m.shape[0]
    total = 0.0
    for p in range(j):
        for q in range(p + 1, j):
            total += max(0.0, theta_m - float(dists[p, q])) / theta_m
    return float(total)


def aodt_penalty(aodt_s: np.ndarray, threshold_s: float, clip_s: float) -> float:
    """Eq. (34) hinge with +∞ clipped. P_AoDT = Σ_k max(0, min(Δ_k, clip) − T_k)."""
    clipped = np.minimum(np.nan_to_num(aodt_s, nan=clip_s, posinf=clip_s), clip_s)
    return float(np.sum(np.maximum(0.0, clipped - threshold_s)))


def violation_penalty(ev: EvalResult, cfg: SimConfig) -> float:
    """QoS / CPU / bandwidth / field. AoDT is not included (see P_AoDT)."""
    c = ev.constraints
    i = max(cfg.num_iot, 1)
    j = max(cfg.num_uav, 1)
    bw_bad = 0.0 if (c.bandwidth_budget_ok and c.bandwidth_support_ok) else 1.0
    out = 0.0 if c.uav_in_field_ok else 1.0
    return float(c.qos_violations) / i + float(c.cpu_unstable_count) / j + bw_bad + out


def copy_allocation(allocation: Allocation) -> Allocation:
    return Allocation(
        association=np.asarray(allocation.association, dtype=float).copy(),
        processing=np.asarray(allocation.processing, dtype=float).copy(),
        bandwidth_hz=np.asarray(allocation.bandwidth_hz, dtype=float).copy(),
    )


def reward_breakdown(
    ev: EvalResult,
    uav_xyz_m: np.ndarray,
    cfg: SimConfig,
    settings: TD3Settings,
    r_max: float,
) -> tuple[float, dict[str, float]]:
    """Alg. 2 penalty, or feasible Mbps (residual-on-SCA / Problem (P))."""
    p_aodt = aodt_penalty(ev.aodt_s, cfg.aodt_threshold_s, settings.aodt_clip_s)
    p_dist = dist_penalty(uav_xyz_m, cfg.uav_min_separation_m)
    v_viol = violation_penalty(ev, cfg)
    rate_term = float(settings.rate_weight * ev.sum_rate_bit_per_s / r_max)
    if settings.reward_mode == "feasible_rate":
        reward = (
            float(ev.sum_rate_mbps)
            if ev.feasible
            else float(settings.infeasible_reward)
        )
    else:
        reward = float(
            rate_term
            - settings.aodt_weight * p_aodt
            - settings.dist_weight * p_dist
            - settings.viol_weight * v_viol
        )
    return reward, {
        "rate_term": rate_term,
        "p_aodt": float(p_aodt),
        "p_dist": float(p_dist),
        "v_viol": float(v_viol),
        "sum_rate_Mbps": float(ev.sum_rate_mbps),
        "feasible": float(ev.feasible),
        "reward_mode": 1.0 if settings.reward_mode == "feasible_rate" else 0.0,
    }


def step_reward(
    ev: EvalResult,
    uav_xyz_m: np.ndarray,
    cfg: SimConfig,
    settings: TD3Settings,
    r_max: float,
) -> float:
    """Reward for the configured TD3Settings.reward_mode."""
    return reward_breakdown(ev, uav_xyz_m, cfg, settings, r_max)[0]


def build_observation(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    allocation: Allocation,
    ev: EvalResult,
    settings: TD3Settings,
) -> np.ndarray:
    cfg = scenario.cfg
    i, j = cfg.num_iot, cfg.num_uav
    ax = max(cfg.area_x_m, 1e-12)
    ay = max(cfg.area_y_m, 1e-12)
    uav = np.asarray(uav_xyz_m, dtype=float)
    iot = np.asarray(scenario.iot_xyz_m, dtype=float)
    uav_xy = np.empty(2 * j, dtype=float)
    uav_xy[0::2] = uav[:, 0] / ax
    uav_xy[1::2] = uav[:, 1] / ay
    iot_xy = np.empty(2 * i, dtype=float)
    iot_xy[0::2] = iot[:, 0] / ax
    iot_xy[1::2] = iot[:, 1] / ay
    lam = np.asarray(scenario.lambdas_per_s, dtype=float)
    lam_scale = max(float(np.max(lam)), 1e-9)
    rho = np.clip(np.asarray(ev.rho, dtype=float), 0.0, settings.rho_clip)
    tk = max(cfg.aodt_threshold_s, 1e-12)
    aodt = np.asarray(ev.aodt_s, dtype=float)
    aodt_ratio = np.where(
        np.isfinite(aodt),
        np.clip(aodt / tk, 0.0, settings.aodt_ratio_clip),
        settings.aodt_ratio_clip,
    )
    a = allocation.hard_association().reshape(-1)
    b = allocation.hard_processing().reshape(-1)
    bw = np.asarray(allocation.bandwidth_hz, dtype=float).reshape(-1) / max(
        cfg.b_sys_hz, 1e-12
    )
    bsys = np.array([cfg.b_sys_hz / settings.b_sys_ref_hz], dtype=float)
    obs = np.concatenate(
        [uav_xy, iot_xy, lam / lam_scale, rho, aodt_ratio, a, b, bw, bsys]
    )
    return np.nan_to_num(obs, nan=0.0, posinf=settings.aodt_ratio_clip, neginf=0.0)


@dataclass
class BestSnapshot:
    uav_xyz_m: np.ndarray
    allocation: Allocation
    true_eval: EvalResult
    reward: float
    found_at_step: int = 0


class UAVAoDTEnv:
    """Iterative placement/association search on one frozen Scenario."""

    def __init__(
        self,
        scenario: Scenario,
        settings: TD3Settings | None = None,
        *,
        seed: int,
    ) -> None:
        self.scenario = scenario
        self.settings = settings or TD3Settings()
        self.seed = int(seed)
        self.obs_dim = observation_size(scenario.cfg)
        self.act_dim = action_size(scenario.cfg, self.settings)
        self.n_move = 2 * scenario.cfg.num_uav
        self.n_assoc = (
            0
            if self.settings.action_heads == "move_only"
            else scenario.cfg.num_iot * scenario.cfg.num_uav
        )
        self.r_max = zenith_r_max_bit_per_s(scenario.cfg)
        self._episode = 0
        self._ep_step = 0
        self._global_step = 0
        self._uav: np.ndarray | None = None
        self._alloc: Allocation | None = None
        self._last_info: dict = {}
        self.best: BestSnapshot | None = None
        self.freeze_best: bool = False
        self._uav0: np.ndarray | None = None
        self._alloc0: Allocation | None = None
        self.origin_eval: EvalResult | None = None
        self.origin_wall_clock_s: float = 0.0
        self.origin_stop_reason: str | None = None

    def _origin_allocation(self) -> Allocation | None:
        if self._alloc0 is None:
            return None
        return copy_allocation(self._alloc0)

    def _init_origin(self) -> None:
        """One UAV origin per frozen IoT layout. SCA is called at most once."""
        if self._uav0 is not None:
            return
        from time import perf_counter

        cfg = self.scenario.cfg
        t0 = perf_counter()
        if self.settings.uav_init == "sca":
            from uavdt.sca.algorithm import solve_sca
            from uavdt.sca.settings import SCASettings

            try:
                result = solve_sca(
                    self.scenario,
                    self.seed,
                    settings=SCASettings(solver=None),
                )
            except RuntimeError as exc:
                if "CPU stability" not in str(exc):
                    raise
                self._uav0 = place_kmeans(self.scenario, int(self.seed))
                self._alloc0 = allocation_from_positions(self.scenario, self._uav0)
                self.origin_stop_reason = "init_cpu_unstable"
                self.origin_wall_clock_s = perf_counter() - t0
                return
            self._uav0 = np.asarray(result.uav_xyz_m, dtype=float).copy()
            self._alloc0 = copy_allocation(result.allocation)
            self.origin_eval = result.true_eval
            self.origin_stop_reason = str(result.diagnostics.get("stop_reason", ""))
            self.origin_wall_clock_s = perf_counter() - t0
            return
        if self.settings.uav_init == "kmeans":
            self._uav0 = place_kmeans(self.scenario, int(self.seed))
        else:
            self._uav0 = place_random(cfg.num_uav, int(self.seed), cfg)
        self._alloc0 = allocation_from_positions(self.scenario, self._uav0)
        self.origin_wall_clock_s = perf_counter() - t0

    def reset(self) -> np.ndarray:
        cfg = self.scenario.cfg
        # Per-instance solve: one UAV init for the frozen IoT layout. Re-sampling
        # q every episode trains a multi-start controller and was collapsing to
        # field-corner bang-bang instead of improving this scenario.
        self._init_origin()
        self._uav = np.asarray(self._uav0, dtype=float).copy()
        if self.settings.assoc_mode == "frozen" or self.settings.process_mode == "frozen":
            if self._alloc0 is None:
                raise RuntimeError("frozen a/b requested but origin allocation is missing")
            seed_alloc = copy_allocation(self._alloc0)
        else:
            seed_alloc = allocation_from_positions(self.scenario, self._uav)
        self._alloc = self._inner_bandwidth(self._uav, seed_alloc)
        ev = evaluate(self.scenario, self._uav, self._alloc)
        if self.origin_eval is None:
            self.origin_eval = ev
        reward, terms = reward_breakdown(
            ev, self._uav, cfg, self.settings, self.r_max
        )
        self._consider_best(self._uav, self._alloc, ev, reward)
        self._ep_step = 0
        self._last_info = {"eval": ev, "reward": float(reward), **terms}
        return build_observation(
            self.scenario, self._uav, self._alloc, ev, self.settings
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self._uav is None or self._alloc is None:
            raise RuntimeError("reset() must be called before step()")
        uav, alloc = decode_action(
            action,
            self._uav,
            self.scenario,
            self.settings,
            origin_xyz_m=self._uav0,
            origin_allocation=self._origin_allocation(),
        )
        alloc = self._inner_bandwidth(uav, alloc)
        ev = evaluate(self.scenario, uav, alloc)
        reward, terms = reward_breakdown(
            ev, uav, self.scenario.cfg, self.settings, self.r_max
        )
        self._uav = uav
        self._alloc = alloc
        self._ep_step += 1
        self._global_step += 1
        done = self._ep_step >= self.settings.horizon
        self._consider_best(uav, alloc, ev, reward)
        obs = build_observation(self.scenario, uav, alloc, ev, self.settings)
        if done:
            self._episode += 1
        info = {"eval": ev, "reward": float(reward), **terms}
        self._last_info = info
        return obs, float(reward), bool(done), info

    def _inner_bandwidth(self, uav: np.ndarray, alloc: Allocation) -> Allocation:
        """Algorithm 2 'update bandwidth' after parsing a_ij / b_ij."""
        a = alloc.hard_association()
        b = alloc.hard_processing()
        cfg = self.scenario.cfg
        mode = self.settings.inner_bandwidth
        if mode == "equal_share":
            return Allocation(a, b, equal_share_bandwidth_hz(a, cfg))
        if mode == "lp":
            from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
            from uavdt.sca.settings import SCASettings

            res = solve_bandwidth_at_fixed_q(
                self.scenario, uav, a, b, SCASettings(solver=None)
            )
            if res.infeasible:
                if self.settings.reward_mode == "feasible_rate":
                    return Allocation(a, b, np.zeros_like(a, dtype=float))
                return Allocation(a, b, equal_share_bandwidth_hz(a, cfg))
            return Allocation(a, b, res.bandwidth_hz)
        se = spectral_efficiency(self.scenario.iot_xyz_m, uav, cfg)
        bw = qos_floor_bandwidth(a, se, cfg)
        return Allocation(a, b, bw)

    def _consider_best(
        self,
        uav: np.ndarray,
        alloc: Allocation,
        ev: EvalResult,
        reward: float,
    ) -> None:
        if self.freeze_best:
            return
        key = (int(ev.feasible), float(ev.sum_rate_bit_per_s))
        if self.best is None:
            better = True
        else:
            prev = (
                int(self.best.true_eval.feasible),
                float(self.best.true_eval.sum_rate_bit_per_s),
            )
            better = key > prev
        if better:
            self.best = BestSnapshot(
                uav_xyz_m=np.asarray(uav, dtype=float).copy(),
                allocation=copy_allocation(alloc),
                true_eval=ev,
                reward=float(reward),
                found_at_step=int(self._global_step),
            )
