"""TD3 environment and agent following Algorithm 2."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    TD3_ACTOR_LR,
    TD3_BATCH_SIZE,
    TD3_BUFFER_SIZE,
    TD3_CRITIC_LR,
    TD3_GAMMA,
    TD3_HIDDEN,
    TD3_NOISE,
    TD3_NOISE_CLIP,
    TD3_POLICY_DELAY,
    TD3_POS_SCALE,
    TD3_R_MAX,
    TD3_TARGET_NOISE,
    TD3_TAU,
    TD3_TOTAL_STEPS,
    TD3_W_AODT,
    TD3_W_DIST,
    TD3_W_VIOL,
    TD3_WARMUP,
)
from src.evaluator import EvalResult, evaluate
from src.repair import complete_solution, clip_positions, enforce_separation
from src.scenario import Scenario
from src.solvers.kmeans import kmeans


def _mlp(in_dim: int, out_dim: int, hidden: int, out_act: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    ]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = TD3_HIDDEN):
        super().__init__()
        self.net = _mlp(state_dim, action_dim, hidden, nn.Tanh())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = TD3_HIDDEN):
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, 1, hidden)
        self.q2 = _mlp(state_dim + action_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.q1(x)


class ReplayBuffer:
    def __init__(self, cap: int, state_dim: int, action_dim: int):
        self.cap = cap
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((cap, state_dim), np.float32)
        self.a = np.zeros((cap, action_dim), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.ns = np.zeros((cap, state_dim), np.float32)

    def add(self, s, a, r, ns):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx]


class UAVAoDTEnv:
    """Paper Alg. 2 environment. Uses the shared evaluator."""

    def __init__(self, scenario: Scenario, n_uav: int | None = None, seed: int = 0):
        self.scenario = scenario
        self.cfg = scenario.cfg
        self.j = n_uav if n_uav is not None else self.cfg.num_uav
        self.i = self.cfg.num_iot
        self.rng = np.random.default_rng(seed)
        self.uav_xy = np.zeros((self.j, 2))
        self.last_result: EvalResult | None = None
        self.state_dim = 2 * self.j + 2 * self.i + self.cfg.num_processes + 1 + self.j + self.i
        # Δx,Δy, assoc logits, proc logits, bandwidth raw
        self.action_dim = 2 * self.j + 3 * self.i * self.j

    def reset(self, uav_xy: np.ndarray | None = None) -> np.ndarray:
        if uav_xy is None:
            self.uav_xy = kmeans(self.scenario.iot_xy, self.j, self.rng)
        else:
            self.uav_xy = np.asarray(uav_xy, dtype=float).reshape(self.j, 2)
        self.uav_xy = clip_positions(self.uav_xy, self.cfg)
        xy, a, b, bw = complete_solution(self.scenario, self.uav_xy)
        self.uav_xy = xy
        self.last_result = evaluate(self.scenario, xy, a, b, bw)
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        r = self.last_result
        aodt = np.nan_to_num(r.aodt, nan=0.0) if r is not None else np.zeros(self.cfg.num_processes)
        rho = r.rho if r is not None else np.zeros(self.j)
        parts = [
            self.uav_xy.reshape(-1) / max(self.cfg.area_x, 1.0),
            self.scenario.iot_xy.reshape(-1) / max(self.cfg.area_x, 1.0),
            aodt / max(self.cfg.aodt_threshold, 1.0),
            np.array([self.cfg.b_sys / 20_000.0]),
            rho,
            self.scenario.lambdas / max(float(self.scenario.lambdas.max()), 1e-6),
        ]
        return np.concatenate(parts).astype(np.float32)

    def parse_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=float).reshape(-1)
        j, i = self.j, self.i
        dxdy = action[: 2 * j].reshape(j, 2)
        assoc_logits = action[2 * j : 2 * j + i * j].reshape(i, j)
        proc_logits = action[2 * j + i * j : 2 * j + 2 * i * j].reshape(i, j)
        bw_raw = action[2 * j + 2 * i * j : 2 * j + 3 * i * j].reshape(i, j)
        # Alg. 2: update UAV positions Δx, Δy × 10, clip to [0, area]
        xy = self.uav_xy + dxdy * TD3_POS_SCALE
        xy = clip_positions(xy, self.cfg)
        xy, a, b, bw = complete_solution(
            self.scenario,
            xy,
            assoc_logits=assoc_logits,
            proc_logits=proc_logits,
            bandwidth=np.abs(bw_raw) * self.cfg.b_sys,
        )
        return xy, a, b, bw

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, EvalResult]:
        xy, a, b, bw = self.parse_action(action)
        result = evaluate(self.scenario, xy, a, b, bw)
        self.uav_xy = xy
        self.last_result = result
        reward = self._reward(result)
        return self.get_obs(), reward, result

    def set_scenario(self, scenario: Scenario) -> np.ndarray:
        if scenario.cfg.num_iot != self.i:
            raise ValueError("TD3 env I must stay fixed for a trained agent")
        self.scenario = scenario
        self.cfg = scenario.cfg
        return self.reset()

    def _reward(self, result: EvalResult) -> float:
        # Alg. 2: r = sum_rate/R_max - 10 P_AoDT - 5 P_dist - 5 V_viol
        p_aodt = float(result.aodt_violations)
        if result.compute_available and np.any(np.isfinite(result.aodt)):
            p_aodt = float(np.sum(np.maximum(0.0, result.aodt - self.cfg.aodt_threshold)))
        p_dist = float(result.sep_violations)
        v_viol = float(
            result.qos_violations
            + int(result.bw_excess > 1e-6)
            + result.cpu_unstable
            + result.association_violations
            + result.processing_violations
            + result.process_consistency_violations
        )
        return (
            result.sum_rate / TD3_R_MAX
            - TD3_W_AODT * p_aodt
            - TD3_W_DIST * p_dist
            - TD3_W_VIOL * v_viol
        )


@dataclass
class TD3TrainLog:
    rewards: list[float]
    sum_rates: list[float]


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, seed: int = 0, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_t = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_t = Critic(state_dim, action_dim).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=TD3_ACTOR_LR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=TD3_CRITIC_LR)
        self.buffer = ReplayBuffer(TD3_BUFFER_SIZE, state_dim, action_dim)
        self.action_dim = action_dim
        self.total_it = 0

    def act(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s).cpu().numpy()[0]
        if noise > 0:
            a = a + self.rng.normal(0.0, noise, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def train_step(self):
        if self.buffer.size < TD3_BATCH_SIZE:
            return
        self.total_it += 1
        s, a, r, ns = self.buffer.sample(TD3_BATCH_SIZE, self.rng)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        ns = torch.as_tensor(ns, device=self.device)

        with torch.no_grad():
            na = self.actor_t(ns)
            noise = (torch.randn_like(na) * TD3_TARGET_NOISE).clamp(-TD3_NOISE_CLIP, TD3_NOISE_CLIP)
            na = (na + noise).clamp(-1.0, 1.0)
            q1, q2 = self.critic_t(ns, na)
            target = r + TD3_GAMMA * torch.min(q1, q2)

        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        if self.total_it % TD3_POLICY_DELAY == 0:
            loss_a = -self.critic.q1_only(s, self.actor(s)).mean()
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()
            with torch.no_grad():
                for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                    pt.data.mul_(1 - TD3_TAU).add_(TD3_TAU * p.data)
                for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                    pt.data.mul_(1 - TD3_TAU).add_(TD3_TAU * p.data)


def train_td3(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    total_steps: int = TD3_TOTAL_STEPS,
) -> tuple[TD3Agent, UAVAoDTEnv, TD3TrainLog]:
    env = UAVAoDTEnv(scenario, n_uav=n_uav, seed=seed)
    agent = TD3Agent(env.state_dim, env.action_dim, seed=seed)
    log = TD3TrainLog(rewards=[], sum_rates=[])
    state = env.reset()
    for t in range(total_steps):
        if t < TD3_WARMUP:
            action = env.rng.uniform(-1.0, 1.0, size=env.action_dim)
        else:
            action = agent.act(state, noise=TD3_NOISE)
        ns, reward, result = env.step(action)
        agent.buffer.add(state, action, reward, ns)
        if t >= TD3_WARMUP:
            agent.train_step()
        log.rewards.append(float(reward))
        log.sum_rates.append(float(result.sum_rate))
        state = ns
    return agent, env, log


def train_td3_across_scenarios(
    cfg,
    train_seeds: tuple[int, ...] | list[int],
    seed: int = 0,
    n_uav: int | None = None,
    total_steps: int = TD3_TOTAL_STEPS,
    resample_every: int = 40,
) -> tuple[TD3Agent, UAVAoDTEnv, TD3TrainLog]:
    """Train one TD3 policy on a pool of IoT deployments (paper-style generalization)."""
    from src.scenario import generate_scenario

    rng = np.random.default_rng(seed)
    seeds = list(train_seeds)
    scenario = generate_scenario(int(seeds[0]), cfg)
    env = UAVAoDTEnv(scenario, n_uav=n_uav, seed=seed)
    agent = TD3Agent(env.state_dim, env.action_dim, seed=seed)
    log = TD3TrainLog(rewards=[], sum_rates=[])
    state = env.reset()
    for t in range(total_steps):
        if t > 0 and t % resample_every == 0:
            scenario = generate_scenario(int(rng.choice(seeds)), cfg)
            state = env.set_scenario(scenario)
        if t < TD3_WARMUP:
            action = env.rng.uniform(-1.0, 1.0, size=env.action_dim)
        else:
            action = agent.act(state, noise=TD3_NOISE)
        ns, reward, result = env.step(action)
        agent.buffer.add(state, action, reward, ns)
        if t >= TD3_WARMUP:
            agent.train_step()
        log.rewards.append(float(reward))
        log.sum_rates.append(float(result.sum_rate))
        state = ns
    return agent, env, log


def solve_td3(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    total_steps: int = TD3_TOTAL_STEPS,
    greedy_steps: int = 20,
    agent: TD3Agent | None = None,
    n_restarts: int = 5,
) -> tuple[np.ndarray, EvalResult, float, TD3TrainLog | None]:
    t0 = time.perf_counter()
    log = None
    if agent is None:
        agent, env, log = train_td3(scenario, seed=seed, n_uav=n_uav, total_steps=total_steps)
    else:
        env = UAVAoDTEnv(scenario, n_uav=n_uav, seed=seed)
        env.reset()
    j = env.j
    cfg = env.cfg
    rng = np.random.default_rng(seed + 17)
    starts = [None]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(
            np.column_stack(
                [rng.uniform(0.0, cfg.area_x, j), rng.uniform(0.0, cfg.area_y, j)]
            )
        )
    best_xy = env.uav_xy.copy()
    best_result = env.last_result
    assert best_result is not None
    best_fit = best_result.sum_rate
    for start in starts:
        state = env.reset(uav_xy=start)
        result = env.last_result
        assert result is not None
        if result.violation_count < best_result.violation_count or (
            result.violation_count == best_result.violation_count and result.sum_rate >= best_fit
        ):
            best_fit = result.sum_rate
            best_xy = env.uav_xy.copy()
            best_result = result
        for _ in range(greedy_steps):
            action = agent.act(state, noise=0.0)
            state, _, result = env.step(action)
            if result.violation_count < best_result.violation_count or (
                result.violation_count == best_result.violation_count and result.sum_rate >= best_fit
            ):
                best_fit = result.sum_rate
                best_xy = env.uav_xy.copy()
                best_result = result
    return best_xy, best_result, time.perf_counter() - t0, log
