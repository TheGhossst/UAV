"""TD3 environment and agent following Algorithm 2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logutil import log, log_td3_step, result_bits
from src.config import (
    AODT_THRESHOLD,
    B_SYS,
    LAMBDA_I,
    TD3_ACTOR_LR,
    TD3_ASSOC_ACTION_SCALE,
    TD3_EPISODE_LEN,
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
    UAV_CPU,
)
from src.comm import link_metrics
from src.evaluator import EvalResult, evaluate
from src.repair import complete_solution, clip_positions, enforce_separation
from src.scenario import Scenario
from src.solvers.kmeans import kmeans

# None/"auto" → CUDA when PyTorch sees a GPU, else CPU. Overridden by --device.
_DEFAULT_DEVICE: str | None = None
_DEVICE_LOGGED = False


def set_default_device(name: str | None) -> None:
    """Set the process-wide TD3 device (`auto`, `cpu`, `cuda`, or `cuda:N`)."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = name


def resolve_device(device: str | None = None) -> torch.device:
    name = device if device is not None else _DEFAULT_DEVICE
    if name in (None, "auto"):
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(name, str) and name.startswith("cuda") and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _log_device(dev: torch.device) -> None:
    global _DEVICE_LOGGED
    if _DEVICE_LOGGED:
        return
    extra = f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""
    log.info("TD3 device: %s%s", dev, extra)
    _DEVICE_LOGGED = True


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

    def __init__(
        self,
        scenario: Scenario,
        n_uav: int | None = None,
        seed: int = 0,
        assoc_action_scale: float | None = None,
        distance_prior: bool = True,
    ):
        self.scenario = scenario
        self.cfg = scenario.cfg
        self.j = n_uav if n_uav is not None else self.cfg.num_uav
        self.i = self.cfg.num_iot
        # Ablation knobs; defaults match config.py (not a retune).
        self.assoc_action_scale = (
            TD3_ASSOC_ACTION_SCALE if assoc_action_scale is None else float(assoc_action_scale)
        )
        self.distance_prior = bool(distance_prior)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.uav_xy = np.zeros((self.j, 2))
        self.last_result: EvalResult | None = None
        # State (32) includes the channel. Without the per-link spectral
        # efficiency the actor would have to infer link quality from raw
        # coordinates before it could allocate bandwidth sensibly, which it does
        # not manage inside TD3_TOTAL_STEPS.
        # +2: T_k / T_k^table and f_j / f_j^table so λ / AoDT / CPU sweeps are
        # visible even when per-IoT λ ratios are all 1.
        self.state_dim = (
            2 * self.j
            + 2 * self.i
            + self.cfg.num_processes
            + 1
            + self.j
            + self.i
            + self.i * self.j
            + 2
        )
        # Δx,Δy, assoc logits, proc logits, bandwidth raw
        self.action_dim = 2 * self.j + 3 * self.i * self.j

    def reset(self, uav_xy: np.ndarray | None = None, *, eval_kmeans: bool = False) -> np.ndarray:
        if uav_xy is None:
            # Eval-time k-means must match solve_kmeans(scenario, seed): a
            # fresh default_rng(self.seed), not whatever self.rng has left
            # after training. Training episode resets keep using self.rng.
            rng = np.random.default_rng(self.seed) if eval_kmeans else self.rng
            self.uav_xy = kmeans(self.scenario.iot_xy, self.j, rng)
        else:
            self.uav_xy = np.asarray(uav_xy, dtype=float).reshape(self.j, 2)
        self.uav_xy = clip_positions(self.uav_xy, self.cfg)
        xy, a, b, bw = complete_solution(self.scenario, self.uav_xy)
        self.uav_xy = xy
        self.last_result = evaluate(self.scenario, xy, a, b, bw)
        return self.get_obs()

    def spectral_efficiency(self, uav_xy: np.ndarray) -> np.ndarray:
        """log2(1 + SNR_ij) in bit/s/Hz, i.e. the rate per Hz of Eq. (6)."""
        ones = np.ones((self.i, uav_xy.shape[0]))
        return link_metrics(self.scenario.iot_xy, uav_xy, ones, self.cfg)["rates"]

    def get_obs(self) -> np.ndarray:
        r = self.last_result
        aodt = np.nan_to_num(r.aodt, nan=0.0) if r is not None else np.zeros(self.cfg.num_processes)
        rho = r.rho if r is not None else np.zeros(self.j)
        se = self.spectral_efficiency(self.uav_xy)
        # SE spans several decades over the area, so feed it in log units.
        se_obs = (np.log10(np.maximum(se, 1e-4)) + 4.0) / 4.0
        parts = [
            self.uav_xy.reshape(-1) / max(self.cfg.area_x, 1.0),
            self.scenario.iot_xy.reshape(-1) / max(self.cfg.area_x, 1.0),
            aodt / max(self.cfg.aodt_threshold, 1.0),
            np.array([self.cfg.b_sys / B_SYS]),
            rho,
            # Table II references so the absolute λ / T_k / f_j of a sweep point
            # are visible. Dividing λ by max(λ) hid the arrival-rate sweep.
            self.scenario.lambdas / max(LAMBDA_I, 1e-6),
            se_obs.reshape(-1),
            np.array([self.cfg.aodt_threshold / max(AODT_THRESHOLD, 1e-6)]),
            np.array([self.cfg.uav_cpu / max(UAV_CPU, 1e-6)]),
        ]
        return np.concatenate(parts).astype(np.float32)

    def _distance_logits(self, uav_xy: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(self.scenario.iot_xy[:, None, :] - uav_xy[None, :, :], axis=-1)
        return -d / max(self.cfg.area_x, 1.0)

    def parse_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=float).reshape(-1)
        j, i = self.j, self.i
        dxdy = action[: 2 * j].reshape(j, 2)
        assoc_off = action[2 * j : 2 * j + i * j].reshape(i, j)
        proc_off = action[2 * j + i * j : 2 * j + 2 * i * j].reshape(i, j)
        bw_raw = action[2 * j + 2 * i * j : 2 * j + 3 * i * j].reshape(i, j)
        # Alg. 2: update UAV positions Δx, Δy × 10, clip to [0, area]
        xy = self.uav_xy + dxdy * TD3_POS_SCALE
        xy = clip_positions(xy, self.cfg)
        base = self._distance_logits(xy) if self.distance_prior else np.zeros((i, j))
        xy, a, b, bw = complete_solution(
            self.scenario,
            xy,
            assoc_logits=base + self.assoc_action_scale * assoc_off,
            proc_logits=base + self.assoc_action_scale * proc_off,
            # 1 + tanh keeps the map monotone and makes a zero action a uniform
            # request; bandwidth_from_weights only spends the R_min surplus on it.
            bandwidth=np.maximum(1.0 + bw_raw, 0.0),
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
        # V_viol uses the continuous R_min shortfall instead of a violation
        # count: a count saturates, so once the UAVs drift out of range every
        # action scores the same and the actor has nothing to descend.
        p_aodt = float(result.aodt_violations)
        if result.compute_available:
            p_aodt = float(result.aodt_excess)
        p_dist = float(result.sep_violations)
        v_viol = float(
            result.qos_shortfall
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
    """Diagnostics from training. ``best_*`` is the training-search archive
    and is **not** the reported TD3 result; ``solve_td3`` returns greedy-policy
    evaluation after training.

    ``in_episode_steps`` / ``start_kinds`` re-bucket the global-step reward
    series by ``t % episode_len`` and by the k-means vs random reset that
    started that episode. They do not change training.
    """

    rewards: list[float]
    sum_rates: list[float]
    best_xy: np.ndarray | None = None
    best_result: EvalResult | None = None
    in_episode_steps: list[int] = field(default_factory=list)
    start_kinds: list[str] = field(default_factory=list)


def _better(candidate: EvalResult, incumbent: EvalResult | None) -> bool:
    """Fewer constraint violations first, then higher sum rate."""
    if incumbent is None:
        return True
    if candidate.violation_count != incumbent.violation_count:
        return candidate.violation_count < incumbent.violation_count
    return candidate.sum_rate > incumbent.sum_rate


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, seed: int = 0, device: str | None = None):
        # Default is CUDA when available (RTX 50-series included). The env/evaluator
        # stay on CPU; only actor/critic updates run on the GPU. Pass device="cpu"
        # (or --device cpu) for bit-repeatable paper sweeps.
        self.device = resolve_device(device)
        _log_device(self.device)
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


def _episode_start(env: UAVAoDTEnv, episode: int) -> np.ndarray:
    """Alternate the k-means start with random starts for exploration."""
    if episode % 2 == 0:
        return env.reset()
    cfg = env.cfg
    xy = np.column_stack(
        [env.rng.uniform(0.0, cfg.area_x, env.j), env.rng.uniform(0.0, cfg.area_y, env.j)]
    )
    return env.reset(uav_xy=xy)


def _episode_bucket(t: int, episode_len: int) -> tuple[int, str]:
    """Map a global step to (t % episode_len, kmeans|random). Observe-only."""
    if episode_len <= 0:
        return t, "continuous"
    ep = t // episode_len
    kind = "kmeans" if ep % 2 == 0 else "random"
    return t % episode_len, kind


def _resolve_alg2_knobs(
    *,
    fidelity_mode: bool,
    episode_len: int,
    assoc_action_scale: float | None,
    distance_prior: bool | None,
    warmup: int | None,
) -> tuple[int, float | None, bool, int]:
    """Default path unchanged. Fidelity mode isolates the three Alg. 2 deviations.

    Warmup is dropped (0), not shrunk: Alg. 2 takes a_t = μ_φ(s_t)+ε from t=1,
    and train_step already no-ops until the buffer has TD3_BATCH_SIZE samples.
    A 500-step uniform-random phase is the extra deviation. TD3_NOISE on the
    actor from t=0 still explores. Callers can pass warmup=... to override.
    """
    if fidelity_mode:
        episode_len = 0
        if assoc_action_scale is None:
            # Direct actor logits, not a 0.25 offset on a nearest-UAV prior.
            assoc_action_scale = 1.0
        if distance_prior is None:
            distance_prior = False
        warmup_steps = 0 if warmup is None else int(warmup)
    else:
        if distance_prior is None:
            distance_prior = True
        warmup_steps = TD3_WARMUP if warmup is None else int(warmup)
    return episode_len, assoc_action_scale, bool(distance_prior), warmup_steps


def train_td3(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    total_steps: int = TD3_TOTAL_STEPS,
    episode_len: int = TD3_EPISODE_LEN,
    device: str | None = None,
    assoc_action_scale: float | None = None,
    distance_prior: bool | None = None,
    fidelity_mode: bool = False,
    warmup: int | None = None,
) -> tuple[TD3Agent, UAVAoDTEnv, TD3TrainLog]:
    episode_len, assoc_action_scale, distance_prior, warmup_steps = _resolve_alg2_knobs(
        fidelity_mode=fidelity_mode,
        episode_len=episode_len,
        assoc_action_scale=assoc_action_scale,
        distance_prior=distance_prior,
        warmup=warmup,
    )
    env = UAVAoDTEnv(
        scenario,
        n_uav=n_uav,
        seed=seed,
        assoc_action_scale=assoc_action_scale,
        distance_prior=distance_prior,
    )
    agent = TD3Agent(env.state_dim, env.action_dim, seed=seed, device=device)
    train_log = TD3TrainLog(rewards=[], sum_rates=[])
    state = env.reset()
    t0 = time.perf_counter()
    log.info("td3 train    I=%d J=%d steps=%d episode=%d", env.i, env.j, total_steps, episode_len)
    if fidelity_mode:
        log.info(
            "td3 fidelity  continuous  prior=%s  scale=%g  warmup=%d",
            env.distance_prior,
            env.assoc_action_scale,
            warmup_steps,
        )
    for t in range(total_steps):
        # Reset before stepping so no buffer transition straddles an episode.
        if episode_len > 0 and t % episode_len == 0:
            state = _episode_start(env, t // episode_len)
        if t < warmup_steps:
            action = env.rng.uniform(-1.0, 1.0, size=env.action_dim)
        else:
            action = agent.act(state, noise=TD3_NOISE)
        ns, reward, result = env.step(action)
        agent.buffer.add(state, action, reward, ns)
        if t >= warmup_steps:
            agent.train_step()
        in_ep, kind = _episode_bucket(t, episode_len)
        train_log.rewards.append(float(reward))
        train_log.sum_rates.append(float(result.sum_rate))
        train_log.in_episode_steps.append(int(in_ep))
        train_log.start_kinds.append(kind)
        if _better(result, train_log.best_result):
            train_log.best_result = result
            train_log.best_xy = env.uav_xy.copy()
        log_td3_step("train", t, total_steps, t0, float(reward), float(result.sum_rate))
        state = ns
    return agent, env, train_log


def train_td3_across_scenarios(
    cfg,
    train_seeds: tuple[int, ...] | list[int],
    seed: int = 0,
    n_uav: int | None = None,
    total_steps: int = TD3_TOTAL_STEPS,
    resample_every: int = TD3_EPISODE_LEN,
    device: str | None = None,
    assoc_action_scale: float | None = None,
    distance_prior: bool = True,
) -> tuple[TD3Agent, UAVAoDTEnv, TD3TrainLog]:
    """Train one TD3 policy on a pool of IoT deployments (paper-style generalization)."""
    from src.scenario import generate_scenario

    rng = np.random.default_rng(seed)
    seeds = list(train_seeds)
    scenario = generate_scenario(int(seeds[0]), cfg)
    env = UAVAoDTEnv(
        scenario,
        n_uav=n_uav,
        seed=seed,
        assoc_action_scale=assoc_action_scale,
        distance_prior=distance_prior,
    )
    agent = TD3Agent(env.state_dim, env.action_dim, seed=seed, device=device)
    train_log = TD3TrainLog(rewards=[], sum_rates=[])
    state = env.reset()
    t0 = time.perf_counter()
    log.info(
        "td3 train-across  I=%d J=%d steps=%d pool=%d resample=%d",
        env.i,
        env.j,
        total_steps,
        len(seeds),
        resample_every,
    )
    for t in range(total_steps):
        # Resampling the deployment doubles as the episode boundary.
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
        in_ep, _kind = _episode_bucket(t, resample_every)
        train_log.rewards.append(float(reward))
        train_log.sum_rates.append(float(result.sum_rate))
        train_log.in_episode_steps.append(int(in_ep))
        train_log.start_kinds.append("resample")
        log_td3_step("pool", t, total_steps, t0, float(reward), float(result.sum_rate))
        state = ns
    return agent, env, train_log


def solve_td3(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    total_steps: int = TD3_TOTAL_STEPS,
    greedy_steps: int = 20,
    agent: TD3Agent | None = None,
    n_restarts: int = 5,
    device: str | None = None,
    eval_trace: dict | None = None,
    assoc_action_scale: float | None = None,
    distance_prior: bool | None = None,
    fidelity_mode: bool = False,
    warmup: int | None = None,
) -> tuple[np.ndarray, EvalResult, float, TD3TrainLog | None]:
    """Train (unless ``agent`` is given) then evaluate the greedy policy.

    Reported result: greedy-policy evaluation, not the training-search best.
    compare / sweeps train per scenario then call this; aodt-compare passes a
    policy from ``train_td3_across_scenarios``. All three use the same eval:
    k-means plus random restarts, each followed by a noiseless rollout, keeping
    the best feasible point along those rollouts.

    ``fidelity_mode`` is opt-in Algorithm 2 training (continuous trajectory,
    un-mediated logits, no uniform warmup). Default False: compare / sweeps
    are unchanged.
    """
    t0 = time.perf_counter()
    train_log = None
    _ep, assoc_action_scale, distance_prior, _w = _resolve_alg2_knobs(
        fidelity_mode=fidelity_mode,
        episode_len=TD3_EPISODE_LEN,
        assoc_action_scale=assoc_action_scale,
        distance_prior=distance_prior,
        warmup=warmup,
    )
    env_kw = dict(
        n_uav=n_uav,
        assoc_action_scale=assoc_action_scale,
        distance_prior=distance_prior,
    )
    if agent is None:
        agent, env, train_log = train_td3(
            scenario, seed=seed, n_uav=n_uav, total_steps=total_steps, device=device,
            assoc_action_scale=assoc_action_scale, distance_prior=distance_prior,
            fidelity_mode=fidelity_mode, warmup=warmup,
        )
    else:
        env = UAVAoDTEnv(scenario, seed=seed, **env_kw)
        env.reset()
        log.info("td3 eval     I=%d J=%d greedy_steps=%d restarts=%d", env.i, env.j, greedy_steps, n_restarts)
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
    # Do not seed from the last training state or log.best_*: those are
    # exploration visits. Only greedy-policy rollouts are reported.
    best_xy: np.ndarray | None = None
    best_result: EvalResult | None = None
    winner: dict | None = None
    step_records: list[dict] = []
    for start_index, start in enumerate(starts):
        kind = "kmeans" if start is None else "random"
        state = env.reset(uav_xy=start, eval_kmeans=(start is None))
        result = env.last_result
        assert result is not None

        def _snap(step: int, source: str) -> dict:
            return {
                "start_index": start_index,
                "kind": kind,
                "step": step,
                "source": source,
                "sum_rate": float(result.sum_rate),
                "feasible": bool(result.feasible),
                "qos": int(result.qos_violations),
                "violation_count": int(result.violation_count),
            }

        snap = _snap(0, "pre_rollout")
        step_records.append(snap)
        if _better(result, best_result):
            best_xy, best_result = env.uav_xy.copy(), result
            winner = snap
        for gs in range(greedy_steps):
            action = agent.act(state, noise=0.0)
            state, _, result = env.step(action)
            snap = _snap(gs + 1, "actor")
            step_records.append(snap)
            if _better(result, best_result):
                best_xy, best_result = env.uav_xy.copy(), result
                winner = snap
    assert best_xy is not None and best_result is not None and winner is not None
    if eval_trace is not None:
        eval_trace["winner"] = winner
        eval_trace["n_restarts"] = int(n_restarts)
        eval_trace["greedy_steps"] = int(greedy_steps)
        eval_trace["records"] = step_records
        log.info(
            "td3 eval-trace winner start=%d kind=%s step=%d source=%s rate=%.3f Mbps",
            winner["start_index"],
            winner["kind"],
            winner["step"],
            winner["source"],
            winner["sum_rate"] / 1e6,
        )
    log.info(
        "td3 done     %s  (greedy, %d restart%s)",
        result_bits(best_result, time.perf_counter() - t0),
        n_restarts,
        "" if n_restarts == 1 else "s",
    )
    return best_xy, best_result, time.perf_counter() - t0, train_log
