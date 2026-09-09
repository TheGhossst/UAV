"""Fujimoto TD3 updates. Hyperparameters live on TD3Settings."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from uavdt.td3.buffer import ReplayBuffer
from uavdt.td3.networks import Actor, Critic
from uavdt.td3.settings import TD3Settings


def _soft_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)


class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        settings: TD3Settings,
        seed: int,
        n_move: int | None = None,
        n_assoc: int | None = None,
    ) -> None:
        self.settings = settings
        self.act_dim = int(act_dim)
        self.n_move = int(act_dim if n_move is None else n_move)
        self.n_assoc = None if n_assoc is None else int(n_assoc)
        if not 1 <= self.n_move <= self.act_dim:
            raise ValueError(f"n_move={self.n_move} not in [1, {self.act_dim}]")
        self.device = torch.device(settings.device)
        torch.manual_seed(int(seed))
        hidden = int(settings.hidden)
        layer_norm = bool(settings.actor_layer_norm)
        c_dim = self.n_move if settings.critic_move_only else self.act_dim
        use_obs = bool(settings.critic_use_obs)
        self.actor = Actor(
            obs_dim, act_dim, hidden, n_move=self.n_move, layer_norm=layer_norm
        ).to(self.device)
        self.actor_t = Actor(
            obs_dim, act_dim, hidden, n_move=self.n_move, layer_norm=layer_norm
        ).to(self.device)
        self.critic1 = Critic(obs_dim, c_dim, hidden, use_obs=use_obs).to(self.device)
        self.critic2 = Critic(obs_dim, c_dim, hidden, use_obs=use_obs).to(self.device)
        self.critic1_t = Critic(obs_dim, c_dim, hidden, use_obs=use_obs).to(self.device)
        self.critic2_t = Critic(obs_dim, c_dim, hidden, use_obs=use_obs).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic1_t.load_state_dict(self.critic1.state_dict())
        self.critic2_t.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=settings.actor_lr)
        self.critic1_opt = torch.optim.Adam(
            self.critic1.parameters(), lr=settings.critic_lr
        )
        self.critic2_opt = torch.optim.Adam(
            self.critic2.parameters(), lr=settings.critic_lr
        )
        self.replay = ReplayBuffer(
            obs_dim,
            act_dim,
            settings.buffer_size,
            self.device,
            np.random.default_rng(int(seed) + 17),
        )
        self.n_updates = 0
        self.last_critic_loss: float | None = None
        self.last_actor_loss: float | None = None
        self.last_preact_mean_abs: float | None = None
        self.best_actor_score: float = float("-inf")
        self.best_actor_state: dict | None = None

    def consider_checkpoint(self, score: float) -> None:
        if not np.isfinite(score) or score <= self.best_actor_score:
            return
        self.best_actor_score = float(score)
        self.best_actor_state = {
            key: value.detach().cpu().clone()
            for key, value in self.actor.state_dict().items()
        }

    def load_best_actor(self) -> None:
        if self.best_actor_state is None:
            return
        self.actor.load_state_dict(self.best_actor_state)

    def pack_for_critic(self, action: torch.Tensor) -> torch.Tensor:
        """Critic input: movement only by default (placement bandit)."""
        n = self.n_move
        if self.settings.critic_move_only or n >= action.shape[-1]:
            return action[..., :n]
        clip = float(self.settings.logit_clip)
        move = action[..., :n]
        disc = action[..., n:] / clip
        if (
            self.settings.process_mode == "cpu_stable"
            and self.n_assoc is not None
            and self.n_assoc < disc.shape[-1]
        ):
            disc = disc.clone()
            disc[..., self.n_assoc :] = 0.0
        return torch.cat([move, disc], dim=-1)

    def _clamp_action(self, act: torch.Tensor) -> torch.Tensor:
        n = self.n_move
        move = act[..., :n].clamp(-1.0, 1.0)
        if n >= act.shape[-1]:
            return move
        clip = float(self.settings.logit_clip)
        disc = act[..., n:].clamp(-clip, clip)
        return torch.cat([move, disc], dim=-1)

    def _explore_noise(self, act: torch.Tensor) -> torch.Tensor:
        nse = torch.randn_like(act) * self.settings.explore_noise
        n = self.n_move
        if n < act.shape[-1]:
            nse = nse.clone()
            nse[..., n:] = nse[..., n:] * float(self.settings.logit_clip)
        return nse

    def _target_noise(self, action: torch.Tensor) -> torch.Tensor:
        raw = torch.randn_like(action)
        n = self.n_move
        nc = float(self.settings.noise_clip)
        tn = float(self.settings.target_noise)
        clip = float(self.settings.logit_clip)
        noise = torch.empty_like(action)
        noise[..., :n] = (tn * raw[..., :n]).clamp(-nc, nc)
        if n < action.shape[-1]:
            noise[..., n:] = (tn * clip * raw[..., n:]).clamp(-nc * clip, nc * clip)
        return noise

    def sample_warmup_action(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n_move
        out = np.empty(self.act_dim, dtype=np.float64)
        out[:n] = rng.uniform(-1.0, 1.0, size=n)
        if n < self.act_dim:
            clip = float(self.settings.logit_clip)
            out[n:] = rng.uniform(-clip, clip, size=self.act_dim - n)
        return out

    def select_action(self, obs: np.ndarray, *, noise: bool) -> np.ndarray:
        x = torch.as_tensor(
            np.asarray(obs, dtype=np.float32).reshape(1, -1),
            device=self.device,
        )
        with torch.no_grad():
            act = self.actor(x)[0]
            if noise:
                act = act + self._explore_noise(act)
            act = self._clamp_action(act)
        return act.cpu().numpy().astype(np.float64)

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic1(obs, self.pack_for_critic(action))

    def update(self) -> None:
        s, a, r, s2, done = self.replay.sample(self.settings.batch_size)
        with torch.no_grad():
            a2 = self._clamp_action(self.actor_t(s2) + self._target_noise(a))
            q1_t = self.critic1_t(s2, self.pack_for_critic(a2))
            q2_t = self.critic2_t(s2, self.pack_for_critic(a2))
            y = r + (1.0 - done) * self.settings.discount * torch.min(q1_t, q2_t)
        a_c = self.pack_for_critic(a)
        loss1 = torch.mean((self.critic1(s, a_c) - y) ** 2)
        loss2 = torch.mean((self.critic2(s, a_c) - y) ** 2)
        self.critic1_opt.zero_grad()
        loss1.backward()
        self.critic1_opt.step()
        self.critic2_opt.zero_grad()
        loss2.backward()
        self.critic2_opt.step()
        self.n_updates += 1
        self.last_critic_loss = float(0.5 * (loss1.item() + loss2.item()))
        if self.n_updates % self.settings.policy_delay != 0:
            return
        pi, z_move, disc = self.actor.forward_parts(s)
        q = self.critic1(s, self.pack_for_critic(pi)).mean()
        pen = self.settings.actor_preact_l2 * z_move.square().mean()
        if disc is not None:
            pen = pen + self.settings.actor_logit_l2 * disc.square().mean()
        actor_loss = -q + pen
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.last_actor_loss = float(actor_loss.item())
        self.last_preact_mean_abs = float(z_move.detach().abs().mean().item())
        _soft_update(self.actor, self.actor_t, self.settings.tau)
        _soft_update(self.critic1, self.critic1_t, self.settings.tau)
        _soft_update(self.critic2, self.critic2_t, self.settings.tau)


def make_td3_agent(env, settings: TD3Settings, seed: int) -> TD3Agent:
    return TD3Agent(
        env.obs_dim,
        env.act_dim,
        settings,
        seed=int(seed),
        n_move=int(env.n_move),
        n_assoc=int(env.n_assoc),
    )
