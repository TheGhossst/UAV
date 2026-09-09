"""Fixed-capacity replay for (s, a, r, s', done)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.rng = rng
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self._i = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._i
        self.obs[i] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.act[i] = np.asarray(act, dtype=np.float32).reshape(-1)
        self.rew[i, 0] = float(rew)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.done[i, 0] = 1.0 if done else 0.0
        self._i = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        to_t = lambda arr: torch.as_tensor(arr[idx], device=self.device)
        return (
            to_t(self.obs),
            to_t(self.act),
            to_t(self.rew),
            to_t(self.next_obs),
            to_t(self.done),
        )
