"""Actor and twin critics. Hidden sizes are TD3Settings, not Table II."""

from __future__ import annotations

import torch
from torch import nn

# Fujimoto et al. 2018 (official TD3 PyTorch): last-layer uniform ±3e-3 so
# tanh starts near 0. Default Linear init saturates tanh; with Δx,Δy × 10 m
# that pins UAVs to the field boundary.
FUJIMOTO_LAST_UNIFORM = 3.0e-3


def _init_last_layer(layer: nn.Linear, bound: float = FUJIMOTO_LAST_UNIFORM) -> None:
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


def _mlp(in_dim: int, out_dim: int, hidden: int, *, tanh: bool) -> nn.Sequential:
    last = nn.Linear(hidden, out_dim)
    _init_last_layer(last)
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        last,
    ]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Shared backbone, tanh on movement, raw logits for assoc/proc.

    Algorithm 2 decodes assoc/proc by argmax, so those heads must not share
    a saturating tanh with Δx,Δy. A dead Jacobian was the n=20 BEFORE failure:
    pre-tanh |z|→12–16, actor(s)→constant ±1, inner rate stuck at ~6.3 Mbps.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int,
        n_move: int | None = None,
        *,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        act_dim = int(act_dim)
        self.n_move = int(act_dim if n_move is None else n_move)
        if not 1 <= self.n_move <= act_dim:
            raise ValueError(f"n_move={self.n_move} not in [1, {act_dim}]")
        self.n_disc = act_dim - self.n_move
        hidden = int(hidden)
        body: list[nn.Module] = [
            nn.Linear(int(obs_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ]
        if layer_norm:
            body.append(nn.LayerNorm(hidden))
        self.backbone = nn.Sequential(*body)
        self.move_head = nn.Linear(hidden, self.n_move)
        _init_last_layer(self.move_head)
        if self.n_disc > 0:
            self.disc_head = nn.Linear(hidden, self.n_disc)
            _init_last_layer(self.disc_head)
        else:
            self.disc_head = None

    def forward_parts(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (action, move pre-tanh, discrete logits or None)."""
        h = self.backbone(obs)
        z_move = self.move_head(h)
        move = torch.tanh(z_move)
        if self.disc_head is None:
            return move, z_move, None
        disc = self.disc_head(h)
        return torch.cat([move, disc], dim=-1), z_move, disc

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        act, _, _ = self.forward_parts(obs)
        return act

    def forward_pre_tanh(self, obs: torch.Tensor) -> torch.Tensor:
        """Movement pre-tanh. Diagnostic / actor penalty."""
        _, z_move, _ = self.forward_parts(obs)
        return z_move


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int,
        *,
        use_obs: bool = True,
    ) -> None:
        super().__init__()
        self.use_obs = bool(use_obs)
        in_dim = (int(obs_dim) + int(act_dim)) if self.use_obs else int(act_dim)
        self.net = _mlp(in_dim, 1, hidden, tanh=False)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if self.use_obs:
            return self.net(torch.cat([obs, act], dim=-1))
        return self.net(act)
