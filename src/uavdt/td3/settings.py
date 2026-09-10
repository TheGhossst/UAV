"""TD3 algorithm knobs. EXTERNAL — not Table II / SimConfig / Problem (P).

Paper Algorithm 2 names σ, d, γ, τ, T, η, N but does not publish values.
Defaults follow Fujimoto et al. 2018 except total_steps (Fig. 4 ≈ 7000)
and buffer_size (1e5; T is short). Domain weights match Algorithm 2.

The proposed method is TD3Settings.residual_on_sca() — residual Δq on
the SCA incumbent, frozen a/b, inner bandwidth LP, feasible-rate reward.
That is a settings/origin change, not a new swarm. Defaults below stay
Algorithm 2 (reproduction chapter).
"""

from __future__ import annotations

from dataclasses import dataclass

from uavdt.config import BANDWIDTH_PRESETS


@dataclass(frozen=True)
class TD3Settings:
    """Algorithm knobs for the TD3 solver. Do not put these on SimConfig."""

    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    # Static Problem (P): each (q,a,b) is a complete score. γ=0.99 made Q(s,π)
    # explode (~1e3) and the actor chased a mediocre constant. Fujimoto's 0.99
    # is for continuing MuJoCo, not a 50-step placement search.
    discount: float = 0.0
    tau: float = 0.005
    policy_delay: int = 2
    explore_noise: float = 0.1
    target_noise: float = 0.2
    noise_clip: float = 0.5
    batch_size: int = 256
    buffer_size: int = 100_000
    hidden: int = 256
    total_steps: int = 7000
    horizon: int = 50
    warmup_steps: int = 256
    # Algorithm 2: Δx, Δy × 10 (both axes). IMPLEMENTATION CHOICE.
    move_scale_m: float = 10.0
    aodt_clip_s: float = 10.0
    rho_clip: float = 2.0
    aodt_ratio_clip: float = 5.0
    b_sys_ref_hz: float = BANDWIDTH_PRESETS["8.8mhz"]
    rate_weight: float = 1.0
    aodt_weight: float = 10.0
    dist_weight: float = 5.0
    viol_weight: float = 5.0
    device: str = "cpu"
    # 0 = silent. Default prints a progress line every 250 env steps.
    log_every: int = 250
    # Official score: trained policy (noise-free rollout), not best-snapshot.
    # "best_snapshot" is the old bookkeeping export (kept for A/B).
    export_mode: str = "policy"
    # Average last N UAV xy of the deterministic eval episode; a/b from last step.
    export_avg_steps: int = 10
    # Keep movement pre-tanh O(1). Dead tanh ( |z|→12–16, actor→±1 ) was the
    # n=20 BEFORE failure. EXTERNAL — not Table II.
    actor_preact_l2: float = 0.05
    actor_logit_l2: float = 0.05
    logit_clip: float = 3.0
    actor_layer_norm: bool = True
    # "residual": Δ from the frozen instance init (k-means). a=0 stays at
    # k-means; chained "delta" walks off it (~4 m/step × 50).
    move_mode: str = "residual"
    # Same discrete helpers as frozen SCA / k-means.
    assoc_mode: str = "nearest"
    assoc_distance_coef: float = 2.0
    process_mode: str = "cpu_stable"
    critic_use_obs: bool = False
    critic_move_only: bool = True
    export_actor: str = "best_checkpoint"
    # Algorithm 1/§VII: k-means is the named placement initialization.
    # Random init left TD3 ~0.05 Mbps behind k-means itself.
    uav_init: str = "kmeans"
    # Algorithm 2: "update bandwidth" inside the env step. leftover-dump is
    # the same QoS-floor + cap heuristic SCA starts with; full LP at export.
    inner_bandwidth: str = "leftover"
    # "alg2": paper Eq. (34) penalty. "feasible_rate": Problem (P) Mbps gate.
    reward_mode: str = "alg2"
    infeasible_reward: float = -100.0
    # "full": Δxy + assoc/proc logits (Alg. 2). "move_only": Δxy in R^{2J}.
    action_heads: str = "full"

    def __post_init__(self) -> None:
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        if self.policy_delay < 1:
            raise ValueError("policy_delay must be >= 1")
        if self.batch_size < 1 or self.buffer_size < 1:
            raise ValueError("batch_size and buffer_size must be >= 1")
        if self.hidden < 1:
            raise ValueError("hidden must be >= 1")
        if self.total_steps < 1 or self.horizon < 1:
            raise ValueError("total_steps and horizon must be >= 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.move_scale_m <= 0.0:
            raise ValueError("move_scale_m must be positive")
        if self.aodt_clip_s <= 0.0 or self.aodt_ratio_clip <= 0.0:
            raise ValueError("AoDT clips must be positive")
        if self.rho_clip <= 0.0:
            raise ValueError("rho_clip must be positive")
        if self.b_sys_ref_hz <= 0.0:
            raise ValueError("b_sys_ref_hz must be positive")
        if self.log_every < 0:
            raise ValueError("log_every must be >= 0")
        if self.export_mode not in {"policy", "best_snapshot"}:
            raise ValueError("export_mode must be 'policy' or 'best_snapshot'")
        if self.export_avg_steps < 1:
            raise ValueError("export_avg_steps must be >= 1")
        if self.actor_preact_l2 < 0.0 or self.actor_logit_l2 < 0.0:
            raise ValueError("actor L2 penalties must be >= 0")
        if self.logit_clip <= 0.0:
            raise ValueError("logit_clip must be positive")
        if self.move_mode not in {"setpoint", "delta", "absolute", "residual"}:
            raise ValueError(
                "move_mode must be 'setpoint', 'delta', 'absolute', or 'residual'"
            )
        if self.assoc_distance_coef < 0.0:
            raise ValueError("assoc_distance_coef must be >= 0")
        if self.process_mode not in {"cpu_stable", "logits", "frozen"}:
            raise ValueError("process_mode must be 'cpu_stable', 'logits', or 'frozen'")
        if self.assoc_mode not in {"nearest", "logits", "logits_plus_dist", "frozen"}:
            raise ValueError(
                "assoc_mode must be nearest, logits, logits_plus_dist, or frozen"
            )
        if self.export_actor not in {"online", "best_checkpoint"}:
            raise ValueError("export_actor must be 'online' or 'best_checkpoint'")
        if self.uav_init not in {"kmeans", "random", "sca"}:
            raise ValueError("uav_init must be 'kmeans', 'random', or 'sca'")
        if self.inner_bandwidth not in {"leftover", "equal_share", "lp"}:
            raise ValueError("inner_bandwidth must be leftover, equal_share, or lp")
        if self.reward_mode not in {"alg2", "feasible_rate"}:
            raise ValueError("reward_mode must be 'alg2' or 'feasible_rate'")
        if self.action_heads not in {"full", "move_only"}:
            raise ValueError("action_heads must be 'full' or 'move_only'")
        if self.action_heads == "move_only" and self.assoc_mode in {
            "logits",
            "logits_plus_dist",
        }:
            raise ValueError("action_heads='move_only' cannot decode assoc logits")
        if self.action_heads == "move_only" and self.process_mode == "logits":
            raise ValueError("action_heads='move_only' cannot decode process logits")

    def is_residual_on_sca(self) -> bool:
        return (
            self.uav_init == "sca"
            and self.assoc_mode == "frozen"
            and self.process_mode == "frozen"
            and self.inner_bandwidth == "lp"
            and self.reward_mode == "feasible_rate"
        )

    @classmethod
    def residual_on_sca(cls, **overrides: object) -> TD3Settings:
        """Proposed method: residual Δq on SCA, frozen a/b, inner LP, (P) reward.

        Algorithm 2 defaults are unchanged; this factory is the opt-in preset.
        """
        kwargs: dict = {
            "uav_init": "sca",
            "move_mode": "residual",
            "assoc_mode": "frozen",
            "process_mode": "frozen",
            "inner_bandwidth": "lp",
            "reward_mode": "feasible_rate",
            "export_mode": "best_snapshot",
            "discount": 0.0,
            "critic_move_only": True,
            "action_heads": "move_only",
        }
        kwargs.update(overrides)
        return cls(**kwargs)
