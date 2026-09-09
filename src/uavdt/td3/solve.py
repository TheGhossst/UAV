"""Per-instance TD3 solve of Problem (P). Scores with evaluate()."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from uavdt.constraints import pairwise_uav_distance_m
from uavdt.evaluator import EvalResult, evaluate
from uavdt.models import Allocation, Scenario
from uavdt.resources import equal_share_bandwidth_hz, nearest_association
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
from uavdt.sca.settings import SCASettings
from uavdt.td3.env import UAVAoDTEnv
from uavdt.td3.settings import TD3Settings


@dataclass
class TD3Result:
    uav_xyz_m: np.ndarray
    allocation: Allocation
    true_eval: EvalResult
    n_steps: int = 0
    diagnostics: dict = field(default_factory=dict)


def log_td3(msg: str) -> None:
    """Flushing ASCII-safe progress line (Windows consoles included)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(round(float(seconds)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def copy_allocation(allocation: Allocation) -> Allocation:
    return Allocation(
        association=np.asarray(allocation.association, dtype=float).copy(),
        processing=np.asarray(allocation.processing, dtype=float).copy(),
        bandwidth_hz=np.asarray(allocation.bandwidth_hz, dtype=float).copy(),
    )


def export_allocation(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    allocation: Allocation,
) -> Allocation:
    """Keep TD3 a_ij / b_ij; replace B_ij with the frozen-q bandwidth LP."""
    a = allocation.hard_association()
    b = allocation.hard_processing()
    res = solve_bandwidth_at_fixed_q(
        scenario, uav_xyz_m, a, b, SCASettings(solver=None)
    )
    if res.infeasible:
        bw = equal_share_bandwidth_hz(a, scenario.cfg)
        return Allocation(a, b, bw)
    return Allocation(a, b, res.bandwidth_hz)


def lp_score(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    allocation: Allocation,
) -> tuple[np.ndarray, Allocation, EvalResult]:
    uav = np.asarray(uav_xyz_m, dtype=float).copy()
    alloc = export_allocation(scenario, uav, allocation)
    return uav, alloc, evaluate(scenario, uav, alloc)


def deterministic_policy_export(env: UAVAoDTEnv, agent, settings: TD3Settings) -> dict:
    """One noise-free episode: last-N UAV xy mean, last-step a/b (inner B)."""
    cfg = env.scenario.cfg
    n_roll = int(settings.horizon)
    n_avg = min(int(settings.export_avg_steps), n_roll)
    env.freeze_best = True
    obs = env.reset()
    qs: list[np.ndarray] = []
    allocs: list[Allocation] = []
    inner_rates: list[float] = []
    inner_feas: list[float] = []
    move_norms: list[float] = []
    n_move = 2 * cfg.num_uav
    for i in range(n_roll):
        action = agent.select_action(obs, noise=False)
        move_norms.append(float(np.linalg.norm(np.asarray(action[:n_move], dtype=float))))
        obs, _reward, done, info = env.step(action)
        if env._uav is None or env._alloc is None:
            raise RuntimeError("policy export missing env state")
        qs.append(np.asarray(env._uav, dtype=float).copy())
        allocs.append(copy_allocation(env._alloc))
        inner_rates.append(float(info["eval"].sum_rate_mbps))
        inner_feas.append(float(info["eval"].feasible))
        if done and i + 1 < n_roll:
            obs = env.reset()
    stacked = np.stack(qs[-n_avg:], axis=0)
    uav = np.mean(stacked, axis=0)
    uav[:, 0] = np.clip(uav[:, 0], 0.0, cfg.area_x_m)
    uav[:, 1] = np.clip(uav[:, 1], 0.0, cfg.area_y_m)
    uav[:, 2] = cfg.uav_height_m
    last_uav = qs[-1]
    last_a = allocs[-1].hard_association()
    nearest = nearest_association(env.scenario.iot_xyz_m, last_uav)
    dists = pairwise_uav_distance_m(last_uav)
    j = last_uav.shape[0]
    pair = [float(dists[p, q]) for p in range(j) for q in range(p + 1, j)]
    return {
        "uav_xyz_m": uav,
        "allocation": allocs[-1],
        "policy_inner_mean_Mbps": float(np.mean(inner_rates[-n_avg:])),
        "policy_inner_last_Mbps": float(inner_rates[-1]),
        "policy_inner_feas_mean": float(np.mean(inner_feas[-n_avg:])),
        "policy_inner_rates_Mbps": [float(x) for x in inner_rates[-n_avg:]],
        "mean_move_action_norm": float(np.mean(move_norms)) if move_norms else 0.0,
        "nearest_agree": float(np.mean(np.argmax(last_a, axis=1) == np.argmax(nearest, axis=1))),
        "min_pairwise_m": float(min(pair)) if pair else None,
        "uav_loads": [float(x) for x in last_a.sum(axis=0)],
        "n_rollout": n_roll,
        "n_avg": n_avg,
    }


def format_progress_line(
    *,
    t: int,
    total: int,
    warmup: int,
    elapsed: float,
    window_r: list[float],
    window_mbps: list[float],
    window_feas: list[float],
    info: dict,
    env: UAVAoDTEnv,
    n_updates: int,
    critic_loss: float | None,
    actor_loss: float | None,
) -> str:
    step = t + 1
    pct = 100.0 * step / max(total, 1)
    phase = "warmup" if t < warmup else "train"
    sps = step / elapsed if elapsed > 0 else 0.0
    remain = (total - step) / sps if sps > 0 else float("inf")
    best = env.best
    best_s = "none"
    if best is not None:
        flag = "yes" if best.true_eval.feasible else "no"
        best_s = f"{best.true_eval.sum_rate_mbps:.3f} {flag}"
    q_s = f"  Q={critic_loss:.3f}" if critic_loss is not None else ""
    pi_s = f" pi={actor_loss:.3f}" if actor_loss is not None else ""
    return (
        f"  step {step:5d}/{total} ({pct:5.1f}%)  {phase:6s}  "
        f"rew={float(np.mean(window_r)):7.3f}  "
        f"Mbps={float(np.mean(window_mbps)):6.3f}  "
        f"feas={float(np.mean(window_feas)):4.2f}  "
        f"Paodt={float(info.get('p_aodt', 0.0)):5.2f} "
        f"Pdist={float(info.get('p_dist', 0.0)):4.2f} "
        f"V={float(info.get('v_viol', 0.0)):4.2f}  "
        f"best={best_s}  "
        f"ep={env._episode}{q_s}{pi_s}  "
        f"upd={n_updates}  {sps:.0f}/s  eta={format_eta(remain)}"
    )


def solve_td3(
    scenario: Scenario,
    seed: int,
    settings: TD3Settings | None = None,
) -> TD3Result:
    """Train a TD3 agent on one scenario.

    Official score is a noise-free policy rollout (last-N UAV xy, last-step
    a/b, frozen-q LP). Best-snapshot export remains in diagnostics and is
    available via TD3Settings.export_mode='best_snapshot'.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TD3 requires PyTorch. Install with pip install 'uavdt[td3]' "
            "or pip install torch."
        ) from exc

    from uavdt.td3.agent import make_td3_agent

    settings = settings or TD3Settings()
    t0 = perf_counter()
    rng = np.random.default_rng(int(seed))
    env = UAVAoDTEnv(scenario, settings, seed=int(seed))
    agent = make_td3_agent(env, settings, int(seed))
    obs = env.reset()
    n_episodes = 1
    warmup = int(settings.warmup_steps)
    log_every = int(settings.log_every)
    cfg = scenario.cfg
    window_r: list[float] = []
    window_mbps: list[float] = []
    window_feas: list[float] = []
    ep_mbps: list[float] = []
    verbose = log_every > 0
    if verbose:
        log_td3(
            f"TD3 train  seed={int(seed)}  I={cfg.num_iot} J={cfg.num_uav} "
            f"K={cfg.num_processes}  steps={settings.total_steps} "
            f"horizon={settings.horizon} warmup={warmup}  "
            f"device={settings.device}  obs={env.obs_dim} act={env.act_dim}  "
            f"log_every={log_every}"
        )
    for t in range(settings.total_steps):
        if t < warmup or agent.replay.size < settings.batch_size:
            action = agent.sample_warmup_action(rng)
        else:
            action = agent.select_action(obs, noise=True)
        next_obs, reward, done, info = env.step(action)
        agent.replay.add(obs, action, reward, next_obs, done)
        if t >= warmup and agent.replay.size >= settings.batch_size:
            agent.update()
        window_r.append(float(reward))
        window_mbps.append(float(info.get("sum_rate_Mbps", info["eval"].sum_rate_mbps)))
        window_feas.append(float(info["eval"].feasible))
        ep_mbps.append(float(info.get("sum_rate_Mbps", info["eval"].sum_rate_mbps)))
        if log_every > 0 and len(window_r) > log_every:
            window_r.pop(0)
            window_mbps.pop(0)
            window_feas.pop(0)
        if done:
            agent.consider_checkpoint(float(np.mean(ep_mbps)))
            ep_mbps = []
            n_episodes += 1
            if t + 1 < settings.total_steps:
                obs = env.reset()
            else:
                obs = next_obs
        else:
            obs = next_obs
        if verbose and ((t + 1) % log_every == 0 or t + 1 == settings.total_steps):
            log_td3(
                format_progress_line(
                    t=t,
                    total=settings.total_steps,
                    warmup=warmup,
                    elapsed=perf_counter() - t0,
                    window_r=window_r,
                    window_mbps=window_mbps,
                    window_feas=window_feas,
                    info=info,
                    env=env,
                    n_updates=int(agent.n_updates),
                    critic_loss=agent.last_critic_loss,
                    actor_loss=agent.last_actor_loss,
                )
            )

    if ep_mbps:
        agent.consider_checkpoint(float(np.mean(ep_mbps)))
    if settings.export_actor == "best_checkpoint":
        agent.load_best_actor()
    if env.best is None:
        raise RuntimeError("TD3 produced no snapshot")
    train_best = env.best
    if verbose:
        log_td3("  export: snapshot LP + deterministic policy rollout ...")
    _snap_uav, snap_alloc, snap_ev = lp_score(
        scenario, train_best.uav_xyz_m, train_best.allocation
    )
    policy = deterministic_policy_export(env, agent, settings)
    _pol_uav, pol_alloc, pol_ev = lp_score(
        scenario, policy["uav_xyz_m"], policy["allocation"]
    )
    n_avg = int(policy["n_avg"])
    if settings.export_mode == "best_snapshot":
        uav = np.asarray(train_best.uav_xyz_m, dtype=float).copy()
        alloc = snap_alloc
        ev = snap_ev
        export_rule = "best_snapshot frozen_q_lp"
    else:
        uav = np.asarray(policy["uav_xyz_m"], dtype=float).copy()
        alloc = pol_alloc
        ev = pol_ev
        export_rule = (
            f"deterministic_policy last_{n_avg}_xy_mean last_step_ab frozen_q_lp"
        )
    elapsed = perf_counter() - t0
    if verbose:
        inner = train_best.true_eval
        log_td3(
            f"TD3 done   seed={int(seed)}  rule={export_rule}  "
            f"export={ev.sum_rate_mbps:.4f} Mbps "
            f"feas={'yes' if ev.feasible else 'no'}  "
            f"snapshot_LP={snap_ev.sum_rate_mbps:.4f} "
            f"policy_LP={pol_ev.sum_rate_mbps:.4f}  "
            f"inner_best={inner.sum_rate_mbps:.4f}  "
            f"policy_inner_lastN={policy['policy_inner_mean_Mbps']:.4f}  "
            f"updates={agent.n_updates}  {elapsed:.1f}s  "
            f"({settings.total_steps / elapsed:.0f} step/s)"
        )
    return TD3Result(
        uav_xyz_m=uav,
        allocation=alloc,
        true_eval=ev,
        n_steps=int(settings.total_steps),
        diagnostics={
            "method": "td3",
            "label": (
                "TD3 Algorithm 2 fill-in: per-instance train, Alg. 2 reward, "
                "Fujimoto knobs in TD3Settings (not Table II)"
            ),
            "n_updates": int(agent.n_updates),
            "n_episodes": int(n_episodes),
            "obs_dim": int(env.obs_dim),
            "act_dim": int(env.act_dim),
            "hidden": int(settings.hidden),
            "n_move": int(env.n_move),
            "actor_layer_norm": bool(settings.actor_layer_norm),
            "actor_preact_l2": float(settings.actor_preact_l2),
            "actor_logit_l2": float(settings.actor_logit_l2),
            "logit_clip": float(settings.logit_clip),
            "move_mode": str(settings.move_mode),
            "assoc_distance_coef": float(settings.assoc_distance_coef),
            "process_mode": str(settings.process_mode),
            "assoc_mode": str(settings.assoc_mode),
            "uav_init": str(settings.uav_init),
            "inner_bandwidth": str(settings.inner_bandwidth),
            "discount": float(settings.discount),
            "critic_use_obs": bool(settings.critic_use_obs),
            "critic_move_only": bool(settings.critic_move_only),
            "export_actor": str(settings.export_actor),
            "best_actor_episode_Mbps": (
                None
                if not np.isfinite(agent.best_actor_score)
                else float(agent.best_actor_score)
            ),
            "total_steps": int(settings.total_steps),
            "horizon": int(settings.horizon),
            "export_bandwidth": "frozen_q_lp",
            "export_mode": str(settings.export_mode),
            "export_rule": export_rule,
            "export_avg_steps": n_avg,
            "best_inner_feasible": bool(train_best.true_eval.feasible),
            "best_inner_sum_rate_Mbps": float(train_best.true_eval.sum_rate_mbps),
            "best_inner_reward": float(train_best.reward),
            "snapshot_export_sum_rate_Mbps": float(snap_ev.sum_rate_mbps),
            "snapshot_export_feasible": bool(snap_ev.feasible),
            "policy_export_sum_rate_Mbps": float(pol_ev.sum_rate_mbps),
            "policy_export_feasible": bool(pol_ev.feasible),
            "policy_inner_mean_Mbps": float(policy["policy_inner_mean_Mbps"]),
            "policy_inner_last_Mbps": float(policy["policy_inner_last_Mbps"]),
            "policy_inner_feas_mean": float(policy["policy_inner_feas_mean"]),
            "policy_mean_move_action_norm": float(policy["mean_move_action_norm"]),
            "policy_nearest_agree": float(policy["nearest_agree"]),
            "policy_min_pairwise_m": policy["min_pairwise_m"],
            "policy_uav_loads": policy["uav_loads"],
            "r_max_bit_per_s": float(env.r_max),
            "wall_clock_s": elapsed,
        },
    )
