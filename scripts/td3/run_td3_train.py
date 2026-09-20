"""Train TD3 on one scenario, log a curve, compare to other methods.

Does not change SimConfig. Algorithm knobs are TD3Settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from uavdt.config import BANDWIDTH_PRESETS, PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.methods import run_method
from uavdt.scenario import generate_scenario
from uavdt.td3.agent import make_td3_agent
from uavdt.td3.env import UAVAoDTEnv
from uavdt.td3.settings import TD3Settings
from uavdt.td3.solve import (
    deterministic_policy_export,
    format_progress_line,
    log_td3,
    lp_score,
)


def _cfg(args: argparse.Namespace) -> SimConfig:
    b_hz = BANDWIDTH_PRESETS[args.bandwidth_preset]
    return SimConfig(
        b_sys_hz=float(b_hz),
        max_bw_share=args.max_bw_share,
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _eval_record(run) -> dict:
    ev = run.true_eval
    c = ev.constraints
    return {
        "method": run.method,
        "sum_rate_Mbps": float(ev.sum_rate_mbps),
        "feasible": bool(ev.feasible),
        "aodt_s": [float(x) for x in ev.aodt_s],
        "max_AoDT_s": float(np.max(ev.aodt_s)),
        "rho": [float(x) for x in ev.rho],
        "qos_violations": int(c.qos_violations),
        "aodt_violations": int(c.aodt_violations),
        "sep_violations": int(c.sep_violations),
        "cpu_unstable_count": int(c.cpu_unstable_count),
        "wall_clock_s": run.diagnostics.get("wall_clock_s"),
        "uav_xyz_m": np.asarray(run.uav_xyz_m, dtype=float).tolist(),
    }


def train(args: argparse.Namespace) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required. pip install torch") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = _cfg(args)
    scenario = generate_scenario(args.seed, cfg)
    settings = TD3Settings(
        total_steps=int(args.total_steps),
        horizon=int(args.horizon),
        hidden=int(args.hidden),
        batch_size=int(args.batch_size),
        warmup_steps=int(args.warmup_steps),
        buffer_size=int(args.buffer_size),
        device=device,
        log_every=int(args.log_every),
    )
    rng = np.random.default_rng(int(args.seed))
    env = UAVAoDTEnv(scenario, settings, seed=int(args.seed))
    agent = make_td3_agent(env, settings, int(args.seed))

    rewards: list[float] = []
    rates: list[float] = []
    feas: list[bool] = []
    episode_returns: list[float] = []
    ep_ret = 0.0
    n_episodes = 1
    warmup = int(settings.warmup_steps)
    t0 = perf_counter()
    obs = env.reset()
    log_every = max(1, int(args.log_every))
    log_td3(
        f"TD3 train  seed={int(args.seed)}  I={cfg.num_iot} J={cfg.num_uav} "
        f"K={cfg.num_processes}  steps={settings.total_steps} "
        f"horizon={settings.horizon} warmup={warmup}  "
        f"device={device}  obs={env.obs_dim} act={env.act_dim}  "
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
        ev = info["eval"]
        rewards.append(float(reward))
        rates.append(float(ev.sum_rate_mbps))
        feas.append(bool(ev.feasible))
        ep_ret += float(reward)
        if done:
            episode_returns.append(ep_ret)
            ep_ret = 0.0
            n_episodes += 1
            if t + 1 < settings.total_steps:
                obs = env.reset()
            else:
                obs = next_obs
        else:
            obs = next_obs
        if (t + 1) % log_every == 0 or t + 1 == settings.total_steps:
            log_td3(
                format_progress_line(
                    t=t,
                    total=settings.total_steps,
                    warmup=warmup,
                    elapsed=perf_counter() - t0,
                    window_r=rewards[-log_every:],
                    window_mbps=rates[-log_every:],
                    window_feas=[float(x) for x in feas[-log_every:]],
                    info=info,
                    env=env,
                    n_updates=int(agent.n_updates),
                    critic_loss=agent.last_critic_loss,
                    actor_loss=agent.last_actor_loss,
                )
            )

    if env.best is None:
        raise RuntimeError("TD3 produced no snapshot")
    train_best = env.best
    _snap_uav, snap_alloc, snap_ev = lp_score(
        scenario, train_best.uav_xyz_m, train_best.allocation
    )
    policy = deterministic_policy_export(env, agent, settings)
    pol_uav, _pol_alloc, pol_ev = lp_score(
        scenario, policy["uav_xyz_m"], policy["allocation"]
    )
    if settings.export_mode == "best_snapshot":
        final_ev = snap_ev
        export_uav = np.asarray(train_best.uav_xyz_m, dtype=float)
        export_rule = "best_snapshot frozen_q_lp"
    else:
        final_ev = pol_ev
        export_uav = np.asarray(pol_uav, dtype=float)
        export_rule = (
            f"deterministic_policy last_{int(policy['n_avg'])}_xy_mean "
            "last_step_ab frozen_q_lp"
        )
    train_s = perf_counter() - t0

    def running_mean(xs: list[float], w: int = 100) -> list[float]:
        out = []
        acc = 0.0
        q: list[float] = []
        for x in xs:
            q.append(x)
            acc += x
            if len(q) > w:
                acc -= q.pop(0)
            out.append(acc / len(q))
        return out

    return {
        "paper": "Khalaf et al. IEEE TNSM 2026 Algorithm 2 fill-in",
        "note": (
            "Per-instance TD3. TD3Settings, not Table II. "
            "Inner B is equal-share; export uses frozen-q LP + evaluate()."
        ),
        "seed": int(args.seed),
        "device": device,
        "torch": str(torch.__version__),
        "cuda": bool(torch.cuda.is_available()),
        "cfg": {
            "area_x_m": cfg.area_x_m,
            "area_y_m": cfg.area_y_m,
            "b_sys_hz": cfg.b_sys_hz,
            "max_bw_share": cfg.max_bw_share,
            "num_iot": cfg.num_iot,
            "num_uav": cfg.num_uav,
            "lambda_i_per_s": cfg.lambda_i_per_s,
            "aodt_threshold_s": cfg.aodt_threshold_s,
        },
        "settings": {
            "total_steps": settings.total_steps,
            "horizon": settings.horizon,
            "hidden": settings.hidden,
            "batch_size": settings.batch_size,
            "warmup_steps": settings.warmup_steps,
            "explore_noise": settings.explore_noise,
            "discount": settings.discount,
        },
        "train": {
            "wall_clock_s": train_s,
            "n_updates": int(agent.n_updates),
            "n_episodes": int(n_episodes),
            "obs_dim": int(env.obs_dim),
            "act_dim": int(env.act_dim),
            "mean_reward": float(np.mean(rewards)),
            "mean_reward_last_500": float(np.mean(rewards[-500:])),
            "mean_reward_warmup": float(np.mean(rewards[:warmup])) if warmup else None,
            "mean_reward_after_warmup": (
                float(np.mean(rewards[warmup:])) if len(rewards) > warmup else None
            ),
            "feasible_fraction": float(np.mean(feas)),
            "feasible_fraction_last_500": float(np.mean(feas[-500:])),
            "mean_Mbps_inner": float(np.mean(rates)),
            "mean_Mbps_inner_last_500": float(np.mean(rates[-500:])),
            "episode_returns": episode_returns,
            "reward_running_mean_100": running_mean(rewards, 100),
            "rate_running_mean_100": running_mean(rates, 100),
            "rewards_subsample": rewards[:: max(1, len(rewards) // 700)],
            "rates_subsample": rates[:: max(1, len(rates) // 700)],
            "best_inner_sum_rate_Mbps": float(env.best.true_eval.sum_rate_mbps),
            "best_inner_feasible": bool(env.best.true_eval.feasible),
            "best_inner_reward": float(env.best.reward),
            "r_max_bit_per_s": float(env.r_max),
        },
        "td3_export": {
            "export_rule": export_rule,
            "sum_rate_Mbps": float(final_ev.sum_rate_mbps),
            "feasible": bool(final_ev.feasible),
            "snapshot_export_sum_rate_Mbps": float(snap_ev.sum_rate_mbps),
            "snapshot_export_feasible": bool(snap_ev.feasible),
            "policy_export_sum_rate_Mbps": float(pol_ev.sum_rate_mbps),
            "policy_export_feasible": bool(pol_ev.feasible),
            "policy_inner_mean_Mbps": float(policy["policy_inner_mean_Mbps"]),
            "aodt_s": [float(x) for x in final_ev.aodt_s],
            "max_AoDT_s": float(np.max(final_ev.aodt_s)),
            "rho": [float(x) for x in final_ev.rho],
            "qos_violations": int(final_ev.constraints.qos_violations),
            "aodt_violations": int(final_ev.constraints.aodt_violations),
            "sep_violations": int(final_ev.constraints.sep_violations),
            "cpu_unstable_count": int(final_ev.constraints.cpu_unstable_count),
            "uav_xyz_m": np.asarray(export_uav, dtype=float).tolist(),
        },
    }


def compare_methods(args: argparse.Namespace) -> dict:
    cfg = _cfg(args)
    scenario = generate_scenario(args.seed, cfg)
    by_method = {}
    for name in ("random", "kmeans", "pso", "sca"):
        t0 = perf_counter()
        print(f"  baseline {name} ...", flush=True)
        run = run_method(scenario, name, args.seed)
        run.diagnostics.setdefault("wall_clock_s", perf_counter() - t0)
        rec = _eval_record(run)
        by_method[name] = rec
        print(
            f"    {name:8s}  {rec['sum_rate_Mbps']:.4f} Mbps  "
            f"feas={rec['feasible']}  maxAoDT={rec['max_AoDT_s']:.3f}s  "
            f"{rec['wall_clock_s']:.2f}s",
            flush=True,
        )
    return by_method


def maybe_plot(payload: dict, fig_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip figure")
        return
    train = payload["train"]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    axes[0].plot(train["reward_running_mean_100"], color="C0", lw=1.4)
    axes[0].set_ylabel("reward (100-step mean)")
    axes[0].axvline(256, color="0.5", ls="--", lw=0.8, label="warmup")
    axes[0].legend(frameon=False)
    axes[1].plot(train["rate_running_mean_100"], color="C1", lw=1.4)
    axes[1].set_ylabel("inner sum rate (Mbps)")
    axes[1].set_xlabel("env step")
    fig.suptitle(
        f"TD3 seed={payload['seed']}  "
        f"export {payload['td3_export']['sum_rate_Mbps']:.3f} Mbps  "
        f"feas={payload['td3_export']['feasible']}"
    )
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {fig_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Train TD3 and compare baselines")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bandwidth-preset", default="8.8mhz", choices=sorted(BANDWIDTH_PRESETS))
    p.add_argument("--max-bw-share", type=float, default=PRIMARY_MAX_BW_SHARE)
    p.add_argument("--total-steps", type=int, default=7000)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-steps", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--out", default="results/td3/train_seed1.json")
    args = p.parse_args()

    print("=== baselines on the same scenario ===", flush=True)
    by_method = compare_methods(args)
    print("=== TD3 train ===", flush=True)
    payload = train(args)
    payload["baselines"] = by_method
    td3 = payload["td3_export"]
    by_method["td3"] = {
        "method": "td3",
        "sum_rate_Mbps": td3["sum_rate_Mbps"],
        "feasible": td3["feasible"],
        "aodt_s": td3["aodt_s"],
        "max_AoDT_s": td3["max_AoDT_s"],
        "rho": td3["rho"],
        "qos_violations": td3["qos_violations"],
        "aodt_violations": td3["aodt_violations"],
        "sep_violations": td3["sep_violations"],
        "cpu_unstable_count": td3["cpu_unstable_count"],
        "wall_clock_s": payload["train"]["wall_clock_s"],
        "uav_xyz_m": td3["uav_xyz_m"],
    }
    payload["comparison"] = {
        name: {
            "sum_rate_Mbps": rec["sum_rate_Mbps"],
            "feasible": rec["feasible"],
            "max_AoDT_s": rec["max_AoDT_s"],
            "wall_clock_s": rec["wall_clock_s"],
        }
        for name, rec in by_method.items()
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    maybe_plot(payload, out.with_name(out.stem + "_curve.png"))
    print("=== comparison (true evaluate, frozen-q LP for placement methods) ===")
    for name, rec in by_method.items():
        print(
            f"{name:8s}  {rec['sum_rate_Mbps']:8.4f} Mbps  "
            f"feas={str(rec['feasible']):5s}  "
            f"maxAoDT={rec['max_AoDT_s']:7.3f}s  "
            f"{rec['wall_clock_s']:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
