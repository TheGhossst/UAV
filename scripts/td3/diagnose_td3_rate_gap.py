"""Why TD3's inner equal-share sits at ~6 Mbps: placement, association, or reward.

Trains seeds 1-3 at I=10 J=3 (same radio as the policy-export probe).
Does not change SimConfig. Not a campaign result.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from uavdt.config import BANDWIDTH_PRESETS, PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.constraints import pairwise_uav_distance_m
from uavdt.evaluator import evaluate
from uavdt.experiments.grids import config_for_counts
from uavdt.experiments.methods import run_method
from uavdt.models import Allocation, Scenario
from uavdt.resources import (
    allocation_from_positions,
    cpu_stable_processing,
    equal_share_bandwidth_hz,
    nearest_association,
)
from uavdt.scenario import generate_scenario
from uavdt.td3.agent import make_td3_agent
from uavdt.td3.env import UAVAoDTEnv, reward_breakdown, zenith_r_max_bit_per_s
from uavdt.td3.settings import TD3Settings
from uavdt.td3.solve import (
    deterministic_policy_export,
    format_progress_line,
    log_td3,
    lp_score,
)

OUT = Path("results/td3/rate_gap_diagnostic.json")
TXT = Path("results/td3/rate_gap_diagnostic.txt")
SEEDS = (1, 2, 3)


def _cfg() -> SimConfig:
    return config_for_counts(
        10,
        3,
        SimConfig(
            b_sys_hz=BANDWIDTH_PRESETS["8.8mhz"],
            max_bw_share=PRIMARY_MAX_BW_SHARE,
        ),
    )


def _eq_eval(scenario: Scenario, uav: np.ndarray, alloc: Allocation | None = None):
    uav = np.asarray(uav, dtype=float)
    if alloc is None:
        alloc = allocation_from_positions(scenario, uav)
    else:
        a = alloc.hard_association()
        b = alloc.hard_processing()
        alloc = Allocation(a, b, equal_share_bandwidth_hz(a, scenario.cfg))
    return evaluate(scenario, uav, alloc)


def _geometry(scenario: Scenario, uav: np.ndarray, alloc: Allocation) -> dict:
    uav = np.asarray(uav, dtype=float)
    a = alloc.hard_association()
    nearest = nearest_association(scenario.iot_xyz_m, uav)
    loads = a.sum(axis=0)
    dists = pairwise_uav_distance_m(uav)
    j = uav.shape[0]
    pair = [float(dists[p, q]) for p in range(j) for q in range(p + 1, j)]
    return {
        "nearest_agree": float(np.mean(np.argmax(a, axis=1) == np.argmax(nearest, axis=1))),
        "uav_loads": [float(x) for x in loads],
        "max_load": float(np.max(loads)),
        "min_pairwise_m": float(min(pair)) if pair else None,
        "mean_pairwise_m": float(np.mean(pair)) if pair else None,
        "xy_std_m": float(np.std(uav[:, :2])),
        "xy": np.asarray(uav[:, :2], dtype=float).round(2).tolist(),
    }


def _reward_scale(scenario: Scenario) -> dict:
    cfg = scenario.cfg
    settings = TD3Settings()
    r_max = zenith_r_max_bit_per_s(cfg)
    rows = []
    for mbps in (6.3, 7.7, 8.9):
        rate_term = settings.rate_weight * (mbps * 1e6) / r_max
        rows.append(
            {
                "Mbps": mbps,
                "rate_term": float(rate_term),
                "vs_one_dist_penalty": float(rate_term / max(settings.dist_weight, 1e-12)),
                "vs_one_aodt_unit": float(rate_term / max(settings.aodt_weight, 1e-12)),
            }
        )
    return {
        "r_max_Mbps": float(r_max / 1e6),
        "aodt_weight": settings.aodt_weight,
        "dist_weight": settings.dist_weight,
        "viol_weight": settings.viol_weight,
        "rate_weight": settings.rate_weight,
        "at_rates": rows,
        "note": (
            "Alg. 2 reward is rate/R_max - 10 P_AoDT - 5 P_dist - 5 V. "
            "R_max is zenith Shannon with full B_sys (implementation choice)."
        ),
    }


def _window_stats(xs: list[float], a: int, b: int | None = None) -> dict:
    sl = xs[a:] if b is None else xs[a:b]
    arr = np.asarray(sl, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _score_uav(scenario: Scenario, name: str, uav: np.ndarray, alloc: Allocation) -> dict:
    eq_own = _eq_eval(scenario, uav, alloc)
    eq_near = _eq_eval(scenario, uav, None)
    _u, lp_alloc, lp_own = lp_score(scenario, uav, alloc)
    near_a = nearest_association(scenario.iot_xyz_m, uav)
    near_b = cpu_stable_processing(scenario, near_a)
    _u2, _lp2, lp_near = lp_score(
        scenario, uav, Allocation(near_a, near_b, equal_share_bandwidth_hz(near_a, scenario.cfg))
    )
    geo = _geometry(scenario, uav, alloc)
    return {
        "name": name,
        "equal_share_own_ab_Mbps": float(eq_own.sum_rate_mbps),
        "equal_share_nearest_Mbps": float(eq_near.sum_rate_mbps),
        "lp_own_ab_Mbps": float(lp_own.sum_rate_mbps),
        "lp_nearest_Mbps": float(lp_near.sum_rate_mbps),
        "eq_own_feasible": bool(eq_own.feasible),
        "eq_nearest_feasible": bool(eq_near.feasible),
        "geometry": geo,
    }


def train_one(seed: int, settings: TD3Settings) -> dict:
    cfg = _cfg()
    scenario = generate_scenario(seed, cfg)
    rng = np.random.default_rng(int(seed))
    env = UAVAoDTEnv(scenario, settings, seed=int(seed))
    agent = make_td3_agent(env, settings, int(seed))
    rewards: list[float] = []
    rates: list[float] = []
    rate_terms: list[float] = []
    p_dists: list[float] = []
    p_aodts: list[float] = []
    v_viols: list[float] = []
    warmup = int(settings.warmup_steps)
    t0 = perf_counter()
    obs = env.reset()
    log_td3(f"=== diagnostic train seed={seed}  R_max={env.r_max/1e6:.3f} Mbps ===")
    for t in range(settings.total_steps):
        if t < warmup or agent.replay.size < settings.batch_size:
            action = agent.sample_warmup_action(rng)
        else:
            action = agent.select_action(obs, noise=True)
        next_obs, reward, done, info = env.step(action)
        agent.replay.add(obs, action, reward, next_obs, done)
        if t >= warmup and agent.replay.size >= settings.batch_size:
            agent.update()
        rewards.append(float(reward))
        rates.append(float(info["sum_rate_Mbps"]))
        rate_terms.append(float(info["rate_term"]))
        p_dists.append(float(info["p_dist"]))
        p_aodts.append(float(info["p_aodt"]))
        v_viols.append(float(info["v_viol"]))
        if done:
            obs = env.reset() if t + 1 < settings.total_steps else next_obs
        else:
            obs = next_obs
        if (t + 1) % 1000 == 0 or t + 1 == settings.total_steps:
            log_td3(
                format_progress_line(
                    t=t,
                    total=settings.total_steps,
                    warmup=warmup,
                    elapsed=perf_counter() - t0,
                    window_r=rewards[-250:],
                    window_mbps=rates[-250:],
                    window_feas=[float(info["feasible"])],
                    info=info,
                    env=env,
                    n_updates=int(agent.n_updates),
                    critic_loss=agent.last_critic_loss,
                    actor_loss=agent.last_actor_loss,
                )
            )

    assert env.best is not None
    train_best = env.best
    policy = deterministic_policy_export(env, agent, settings)
    pol_uav = np.asarray(policy["uav_xyz_m"], dtype=float)
    pol_alloc = policy["allocation"]

    # Live last-N inner steps used the unaveraged trajectory; re-roll for move size.
    env.freeze_best = True
    obs = env.reset()
    move_norms: list[float] = []
    step_rates: list[float] = []
    agrees: list[float] = []
    step_dxy: list[float] = []
    prev = None
    last_alloc = None
    last_uav = None
    for _ in range(settings.horizon):
        action = agent.select_action(obs, noise=False)
        move_norms.append(float(np.linalg.norm(action[: 2 * cfg.num_uav])))
        obs, _r, done, info = env.step(action)
        assert env._uav is not None and env._alloc is not None
        cur = np.asarray(env._uav, dtype=float).copy()
        if prev is not None:
            step_dxy.append(float(np.linalg.norm(cur[:, :2] - prev[:, :2], axis=1).mean()))
        prev = cur
        last_uav = cur
        last_alloc = env._alloc
        geo = _geometry(scenario, env._uav, env._alloc)
        agrees.append(geo["nearest_agree"])
        step_rates.append(float(info["sum_rate_Mbps"]))
        if done:
            break

    baselines = {}
    for name in ("random", "kmeans", "pso", "sca"):
        run = run_method(scenario, name, seed)
        baselines[name] = _score_uav(scenario, name, run.uav_xyz_m, run.allocation)

    snap = _score_uav(scenario, "td3_snapshot", train_best.uav_xyz_m, train_best.allocation)
    pol = _score_uav(scenario, "td3_policy", pol_uav, pol_alloc)
    last_live = _score_uav(scenario, "td3_policy_last_step", last_uav, last_alloc)

    return {
        "seed": seed,
        "r_max_Mbps": float(env.r_max / 1e6),
        "curves": {
            "warmup_Mbps": _window_stats(rates, 0, warmup),
            "last500_Mbps": _window_stats(rates, -500),
            "last500_reward": _window_stats(rewards, -500),
            "last500_rate_term": _window_stats(rate_terms, -500),
            "last500_p_dist": _window_stats(p_dists, -500),
            "last500_p_aodt": _window_stats(p_aodts, -500),
            "last500_v_viol": _window_stats(v_viols, -500),
        },
        "policy_rollout": {
            "inner_lastN_Mbps": float(policy["policy_inner_mean_Mbps"]),
            "mean_action_move_norm": float(np.mean(move_norms)),
            "mean_step_xy_m": float(np.mean(step_dxy)) if step_dxy else 0.0,
            "mean_nearest_agree": float(np.mean(agrees)),
            "step_Mbps": [float(x) for x in step_rates],
        },
        "baselines": baselines,
        "td3_snapshot": snap,
        "td3_policy": pol,
        "td3_policy_last_step": last_live,
        "wall_clock_s": perf_counter() - t0,
    }


def _lines(payload: dict) -> list[str]:
    sc = payload["reward_scale"]
    lines = [
        "TD3 inner-rate gap diagnostic (I=10 J=3, 8.8 MHz, 25% cap, seeds 1-3)",
        sc["note"],
        f"R_max (zenith full B_sys) = {sc['r_max_Mbps']:.3f} Mbps",
        (
            f"weights: rate={sc['rate_weight']} aodt={sc['aodt_weight']} "
            f"dist={sc['dist_weight']} viol={sc['viol_weight']}"
        ),
    ]
    for row in sc["at_rates"]:
        lines.append(
            f"  {row['Mbps']:.1f} Mbps -> rate_term={row['rate_term']:.4f}  "
            f"(that / dist_w={row['vs_one_dist_penalty']:.4f}, "
            f"/ aodt_w={row['vs_one_aodt_unit']:.4f})"
        )
    lines.append("")
    for rec in payload["seeds"]:
        lines.append(f"=== seed {rec['seed']} ===")
        c = rec["curves"]
        lines.append(
            f"  inner Mbps  warmup={c['warmup_Mbps']['mean']:.3f}  "
            f"last500={c['last500_Mbps']['mean']:.3f}  "
            f"rate_term last500={c['last500_rate_term']['mean']:.4f}  "
            f"P_dist={c['last500_p_dist']['mean']:.3f}  "
            f"P_AoDT={c['last500_p_aodt']['mean']:.3f}  "
            f"V={c['last500_v_viol']['mean']:.3f}"
        )
        pr = rec["policy_rollout"]
        lines.append(
            f"  det. rollout  inner={pr['inner_lastN_Mbps']:.3f}  "
            f"|move_action|={pr['mean_action_move_norm']:.3f}  "
            f"nearest_agree={pr['mean_nearest_agree']:.2f}"
        )
        lines.append(
            "  equal-share Mbps: own a/b vs nearest a/b  |  LP own vs LP nearest"
        )
        for key in ("random", "kmeans", "pso", "sca"):
            b = rec["baselines"][key]
            lines.append(
                f"    {key:8s}  eq_own={b['equal_share_own_ab_Mbps']:.3f}  "
                f"eq_near={b['equal_share_nearest_Mbps']:.3f}  "
                f"lp_own={b['lp_own_ab_Mbps']:.3f}  lp_near={b['lp_nearest_Mbps']:.3f}  "
                f"agree={b['geometry']['nearest_agree']:.2f}  "
                f"loads={b['geometry']['uav_loads']}"
            )
        for key, label in (
            ("td3_snapshot", "snap"),
            ("td3_policy", "policy"),
            ("td3_policy_last_step", "last"),
        ):
            b = rec[key]
            lines.append(
                f"    {label:8s}  eq_own={b['equal_share_own_ab_Mbps']:.3f}  "
                f"eq_near={b['equal_share_nearest_Mbps']:.3f}  "
                f"lp_own={b['lp_own_ab_Mbps']:.3f}  lp_near={b['lp_nearest_Mbps']:.3f}  "
                f"agree={b['geometry']['nearest_agree']:.2f}  "
                f"loads={b['geometry']['uav_loads']}  "
                f"min_d={b['geometry']['min_pairwise_m']}"
            )
        pol = rec["td3_policy"]
        rnd = rec["baselines"]["random"]
        lines.append(
            f"  gap policy eq_own vs random eq_near: "
            f"{pol['equal_share_own_ab_Mbps'] - rnd['equal_share_nearest_Mbps']:+.3f} Mbps"
        )
        lines.append(
            f"  gap policy eq_near vs random eq_near: "
            f"{pol['equal_share_nearest_Mbps'] - rnd['equal_share_nearest_Mbps']:+.3f} Mbps"
            "  (placement-only, nearest a/b)"
        )
        lines.append(
            f"  assoc tax (eq_near - eq_own) on policy: "
            f"{pol['equal_share_nearest_Mbps'] - pol['equal_share_own_ab_Mbps']:+.3f} Mbps"
        )
        lines.append("")
    return lines


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="TD3 inner-rate gap diagnostic")
    p.add_argument("--seeds", default="1,2,3")
    args = p.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = TD3Settings(device=device, log_every=1000, export_mode="policy")
    cfg = _cfg()
    sc0 = generate_scenario(1, cfg)
    payload = {
        "reward_scale": _reward_scale(sc0),
        "device": device,
        "torch": str(torch.__version__),
        "seeds_requested": list(seeds),
        "seeds": [],
    }
    log_td3(
        f"R_max={payload['reward_scale']['r_max_Mbps']:.3f} Mbps  "
        f"6.3 Mbps rate_term={payload['reward_scale']['at_rates'][0]['rate_term']:.4f}"
    )
    for seed in seeds:
        payload["seeds"].append(train_one(seed, settings))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = "\n".join(_lines(payload)) + "\n"
    TXT.write_text(text, encoding="utf-8")
    log_td3(text)
    log_td3(f"wrote {OUT}")
    log_td3(f"wrote {TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
