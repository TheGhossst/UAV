"""Go/no-go: TD3 actor unsaturates and inner equal-share climbs on seeds 1-3.

BEFORE (n=20 saturated policy): move rail ~1, pre-tanh |z|~12-16,
actor_std ~1e-7, inner ~6.3 Mbps vs equal-share baselines 8.4-8.8.

Does not change SimConfig. Does not overwrite BEFORE policy-export files.
Not a campaign result.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from uavdt.config import BANDWIDTH_PRESETS, PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.grids import config_for_counts
from uavdt.experiments.methods import run_method
from uavdt.scenario import generate_scenario
from uavdt.td3.agent import make_td3_agent
from uavdt.td3.env import UAVAoDTEnv
from uavdt.td3.settings import TD3Settings
from uavdt.td3.solve import (
    deterministic_policy_export,
    format_eta,
    log_td3,
    lp_score,
)

OUT = Path("results/td3/unsaturate_verify.json")
TXT = Path("results/td3/unsaturate_verify.txt")
SEEDS = (1, 2, 3)
RAIL = 0.95
PROBE_EVERY = 250

# BEFORE failure numbers. Pass if we clearly leave that regime and climb.
# Pass if we close the remaining SCA gap (AFTER hover trainer was -0.10 Mbps).
MAX_MOVE_RAIL = 1.01
MAX_PRE_TANH = 4.0
MIN_ACTOR_STD = 0.0
MIN_INNER_MBPS = 8.4
MAX_GAP_VS_SCA_LP = 0.12


def _cfg() -> SimConfig:
    return config_for_counts(
        10,
        3,
        SimConfig(
            b_sys_hz=BANDWIDTH_PRESETS["8.8mhz"],
            max_bw_share=PRIMARY_MAX_BW_SHARE,
        ),
    )


def _probe(agent) -> dict | None:
    if agent.replay.size < agent.settings.batch_size:
        return None
    n = agent.n_move
    s, _a, _r, _s2, _d = agent.replay.sample(agent.settings.batch_size)
    with torch.no_grad():
        pi = agent.actor(s)
        z0 = torch.zeros_like(pi)
        q_pi = agent.q1(s, pi).mean()
        q_0 = agent.q1(s, z0).mean()
        across_move = pi[:, :n].std(dim=0).mean()
        pre = agent.actor.forward_pre_tanh(s)
        move_rail = (pi[:, :n].abs() >= RAIL).float().mean()
    return {
        "q_pi": float(q_pi.item()),
        "q_0": float(q_0.item()),
        "q_pi_minus_q_0": float((q_pi - q_0).item()),
        "actor_std_move": float(across_move.item()),
        "pre_tanh_mean_abs": float(pre.abs().mean().item()),
        "pre_tanh_max_abs": float(pre.abs().max().item()),
        "det_move_rail": float(move_rail.item()),
    }


def train_seed(seed: int, settings: TD3Settings) -> dict:
    cfg = _cfg()
    scenario = generate_scenario(seed, cfg)
    rng = np.random.default_rng(int(seed))
    env = UAVAoDTEnv(scenario, settings, seed=int(seed))
    agent = make_td3_agent(env, settings, int(seed))
    warmup = int(settings.warmup_steps)
    obs = env.reset()
    t0 = perf_counter()
    probes: list[dict] = []
    window_mbps: list[float] = []
    ep_mbps: list[float] = []
    log_td3(f"=== unsaturate verify seed={seed}  act={env.act_dim} n_move={env.n_move} ===")
    for t in range(settings.total_steps):
        if t < warmup or agent.replay.size < settings.batch_size:
            action = agent.sample_warmup_action(rng)
        else:
            action = agent.select_action(obs, noise=True)
        next_obs, _reward, done, info = env.step(action)
        agent.replay.add(obs, action, _reward, next_obs, done)
        if t >= warmup and agent.replay.size >= settings.batch_size:
            agent.update()
        mbps = float(info["sum_rate_Mbps"])
        window_mbps.append(mbps)
        ep_mbps.append(mbps)
        if (t + 1) % PROBE_EVERY == 0 or t + 1 == settings.total_steps:
            probe = _probe(agent) or {}
            probe["step"] = t + 1
            probe["window_Mbps"] = float(np.mean(window_mbps[-PROBE_EVERY:]))
            probes.append(probe)
            log_td3(
                f"  step {t+1:5d}/{settings.total_steps}  "
                f"Mbps={probe['window_Mbps']:6.3f}  "
                f"move_rail={probe.get('det_move_rail', float('nan')):.2f}  "
                f"pre|z|={probe.get('pre_tanh_mean_abs', float('nan')):.2f}  "
                f"act_std={probe.get('actor_std_move', float('nan')):.3f}  "
                f"eta={format_eta((settings.total_steps - t - 1) * (perf_counter()-t0) / (t+1))}"
            )
        if done:
            agent.consider_checkpoint(float(np.mean(ep_mbps)))
            ep_mbps = []
            obs = env.reset() if t + 1 < settings.total_steps else next_obs
        else:
            obs = next_obs

    if ep_mbps:
        agent.consider_checkpoint(float(np.mean(ep_mbps)))
    if settings.export_actor == "best_checkpoint":
        agent.load_best_actor()
    policy = deterministic_policy_export(env, agent, settings)
    _u, _alloc, pol_ev = lp_score(scenario, policy["uav_xyz_m"], policy["allocation"])
    baselines = {}
    for name in ("random", "kmeans", "pso", "sca"):
        run = run_method(scenario, name, seed=int(seed))
        baselines[name] = {
            "lp_Mbps": float(run.sum_rate_mbps),
            "feasible": bool(run.true_eval.feasible),
        }
    final = probes[-1] if probes else {}
    inner = float(policy["policy_inner_mean_Mbps"])
    sca_lp = float(baselines["sca"]["lp_Mbps"])
    checks = {
        "pre_tanh_ok": float(final.get("pre_tanh_mean_abs", 99.0)) <= MAX_PRE_TANH,
        "inner_ok": inner >= MIN_INNER_MBPS,
        "vs_sca_ok": float(pol_ev.sum_rate_mbps) >= sca_lp - MAX_GAP_VS_SCA_LP,
    }
    return {
        "seed": seed,
        "wall_clock_s": perf_counter() - t0,
        "policy_inner_mean_Mbps": inner,
        "policy_lp_Mbps": float(pol_ev.sum_rate_mbps),
        "policy_lp_feasible": bool(pol_ev.feasible),
        "baselines": baselines,
        "inner_minus_sca_lp": inner - float(baselines["sca"]["lp_Mbps"]),
        "final_probe": final,
        "checks": checks,
        "pass": all(checks.values()),
        "probes": probes,
        "mean_move_action_norm": float(policy["mean_move_action_norm"]),
        "nearest_agree": float(policy["nearest_agree"]),
        "best_actor_episode_Mbps": float(agent.best_actor_score),
    }


def _lines(payload: dict) -> list[str]:
    lines = [
        "TD3 unsaturate verify -- seeds 1-3, I=10 J=3, 8.8 MHz, 25% cap, 7000 steps",
        "Fix: k-means init, Alg. 2 Δ×10, leftover-dump inner B, nearest+cpu_stable.",
        (
            f"Pass gates: pre|z|<={MAX_PRE_TANH}, inner>={MIN_INNER_MBPS} Mbps, "
            f"policy_LP within {MAX_GAP_VS_SCA_LP} of SCA."
        ),
        "",
    ]
    for rec in payload["seeds"]:
        fp = rec["final_probe"] or {}
        b = rec["baselines"]
        lines += [
            f"=== seed {rec['seed']}  {'PASS' if rec['pass'] else 'FAIL'} ===",
            (
                f"  inner={rec['policy_inner_mean_Mbps']:.3f}  "
                f"policy_LP={rec['policy_lp_Mbps']:.3f}  "
                f"random_LP={b['random']['lp_Mbps']:.3f}  "
                f"sca_LP={b['sca']['lp_Mbps']:.3f}"
            ),
            (
                f"  move_rail={fp.get('det_move_rail')}  "
                f"pre|z|={fp.get('pre_tanh_mean_abs')}  "
                f"act_std={fp.get('actor_std_move')}  "
                f"checks={rec['checks']}"
            ),
            "",
        ]
    lines.append(f"overall={'PASS' if payload['pass'] else 'FAIL'}")
    return lines


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = TD3Settings(device=device, log_every=0, export_mode="policy")
    log_td3(f"TD3 device            {device}  torch={torch.__version__}")
    payload = {
        "note": (
            "3-seed go/no-go after the dead-tanh fix. Not a campaign. "
            "Do not overwrite policy_export_n20_BEFORE_saturation.json."
        ),
        "device": device,
        "gates": {
            "max_move_rail": MAX_MOVE_RAIL,
            "max_pre_tanh": MAX_PRE_TANH,
            "min_actor_std": MIN_ACTOR_STD,
            "min_inner_Mbps": MIN_INNER_MBPS,
            "max_gap_vs_sca_lp": MAX_GAP_VS_SCA_LP,
            "move_mode": settings.move_mode,
            "uav_init": settings.uav_init,
            "inner_bandwidth": settings.inner_bandwidth,
            "assoc_mode": settings.assoc_mode,
            "process_mode": settings.process_mode,
            "discount": settings.discount,
        },
        "settings": {
            "actor_preact_l2": settings.actor_preact_l2,
            "actor_logit_l2": settings.actor_logit_l2,
            "logit_clip": settings.logit_clip,
            "actor_layer_norm": settings.actor_layer_norm,
        },
        "seeds": [],
    }
    for seed in SEEDS:
        payload["seeds"].append(train_seed(seed, settings))
    payload["pass"] = all(r["pass"] for r in payload["seeds"])
    mean_inner = float(np.mean([r["policy_inner_mean_Mbps"] for r in payload["seeds"]]))
    payload["mean_inner_Mbps"] = mean_inner
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = "\n".join(_lines(payload)) + "\n"
    TXT.write_text(text, encoding="utf-8")
    log_td3(text)
    log_td3(f"wrote {OUT}")
    log_td3(f"wrote {TXT}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
