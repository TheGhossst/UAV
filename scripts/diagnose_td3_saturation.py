"""Raw TD3 action distributions over training. Diagnosis only, no new fix.

Matches the n=20 saturated "before" trainer: I=10 J=3, 8.8 MHz, 25% cap,
Fujimoto last-layer init, frozen instance UAV init, 7000 steps.

Does not change SimConfig. Do not treat this as a campaign result.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from uavdt.config import BANDWIDTH_PRESETS, PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.grids import config_for_counts
from uavdt.scenario import generate_scenario
from uavdt.td3.agent import TD3Agent, make_td3_agent
from uavdt.td3.decode import _slices
from uavdt.td3.env import UAVAoDTEnv
from uavdt.td3.settings import TD3Settings
from uavdt.td3.solve import format_eta, log_td3

OUT = Path("results/td3/saturation_actions.json")
TXT = Path("results/td3/saturation_actions.txt")
FIG_DIR = Path("results/figures/td3_saturation")
SEEDS = (1, 2, 3)
RAIL = 0.95
PROBE_EVERY = 250
CURVE_EVERY = 50


def _cfg() -> SimConfig:
    return config_for_counts(
        10,
        3,
        SimConfig(
            b_sys_hz=BANDWIDTH_PRESETS["8.8mhz"],
            max_bw_share=PRIMARY_MAX_BW_SHARE,
        ),
    )


def _head_stats(vec: np.ndarray, n_move: int, n_assoc: int) -> dict:
    move = vec[:n_move]
    assoc = vec[n_move : n_move + n_assoc]
    proc = vec[n_move + n_assoc :]

    def pack(x: np.ndarray) -> dict:
        ax = np.abs(x)
        return {
            "mean": float(np.mean(x)),
            "mean_abs": float(np.mean(ax)),
            "std": float(np.std(x)),
            "frac_rail": float(np.mean(ax >= RAIL)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    return {"move": pack(move), "assoc": pack(assoc), "proc": pack(proc), "all": pack(vec)}


def _last_linear_rms(actor: torch.nn.Module) -> float:
    parts = [actor.move_head]
    if actor.disc_head is not None:
        parts.append(actor.disc_head)
    sq = []
    for layer in parts:
        sq.append(layer.weight.detach().square().mean())
        sq.append(layer.bias.detach().square().mean())
    return float(torch.sqrt(sum(sq) / len(sq)).item())


def _q_probe(agent: TD3Agent) -> dict | None:
    if agent.replay.size < agent.settings.batch_size:
        return None
    n = agent.n_move
    s, a_buf, _r, _s2, _d = agent.replay.sample(agent.settings.batch_size)
    with torch.no_grad():
        pi = agent.actor(s)
        z = torch.zeros_like(pi)
        q_pi = agent.q1(s, pi).mean()
        q_0 = agent.q1(s, z).mean()
        q_buf = agent.q1(s, a_buf).mean()
        q_sign = agent.q1(s, pi.sign()).mean()
        q_half = agent.q1(s, 0.5 * pi).mean()
        across = pi.std(dim=0).mean()
        across_move = pi[:, :n].std(dim=0).mean()
    pi_g = agent.actor(s)
    q = agent.q1(s, pi_g).mean()
    grad = torch.autograd.grad(q, pi_g, retain_graph=False)[0]
    with torch.no_grad():
        rail_push = (grad[:, :n] * pi_g[:, :n].sign()).mean()
        grad_abs = grad[:, :n].abs().mean()
        pre = agent.actor.forward_pre_tanh(s)
    return {
        "q_pi": float(q_pi.item()),
        "q_0": float(q_0.item()),
        "q_buffer": float(q_buf.item()),
        "q_sign_pi": float(q_sign.item()),
        "q_half_pi": float(q_half.item()),
        "q_pi_minus_q_0": float((q_pi - q_0).item()),
        "rail_push": float(rail_push.item()),
        "grad_abs": float(grad_abs.item()),
        "actor_std_across_batch": float(across.item()),
        "actor_std_move": float(across_move.item()),
        "pre_tanh_mean_abs": float(pre.abs().mean().item()),
        "pre_tanh_max_abs": float(pre.abs().max().item()),
        "last_linear_rms": _last_linear_rms(agent.actor),
    }


def train_seed(seed: int, settings: TD3Settings) -> dict:
    cfg = _cfg()
    scenario = generate_scenario(seed, cfg)
    n_move, n_assoc, n_proc = _slices(cfg)
    rng = np.random.default_rng(int(seed))
    env = UAVAoDTEnv(scenario, settings, seed=int(seed))
    agent = make_td3_agent(env, settings, int(seed))
    warmup = int(settings.warmup_steps)
    obs = env.reset()
    t0 = perf_counter()
    curves: list[dict] = []
    probes: list[dict] = []
    snapshots: list[dict] = []
    log_td3(f"=== saturation train seed={seed}  act={env.act_dim}  "
            f"move={n_move} assoc={n_assoc} proc={n_proc} ===")

    window_det: list[np.ndarray] = []
    window_exe: list[np.ndarray] = []
    window_mbps: list[float] = []
    window_rew: list[float] = []

    for t in range(settings.total_steps):
        with torch.no_grad():
            x = torch.as_tensor(
                np.asarray(obs, dtype=np.float32).reshape(1, -1),
                device=agent.device,
            )
            det = agent.actor(x)[0].cpu().numpy().astype(np.float64)
        if t < warmup or agent.replay.size < settings.batch_size:
            action = agent.sample_warmup_action(rng)
            phase = "warmup"
        else:
            action = agent.select_action(obs, noise=True)
            phase = "train"
        next_obs, reward, done, info = env.step(action)
        agent.replay.add(obs, action, reward, next_obs, done)
        if t >= warmup and agent.replay.size >= settings.batch_size:
            agent.update()
        window_det.append(det)
        window_exe.append(np.asarray(action, dtype=float))
        window_mbps.append(float(info["sum_rate_Mbps"]))
        window_rew.append(float(reward))

        if (t + 1) % CURVE_EVERY == 0:
            det_w = np.stack(window_det, axis=0)
            exe_w = np.stack(window_exe, axis=0)
            rec = {
                "step": t + 1,
                "phase": phase,
                "Mbps": float(np.mean(window_mbps)),
                "reward": float(np.mean(window_rew)),
                "det": _head_stats(det_w.mean(axis=0), n_move, n_assoc),
                "det_frac_rail": {
                    "move": float(np.mean(np.abs(det_w[:, :n_move]) >= RAIL)),
                    "assoc": float(
                        np.mean(np.abs(det_w[:, n_move : n_move + n_assoc]) >= RAIL)
                    ),
                    "proc": float(
                        np.mean(np.abs(det_w[:, n_move + n_assoc :]) >= RAIL)
                    ),
                },
                "exe_frac_rail": {
                    "move": float(np.mean(np.abs(exe_w[:, :n_move]) >= RAIL)),
                    "assoc": float(
                        np.mean(np.abs(exe_w[:, n_move : n_move + n_assoc]) >= RAIL)
                    ),
                    "proc": float(
                        np.mean(np.abs(exe_w[:, n_move + n_assoc :]) >= RAIL)
                    ),
                },
                "det_mean_abs": {
                    "move": float(np.mean(np.abs(det_w[:, :n_move]))),
                    "assoc": float(np.mean(np.abs(det_w[:, n_move : n_move + n_assoc]))),
                    "proc": float(np.mean(np.abs(det_w[:, n_move + n_assoc :]))),
                },
            }
            curves.append(rec)
            window_det = []
            window_exe = []
            window_mbps = []
            window_rew = []

        if (t + 1) % PROBE_EVERY == 0 or t + 1 == settings.total_steps:
            probe = _q_probe(agent) or {}
            probe["step"] = t + 1
            probe["phase"] = phase
            probes.append(probe)
            snapshots.append(
                {
                    "step": t + 1,
                    "det_action": [float(x) for x in det],
                }
            )
            log_td3(
                f"  step {t+1:5d}/{settings.total_steps}  {phase:6s}  "
                f"Mbps={float(info['sum_rate_Mbps']):6.3f}  "
                f"det_rail_move={curves[-1]['det_frac_rail']['move']:.2f}  "
                f"assoc={curves[-1]['det_frac_rail']['assoc']:.2f}  "
                f"pre|z|={probe.get('pre_tanh_mean_abs', float('nan')):.2f}  "
                f"Qpi-Q0={probe.get('q_pi_minus_q_0', float('nan')):.2f}  "
                f"rail_push={probe.get('rail_push', float('nan')):.3f}  "
                f"act_std={probe.get('actor_std_across_batch', float('nan')):.3f}  "
                f"eta={format_eta((settings.total_steps - t - 1) * (perf_counter()-t0) / (t+1))}"
            )

        if done:
            obs = env.reset() if t + 1 < settings.total_steps else next_obs
        else:
            obs = next_obs

    def first_rail(head: str, thresh: float = 0.8) -> int | None:
        for rec in curves:
            if rec["phase"] != "train":
                continue
            if rec["det_frac_rail"][head] >= thresh:
                return int(rec["step"])
        return None

    return {
        "seed": seed,
        "n_move": n_move,
        "n_assoc": n_assoc,
        "n_proc": n_proc,
        "warmup_steps": warmup,
        "wall_clock_s": perf_counter() - t0,
        "first_step_move_rail_80pct": first_rail("move"),
        "first_step_assoc_rail_80pct": first_rail("assoc"),
        "first_step_proc_rail_80pct": first_rail("proc"),
        "final_det_frac_rail": curves[-1]["det_frac_rail"] if curves else None,
        "final_probe": probes[-1] if probes else None,
        "curves": curves,
        "probes": probes,
        "det_snapshots": snapshots,
    }


def _lines(payload: dict) -> list[str]:
    lines = [
        "TD3 saturation diagnosis -- raw actions over training",
        "Same trainer as n=20 BEFORE baseline (frozen instance init, Fujimoto last layer).",
        "I=10 J=3, 8.8 MHz, 25% cap, 7000 steps, seeds 1-3. rail = |a| >= 0.95",
        "det = actor tanh output (no explore noise). exe = executed action.",
        "rail_push = mean(dQ/da * sign(a)); >0 means the critic pushes toward +/- 1.",
        "actor_std_across_batch ~ 0 with |a|~1 means a state-independent rail policy.",
        "",
    ]
    for rec in payload["seeds"]:
        fp = rec["final_probe"] or {}
        fr = rec["final_det_frac_rail"] or {}
        lines += [
            f"=== seed {rec['seed']} ===",
            (
                f"  first train step with det rail>=80%:  "
                f"move={rec['first_step_move_rail_80pct']}  "
                f"assoc={rec['first_step_assoc_rail_80pct']}  "
                f"proc={rec['first_step_proc_rail_80pct']}"
            ),
            (
                f"  final det frac rail  move={fr.get('move')}  "
                f"assoc={fr.get('assoc')}  proc={fr.get('proc')}"
            ),
            (
                f"  final Q(s,pi)={fp.get('q_pi')}  Q(s,0)={fp.get('q_0')}  "
                f"Q(s,sign(pi))={fp.get('q_sign_pi')}  Q(s,0.5 pi)={fp.get('q_half_pi')}"
            ),
            (
                f"  final rail_push={fp.get('rail_push')}  "
                f"pre_tanh_mean_abs={fp.get('pre_tanh_mean_abs')}  "
                f"pre_tanh_max_abs={fp.get('pre_tanh_max_abs')}  "
                f"last_linear_rms={fp.get('last_linear_rms')}  "
                f"actor_std_across_batch={fp.get('actor_std_across_batch')}"
            ),
            "",
        ]
    return lines


def _plot(payload: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_td3("matplotlib not installed; skip figures")
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    for rec in payload["seeds"]:
        steps = [c["step"] for c in rec["curves"] if c["phase"] == "train"]
        if not steps:
            continue
        label = f"seed {rec['seed']}"
        axes[0].plot(
            steps,
            [c["det_frac_rail"]["move"] for c in rec["curves"] if c["phase"] == "train"],
            label=label,
        )
        axes[1].plot(
            steps,
            [c["det_frac_rail"]["assoc"] for c in rec["curves"] if c["phase"] == "train"],
            label=label,
        )
        axes[2].plot(
            steps,
            [c["Mbps"] for c in rec["curves"] if c["phase"] == "train"],
            label=label,
        )
    axes[0].axhline(1.0, color="0.7", lw=0.6)
    axes[0].set_ylabel("det move |a|>=0.95")
    axes[1].set_ylabel("det assoc |a|>=0.95")
    axes[2].set_ylabel("inner Mbps")
    axes[2].set_xlabel("env step")
    axes[0].legend(frameon=False)
    axes[0].set_title("Deterministic actor rails and inner rate (train phase)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rails_and_rate.png", dpi=140)
    fig.savefig(FIG_DIR / "rails_and_rate.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    for rec in payload["seeds"]:
        steps = [p["step"] for p in rec["probes"] if "q_pi" in p]
        if not steps:
            continue
        label = f"seed {rec['seed']}"
        axes[0].plot(steps, [p["q_pi_minus_q_0"] for p in rec["probes"] if "q_pi" in p], label=label)
        axes[1].plot(steps, [p["rail_push"] for p in rec["probes"] if "q_pi" in p], label=label)
        axes[2].plot(steps, [p["pre_tanh_mean_abs"] for p in rec["probes"] if "q_pi" in p], label=label)
    axes[0].axhline(0.0, color="0.7", lw=0.6)
    axes[1].axhline(0.0, color="0.7", lw=0.6)
    axes[0].set_ylabel("Q(s,pi) - Q(s,0)")
    axes[1].set_ylabel("rail_push mean(dQ/da * sign a)")
    axes[2].set_ylabel("mean |pre-tanh|")
    axes[2].set_xlabel("env step")
    axes[0].legend(frameon=False)
    axes[0].set_title("Critic preference for the current actor vs zero action")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "critic_and_pretanh.png", dpi=140)
    fig.savefig(FIG_DIR / "critic_and_pretanh.pdf")
    plt.close(fig)
    log_td3(f"wrote figures in {FIG_DIR}")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = TD3Settings(device=device, log_every=0, export_mode="policy")
    log_td3(f"TD3 device            {device}  torch={torch.__version__}")
    payload = {
        "note": (
            "Saturation diagnosis for the n=20 BEFORE policy-export trainer. "
            "Not a new campaign. Not a claimed fix."
        ),
        "rail": RAIL,
        "device": device,
        "seeds": [],
    }
    for seed in SEEDS:
        payload["seeds"].append(train_seed(seed, settings))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = "\n".join(_lines(payload)) + "\n"
    TXT.write_text(text, encoding="utf-8")
    log_td3(text)
    _plot(payload)
    log_td3(f"wrote {OUT}")
    log_td3(f"wrote {TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
