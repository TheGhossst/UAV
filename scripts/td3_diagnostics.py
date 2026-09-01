"""TD3 Algorithm-2 fidelity diagnostics (observe-only; does not retune).

Re-runs train_td3 / solve_td3 with extra logs, a no-op k-means control, and
two distance-prior ablations. Config defaults are not changed.

    python -m scripts.td3_diagnostics --mode q1q2
    python -m scripts.td3_diagnostics --mode full --td3-steps 7000 --device auto
    python -m scripts.td3_diagnostics --mode alg2 --td3-steps 7000 --device auto
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DEFAULT,
    TD3_ASSOC_ACTION_SCALE,
    TD3_EPISODE_LEN,
    TD3_TOTAL_STEPS,
    TD3_WARMUP,
)
from src.logutil import configure_logging, log
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.td3 import UAVAoDTEnv, solve_td3

OUT_DEFAULT = Path("results") / "td3_diagnostics"
TASK1_SEEDS = tuple(range(100, 105))
TASK2_SEEDS = tuple(range(100, 110))
SCALE_X10 = 10.0 * TD3_ASSOC_ACTION_SCALE  # 2.5; default 0.25 is unchanged


def solve_td3_noop(scenario, seed: int, n_uav: int | None = None):
    """n_restarts=1, greedy_steps=0 equivalent: k-means + complete_solution, no actor.

    Separate path from ``solve_td3`` so the reported greedy eval is untouched.
    """
    env = UAVAoDTEnv(scenario, n_uav=n_uav, seed=seed)
    env.reset()
    assert env.last_result is not None
    return env.uav_xy.copy(), env.last_result


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _phase(t: int, total: int, warmup: int) -> str:
    if warmup > 0 and t < warmup:
        return "warmup"
    start = max(int(warmup), 0)
    span = max(total - start, 1)
    mid = start + span // 2
    return "early" if t < mid else "late"


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _train_rows(
    seed: int, train_log, variant: str, episode_len: int, warmup: int = TD3_WARMUP
) -> list[dict]:
    rows = []
    for t, (reward, rate, in_ep, kind) in enumerate(
        zip(
            train_log.rewards,
            train_log.sum_rates,
            train_log.in_episode_steps,
            train_log.start_kinds,
        )
    ):
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "t": t,
                "in_episode_step": int(in_ep),
                "episode": int(t // episode_len) if episode_len else t,
                "start_kind": kind,
                "phase": _phase(t, len(train_log.rewards), warmup),
                "reward": float(reward),
                "sum_rate": float(rate),
            }
        )
    return rows


def _best_prerollout(records: list[dict]) -> dict | None:
    pre = [r for r in records if r["source"] == "pre_rollout"]
    if not pre:
        return None
    # Same order as solve_td3: fewer violations, then higher rate.
    return min(pre, key=lambda r: (r["violation_count"], -r["sum_rate"]))


def _run_solve(
    scenario,
    seed: int,
    args,
    variant: str,
    assoc_action_scale: float | None = None,
    distance_prior: bool | None = None,
    fidelity_mode: bool = False,
    warmup: int | None = None,
) -> tuple[list[dict], dict, dict]:
    trace: dict = {}
    _xy, result, runtime, train_log = solve_td3(
        scenario,
        seed=seed,
        n_uav=args.n_uav,
        total_steps=args.td3_steps,
        device=args.device,
        eval_trace=trace,
        assoc_action_scale=assoc_action_scale,
        distance_prior=distance_prior,
        fidelity_mode=fidelity_mode,
        warmup=warmup,
    )
    assert train_log is not None
    episode_len = 0 if fidelity_mode else TD3_EPISODE_LEN
    wup = 0 if fidelity_mode else TD3_WARMUP
    if warmup is not None:
        wup = int(warmup)
    rows = _train_rows(seed, train_log, variant, episode_len, warmup=wup)
    winner = dict(trace["winner"])
    winner.update(
        {
            "seed": seed,
            "variant": variant,
            "td3_sum_rate": float(result.sum_rate),
            "td3_feasible": bool(result.feasible),
            "td3_qos": int(result.qos_violations),
            "td3_violations": int(result.violation_count),
            "runtime_s": float(runtime),
        }
    )
    best_pre = _best_prerollout(trace["records"])
    meta = {
        "seed": seed,
        "variant": variant,
        "winner": winner,
        "best_prerollout": best_pre,
        "n_restarts": trace["n_restarts"],
        "greedy_steps": trace["greedy_steps"],
        "records": trace["records"],
        "td3_sum_rate": float(result.sum_rate),
        "td3_feasible": bool(result.feasible),
        "td3_qos": int(result.qos_violations),
        "td3_violations": int(result.violation_count),
    }
    _free_cuda()
    return rows, winner, meta


def aggregate_within_episode(rows: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    buckets: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if int(r["seed"]) not in seeds:
            continue
        key = (r["phase"], r["start_kind"], int(r["in_episode_step"]))
        buckets[key].append((float(r["reward"]), float(r["sum_rate"])))
        key_all = ("all_post_warmup", r["start_kind"], int(r["in_episode_step"]))
        if r["phase"] != "warmup":
            buckets[key_all].append((float(r["reward"]), float(r["sum_rate"])))
        key_all2 = ("all", r["start_kind"], int(r["in_episode_step"]))
        buckets[key_all2].append((float(r["reward"]), float(r["sum_rate"])))
    out = []
    for (phase, kind, step), vals in sorted(buckets.items()):
        rewards = [v[0] for v in vals]
        rates = [v[1] for v in vals]
        out.append(
            {
                "phase": phase,
                "start_kind": kind,
                "in_episode_step": step,
                "n": len(vals),
                "mean_reward": statistics.fmean(rewards),
                "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
                "mean_sum_rate": statistics.fmean(rates),
            }
        )
    return out


def climb_table(agg: list[dict], last_step: int) -> list[dict]:
    by = {(r["phase"], r["start_kind"], int(r["in_episode_step"])): r for r in agg}
    phases = sorted({r["phase"] for r in agg})
    kinds = sorted({r["start_kind"] for r in agg})
    out = []
    for phase in phases:
        for kind in kinds:
            a = by.get((phase, kind, 0))
            b = by.get((phase, kind, last_step))
            if a is None or b is None:
                continue
            out.append(
                {
                    "phase": phase,
                    "start_kind": kind,
                    "reward_step0": a["mean_reward"],
                    "reward_step_last": b["mean_reward"],
                    "reward_climb": b["mean_reward"] - a["mean_reward"],
                    "sum_rate_step0": a["mean_sum_rate"],
                    "sum_rate_step_last": b["mean_sum_rate"],
                    "sum_rate_climb": b["mean_sum_rate"] - a["mean_sum_rate"],
                    "n_step0": a["n"],
                    "n_step_last": b["n"],
                }
            )
    return out


def episode_climbs(rows: list[dict], seeds: tuple[int, ...], last_step: int) -> list[dict]:
    groups: dict[tuple, dict[int, tuple[float, float]]] = defaultdict(dict)
    for r in rows:
        if int(r["seed"]) not in seeds:
            continue
        key = (int(r["seed"]), int(r["episode"]), r["start_kind"], r["phase"])
        groups[key][int(r["in_episode_step"])] = (float(r["reward"]), float(r["sum_rate"]))
    out = []
    for (seed, ep, kind, phase), by_step in sorted(groups.items()):
        if 0 not in by_step or last_step not in by_step:
            continue
        r0, s0 = by_step[0]
        r1, s1 = by_step[last_step]
        out.append(
            {
                "seed": seed,
                "episode": ep,
                "start_kind": kind,
                "phase": phase,
                "reward_climb": r1 - r0,
                "sum_rate_climb": s1 - s0,
                "reward_start": r0,
                "reward_end": r1,
            }
        )
    return out


def plot_within_episode(agg: list[dict], path: Path, ylabel: str, value_key: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    phases = ["warmup", "early", "late"]
    colors = {"warmup": "#888888", "early": "#1f77b4", "late": "#d62728"}
    for ax, kind in zip(axes, ("kmeans", "random")):
        for phase in phases:
            xs, ys = [], []
            for r in agg:
                if r["phase"] == phase and r["start_kind"] == kind:
                    xs.append(int(r["in_episode_step"]))
                    ys.append(float(r[value_key]))
            if not xs:
                continue
            order = np.argsort(xs)
            ax.plot(
                np.asarray(xs)[order],
                np.asarray(ys)[order],
                label=phase,
                color=colors[phase],
                lw=1.8,
            )
        ax.set_title(f"{kind}-start episodes")
        ax.set_xlabel("in-episode step (t % episode_len)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_episode_climbs(climbs: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    for kind, color in (("kmeans", "#1f77b4"), ("random", "#ff7f0e")):
        pts = [c for c in climbs if c["start_kind"] == kind]
        if not pts:
            continue
        ax.scatter(
            [c["episode"] for c in pts],
            [c["reward_climb"] for c in pts],
            s=12,
            alpha=0.45,
            label=kind,
            color=color,
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("episode index")
    ax.set_ylabel("reward climb (step last − step 0)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_reward_global(rows: list[dict], seeds: tuple[int, ...], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    by_t: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if int(r["seed"]) not in seeds:
            continue
        by_t[int(r["t"])].append(float(r["reward"]))
    ts = sorted(by_t)
    means = [statistics.fmean(by_t[t]) for t in ts]
    ax.plot(ts, means, lw=1.0, color="#1f77b4", label="mean reward")
    if len(means) >= 50:
        k = 50
        roll = np.convolve(means, np.ones(k) / k, mode="valid")
        ax.plot(ts[k - 1 :], roll, lw=1.6, color="#d62728", label=f"{k}-step rolling mean")
    ax.set_xlabel("global training step")
    ax.set_ylabel("mean reward (seeds)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def window_table(rows: list[dict], seeds: tuple[int, ...], total: int) -> list[dict]:
    """Equal-length early vs late windows of the continuous trajectory."""
    half = total // 2
    windows = [
        ("first_half", 0, half),
        ("second_half", half, total),
    ]
    if total >= 4000:
        windows.extend(
            [
                ("first_2000", 0, 2000),
                ("last_2000", total - 2000, total),
            ]
        )
    out = []
    for name, lo, hi in windows:
        rewards, rates = [], []
        for r in rows:
            if int(r["seed"]) not in seeds:
                continue
            t = int(r["t"])
            if lo <= t < hi:
                rewards.append(float(r["reward"]))
                rates.append(float(r["sum_rate"]))
        if not rewards:
            continue
        out.append(
            {
                "window": name,
                "t_lo": lo,
                "t_hi": hi,
                "n": len(rewards),
                "mean_reward": statistics.fmean(rewards),
                "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
                "mean_sum_rate": statistics.fmean(rates),
                "mean_mbps": statistics.fmean(rates) / 1e6,
            }
        )
    return out


def plot_binned_reward(rows: list[dict], seeds: tuple[int, ...], path: Path, bin_size: int = 50) -> None:
    """50-step bins of a continuous trajectory (not episode resets)."""
    bins: dict[int, list[float]] = defaultdict(list)
    rates: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if int(r["seed"]) not in seeds:
            continue
        b = int(r["t"]) // bin_size
        bins[b].append(float(r["reward"]))
        rates[b].append(float(r["sum_rate"]))
    if not bins:
        return
    xs = sorted(bins)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    axes[0].plot([x * bin_size for x in xs], [statistics.fmean(bins[x]) for x in xs], lw=1.6)
    axes[0].set_ylabel("mean reward")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        [x * bin_size for x in xs],
        [statistics.fmean(rates[x]) / 1e6 for x in xs],
        lw=1.6,
        color="#ff7f0e",
    )
    axes[1].set_ylabel("mean sum rate (Mbps)")
    axes[1].set_xlabel(f"training step (bin start, {bin_size} steps)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args) -> None:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=str(out / "run.log"), td3_log_every=args.td3_log_every)
    log.info(
        "td3 diagnostics  steps=%d  J=%d  task1=%s  task2=%s  task4=%s  device=%s",
        args.td3_steps,
        args.n_uav,
        args.seeds_task1,
        args.seeds_task2,
        args.seeds_task4,
        args.device,
    )

    train_rows: list[dict] = []
    winner_rows: list[dict] = []
    compare_rows: list[dict] = []
    traces: list[dict] = []

    for seed in args.seeds_task2:
        log.info("--- seed %d  variant=default ---", seed)
        scenario = generate_scenario(seed, DEFAULT)
        xy_noop, res_noop = solve_td3_noop(scenario, seed, n_uav=args.n_uav)
        _xy_k, res_k, _rt_k = solve_kmeans(scenario, seed=seed, n_uav=args.n_uav)
        rows, winner, meta = _run_solve(scenario, seed, args, "default")
        train_rows.extend(rows)
        traces.append(meta)
        best_pre = meta["best_prerollout"]
        winner_rows.append(
            {
                "seed": seed,
                "variant": "default",
                "winner_start_index": winner["start_index"],
                "winner_kind": winner["kind"],
                "winner_step": winner["step"],
                "winner_source": winner["source"],
                "winner_sum_rate": winner["sum_rate"],
                "td3_sum_rate": winner["td3_sum_rate"],
                "td3_mbps": winner["td3_sum_rate"] / 1e6,
                "td3_feasible": int(winner["td3_feasible"]),
                "td3_qos": winner["td3_qos"],
                "pre_kmeans_sum_rate": next(
                    r["sum_rate"]
                    for r in meta["records"]
                    if r["start_index"] == 0 and r["source"] == "pre_rollout"
                ),
                "best_prerollout_sum_rate": best_pre["sum_rate"] if best_pre else "",
                "best_prerollout_kind": best_pre["kind"] if best_pre else "",
                "runtime_s": winner["runtime_s"],
            }
        )
        compare_rows.append(
            {
                "seed": seed,
                "noop_mbps": res_noop.sum_rate / 1e6,
                "noop_feasible": int(res_noop.feasible),
                "noop_qos": int(res_noop.qos_violations),
                "kmeans_mbps": res_k.sum_rate / 1e6,
                "kmeans_feasible": int(res_k.feasible),
                "td3_mbps": winner["td3_sum_rate"] / 1e6,
                "td3_feasible": int(winner["td3_feasible"]),
                "td3_qos": int(winner["td3_qos"]),
                "best_prerollout_mbps": (best_pre["sum_rate"] / 1e6) if best_pre else "",
                "gap_td3_minus_noop_mbps": winner["td3_sum_rate"] / 1e6 - res_noop.sum_rate / 1e6,
                "gap_td3_minus_kmeans_mbps": winner["td3_sum_rate"] / 1e6 - res_k.sum_rate / 1e6,
                "gap_td3_minus_best_prerollout_mbps": (
                    winner["td3_sum_rate"] / 1e6 - best_pre["sum_rate"] / 1e6 if best_pre else ""
                ),
                "winner_source": winner["source"],
                "winner_kind": winner["kind"],
                "winner_start_index": winner["start_index"],
                "winner_step": winner["step"],
            }
        )

    _write_csv(out / "task1_train_steps.csv", train_rows)
    _write_csv(out / "task2_winners.csv", winner_rows)
    _write_csv(out / "task3_noop_vs_td3.csv", compare_rows)
    (out / "task2_eval_traces.jsonl").write_text(
        "\n".join(json.dumps(t, default=str) for t in traces) + "\n",
        encoding="utf-8",
    )

    last_step = max(0, TD3_EPISODE_LEN - 1)
    agg = aggregate_within_episode(train_rows, args.seeds_task1)
    climbs = climb_table(agg, last_step)
    per_ep = episode_climbs(train_rows, args.seeds_task1, last_step)
    _write_csv(out / "task1_within_episode.csv", agg)
    _write_csv(out / "task1_climb.csv", climbs)
    _write_csv(out / "task1_episode_climbs.csv", per_ep)
    plot_within_episode(
        agg, out / "task1_reward_vs_in_episode.png", "mean reward", "mean_reward"
    )
    plot_within_episode(
        agg,
        out / "task1_sumrate_vs_in_episode.png",
        "mean sum rate (bit/s)",
        "mean_sum_rate",
    )
    plot_episode_climbs(per_ep, out / "task1_reward_climb_vs_episode.png")
    plot_reward_global(train_rows, args.seeds_task1, out / "task1_reward_vs_global_step.png")

    # Task 4: scale ×10 and distance prior off. Same seeds as task 1.
    ablation_train: list[dict] = []
    ablation_eval: list[dict] = []
    variants = [
        ("scale_x10", SCALE_X10, True),
        ("no_prior", TD3_ASSOC_ACTION_SCALE, False),
    ]
    for vname, scale, prior in variants:
        for seed in args.seeds_task4:
            log.info("--- seed %d  variant=%s ---", seed, vname)
            scenario = generate_scenario(seed, DEFAULT)
            rows, winner, meta = _run_solve(
                scenario,
                seed,
                args,
                vname,
                assoc_action_scale=scale,
                distance_prior=prior,
            )
            ablation_train.extend(rows)
            default_cmp = next(r for r in compare_rows if r["seed"] == seed)
            ablation_eval.append(
                {
                    "seed": seed,
                    "variant": vname,
                    "assoc_action_scale": scale,
                    "distance_prior": int(prior),
                    "td3_mbps": winner["td3_sum_rate"] / 1e6,
                    "td3_feasible": int(winner["td3_feasible"]),
                    "td3_qos": int(winner["td3_qos"]),
                    "winner_source": winner["source"],
                    "winner_kind": winner["kind"],
                    "winner_step": winner["step"],
                    "default_td3_mbps": default_cmp["td3_mbps"],
                    "gap_vs_default_mbps": winner["td3_sum_rate"] / 1e6 - default_cmp["td3_mbps"],
                    "runtime_s": winner["runtime_s"],
                }
            )
            (out / f"task4_{vname}_seed{seed}_trace.json").write_text(
                json.dumps(meta, default=str, indent=2),
                encoding="utf-8",
            )

    _write_csv(out / "task4_train_steps.csv", ablation_train)
    _write_csv(out / "task4_eval.csv", ablation_eval)
    for vname, _, _ in variants:
        vrows = [r for r in ablation_train if r["variant"] == vname]
        vagg = aggregate_within_episode(vrows, args.seeds_task4)
        _write_csv(out / f"task4_{vname}_within_episode.csv", vagg)
        plot_within_episode(
            vagg,
            out / f"task4_{vname}_reward_vs_in_episode.png",
            "mean reward",
            "mean_reward",
        )
        plot_reward_global(
            vrows, args.seeds_task4, out / f"task4_{vname}_reward_vs_global_step.png"
        )

    summary = {
        "config": {
            "radio": "calibrated (DEFAULT)",
            "compute": False,
            "n_uav": args.n_uav,
            "td3_steps": args.td3_steps,
            "episode_len": TD3_EPISODE_LEN,
            "assoc_action_scale_default": TD3_ASSOC_ACTION_SCALE,
            "assoc_action_scale_x10": SCALE_X10,
            "warmup": TD3_WARMUP,
            "seeds_task1": list(args.seeds_task1),
            "seeds_task2": list(args.seeds_task2),
            "seeds_task4": list(args.seeds_task4),
        },
        "task2_winner_source_counts": _count(winner_rows, "winner_source"),
        "task2_winner_kind_counts": _count(winner_rows, "winner_kind"),
        "task2_winner_step0_count": sum(1 for r in winner_rows if int(r["winner_step"]) == 0),
        "task3_mean_noop_mbps": statistics.fmean(r["noop_mbps"] for r in compare_rows),
        "task3_mean_kmeans_mbps": statistics.fmean(r["kmeans_mbps"] for r in compare_rows),
        "task3_mean_td3_mbps": statistics.fmean(r["td3_mbps"] for r in compare_rows),
        "task3_mean_gap_td3_minus_noop_mbps": statistics.fmean(
            r["gap_td3_minus_noop_mbps"] for r in compare_rows
        ),
        "task1_climbs": climbs,
        "task4_mean_mbps": {
            vname: statistics.fmean(r["td3_mbps"] for r in ablation_eval if r["variant"] == vname)
            for vname, _, _ in variants
        },
        "task4_mean_gap_vs_default_mbps": {
            vname: statistics.fmean(
                r["gap_vs_default_mbps"] for r in ablation_eval if r["variant"] == vname
            )
            for vname, _, _ in variants
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)
    log.info(
        "task2 sources %s  step0_wins=%d/%d",
        summary["task2_winner_source_counts"],
        summary["task2_winner_step0_count"],
        len(winner_rows),
    )
    log.info(
        "task3 mean Mbps  noop=%.3f  kmeans=%.3f  td3=%.3f  gap(td3-noop)=%+.3f",
        summary["task3_mean_noop_mbps"],
        summary["task3_mean_kmeans_mbps"],
        summary["task3_mean_td3_mbps"],
        summary["task3_mean_gap_td3_minus_noop_mbps"],
    )


# Meaningful movement: 0.05 Mbps. Below this, a restart is FLAT.
RATE_EPS = 5e4


def classify_greedy_restart(steps: list[dict], rate_eps: float = RATE_EPS) -> dict:
    """Classify one restart's (step 0..N) sum_rate / violation_count trajectory.

    Labels match the Q1 vocabulary in the follow-up diagnostic task.
    """
    ordered = sorted(steps, key=lambda r: int(r["step"]))
    rates = [float(r["sum_rate"]) for r in ordered]
    viols = [int(r["violation_count"]) for r in ordered]
    r0, v0 = rates[0], viols[0]
    actor_rates = rates[1:]
    actor_viols = viols[1:]
    max_dev = max((abs(r - r0) for r in actor_rates), default=0.0)
    viol_changed = any(v != v0 for v in actor_viols)

    def _better_than_start(r: float, v: int) -> bool:
        if v != v0:
            return v < v0
        return r > r0 + rate_eps

    beat_idxs = [i for i, (r, v) in enumerate(zip(rates, viols)) if i > 0 and _better_than_start(r, v)]
    crossed_up = [i for i, r in enumerate(rates) if i > 0 and r > r0 + rate_eps]
    crossed_up_higher_viol = [i for i in crossed_up if viols[i] > v0]

    if max_dev < rate_eps and not viol_changed:
        shape = "FLAT"
    elif beat_idxs:
        shape = "BEATS_OWN_START"
    elif crossed_up and crossed_up_higher_viol and len(crossed_up_higher_viol) == len(crossed_up):
        shape = "IMPROVING_BUT_LOSES"
    elif actor_rates and min(actor_rates) < r0 - rate_eps:
        trough = min(actor_rates)
        t_i = actor_rates.index(trough)
        after = actor_rates[t_i:]
        recovered = max(after) - trough
        gap = r0 - trough
        if recovered > rate_eps and gap > 0 and (max(after) - trough) >= 0.25 * gap:
            shape = "DIP_PARTIAL_RECOVER"
        else:
            shape = "DIP_NO_RECOVER"
    elif actor_rates and max(actor_rates) > r0 + rate_eps:
        shape = "IMPROVING_BUT_LOSES"
    else:
        shape = "FLAT"

    first_drop = (actor_rates[0] - r0) if actor_rates else 0.0
    return {
        "shape": shape,
        "step0_sum_rate": r0,
        "step0_violations": v0,
        "step1_sum_rate": actor_rates[0] if actor_rates else r0,
        "final_sum_rate": rates[-1],
        "final_violations": viols[-1],
        "min_sum_rate": min(rates),
        "max_sum_rate": max(rates),
        "first_step_delta_mbps": first_drop / 1e6,
        "max_abs_dev_mbps": max_dev / 1e6,
        "beat_own_start": int(bool(beat_idxs)),
        "n_better_than_start": len(beat_idxs),
        "n_rate_above_start": len(crossed_up),
        "n_rate_above_start_higher_viol": len(crossed_up_higher_viol),
    }


def plot_seed_trajectories(seed: int, records: list[dict], path: Path) -> None:
    by_start: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_start[int(r["start_index"])].append(r)
    starts = sorted(by_start)
    n = len(starts)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 2.1 * max(n, 1)), sharex=True)
    if n == 1:
        axes = np.array([axes])
    for row, si in enumerate(starts):
        steps = sorted(by_start[si], key=lambda r: int(r["step"]))
        xs = [int(s["step"]) for s in steps]
        rates = [float(s["sum_rate"]) / 1e6 for s in steps]
        viols = [int(s["violation_count"]) for s in steps]
        kind = steps[0]["kind"]
        ax_r, ax_v = axes[row]
        ax_r.plot(xs, rates, color="#1f77b4", lw=1.6)
        ax_r.axhline(rates[0], color="#d62728", ls="--", lw=1.0, label="step-0")
        ax_r.set_ylabel("Mbps")
        ax_r.set_title(f"seed {seed}  restart {si} ({kind})")
        ax_r.grid(True, alpha=0.3)
        if row == 0:
            ax_r.legend(frameon=False, loc="best")
        ax_v.step(xs, viols, where="mid", color="#ff7f0e", lw=1.6)
        ax_v.axhline(viols[0], color="#d62728", ls="--", lw=1.0)
        ax_v.set_ylabel("violations")
        ax_v.set_ylim(-0.2, max(viols + [1]) + 0.4)
        ax_v.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("greedy step (0 = pre-rollout)")
    axes[-1, 1].set_xlabel("greedy step")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_q1q2(args) -> None:
    """Classify existing greedy traces (Q1) and re-measure k-means RNG gap (Q2).

    Does not retrain. Uses ``task2_eval_traces.jsonl`` from the prior pass.
    """
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=str(out / "q1q2.log"), td3_log_every=args.td3_log_every)
    traces_path = out / "task2_eval_traces.jsonl"
    if not traces_path.exists():
        raise SystemExit(f"missing {traces_path}; run without --mode q1q2 first")

    traces = []
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    traces = [t for t in traces if t.get("variant", "default") == "default"]
    traces = [t for t in traces if int(t["seed"]) in set(args.seeds_task2)]

    step_rows: list[dict] = []
    class_rows: list[dict] = []
    for t in traces:
        seed = int(t["seed"])
        recs = t["records"]
        plot_seed_trajectories(seed, recs, out / f"q1_seed{seed}_trajectories.png")
        by_start: dict[int, list[dict]] = defaultdict(list)
        for r in recs:
            by_start[int(r["start_index"])].append(r)
            step_rows.append(
                {
                    "seed": seed,
                    "restart_index": int(r["start_index"]),
                    "kind": r["kind"],
                    "step": int(r["step"]),
                    "source": r["source"],
                    "sum_rate": float(r["sum_rate"]),
                    "mbps": float(r["sum_rate"]) / 1e6,
                    "violation_count": int(r["violation_count"]),
                    "qos": int(r["qos"]),
                    "feasible": int(bool(r["feasible"])),
                }
            )
        for si, steps in sorted(by_start.items()):
            stats = classify_greedy_restart(steps)
            class_rows.append(
                {
                    "seed": seed,
                    "restart_index": si,
                    "kind": steps[0]["kind"],
                    **stats,
                    "step0_mbps": stats["step0_sum_rate"] / 1e6,
                    "final_mbps": stats["final_sum_rate"] / 1e6,
                    "min_mbps": stats["min_sum_rate"] / 1e6,
                    "max_mbps": stats["max_sum_rate"] / 1e6,
                }
            )

    _write_csv(out / "q1_greedy_steps.csv", step_rows)
    _write_csv(out / "q1_classifications.csv", class_rows)

    counts = _count(class_rows, "shape")
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in class_rows:
        by_kind[r["kind"]][r["shape"]] += 1
    log.info("Q1 shapes (all restarts): %s  n=%d", dict(counts), len(class_rows))
    for kind, c in sorted(by_kind.items()):
        log.info("  %s: %s", kind, dict(c))

    # Q2: standalone k-means vs pre-rollout k-means inside the trained TD3 env.
    rng_rows = []
    for t in traces:
        seed = int(t["seed"])
        scenario = generate_scenario(seed, DEFAULT)
        _xy_k, res_k, _ = solve_kmeans(scenario, seed=seed, n_uav=args.n_uav)
        xy_fresh, res_fresh = solve_td3_noop(scenario, seed, n_uav=args.n_uav)
        pre = next(
            r
            for r in t["records"]
            if int(r["start_index"]) == 0 and r["source"] == "pre_rollout"
        )
        rng_rows.append(
            {
                "seed": seed,
                "standalone_kmeans_mbps": res_k.sum_rate / 1e6,
                "fresh_env_reset_mbps": res_fresh.sum_rate / 1e6,
                "td3_eval_kmeans_mbps": float(pre["sum_rate"]) / 1e6,
                "same_draw_standalone_vs_fresh_env": int(
                    abs(res_k.sum_rate - res_fresh.sum_rate) < 1.0
                ),
                "delta_td3_kmeans_minus_standalone_mbps": float(pre["sum_rate"]) / 1e6
                - res_k.sum_rate / 1e6,
                "xy_match_standalone_vs_fresh": int(np.allclose(xy_fresh, _xy_k)),
            }
        )
    _write_csv(out / "q2_rng_audit.csv", rng_rows)
    deltas = [r["delta_td3_kmeans_minus_standalone_mbps"] for r in rng_rows]
    summary = {
        "q1_n_restarts": len(class_rows),
        "q1_shape_counts": dict(counts),
        "q1_shape_counts_by_kind": {k: dict(v) for k, v in by_kind.items()},
        "q1_kmeans_first_step_delta_mbps_mean": statistics.fmean(
            r["first_step_delta_mbps"] for r in class_rows if r["kind"] == "kmeans"
        ),
        "q1_random_first_step_delta_mbps_mean": statistics.fmean(
            r["first_step_delta_mbps"] for r in class_rows if r["kind"] == "random"
        ),
        "q2_n": len(rng_rows),
        "q2_mean_delta_mbps": statistics.fmean(deltas),
        "q2_mean_abs_delta_mbps": statistics.fmean(abs(d) for d in deltas),
        "q2_min_delta_mbps": min(deltas),
        "q2_max_delta_mbps": max(deltas),
        "q2_same_draw_count": sum(r["same_draw_standalone_vs_fresh_env"] for r in rng_rows),
    }
    (out / "q1q2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "Q2  mean delta (TD3-eval k-means − standalone)=%+.3f Mbps  |delta| mean=%.3f  range=[%+.3f, %+.3f]",
        summary["q2_mean_delta_mbps"],
        summary["q2_mean_abs_delta_mbps"],
        summary["q2_min_delta_mbps"],
        summary["q2_max_delta_mbps"],
    )
    log.info("wrote Q1/Q2 artifacts under %s", out)


def run_alg2(args) -> None:
    """Retrain+eval the Algorithm 2 fidelity variant (opt-in; default path unused)."""
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=str(out / "run.log"), td3_log_every=args.td3_log_every)
    log.info(
        "td3 alg2-fidelity  steps=%d  J=%d  seeds=%s  device=%s",
        args.td3_steps,
        args.n_uav,
        args.seeds_task2,
        args.device,
    )

    train_rows: list[dict] = []
    winner_rows: list[dict] = []
    compare_rows: list[dict] = []
    traces: list[dict] = []
    variant = "alg2_fidelity"

    for seed in args.seeds_task2:
        log.info("--- seed %d  variant=%s ---", seed, variant)
        scenario = generate_scenario(seed, DEFAULT)
        xy_noop, res_noop = solve_td3_noop(scenario, seed, n_uav=args.n_uav)
        _xy_k, res_k, _rt_k = solve_kmeans(scenario, seed=seed, n_uav=args.n_uav)
        rows, winner, meta = _run_solve(scenario, seed, args, variant, fidelity_mode=True)
        train_rows.extend(rows)
        traces.append(meta)
        best_pre = meta["best_prerollout"]
        winner_rows.append(
            {
                "seed": seed,
                "variant": variant,
                "winner_start_index": winner["start_index"],
                "winner_kind": winner["kind"],
                "winner_step": winner["step"],
                "winner_source": winner["source"],
                "winner_sum_rate": winner["sum_rate"],
                "td3_sum_rate": winner["td3_sum_rate"],
                "td3_mbps": winner["td3_sum_rate"] / 1e6,
                "td3_feasible": int(winner["td3_feasible"]),
                "td3_qos": winner["td3_qos"],
                "pre_kmeans_sum_rate": next(
                    r["sum_rate"]
                    for r in meta["records"]
                    if r["start_index"] == 0 and r["source"] == "pre_rollout"
                ),
                "best_prerollout_sum_rate": best_pre["sum_rate"] if best_pre else "",
                "best_prerollout_kind": best_pre["kind"] if best_pre else "",
                "runtime_s": winner["runtime_s"],
            }
        )
        compare_rows.append(
            {
                "seed": seed,
                "noop_mbps": res_noop.sum_rate / 1e6,
                "noop_feasible": int(res_noop.feasible),
                "noop_qos": int(res_noop.qos_violations),
                "kmeans_mbps": res_k.sum_rate / 1e6,
                "kmeans_feasible": int(res_k.feasible),
                "td3_mbps": winner["td3_sum_rate"] / 1e6,
                "td3_feasible": int(winner["td3_feasible"]),
                "td3_qos": winner["td3_qos"],
                "best_prerollout_mbps": (best_pre["sum_rate"] / 1e6) if best_pre else "",
                "gap_td3_minus_noop_mbps": winner["td3_sum_rate"] / 1e6 - res_noop.sum_rate / 1e6,
                "gap_td3_minus_kmeans_mbps": winner["td3_sum_rate"] / 1e6 - res_k.sum_rate / 1e6,
                "gap_td3_minus_best_prerollout_mbps": (
                    winner["td3_sum_rate"] / 1e6 - best_pre["sum_rate"] / 1e6 if best_pre else ""
                ),
                "winner_source": winner["source"],
                "winner_kind": winner["kind"],
                "winner_start_index": winner["start_index"],
                "winner_step": winner["step"],
            }
        )
        (out / f"eval_trace_seed{seed}.json").write_text(
            json.dumps(meta, default=str, indent=2), encoding="utf-8"
        )

    _write_csv(out / "task1_train_steps.csv", train_rows)
    _write_csv(out / "task2_winners.csv", winner_rows)
    _write_csv(out / "task3_noop_vs_td3.csv", compare_rows)
    (out / "task2_eval_traces.jsonl").write_text(
        "\n".join(json.dumps(t, default=str) for t in traces) + "\n",
        encoding="utf-8",
    )

    windows = window_table(train_rows, args.seeds_task1, args.td3_steps)
    _write_csv(out / "task1_windows.csv", windows)
    plot_reward_global(train_rows, args.seeds_task1, out / "task1_reward_vs_global_step.png")
    plot_binned_reward(train_rows, args.seeds_task1, out / "task1_reward_vs_50step_bin.png")
    plot_reward_global(train_rows, args.seeds_task2, out / "task1_reward_vs_global_step_all10.png")

    # Q1 classification on this variant's greedy traces.
    step_rows: list[dict] = []
    class_rows: list[dict] = []
    for t in traces:
        seed = int(t["seed"])
        recs = t["records"]
        plot_seed_trajectories(seed, recs, out / f"q1_seed{seed}_trajectories.png")
        by_start: dict[int, list[dict]] = defaultdict(list)
        for r in recs:
            by_start[int(r["start_index"])].append(r)
            step_rows.append(
                {
                    "seed": seed,
                    "restart_index": int(r["start_index"]),
                    "kind": r["kind"],
                    "step": int(r["step"]),
                    "source": r["source"],
                    "sum_rate": float(r["sum_rate"]),
                    "mbps": float(r["sum_rate"]) / 1e6,
                    "violation_count": int(r["violation_count"]),
                    "qos": int(r["qos"]),
                    "feasible": int(bool(r["feasible"])),
                }
            )
        for si, steps in sorted(by_start.items()):
            stats = classify_greedy_restart(steps)
            class_rows.append(
                {
                    "seed": seed,
                    "restart_index": si,
                    "kind": steps[0]["kind"],
                    **stats,
                    "step0_mbps": stats["step0_sum_rate"] / 1e6,
                    "final_mbps": stats["final_sum_rate"] / 1e6,
                    "min_mbps": stats["min_sum_rate"] / 1e6,
                    "max_mbps": stats["max_sum_rate"] / 1e6,
                }
            )
    _write_csv(out / "q1_greedy_steps.csv", step_rows)
    _write_csv(out / "q1_classifications.csv", class_rows)

    counts = _count(class_rows, "shape")
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in class_rows:
        by_kind[r["kind"]][r["shape"]] += 1
    km_rows = [r for r in class_rows if r["kind"] == "kmeans"]
    rnd_rows = [r for r in class_rows if r["kind"] == "random"]

    actor_wins = sum(1 for r in winner_rows if r["winner_source"] == "actor")
    preroll_wins = sum(1 for r in winner_rows if r["winner_source"] == "pre_rollout")
    kmeans_start_wins = sum(
        1 for r in winner_rows if r["winner_kind"] == "kmeans" and int(r["winner_step"]) == 0
    )

    half = next((w for w in windows if w["window"] == "first_half"), None)
    late = next((w for w in windows if w["window"] == "second_half"), None)
    reward_delta = (
        (late["mean_reward"] - half["mean_reward"]) if half and late else None
    )
    rate_delta = (
        (late["mean_sum_rate"] - half["mean_sum_rate"]) if half and late else None
    )

    summary = {
        "config": {
            "radio": "calibrated (DEFAULT)",
            "compute": False,
            "n_uav": args.n_uav,
            "td3_steps": args.td3_steps,
            "fidelity_mode": True,
            "episode_len": 0,
            "distance_prior": False,
            "assoc_action_scale": 1.0,
            "warmup": 0,
            "track_a_eval_kmeans_fixed": True,
            "seeds_task1": list(args.seeds_task1),
            "seeds_task2": list(args.seeds_task2),
        },
        "task1_windows": windows,
        "task1_late_minus_early_mean_reward": reward_delta,
        "task1_late_minus_early_mean_sum_rate": rate_delta,
        "task2_winner_source_counts": _count(winner_rows, "winner_source"),
        "task2_winner_kind_counts": _count(winner_rows, "winner_kind"),
        "task2_actor_wins": actor_wins,
        "task2_prerollout_wins": preroll_wins,
        "task2_kmeans_step0_wins": kmeans_start_wins,
        "task2_n_seeds": len(winner_rows),
        "task3_mean_noop_mbps": statistics.fmean(r["noop_mbps"] for r in compare_rows),
        "task3_mean_kmeans_mbps": statistics.fmean(r["kmeans_mbps"] for r in compare_rows),
        "task3_mean_td3_mbps": statistics.fmean(r["td3_mbps"] for r in compare_rows),
        "task3_mean_gap_td3_minus_noop_mbps": statistics.fmean(
            r["gap_td3_minus_noop_mbps"] for r in compare_rows
        ),
        "task3_mean_gap_td3_minus_kmeans_mbps": statistics.fmean(
            r["gap_td3_minus_kmeans_mbps"] for r in compare_rows
        ),
        "q1_n_restarts": len(class_rows),
        "q1_shape_counts": dict(counts),
        "q1_shape_counts_by_kind": {k: dict(v) for k, v in by_kind.items()},
        "q1_kmeans_first_step_delta_mbps_mean": (
            statistics.fmean(r["first_step_delta_mbps"] for r in km_rows) if km_rows else None
        ),
        "q1_kmeans_n_beats_own_start": sum(int(r["beat_own_start"]) for r in km_rows),
        "q1_kmeans_n": len(km_rows),
        "q1_random_first_step_delta_mbps_mean": (
            statistics.fmean(r["first_step_delta_mbps"] for r in rnd_rows) if rnd_rows else None
        ),
        "q1_random_n_beats_own_start": sum(int(r["beat_own_start"]) for r in rnd_rows),
        "q1_random_n": len(rnd_rows),
        "prior_q1_kmeans_first_step_delta_mbps_mean": -1.86,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)
    log.info(
        "task2 sources %s  actor_wins=%d/%d  kmeans_step0=%d/%d",
        summary["task2_winner_source_counts"],
        actor_wins,
        len(winner_rows),
        kmeans_start_wins,
        len(winner_rows),
    )
    log.info(
        "task3 mean Mbps  noop=%.3f  kmeans=%.3f  td3=%.3f  gap(td3-kmeans)=%+.3f",
        summary["task3_mean_noop_mbps"],
        summary["task3_mean_kmeans_mbps"],
        summary["task3_mean_td3_mbps"],
        summary["task3_mean_gap_td3_minus_kmeans_mbps"],
    )
    log.info(
        "task1 late-early reward=%s  Q1 kmeans first-step mean=%+.3f Mbps (prior was -1.86)  shapes=%s",
        f"{reward_delta:+.3f}" if reward_delta is not None else "n/a",
        summary["q1_kmeans_first_step_delta_mbps_mean"] or 0.0,
        dict(counts),
    )


def _count(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[str(r[key])] += 1
    return dict(out)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-uav", type=int, default=DEFAULT.num_uav)
    p.add_argument("--td3-steps", type=int, default=TD3_TOTAL_STEPS)
    p.add_argument("--td3-log-every", type=int, default=500)
    p.add_argument(
        "--mode",
        choices=("full", "q1q2", "alg2"),
        default="q1q2",
        help="full = default-path retrain+eval. q1q2 = classify existing traces. "
        "alg2 = Algorithm 2 fidelity variant retrain+eval (writes a subfolder).",
    )
    p.add_argument("--seeds-task1", type=int, nargs="*", default=list(TASK1_SEEDS))
    p.add_argument("--seeds-task2", type=int, nargs="*", default=list(TASK2_SEEDS))
    p.add_argument("--seeds-task4", type=int, nargs="*", default=list(TASK1_SEEDS))
    args = p.parse_args()
    args.seeds_task1 = tuple(args.seeds_task1)
    args.seeds_task2 = tuple(args.seeds_task2)
    args.seeds_task4 = tuple(args.seeds_task4)
    return args


if __name__ == "__main__":
    args = _parse()
    if args.mode == "alg2" and args.out == OUT_DEFAULT:
        args.out = OUT_DEFAULT / "alg2_fidelity"
    if args.mode == "q1q2":
        run_q1q2(args)
    elif args.mode == "alg2":
        run_alg2(args)
    else:
        run(args)
