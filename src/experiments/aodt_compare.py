"""AoDT-on comparison: Random, K-means, joint PSO, SCA, TD3. No new algorithm."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import PAPER_SCENARIO_SEEDS, SimConfig
from src.logutil import Counter, banner, log
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.pso import solve_pso_joint
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca
from src.solvers.td3 import solve_td3, train_td3_across_scenarios


METHODS = ("random", "kmeans", "pso", "sca", "td3")
TRAIN_SEEDS = tuple(range(200, 220))  # disjoint from eval seeds 100-119


def _solve(name, scenario, seed, args, n_uav=None, td3_agent=None):
    if name == "random":
        return solve_random(scenario, seed=seed, n_uav=n_uav)
    if name == "kmeans":
        return solve_kmeans(scenario, seed=seed, n_uav=n_uav)
    if name == "pso":
        xy, result, rt, _ = solve_pso_joint(
            scenario, seed=seed, n_uav=n_uav, n_particles=args.particles, n_iter=args.iters
        )
        return xy, result, rt
    if name == "sca":
        return solve_sca(scenario, seed=seed, n_uav=n_uav)
    if name == "td3":
        xy, result, rt, _ = solve_td3(
            scenario,
            seed=seed,
            n_uav=n_uav,
            total_steps=args.td3_steps,
            agent=td3_agent,
        )
        return xy, result, rt
    raise ValueError(name)


def encode_uav_xy(xy: np.ndarray, height: float) -> str:
    arr = np.asarray(xy, dtype=float).reshape(-1, 2)
    return json.dumps([[float(x), float(y), float(height)] for x, y in arr])


def expand_positions(metric_rows: list[dict]) -> list[dict]:
    """One row per UAV: seed, method, uav_id, x, y, z."""
    out = []
    for r in metric_rows:
        coords = json.loads(r["uav_xy"])
        for j, xyz in enumerate(coords):
            out.append(
                {
                    "seed": r["seed"],
                    "method": r["method"],
                    "uav_id": j,
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                    "n_uav": r.get("n_uav"),
                    "n_iot": r.get("n_iot"),
                    "feasible": r.get("feasible"),
                    "sum_rate": r.get("sum_rate"),
                }
            )
    return out


def _row(seed, method, result, rt, uav_xy, height: float, **extra):
    aodt_mean = float(np.nanmean(result.aodt)) if result.compute_available else None
    aodt_max = float(np.nanmax(result.aodt)) if result.compute_available else None
    return {
        "seed": seed,
        "method": method,
        "sum_rate": result.sum_rate,
        "feasible_sum_rate": result.sum_rate if result.feasible else None,
        "qos": result.qos_violations,
        "aodt_violations": result.aodt_violations,
        "cpu_unstable": result.cpu_unstable,
        "feasible": int(result.feasible),
        "runtime": rt,
        "aodt_mean": aodt_mean,
        "aodt_max": aodt_max,
        "min_rate": result.min_assoc_rate,
        "uav_xy": encode_uav_xy(uav_xy, height),
        **extra,
    }


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict], group_key: str | None = None) -> list[dict]:
    if group_key is None:
        keys = {(r["method"], None) for r in rows}
    else:
        keys = {(r["method"], r[group_key]) for r in rows}
    out = []
    for method, g in sorted(keys, key=lambda t: (t[0], t[1] if t[1] is not None else 0)):
        subset = [
            r
            for r in rows
            if r["method"] == method and (group_key is None or r[group_key] == g)
        ]
        rates = np.array([r["sum_rate"] for r in subset], float)
        feas = np.array([r["feasible"] for r in subset], float)
        feas_rates = np.array(
            [r["sum_rate"] for r in subset if r["feasible"]], dtype=float
        )
        qos = np.array([r["qos"] for r in subset], float)
        aodt_v = np.array([r["aodt_violations"] for r in subset], float)
        rts = np.array([r["runtime"] for r in subset], float)
        aodt = [r["aodt_mean"] for r in subset if r.get("aodt_mean") is not None]
        rec = {
            "method": method,
            "n": len(subset),
            "sum_rate_mean": float(rates.mean()),
            "sum_rate_std": float(rates.std(ddof=1) if rates.size > 1 else 0.0),
            "feasible_frac": float(feas.mean()),
            "feasible_n": int(feas.sum()),
            "feasible_sum_rate_mean": float(feas_rates.mean()) if feas_rates.size else None,
            "feasible_sum_rate_std": float(feas_rates.std(ddof=1)) if feas_rates.size > 1 else None,
            "qos_mean": float(qos.mean()),
            "aodt_viol_mean": float(aodt_v.mean()),
            "aodt_mean": float(np.mean(aodt)) if aodt else None,
            "runtime_mean": float(rts.mean()),
        }
        if group_key is not None:
            rec[group_key] = g
        out.append(rec)
    return out


def write_markdown_table(path: Path, rows: list[dict], title: str):
    lines = [
        f"# {title}",
        "",
        "AoDT enabled with experimental `S_i=2000` bytes and `L=2e6` cycles (not Table II).",
        "PSO is the joint encoder (positions + association + processing + bandwidth).",
        "TD3 is one policy trained on seeds 200–219, then greedy-evaluated on the listed scenario seeds.",
        "",
        "| Method | Raw sum rate | Std | Feasible frac | Feasible sum rate | AoDT mean | QoS | AoDT viol | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        feas_r = r["feasible_sum_rate_mean"]
        feas_s = "n/a" if feas_r is None else f"{feas_r:.1f}"
        aodt = "n/a" if r["aodt_mean"] is None else f"{r['aodt_mean']:.3f}"
        extra = ""
        if "n_uav" in r:
            extra = f" (J={r['n_uav']})"
        if "n_iot" in r:
            extra = f" (I={r['n_iot']})"
        lines.append(
            f"| {r['method']}{extra} | {r['sum_rate_mean']:.1f} | {r['sum_rate_std']:.1f} | "
            f"{r['feasible_frac']:.2f} | {feas_s} | {aodt} | {r['qos_mean']:.2f} | "
            f"{r['aodt_viol_mean']:.2f} | {r['runtime_mean']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, summary: list[dict], xkey: str, xlabel: str, ykey: str, ylabel: str, png: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    methods = sorted({r["method"] for r in summary})
    for m in methods:
        sub = [r for r in summary if r["method"] == m]
        xs, ys, es = [], [], []
        for r in sub:
            val = r.get(ykey)
            if val is None:
                continue
            xs.append(float(r[xkey]))
            ys.append(float(val) / (1e6 if "rate" in ykey else 1.0))
            if "sum_rate" in ykey:
                std = r.get("feasible_sum_rate_std") if ykey.startswith("feasible") else r.get("sum_rate_std")
                es.append((float(std) if std else 0.0) / 1e6)
            else:
                es.append(0.0)
        if not xs:
            continue
        order = np.argsort(xs)
        xs, ys, es = np.array(xs)[order], np.array(ys)[order], np.array(es)[order]
        if np.any(es):
            ax.errorbar(xs, ys, yerr=es, marker="o", label=m)
        else:
            ax.plot(xs, ys, marker="o", label=m)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / png, dpi=120)
    plt.close(fig)


def _get_td3_agent(cfg: SimConfig, n_uav: int, args, cache: dict):
    key = (cfg.num_iot, n_uav, cfg.aodt_threshold)
    if key in cache:
        return cache[key]
    log.info("Training TD3 pool  I=%d J=%d steps=%d seeds=%s", cfg.num_iot, n_uav, args.td3_steps, f"{TRAIN_SEEDS[0]}–{TRAIN_SEEDS[-1]}")
    agent, _env, train_log = train_td3_across_scenarios(
        cfg,
        TRAIN_SEEDS,
        seed=0,
        n_uav=n_uav,
        total_steps=args.td3_steps,
    )
    out_dir = Path(getattr(args, "out", "results")) / "aodt"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / f"td3_train_I{cfg.num_iot}_J{n_uav}.csv",
        [{"step": i, "reward": r, "sum_rate": s} for i, (r, s) in enumerate(zip(train_log.rewards, train_log.sum_rates))],
    )
    cache[key] = agent
    return agent


def run_fixed(cfg: SimConfig, args, seeds: tuple[int, ...], n_uav: int, cache: dict) -> list[dict]:
    cfg_j = replace(cfg, num_uav=n_uav)
    agent = _get_td3_agent(cfg_j, n_uav, args, cache)
    rows = []
    jobs = Counter(f"AoDT  I={cfg.num_iot} J={n_uav}", len(seeds) * len(METHODS))
    for seed in seeds:
        scenario = generate_scenario(seed, cfg_j)
        for name in METHODS:
            _xy, result, rt = _solve(
                name, scenario, seed, args, n_uav=n_uav, td3_agent=agent if name == "td3" else None
            )
            rec = _row(
                seed, name, result, rt, _xy, cfg_j.uav_height, n_uav=n_uav, n_iot=cfg.num_iot
            )
            rows.append(rec)
            jobs.tick(f"seed={seed}  {name:8}", result, rt)
    return rows


def run_aodt_comparison(cfg: SimConfig, args):
    out = Path(getattr(args, "out", "results")) / "aodt"
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg.with_compute()
    if getattr(args, "aodt_short", False):
        seeds = tuple(range(100, 105))
    else:
        seeds = PAPER_SCENARIO_SEEDS

    banner("AoDT comparison")
    cache: dict = {}
    default_rows = run_fixed(cfg, args, seeds, n_uav=cfg.num_uav, cache=cache)
    _write_csv(out / "raw_default.csv", default_rows)
    _write_csv(out / "positions_default.csv", expand_positions(default_rows))
    default_sum = summarize(default_rows)
    _write_csv(out / "summary_default.csv", default_sum)
    write_markdown_table(out / "table_default.md", default_sum, "Default I=10, J=3, AoDT on")

    j_rows = []
    for j in (1, 2, 3, 4, 5):
        j_rows.extend(run_fixed(cfg, args, seeds, n_uav=j, cache=cache))
    _write_csv(out / "raw_vs_uav.csv", j_rows)
    _write_csv(out / "positions_vs_uav.csv", expand_positions(j_rows))
    j_sum = summarize(j_rows, "n_uav")
    _write_csv(out / "summary_vs_uav.csv", j_sum)
    write_markdown_table(out / "table_vs_uav.md", j_sum, "Scale with number of UAVs J")
    _plot(out, j_sum, "n_uav", "Number of UAVs J", "sum_rate_mean", "Raw sum rate (Mbps)", "fig_raw_vs_uav.png")
    _plot(
        out,
        j_sum,
        "n_uav",
        "Number of UAVs J",
        "feasible_sum_rate_mean",
        "Feasible sum rate (Mbps)",
        "fig_feasible_vs_uav.png",
    )
    _plot(out, j_sum, "n_uav", "Number of UAVs J", "feasible_frac", "Feasible fraction", "fig_feasfrac_vs_uav.png")

    i_rows = []
    iots = (10, 16, 20, 24, 32)
    jobs = Counter("AoDT  vs I  (J=3)", len(iots) * len(seeds) * len(METHODS))
    for n in iots:
        k = cfg.num_processes
        per = n // k
        cfg_i = replace(cfg, num_iot=n, iots_per_process=per, num_uav=3)
        agent = _get_td3_agent(cfg_i, 3, args, cache)
        for seed in seeds:
            scenario = generate_scenario(seed, cfg_i)
            for name in METHODS:
                _xy, result, rt = _solve(
                    name, scenario, seed, args, n_uav=3, td3_agent=agent if name == "td3" else None
                )
                i_rows.append(
                    _row(seed, name, result, rt, _xy, cfg_i.uav_height, n_uav=3, n_iot=n)
                )
                jobs.tick(f"I={n}  seed={seed}  {name:8}", result, rt)
    _write_csv(out / "raw_vs_iot.csv", i_rows)
    _write_csv(out / "positions_vs_iot.csv", expand_positions(i_rows))
    i_sum = summarize(i_rows, "n_iot")
    _write_csv(out / "summary_vs_iot.csv", i_sum)
    write_markdown_table(out / "table_vs_iot.md", i_sum, "Scale with number of IoTs I (J=3)")
    _plot(out, i_sum, "n_iot", "Number of IoTs I", "sum_rate_mean", "Raw sum rate (Mbps)", "fig_raw_vs_iot.png")
    _plot(
        out,
        i_sum,
        "n_iot",
        "Number of IoTs I",
        "feasible_sum_rate_mean",
        "Feasible sum rate (Mbps)",
        "fig_feasible_vs_iot.png",
    )
    _plot(out, i_sum, "n_iot", "Number of IoTs I", "feasible_frac", "Feasible fraction", "fig_feasfrac_vs_iot.png")

    meta = {
        "task_size_bytes": cfg.task_size_bytes,
        "task_cycles": cfg.task_cycles,
        "aodt_threshold": cfg.aodt_threshold,
        "n_eval_seeds": len(seeds),
        "td3_train_seeds": list(TRAIN_SEEDS),
        "td3_steps": args.td3_steps,
        "pso": "joint",
        "positions": "positions_default.csv / positions_vs_uav.csv / positions_vs_iot.csv (one row per UAV)",
        "note": "S_i and L are experimental (not Table II). Proposed method not included.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote AoDT comparison under %s", out)
