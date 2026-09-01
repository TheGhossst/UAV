"""Paper-style sweeps (Figs. 6–10) using the shared evaluator."""

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
from src.solvers.pso import solve_pso_placement
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca
from src.solvers.td3 import solve_td3


def _methods(args):
    names = ["random", "kmeans", "pso", "sca"]
    if getattr(args, "with_td3", False):
        names.append("td3")
    return names


def _solve(name, scenario, seed, args, n_uav=None, td3_agent=None):
    if name == "random":
        return solve_random(scenario, seed=seed, n_uav=n_uav)
    if name == "kmeans":
        return solve_kmeans(scenario, seed=seed, n_uav=n_uav)
    if name == "pso":
        xy, result, rt, _ = solve_pso_placement(
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


def _seeds(args) -> tuple[int, ...]:
    if getattr(args, "paper_runs", False):
        return PAPER_SCENARIO_SEEDS
    return tuple(range(100, 105))


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summarize(rows: list[dict], group_key: str) -> list[dict]:
    out = []
    keys = {(r["method"], r[group_key]) for r in rows}
    for method, g in sorted(keys, key=lambda t: (t[0], t[1])):
        subset = [r for r in rows if r["method"] == method and r[group_key] == g]
        rates = np.array([r["sum_rate"] for r in subset], dtype=float)
        rts = np.array([r["runtime"] for r in subset], dtype=float)
        qos = np.array([r["qos"] for r in subset], dtype=float)
        feas = np.array([r["feasible"] for r in subset], dtype=float)
        aodt_viol = np.array([r.get("aodt_viol", 0) for r in subset], dtype=float)
        cpu = np.array([r.get("cpu_unstable", 0) for r in subset], dtype=float)
        aodt = [r["aodt_mean"] for r in subset if r["aodt_mean"] is not None]
        feasible_rates = np.array(
            [r["sum_rate"] for r in subset if r["feasible"]], dtype=float
        )
        out.append(
            {
                "method": method,
                group_key: g,
                "sum_rate_mean": float(rates.mean()),
                "sum_rate_std": float(rates.std(ddof=1) if rates.size > 1 else 0.0),
                "feasible_n": int(feas.sum()),
                "feasible_frac": float(feas.mean()),
                "feasible_sum_rate_mean": float(feasible_rates.mean()) if feasible_rates.size else float("nan"),
                "qos_mean": float(qos.mean()),
                "aodt_viol_mean": float(aodt_viol.mean()),
                "cpu_unstable_mean": float(cpu.mean()),
                "runtime_mean": float(rts.mean()),
                "aodt_mean": float(np.mean(aodt)) if aodt else None,
                "n": len(subset),
            }
        )
    return out


def encode_uav_xy(xy, height: float) -> str:
    arr = np.asarray(xy, dtype=float).reshape(-1, 2)
    return json.dumps([[float(x), float(y), float(height)] for x, y in arr])


def expand_positions(metric_rows: list[dict]) -> list[dict]:
    out = []
    for r in metric_rows:
        coords = json.loads(r["uav_xy"])
        for j, xyz in enumerate(coords):
            rec = {
                "seed": r["seed"],
                "method": r["method"],
                "uav_id": j,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "sum_rate": r.get("sum_rate"),
                "feasible": r.get("feasible"),
            }
            for k in ("n_uav", "n_iot", "setting"):
                if k in r:
                    rec[k] = r[k]
            out.append(rec)
    return out


def _row(seed, method, result, rt, uav_xy, height: float, **extra):
    return {
        "seed": seed,
        "method": method,
        "sum_rate": result.sum_rate,
        "qos": result.qos_violations,
        "feasible": int(result.feasible),
        "aodt_viol": result.aodt_violations,
        "aodt_excess": result.aodt_excess,
        "cpu_unstable": result.cpu_unstable,
        "runtime": rt,
        "aodt_mean": float(np.nanmean(result.aodt)) if result.compute_available else None,
        "uav_xy": encode_uav_xy(uav_xy, height),
        **extra,
    }


def sweep_num_uav(cfg: SimConfig, args, out_dir: Path) -> list[dict]:
    rows = []
    js = (1, 2, 3, 4, 5)
    seeds = _seeds(args)
    methods = _methods(args)
    jobs = Counter("Fig.6  sum rate vs J", len(js) * len(seeds) * len(methods))
    for j in js:
        cfg_j = replace(cfg, num_uav=j)
        for seed in seeds:
            scenario = generate_scenario(seed, cfg_j)
            for name in methods:
                _xy, result, rt = _solve(name, scenario, seed, args, n_uav=j)
                rows.append(_row(seed, name, result, rt, _xy, cfg_j.uav_height, n_uav=j))
                jobs.tick(f"J={j}  seed={seed}  {name:8}", result, rt)
    _write_csv(out_dir / "raw_sumrate_vs_uav.csv", rows)
    _write_csv(out_dir / "positions_sumrate_vs_uav.csv", expand_positions(rows))
    _write_csv(out_dir / "sumrate_vs_uav.csv", _summarize(rows, "n_uav"))
    return rows


def sweep_num_iot(cfg: SimConfig, args, out_dir: Path) -> list[dict]:
    rows = []
    ns = (10, 16, 20, 24, 32)
    seeds = _seeds(args)
    methods = _methods(args)
    jobs = Counter("Fig.7  sum rate vs I  (J=3)", len(ns) * len(seeds) * len(methods))
    for n in ns:
        k = cfg.num_processes
        per = n // k
        cfg_i = replace(cfg, num_iot=n, iots_per_process=per, num_uav=3)
        for seed in seeds:
            scenario = generate_scenario(seed, cfg_i)
            for name in methods:
                _xy, result, rt = _solve(name, scenario, seed, args, n_uav=3)
                rows.append(_row(seed, name, result, rt, _xy, cfg_i.uav_height, n_iot=n))
                jobs.tick(f"I={n}  seed={seed}  {name:8}", result, rt)
    _write_csv(out_dir / "raw_sumrate_vs_iot.csv", rows)
    _write_csv(out_dir / "positions_sumrate_vs_iot.csv", expand_positions(rows))
    _write_csv(out_dir / "sumrate_vs_iot.csv", _summarize(rows, "n_iot"))
    return rows


def sweep_lambda(cfg: SimConfig, args, out_dir: Path) -> list[dict]:
    rows = []
    lams = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    seeds = _seeds(args)
    methods = _methods(args)
    jobs = Counter("Fig.8  sum rate vs λ", len(lams) * len(seeds) * len(methods))
    for lam in lams:
        cfg_l = replace(cfg, lambda_i=lam)
        for seed in seeds:
            scenario = generate_scenario(seed, cfg_l)
            for name in methods:
                _xy, result, rt = _solve(name, scenario, seed, args)
                rows.append(_row(seed, name, result, rt, _xy, cfg_l.uav_height, lambda_i=lam))
                jobs.tick(f"lam={lam:g}  seed={seed}  {name:8}", result, rt)
    _write_csv(out_dir / "raw_sumrate_vs_lambda.csv", rows)
    _write_csv(out_dir / "positions_sumrate_vs_lambda.csv", expand_positions(rows))
    _write_csv(out_dir / "sumrate_vs_lambda.csv", _summarize(rows, "lambda_i"))
    return rows


def sweep_aodt(cfg: SimConfig, args, out_dir: Path) -> list[dict]:
    rows = []
    tks = (0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0)
    seeds = _seeds(args)
    methods = _methods(args)
    jobs = Counter("Fig.9  sum rate vs T_k", len(tks) * len(seeds) * len(methods))
    for tk in tks:
        cfg_t = replace(cfg, aodt_threshold=tk)
        for seed in seeds:
            scenario = generate_scenario(seed, cfg_t)
            for name in methods:
                _xy, result, rt = _solve(name, scenario, seed, args)
                rows.append(_row(seed, name, result, rt, _xy, cfg_t.uav_height, aodt_threshold=tk))
                jobs.tick(f"Tk={tk:g}s  seed={seed}  {name:8}", result, rt)
    _write_csv(out_dir / "raw_sumrate_vs_aodt.csv", rows)
    _write_csv(out_dir / "positions_sumrate_vs_aodt.csv", expand_positions(rows))
    _write_csv(out_dir / "sumrate_vs_aodt.csv", _summarize(rows, "aodt_threshold"))
    return rows


def sweep_cpu(cfg: SimConfig, args, out_dir: Path) -> list[dict]:
    rows = []
    cpus = (1e8, 1.5e8, 2e8, 2.5e8)
    seeds = _seeds(args)
    methods = _methods(args)
    jobs = Counter("Fig.10  sum rate vs f_j", len(cpus) * len(seeds) * len(methods))
    for cpu in cpus:
        cfg_c = replace(cfg, uav_cpu=cpu)
        for seed in seeds:
            scenario = generate_scenario(seed, cfg_c)
            for name in methods:
                _xy, result, rt = _solve(name, scenario, seed, args)
                rows.append(_row(seed, name, result, rt, _xy, cfg_c.uav_height, uav_cpu=cpu))
                jobs.tick(f"fj={cpu:g}  seed={seed}  {name:8}", result, rt)
    _write_csv(out_dir / "raw_sumrate_vs_cpu.csv", rows)
    _write_csv(out_dir / "positions_sumrate_vs_cpu.csv", expand_positions(rows))
    _write_csv(out_dir / "sumrate_vs_cpu.csv", _summarize(rows, "uav_cpu"))
    return rows


def plot_summaries(out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    specs = [
        ("sumrate_vs_uav.csv", "n_uav", "Number of UAVs", "fig6_sumrate_vs_uav.png"),
        ("sumrate_vs_iot.csv", "n_iot", "Number of IoTs", "fig7_sumrate_vs_iot.png"),
        ("sumrate_vs_lambda.csv", "lambda_i", "Task arrival rate", "fig8_sumrate_vs_lambda.png"),
        ("sumrate_vs_aodt.csv", "aodt_threshold", "AoDT threshold (s)", "fig9_sumrate_vs_aodt.png"),
        ("sumrate_vs_cpu.csv", "uav_cpu", "UAV CPU (cycles/s)", "fig10_sumrate_vs_cpu.png"),
    ]
    for fname, xkey, xlabel, png in specs:
        path = out_dir / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        methods = sorted({r["method"] for r in rows})
        for m in methods:
            sub = [r for r in rows if r["method"] == m]
            xs = [float(r[xkey]) for r in sub]
            ys = [float(r["sum_rate_mean"]) / 1e6 for r in sub]
            es = [float(r["sum_rate_std"]) / 1e6 for r in sub]
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            es = np.array(es)[order]
            ax.errorbar(xs, ys, yerr=es, marker="o", label=m)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sum rate (Mbps)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / png, dpi=120)
        plt.close(fig)


def write_comparison_table(out_dir: Path, rows: list[dict], filename: str = "comparison_table.md"):
    methods = sorted({r["method"] for r in rows})
    lines = [
        "| Method | Sum Rate (bit/s) | Sum Rate (Mbps) | Feasible | AoDT | QoS | CPU unstable | Runtime (s) | Std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        rates = np.array([r["sum_rate"] for r in sub], float)
        qos = np.array([r["qos"] for r in sub], float)
        rt = np.array([r["runtime"] for r in sub], float)
        feas = np.array([r["feasible"] for r in sub], float)
        cpu = np.array([r.get("cpu_unstable", 0) for r in sub], float)
        aodt_vals = [r["aodt_mean"] for r in sub if r.get("aodt_mean") is not None]
        aodt = f"{np.mean(aodt_vals):.3f}" if aodt_vals else "n/a"
        std = float(rates.std(ddof=1)) if rates.size > 1 else 0.0
        lines.append(
            f"| {m} | {rates.mean():.1f} | {rates.mean() / 1e6:.3f} | {feas.mean():.2f} | {aodt} | "
            f"{qos.mean():.2f} | {cpu.mean():.2f} | {rt.mean():.3f} | {std:.1f} |"
        )
    lines.append("| proposed | | | | | | | | |")
    (out_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_sweeps(cfg: SimConfig, args):
    out_dir = Path(getattr(args, "out", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    banner("default I=10 J=3 comparison")
    base_rows = []
    seeds = _seeds(args)
    methods = _methods(args)
    jobs = Counter("default comparison", len(seeds) * len(methods))
    for seed in seeds:
        scenario = generate_scenario(seed, cfg)
        for name in methods:
            _xy, result, rt = _solve(name, scenario, seed, args)
            base_rows.append(_row(seed, name, result, rt, _xy, cfg.uav_height, setting="default"))
            jobs.tick(f"seed={seed}  {name:8}", result, rt)
    _write_csv(out_dir / "raw_default_comparison.csv", base_rows)
    _write_csv(out_dir / "positions_default_comparison.csv", expand_positions(base_rows))
    write_comparison_table(out_dir, base_rows)

    # Convergence and placement artifacts for a frozen development seed
    conv_scenario = generate_scenario(100, cfg)
    xy_pso, _res_pso, _rt, hist = solve_pso_placement(
        conv_scenario, seed=100, n_particles=args.particles, n_iter=args.iters
    )
    conv_rows = [
        {"iter": i, "fitness": f, "sum_rate": r}
        for i, (f, r) in enumerate(zip(hist.best_fitness, hist.best_sum_rate))
    ]
    _write_csv(out_dir / "pso_convergence.csv", conv_rows)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["sum_rate"] / 1e6 for r in conv_rows])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best sum rate (Mbps)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "pso_convergence.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(conv_scenario.iot_xy[:, 0], conv_scenario.iot_xy[:, 1], c=conv_scenario.process_of_iot, label="IoT")
        ax.scatter(xy_pso[:, 0], xy_pso[:, 1], marker="^", s=80, c="red", label="UAV (PSO)")
        ax.set_xlim(0, cfg.area_x)
        ax.set_ylim(0, cfg.area_y)
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title("Seed 100 placement")
        fig.tight_layout()
        fig.savefig(out_dir / "placement_seed100.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass

    sweep_num_uav(cfg, args, out_dir)
    sweep_num_iot(cfg, args, out_dir)
    if cfg.use_compute_model:
        sweep_lambda(cfg, args, out_dir)
        sweep_aodt(cfg, args, out_dir)
        sweep_cpu(cfg, args, out_dir)
    plot_summaries(out_dir)
    meta = {
        "note": "TASK_SIZE_BYTES and TASK_CYCLES are not in Table II; compute sweeps use experimental defaults only if --compute.",
        "radio": {
            "b_sys_hz": cfg.b_sys,
            "noise_power_w": cfg.noise_power,
            "bandwidth_scope": cfg.bandwidth_scope,
            "max_bw_share": cfg.max_bw_share,
            "note": "see docs/calibration.md; literal Table II is --radio-profile table2",
        },
        "use_compute_model": cfg.use_compute_model,
        "n_seeds": len(_seeds(args)),
        "methods": _methods(args),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote sweep outputs under %s", out_dir)
