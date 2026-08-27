"""CLI: frozen-scenario check, solvers, and comparison runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import (
    DEFAULT,
    DEV_SCENARIO_SEEDS,
    FROZEN_UAV_XY,
    PAPER_SCENARIO_SEEDS,
    PSO_N_ITER,
    PSO_N_PARTICLES,
    RADIO_PROFILES,
    SimConfig,
    TD3_TOTAL_STEPS,
)
from src.evaluator import evaluate
from src.repair import complete_solution
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.pso import solve_pso_joint, solve_pso_placement
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca
from src.solvers.td3 import solve_td3


def _cfg_from_args(args: argparse.Namespace) -> SimConfig:
    cfg = DEFAULT
    if getattr(args, "radio_profile", None):
        cfg = cfg.with_radio_profile(args.radio_profile)
    if getattr(args, "compute", False):
        cfg = cfg.with_compute()
    if getattr(args, "num_uav", None):
        cfg = replace(cfg, num_uav=args.num_uav)
    if getattr(args, "num_iot", None):
        n = args.num_iot
        k = cfg.num_processes
        per = n // k
        cfg = replace(cfg, num_iot=n, iots_per_process=per if per * k == n else cfg.iots_per_process)
    if getattr(args, "lambda_i", None) is not None:
        cfg = replace(cfg, lambda_i=args.lambda_i)
    if getattr(args, "aodt_threshold", None) is not None:
        cfg = replace(cfg, aodt_threshold=args.aodt_threshold)
    if getattr(args, "uav_cpu", None) is not None:
        cfg = replace(cfg, uav_cpu=args.uav_cpu)
    if getattr(args, "los_unit", None):
        cfg = replace(cfg, los_angle_unit=args.los_unit)
    return cfg


def run_single(cfg: SimConfig, seed: int = 100) -> dict:
    scenario = generate_scenario(seed, cfg)
    uav_xy = np.array([[FROZEN_UAV_XY[0], FROZEN_UAV_XY[1]]])
    xy, a, b, bw = complete_solution(scenario, uav_xy)
    result = evaluate(scenario, xy, a, b, bw)
    rows = []
    for i in range(cfg.num_iot):
        rows.append(
            {
                "iot": i,
                "x": float(scenario.iot_xy[i, 0]),
                "y": float(scenario.iot_xy[i, 1]),
                "distance": float(result.extras["distance"][i, 0]),
                "p_los": float(result.extras["p_los"][i, 0]),
                "l_avg": float(result.extras["l_avg"][i, 0]),
                "rate": float(result.rates[i, 0]),
            }
        )
    print("Frozen UAV at (250, 250, 100), seed", seed)
    print(f"{'iot':>4} {'x':>8} {'y':>8} {'d':>10} {'PLoS':>8} {'Lavg':>10} {'rate':>12}")
    for r in rows:
        print(
            f"{r['iot']:4d} {r['x']:8.2f} {r['y']:8.2f} {r['distance']:10.3f} "
            f"{r['p_los']:8.4f} {r['l_avg']:10.3f} {r['rate']:12.3f}"
        )
    print(f"sum_rate = {result.sum_rate:.6f} bit/s")
    print(f"min_assoc_rate = {result.min_assoc_rate:.6f} bit/s")
    print(f"qos_violations = {result.qos_violations}")
    return {"rows": rows, "sum_rate": result.sum_rate, "feasible": result.feasible}


def _print_result(name: str, xy: np.ndarray, result, runtime: float):
    aodt = result.aodt.tolist() if result.compute_available else "n/a (S_i/L unspecified)"
    print(f"{name}: sum_rate={result.sum_rate:.3f} bit/s  min_rate={result.min_assoc_rate:.3f}  "
          f"feasible={result.feasible}  qos={result.qos_violations}  "
          f"aodt={aodt}  runtime={runtime:.3f}s")
    print("  UAV xy:", np.array2string(xy, precision=2))


def run_solver(mode: str, cfg: SimConfig, seed: int, args: argparse.Namespace):
    scenario = generate_scenario(seed, cfg)
    if mode == "random":
        xy, result, rt = solve_random(scenario, seed=seed)
        _print_result("random", xy, result, rt)
        return result
    if mode == "kmeans":
        xy, result, rt = solve_kmeans(scenario, seed=seed)
        _print_result("kmeans", xy, result, rt)
        return result
    if mode == "pso":
        xy, result, rt, hist = solve_pso_placement(
            scenario, seed=seed, n_particles=args.particles, n_iter=args.iters
        )
        _print_result("pso-placement", xy, result, rt)
        print(f"  conv_final_fitness={hist.best_fitness[-1] if hist.best_fitness else None}")
        return result
    if mode == "pso-joint":
        xy, result, rt, hist = solve_pso_joint(
            scenario, seed=seed, n_particles=args.particles, n_iter=args.iters
        )
        _print_result("pso-joint", xy, result, rt)
        return result
    if mode == "sca":
        xy, result, rt = solve_sca(scenario, seed=seed)
        _print_result("sca", xy, result, rt)
        return result
    if mode == "td3":
        xy, result, rt, log = solve_td3(scenario, seed=seed, total_steps=args.td3_steps)
        _print_result("td3", xy, result, rt)
        if log and log.rewards:
            print(f"  last_reward={log.rewards[-1]:.4f}")
        return result
    if mode == "proposed":
        from src.solvers.proposed import solve_proposed

        solve_proposed(scenario, seed=seed)
    raise SystemExit(f"unknown mode {mode}")


def run_compare(cfg: SimConfig, seeds: tuple[int, ...], args: argparse.Namespace) -> list[dict]:
    methods = ["random", "kmeans", "pso", "sca"]
    if args.with_td3:
        methods.append("td3")
    rows = []
    for seed in seeds:
        scenario = generate_scenario(seed, cfg)
        for name in methods:
            t0 = time.perf_counter()
            if name == "random":
                xy, result, rt = solve_random(scenario, seed=seed)
            elif name == "kmeans":
                xy, result, rt = solve_kmeans(scenario, seed=seed)
            elif name == "pso":
                xy, result, rt, _ = solve_pso_placement(
                    scenario, seed=seed, n_particles=args.particles, n_iter=args.iters
                )
            elif name == "sca":
                xy, result, rt = solve_sca(scenario, seed=seed)
            else:
                xy, result, rt, _ = solve_td3(scenario, seed=seed, total_steps=args.td3_steps)
            row = {
                "seed": seed,
                "method": name,
                "sum_rate": result.sum_rate,
                "min_rate": result.min_assoc_rate,
                "qos": result.qos_violations,
                "feasible": result.feasible,
                "runtime": rt,
                "aodt_mean": float(np.nanmean(result.aodt)) if result.compute_available else None,
                "uav_xy": json.dumps(
                    [[float(x), float(y), float(cfg.uav_height)] for x, y in np.asarray(xy).reshape(-1, 2)]
                ),
            }
            rows.append(row)
            print(json.dumps(row))
            _ = t0
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UAV-aided digital twin reproduction")
    p.add_argument(
        "--mode",
        default="single",
        choices=["single", "random", "kmeans", "pso", "pso-joint", "sca", "td3", "proposed", "compare", "sweeps", "aodt-compare"],
    )
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--compute", action="store_true", help="Enable experimental S_i and L (not Table II)")
    p.add_argument(
        "--radio-profile",
        choices=sorted(RADIO_PROFILES),
        default=None,
        help="calibrated (default) reproduces the Mbps-scale figures; table2 is the literal Table II reading",
    )
    p.add_argument("--num-uav", type=int, default=None)
    p.add_argument("--num-iot", type=int, default=None)
    p.add_argument("--lambda-i", type=float, default=None)
    p.add_argument("--aodt-threshold", type=float, default=None)
    p.add_argument("--uav-cpu", type=float, default=None)
    p.add_argument("--los-unit", choices=["rad", "deg"], default=None)
    p.add_argument("--particles", type=int, default=PSO_N_PARTICLES)
    p.add_argument("--iters", type=int, default=PSO_N_ITER)
    p.add_argument("--td3-steps", type=int, default=TD3_TOTAL_STEPS)
    p.add_argument("--paper-runs", action="store_true", help="Use 20 scenario seeds")
    p.add_argument("--with-td3", action="store_true")
    p.add_argument("--aodt-short", action="store_true", help="AoDT compare with 5 seeds instead of 20")
    p.add_argument("--out", type=str, default="results")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _cfg_from_args(args)
    if args.mode == "single":
        run_single(cfg, seed=args.seed)
        return 0
    if args.mode == "compare":
        seeds = PAPER_SCENARIO_SEEDS if args.paper_runs else DEV_SCENARIO_SEEDS
        rows = run_compare(cfg, seeds, args)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        from src.experiments.sweeps import _write_csv, expand_positions, write_comparison_table

        _write_csv(out / ("compare_raw_20runs.csv" if args.paper_runs else "compare_raw.csv"), rows)
        _write_csv(
            out / ("positions_compare_20runs.csv" if args.paper_runs else "positions_compare.csv"),
            expand_positions(rows),
        )
        write_comparison_table(
            out,
            rows,
            filename="comparison_table_20runs.md" if args.paper_runs else "comparison_table.md",
        )
        return 0
    if args.mode == "sweeps":
        from src.experiments.sweeps import run_all_sweeps

        run_all_sweeps(cfg, args)
        return 0
    if args.mode == "aodt-compare":
        from src.experiments.aodt_compare import run_aodt_comparison

        run_aodt_comparison(cfg, args)
        return 0
    run_solver(args.mode, cfg, args.seed, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
