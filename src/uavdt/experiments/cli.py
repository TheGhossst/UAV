"""Command-line entry: evaluate one seed or average several."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from uavdt.config import (
    BANDWIDTH_PRESETS,
    DEFAULT,
    EXTERNAL_TASK_CYCLES,
    EXTERNAL_TASK_SIZE_BITS,
    PRIMARY_MAX_BW_SHARE,
    SENSITIVITY_MAX_BW_SHARE,
    SimConfig,
)
from uavdt.experiments import run_one, run_seeds
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign
from uavdt.experiments.grids import AXES
from uavdt.experiments.methods import METHODS
from uavdt.experiments.spot import spot_validate_sca
from uavdt.placement.pso import PSOSettings
from uavdt.sca import SCASettings, solve_sca
from uavdt.sca.algorithm import write_history
from uavdt.sca.debug import print_human_table, run_sca_seq_debug
from uavdt.scenario import generate_scenario


def _cfg_from_args(args: argparse.Namespace) -> SimConfig:
    b_hz = args.bandwidth
    if args.bandwidth_preset is not None:
        b_hz = BANDWIDTH_PRESETS[args.bandwidth_preset]
    task_bits = args.task_size_bits
    if args.task_size_bytes is not None:
        task_bits = args.task_size_bytes * 8.0
    area = args.area_m
    if area is None:
        area_x, area_y = DEFAULT.area_x_m, DEFAULT.area_y_m
    else:
        area_x = area_y = float(area)
    return SimConfig(
        area_x_m=float(area_x),
        area_y_m=float(area_y),
        b_sys_hz=float(b_hz),
        task_size_bits=float(task_bits),
        task_cycles=float(args.task_cycles),
        lambda_i_per_s=float(args.lambda_i),
        aodt_threshold_s=float(args.aodt_threshold),
        los_angle_unit=args.los_angle_unit,
        max_bw_share=args.max_bw_share,
        download_time_s=float(args.download_time),
    )


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bandwidth",
        type=float,
        default=DEFAULT.b_sys_hz,
        help="B_sys in Hz (20000, 2400000, or 8800000)",
    )
    p.add_argument(
        "--bandwidth-preset",
        choices=sorted(BANDWIDTH_PRESETS),
        default=None,
        help="Named B_sys: 20khz, 2.4mhz, 8.8mhz (overrides --bandwidth)",
    )
    p.add_argument(
        "--task-size-bits",
        type=float,
        default=EXTERNAL_TASK_SIZE_BITS,
        help="S_i in bits for D=S/r. EXTERNAL, not Table II.",
    )
    p.add_argument(
        "--task-size-bytes",
        type=float,
        default=None,
        help="If set, S_i = bytes*8. Interpretation, not Eq. (11).",
    )
    p.add_argument(
        "--task-cycles",
        type=float,
        default=EXTERNAL_TASK_CYCLES,
        help="L in cycles/task. EXTERNAL, not Table II.",
    )
    p.add_argument("--lambda-i", type=float, default=DEFAULT.lambda_i_per_s)
    p.add_argument("--aodt-threshold", type=float, default=DEFAULT.aodt_threshold_s)
    p.add_argument(
        "--max-bw-share",
        type=float,
        default=None,
        help=(
            "EXTERNAL PARAMETER: optional per-link cap as a fraction of "
            f"B_sys (e.g. {PRIMARY_MAX_BW_SHARE:g} primary, "
            f"{SENSITIVITY_MAX_BW_SHARE:g} tighter-cap sensitivity). "
            "Not in Problem (P) or Table II."
        ),
    )
    p.add_argument("--los-angle-unit", choices=("rad", "deg"), default="rad")
    p.add_argument(
        "--area-m",
        type=float,
        default=None,
        help=(
            "Square field side in metres. Default 100 (this reproduction). "
            "Paper §VII uses 500."
        ),
    )
    p.add_argument("--placement", choices=("random", "kmeans"), default="random")
    p.add_argument(
        "--download-time",
        type=float,
        default=0.0,
        help="Eq. (12) UAV→BS download Z (s). Paper neglects this; default 0.",
    )


def _print_eval(seed: int, cfg: SimConfig, uav, result, *, verbose: bool) -> None:
    c = result.constraints
    print(f"seed                 {seed}")
    print(f"area_m               {cfg.area_x_m:g} x {cfg.area_y_m:g}")
    print(f"B_sys                {_fmt_hz(cfg.b_sys_hz)}")
    print(f"sigma                {cfg.sigma}")
    print(f"noise_power_W        {cfg.noise_power_w}  (= sigma**2)")
    print(f"S_bits (EXTERNAL)    {cfg.task_size_bits:g}")
    print(f"L_cycles (EXTERNAL)  {cfg.task_cycles:g}")
    print(f"mu_per_s             {result.mu_per_s:g}")
    print(f"sum_rate_bit_per_s   {result.sum_rate_bit_per_s:.6g}")
    print(f"sum_rate_Mbps        {result.sum_rate_mbps:.6g}")
    print(f"AoDT_s               {np_list(result.aodt_s)}")
    print(f"AoDT <= T_k          {np_list(result.aodt_satisfied)}")
    print(f"Z_download_s         {cfg.download_time_s:g}  (Eq. 12; paper 0)")
    print(f"rho                  {np_list(result.rho)}")
    print(f"feasible             {result.feasible}")
    print(
        "violations           "
        f"qos={c.qos_violations} aodt={c.aodt_violations} "
        f"sep={c.sep_violations} cpu={c.cpu_unstable_count} "
        f"bw_excess_Hz={c.bw_excess_hz:.4g}"
    )
    if verbose:
        print("uav_xyz_m")
        print(uav)
        print("assoc_rates_bit_per_s", np_list(result.assoc_rates_bit_per_s))
        print("upload_times_s", np_list(result.upload_times_s))
        print("min_distance_m", float(result.distances_m.min()))
        print("max_distance_m", float(result.distances_m.max()))


def _fmt_hz(hz: float) -> str:
    if hz >= 1.0e6:
        return f"{hz:g} Hz ({hz / 1.0e6:g} MHz)"
    return f"{hz:g} Hz ({hz / 1.0e3:g} kHz)"


def np_list(arr) -> str:
    parts = []
    for x in arr:
        if isinstance(x, (bool, np.bool_)):
            parts.append("true" if bool(x) else "false")
        else:
            parts.append(f"{float(x):.6g}")
    return "[" + ", ".join(parts) + "]"


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    uav, result = run_one(args.seed, cfg, args.placement)
    _print_eval(args.seed, cfg, uav, result, verbose=args.verbose)
    return 0


def cmd_multi_seed(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    seeds = tuple(range(args.seed_start, args.seed_start + args.n_runs))
    summary = run_seeds(seeds, cfg, args.placement)
    payload = {
        "seeds": list(summary.seeds),
        "B_sys_Hz": cfg.b_sys_hz,
        "placement": args.placement,
        "mean_sum_rate_bit_per_s": summary.mean_sum_rate_bit_per_s,
        "std_sum_rate_bit_per_s": summary.std_sum_rate_bit_per_s,
        "mean_sum_rate_Mbps": summary.mean_sum_rate_bit_per_s / 1e6,
        "std_sum_rate_Mbps": summary.std_sum_rate_bit_per_s / 1e6,
        "mean_AoDT_s": [float(x) for x in summary.mean_aodt_s],
        "std_AoDT_s": [float(x) for x in summary.std_aodt_s],
        "feasible_fraction": float(summary.feasible.mean()),
        "task_size_bits_EXTERNAL": cfg.task_size_bits,
        "task_cycles_EXTERNAL": cfg.task_cycles,
        "note": "Paper §VII uses 20 random runs per plotted point.",
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_bandwidth_sweep(args: argparse.Namespace) -> int:
    args.bandwidth_preset = None
    for name, hz in BANDWIDTH_PRESETS.items():
        args.bandwidth = hz
        cfg = _cfg_from_args(args)
        uav, result = run_one(args.seed, cfg, args.placement)
        print(f"=== preset {name}  B_sys={hz:g} Hz ===")
        _print_eval(args.seed, cfg, uav, result, verbose=False)
        print()
    return 0


def cmd_sca(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    scenario = generate_scenario(args.seed, cfg)
    solver = args.solver
    if solver in {None, "cvxpy", "python", "none"}:
        solver = None
    settings = SCASettings(
        max_iterations=int(args.max_iterations),
        epsilon=float(args.epsilon),
        step_size_m=float(args.step_size),
        solver=solver,
    )
    result = solve_sca(scenario, args.seed, settings=settings)
    write_history(result, args.history_json)
    write_history(result, args.history_csv)
    ev = result.true_eval
    c = ev.constraints
    d = result.diagnostics
    stop = str(d.get("stop_reason", ""))
    print("method                Algorithm 1 SCA of Problem (P)")
    print(f"B_sys                 {_fmt_hz(cfg.b_sys_hz)}")
    if cfg.max_bw_share is None:
        print("max_bw_share          none (paper (26)–(27) only)")
    else:
        print(
            f"max_bw_share          {cfg.max_bw_share:g}  "
            f"(B_ij <= {cfg.link_bandwidth_cap_hz:.6g} Hz; "
            "EXTERNAL PARAMETER, not Problem (P))"
        )
    print(f"solver_status         {result.solver_status}")
    print(f"solver_name           {result.solver_name}")
    print(f"solver_backend        {d.get('solver_backend')}")
    if d.get("se_max_abs_diff") is not None:
        print(f"se_max_abs_diff       {d.get('se_max_abs_diff')}")
    print(f"n_iterations          {result.n_iterations}")
    print(f"stop_reason           {stop}")
    print(stop)
    print(f"accepted_steps        {d.get('accepted_steps')}")
    print(f"rejected_steps        {d.get('rejected_steps')}")
    print(f"step_size_reductions  {d.get('step_size_reductions')}")
    if d.get("final_step_m") is not None:
        print(f"final_step_m          {d.get('final_step_m')}")
    init_obj = result.history[0].true_objective if result.history else float("nan")
    print(f"initial_true_obj      {init_obj:.6g}")
    print(f"final_true_obj        {result.true_objective:.6g}")
    print(f"improvement           {result.true_objective - init_obj:.6g}")
    print(f"true_sum_rate_Mbps    {ev.sum_rate_mbps:.6g}")
    print(f"AoDT_s                {np_list(ev.aodt_s)}")
    print(f"rho                   {np_list(ev.rho)}")
    print(f"min_uav_separation_m  {result.history[-1].current_min_separation:.6g}")
    print(f"bandwidth_used_Hz     {result.history[-1].bandwidth_usage:.6g}")
    print(f"max_link_B_Hz         {float(result.allocation.bandwidth_hz.max()):.6g}")
    print(f"feasible              {ev.feasible}")
    print(
        "violations            "
        f"qos={c.qos_violations} aodt={c.aodt_violations} "
        f"sep={c.sep_violations} cpu={c.cpu_unstable_count} "
        f"bw_excess_Hz={c.bw_excess_hz:.4g}"
    )
    print("uav_xyz_m")
    print(result.uav_xyz_m)
    print("bandwidth_hz")
    print(result.allocation.bandwidth_hz)
    print("assoc_rates_bit_per_s")
    print(ev.assoc_rates_bit_per_s)
    print(f"history_json          {args.history_json}")
    print(f"history_csv           {args.history_csv}")
    return 0


def cmd_sca_seq_debug(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    solver = args.solver
    if solver in {None, "cvxpy", "python", "none"}:
        solver = None
    payload = run_sca_seq_debug(
        args.seed,
        cfg,
        max_iterations=int(args.max_iterations),
        step_size_m=float(args.step_size),
        out_json=args.out_json,
        solver=solver,
    )
    print_human_table(payload)
    return 0


def _parse_methods(raw: str) -> tuple[str, ...]:
    names = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    for n in names:
        if n not in METHODS:
            raise ValueError(f"unknown method {n!r}; expected {METHODS}")
    return names


def _parse_axes(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return AXES
    names = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    for n in names:
        if n not in AXES:
            raise ValueError(f"unknown axis {n!r}; expected {AXES} or all")
    return names


def cmd_campaign(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    solver = args.solver
    if solver in {None, "cvxpy", "python", "none"}:
        solver = None
    settings = CampaignSettings(
        n_runs=int(args.n_runs),
        seed_start=int(args.seed_start),
        methods=_parse_methods(args.methods),
        sca_settings=SCASettings(
            max_iterations=int(args.max_iterations),
            epsilon=float(args.epsilon),
            step_size_m=float(args.step_size),
            solver=solver,
        ),
        pso_settings=PSOSettings(),
    )
    payload = run_campaign(_parse_axes(args.axis), cfg, settings)
    out = write_campaign(payload, args.out)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.csv')}")
    return 0


def cmd_spot_validate(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    payload = spot_validate_sca(
        int(args.seed),
        cfg,
        max_iterations=int(args.max_iterations),
        step_size_m=float(args.step_size),
    )
    print(json.dumps(payload, indent=2))
    if payload.get("agreement") == "objective_mismatch":
        return 2
    return 0


def cmd_aodt_compare(args: argparse.Namespace) -> int:
    from uavdt.aodt import average_aodt_eq15_s, average_aodt_fcfs_closed_s
    from uavdt.aodt_sim import DISCIPLINES, compare_closed_and_sim
    from uavdt.evaluator import evaluate
    from uavdt.experiments.fig11 import lambdas_for_pattern
    from uavdt.placement.kmeans import place_kmeans
    from uavdt.placement.random import place_random

    cfg = _cfg_from_args(args)
    lam = None
    if args.lambda_pattern is not None:
        lam = lambdas_for_pattern(cfg, args.lambda_pattern)
    scenario = generate_scenario(args.seed, cfg, lambdas_per_s=lam)
    if args.placement == "kmeans":
        uav = place_kmeans(scenario, args.seed)
    else:
        uav = place_random(cfg.num_uav, args.seed, cfg)
    ev = evaluate(scenario, uav)
    a = ev.extras["association"]
    b = ev.extras["processing"]
    mu_vec = np.full(uav.shape[0], ev.mu_per_s)
    cmp = compare_closed_and_sim(
        scenario,
        a,
        b,
        ev.rates_bit_per_s,
        mu_vec,
        disciplines=DISCIPLINES,
        horizon_s=float(args.horizon),
        warmup_s=float(args.warmup),
        seed=args.seed,
        queue_scope=args.queue_scope,
    )
    eq15 = average_aodt_eq15_s(scenario, a, b, ev.rates_bit_per_s, mu_vec)
    fcfs_c = average_aodt_fcfs_closed_s(scenario, a, b, ev.rates_bit_per_s, mu_vec)
    print("AoDT closed forms vs event-driven queues")
    print(f"seed                 {args.seed}")
    print(f"placement            {args.placement}")
    print(f"lambda_pattern       {args.lambda_pattern or 'uniform cfg.lambda_i'}")
    print(f"queue_scope          {args.queue_scope}")
    print(f"Z_download_s         {cfg.download_time_s:g}")
    print(f"eq17_s               {np_list(ev.aodt_s)}  (Problem P)")
    print(f"eq15_s               {np_list(eq15)}")
    print(f"fcfs_closed_s        {np_list(fcfs_c)}")
    print(f"feasible             {ev.feasible}")
    for d in DISCIPLINES:
        row = cmp["sim"][d]
        print(
            f"sim_{d:7s}          max={row['mean_max_process_age_s']:.6g}  "
            f"mean_src={row['mean_source_age_flat_s']:.6g}  "
            f"per_process={row['mean_process_age_s']}  "
            f"delivered={row['n_delivered']} dropped={row['n_dropped']}"
        )
    return 0


def cmd_fig11(args: argparse.Namespace) -> int:
    from uavdt.experiments.fig11 import run_fig11, sensibility_checks, write_fig11

    cfg = _cfg_from_args(args)
    payload = run_fig11(
        cfg,
        n_runs=int(args.n_runs),
        seed_start=int(args.seed_start),
        horizon_s=float(args.horizon),
        warmup_s=float(args.warmup),
    )
    report = sensibility_checks(payload)
    payload["sensibility"] = report
    out = write_fig11(payload, args.out)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.csv')}")
    print(f"sensibility          {report['n_pass']}/{report['n_checks']} pass")
    for chk in report["checks"]:
        flag = "ok" if chk["ok"] else "FAIL"
        line = f"  [{flag}] {chk['name']}"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))
    return 0 if report["all_ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="uavdt",
        description="Core simulator for Khalaf et al. UAV-DT IoT model.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="One seed, one bandwidth")
    _add_shared(ev)
    ev.add_argument("--seed", type=int, default=1)
    ev.add_argument("--verbose", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    ms = sub.add_parser("multi-seed", help="Mean/std over consecutive seeds")
    _add_shared(ms)
    ms.add_argument("--seed-start", type=int, default=1)
    ms.add_argument("--n-runs", type=int, default=20)
    ms.set_defaults(func=cmd_multi_seed)

    sw = sub.add_parser("bandwidth-sweep", help="Run 20 kHz, 2.4 MHz, 8.8 MHz")
    _add_shared(sw)
    sw.add_argument("--seed", type=int, default=1)
    sw.set_defaults(func=cmd_bandwidth_sweep)

    sc = sub.add_parser(
        "sca",
        help="Algorithm 1 SCA; scores with the true evaluator",
    )
    _add_shared(sc)
    sc.add_argument("--seed", type=int, default=1)
    sc.add_argument("--max-iterations", type=int, default=30)
    sc.add_argument("--epsilon", type=float, default=1e-4)
    sc.add_argument("--step-size", type=float, default=20.0, help="L-inf SCA neighborhood of q (m)")
    sc.add_argument(
        "--solver",
        type=str,
        default="matlab",
        help="matlab/MOSEK: one-session MATLAB CVX+MOSEK; cvxpy: Python HiGHS LP",
    )
    sc.add_argument("--history-json", type=str, default="results/sca_history.json")
    sc.add_argument("--history-csv", type=str, default="results/sca_history.csv")
    sc.set_defaults(func=cmd_sca)

    dbg = sub.add_parser(
        "sca-seq-debug",
        help="Algorithm 1 SCA iteration log (true-feasible gate)",
    )
    _add_shared(dbg)
    dbg.add_argument("--seed", type=int, default=1)
    dbg.add_argument("--max-iterations", type=int, default=30)
    dbg.add_argument("--step-size", type=float, default=20.0)
    dbg.add_argument(
        "--solver",
        type=str,
        default="matlab",
        help="matlab/MOSEK: one-session MATLAB CVX+MOSEK; cvxpy: Python HiGHS LP",
    )
    dbg.add_argument("--out-json", type=str, default="results/sca_seq_debug.json")
    dbg.set_defaults(func=cmd_sca_seq_debug)

    camp = sub.add_parser(
        "campaign",
        help="§VII-style sweeps: J, I, λ, T_k, CPU; methods random/kmeans/pso/sca",
    )
    _add_shared(camp)
    camp.add_argument(
        "--axis",
        type=str,
        default="all",
        help="uavs,iots,lambda,aodt,cpu or all (paper Figs. 6–10 axes)",
    )
    camp.add_argument(
        "--methods",
        type=str,
        default="random,kmeans,pso,sca",
        help="Comma-separated: random,kmeans,pso,sca",
    )
    camp.add_argument("--n-runs", type=int, default=5, help="Paper uses 20; default 5")
    camp.add_argument("--seed-start", type=int, default=1)
    camp.add_argument("--max-iterations", type=int, default=30)
    camp.add_argument("--epsilon", type=float, default=1e-4)
    camp.add_argument("--step-size", type=float, default=20.0)
    camp.add_argument(
        "--solver",
        type=str,
        default="cvxpy",
        help="SCA backend: cvxpy (campaign default) or matlab/MOSEK",
    )
    camp.add_argument("--out", type=str, default="results/campaign.json")
    camp.set_defaults(func=cmd_campaign)

    sp = sub.add_parser(
        "spot-validate",
        help="CVXPY vs MATLAB CVX/MOSEK spot-check of frozen SCA",
    )
    _add_shared(sp)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--max-iterations", type=int, default=12)
    sp.add_argument("--step-size", type=float, default=20.0)
    sp.set_defaults(func=cmd_spot_validate)

    ac = sub.add_parser(
        "aodt-compare",
        help="Eqs. (14)–(17) vs FCFS / FCFS-P / LCFS-S event simulation",
    )
    _add_shared(ac)
    ac.add_argument("--seed", type=int, default=1)
    ac.add_argument("--horizon", type=float, default=120.0)
    ac.add_argument("--warmup", type=float, default=24.0)
    ac.add_argument(
        "--queue-scope",
        choices=("process", "uav"),
        default="process",
        help="process: Eq. (17) isolation; uav: share server (constraint 24)",
    )
    ac.add_argument(
        "--lambda-pattern",
        choices=("uniform_fast", "uniform_slow", "heterogeneous"),
        default=None,
        help="Fig. 11 arrival pattern; default is uniform cfg.lambda_i",
    )
    ac.set_defaults(func=cmd_aodt_compare)

    f11 = sub.add_parser(
        "fig11",
        help="Fig. 11 uniform-fast / slow / heterogeneous λ vs J (AoDT check)",
    )
    _add_shared(f11)
    f11.add_argument("--n-runs", type=int, default=20, help="Paper uses 20")
    f11.add_argument("--seed-start", type=int, default=1)
    f11.add_argument("--horizon", type=float, default=80.0)
    f11.add_argument("--warmup", type=float, default=16.0)
    f11.add_argument("--out", type=str, default="results/fig11.json")
    f11.set_defaults(func=cmd_fig11)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
