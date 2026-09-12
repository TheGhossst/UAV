"""Command-line entry: evaluate one seed or average several."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

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
from uavdt.experiments.grids import AXES, config_for_counts
from uavdt.experiments.methods import KNOWN_METHODS
from uavdt.experiments.spot import spot_validate_sca
from uavdt.placement.pso import PSOSettings
from uavdt.sca import SCASettings, solve_sca
from uavdt.sca.algorithm import write_history
from uavdt.sca_anchor import AnchorSettings, solve_sca_anchor
from uavdt.sca_multistart import MultiStartSettings, solve_sca_multistart
from uavdt.sca.debug import print_human_table, run_sca_seq_debug
from uavdt.scenario import generate_scenario
from uavdt.td3.settings import TD3Settings


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
        help="B_sys in Hz (20000, 2400000, 7000000, or 8800000)",
    )
    p.add_argument(
        "--bandwidth-preset",
        choices=sorted(BANDWIDTH_PRESETS),
        default=None,
        help="Named B_sys: 20khz, 2.4mhz, 7mhz, 8.8mhz (overrides --bandwidth)",
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


def _add_td3_preset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--td3-preset",
        choices=("alg2", "residual-on-sca"),
        default="alg2",
        help=(
            "alg2: Algorithm 2 reproduction (k-means residual, leftover B, "
            "penalty reward). residual-on-sca: proposed interface to (P)."
        ),
    )
    p.add_argument(
        "--inner-bandwidth",
        choices=("leftover", "equal_share", "lp"),
        default=None,
        help="Override TD3Settings.inner_bandwidth (default follows the preset).",
    )
    p.add_argument(
        "--uav-init",
        choices=("kmeans", "random", "sca"),
        default=None,
        help="Override TD3Settings.uav_init (default follows the preset).",
    )
    p.add_argument(
        "--reward-mode",
        choices=("alg2", "feasible_rate"),
        default=None,
        help="Override TD3Settings.reward_mode (default follows the preset).",
    )
    p.add_argument(
        "--export-mode",
        choices=("policy", "best_snapshot"),
        default=None,
        help="Override TD3Settings.export_mode (default follows the preset).",
    )


def _td3_settings_from_args(args: argparse.Namespace) -> TD3Settings:
    preset = str(getattr(args, "td3_preset", "alg2") or "alg2")
    if preset == "residual-on-sca":
        settings = TD3Settings.residual_on_sca()
    elif preset == "alg2":
        settings = TD3Settings()
    else:
        raise ValueError(f"unknown td3 preset {preset!r}")
    updates: dict = {}
    for name in (
        "total_steps",
        "horizon",
        "hidden",
        "batch_size",
        "warmup_steps",
        "buffer_size",
        "log_every",
        "export_avg_steps",
    ):
        if hasattr(args, name):
            val = getattr(args, name)
            if val is not None:
                updates[name] = val
    for name in ("export_mode", "inner_bandwidth", "uav_init", "reward_mode"):
        val = getattr(args, name, None)
        if val is not None:
            updates[name] = val
    if updates:
        settings = replace(settings, **updates)
    return settings


def _add_multistart_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--multistart-random",
        type=int,
        default=2,
        help="Extra random-placement SCA inits (Experiment A default 2).",
    )
    p.add_argument(
        "--multistart-kmeans",
        type=int,
        default=2,
        help="Extra k-means SCA inits with offset seeds (Experiment A default 2).",
    )


def _multistart_from_args(args: argparse.Namespace) -> MultiStartSettings:
    return MultiStartSettings(
        n_random=int(getattr(args, "multistart_random", 2)),
        n_kmeans=int(getattr(args, "multistart_kmeans", 2)),
        include_frozen=not bool(getattr(args, "no_frozen_start", False)),
    )


def _add_anchor_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--anchor-top-k",
        type=int,
        default=3,
        help="Polish the K best LP-scored zenith J-subsets with frozen SCA.",
    )
    p.add_argument(
        "--anchor-max-enumerate",
        type=int,
        default=1000,
        help="Full C(I,J) enum if at most this many subsets; else beam search.",
    )
    p.add_argument(
        "--anchor-beam-width",
        type=int,
        default=10,
        help="Beam width when C(I,J) exceeds --anchor-max-enumerate.",
    )
    p.add_argument(
        "--anchor-process-cohesive",
        action="store_true",
        help=(
            "Also LP-score process-cohesive a_ij on each zenith set "
            "(T_k=0.8 probe). Default off."
        ),
    )


def _anchor_from_args(args: argparse.Namespace) -> AnchorSettings:
    return AnchorSettings(
        top_k=int(getattr(args, "anchor_top_k", 3)),
        max_enumerate=int(getattr(args, "anchor_max_enumerate", 1000)),
        beam_width=int(getattr(args, "anchor_beam_width", 10)),
        include_frozen=not bool(getattr(args, "no_frozen_start", False)),
        process_cohesive_candidate=bool(
            getattr(args, "anchor_process_cohesive", False)
        ),
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


def cmd_sca_joint(args: argparse.Namespace) -> int:
    from uavdt.sca_joint import solve_sca_joint

    cfg = _cfg_from_args(args)
    scenario = generate_scenario(args.seed, cfg)
    solver = args.solver
    if solver in {None, "cvxpy", "python", "none"}:
        solver = None
    if solver in {"matlab", "MATLAB", "mosek", "MOSEK"}:
        raise SystemExit("sca-joint is CVXPY-only; frozen SCA keeps the MATLAB path")
    settings = SCASettings(
        max_iterations=int(args.max_iterations),
        epsilon=float(args.epsilon),
        step_size_m=float(args.step_size),
        solver=solver,
        process_cohesive_candidate=bool(
            getattr(args, "process_cohesive_candidate", False)
        ),
    )
    result = solve_sca_joint(scenario, args.seed, settings=settings)
    write_history(result, args.history_json)
    write_history(result, args.history_csv)
    ev = result.true_eval
    c = ev.constraints
    d = result.diagnostics
    stop = str(d.get("stop_reason", ""))
    print("method                sca_joint (methodology probe; not headline SCA)")
    print(f"B_sys                 {_fmt_hz(cfg.b_sys_hz)}")
    print(f"solver_status         {result.solver_status}")
    print(f"n_iterations          {result.n_iterations}")
    print(f"stop_reason           {stop}")
    print(f"accepted_steps        {d.get('accepted_steps')}")
    print(f"process_cohesive_cand {d.get('process_cohesive_candidate')}")
    print(f"rematch_accepted      {d.get('rematch_accepted')}")
    print(f"rematch_kinds         {d.get('rematch_kinds')}")
    print(f"association_frozen    {d.get('association_init_equals_final')}")
    print(f"processing_frozen     {d.get('processing_init_equals_final')}")
    print(f"final_true_obj        {result.true_objective:.6g}")
    print(f"true_sum_rate_Mbps    {ev.sum_rate_mbps:.6g}")
    print(f"feasible              {ev.feasible}")
    print(
        "violations            "
        f"qos={c.qos_violations} aodt={c.aodt_violations} "
        f"sep={c.sep_violations} cpu={c.cpu_unstable_count} "
        f"bw_excess_Hz={c.bw_excess_hz:.4g}"
    )
    print(f"wall_clock_s          {d.get('wall_clock_s')}")
    print(f"history_json          {args.history_json}")
    print(f"history_csv           {args.history_csv}")
    return 0


def cmd_td3(args: argparse.Namespace) -> int:
    from uavdt.td3.solve import solve_td3

    cfg = _cfg_from_args(args)
    scenario = generate_scenario(args.seed, cfg)
    settings = _td3_settings_from_args(args)
    result = solve_td3(scenario, args.seed, settings=settings)
    ev = result.true_eval
    c = ev.constraints
    d = result.diagnostics
    if settings.is_residual_on_sca():
        print(
            "method                TD3 residual-on-SCA "
            "(proposed interface to (P); not Algorithm 2)"
        )
    else:
        print("method                TD3 (Algorithm 2 fill-in; TD3Settings, not Table II)")
    print(f"B_sys                 {_fmt_hz(cfg.b_sys_hz)}")
    print(f"total_steps           {d.get('total_steps')}")
    print(f"n_updates             {d.get('n_updates')}")
    print(f"n_episodes            {d.get('n_episodes')}")
    print(f"obs_dim               {d.get('obs_dim')}")
    print(f"act_dim               {d.get('act_dim')}")
    print(f"hidden                {d.get('hidden')}")
    print(f"export_rule           {d.get('export_rule')}")
    print(f"true_sum_rate_Mbps    {ev.sum_rate_mbps:.6g}")
    print(f"snapshot_LP_Mbps      {d.get('snapshot_export_sum_rate_Mbps')}")
    print(f"policy_LP_Mbps        {d.get('policy_export_sum_rate_Mbps')}")
    print(f"AoDT_s                {np_list(ev.aodt_s)}")
    print(f"rho                   {np_list(ev.rho)}")
    print(f"feasible              {ev.feasible}")
    print(
        "violations            "
        f"qos={c.qos_violations} aodt={c.aodt_violations} "
        f"sep={c.sep_violations} cpu={c.cpu_unstable_count} "
        f"bw_excess_Hz={c.bw_excess_hz:.4g}"
    )
    print(f"wall_clock_s          {d.get('wall_clock_s')}")
    print("uav_xyz_m")
    print(result.uav_xyz_m)
    return 0


def cmd_sca_multistart(args: argparse.Namespace) -> int:
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
    ms = MultiStartSettings(
        n_random=int(args.multistart_random),
        n_kmeans=int(args.multistart_kmeans),
        include_frozen=not bool(args.no_frozen_start),
    )
    result = solve_sca_multistart(
        scenario, args.seed, settings=settings, multistart=ms
    )
    write_history(result, args.history_json)
    write_history(result, args.history_csv)
    ev = result.true_eval
    c = ev.constraints
    d = result.diagnostics
    print("method                sca_multistart (keep-best extra inits; not headline SCA)")
    print(f"B_sys                 {_fmt_hz(cfg.b_sys_hz)}")
    print(f"n_starts              {d.get('n_starts')}")
    print(f"winner_kind           {d.get('winner_kind')}")
    print(f"winner_init_seed      {d.get('winner_init_seed')}")
    frozen = d.get("frozen_Mbps")
    if frozen is not None:
        print(f"frozen_SCA_Mbps       {float(frozen):.6g}")
        print(f"delta_vs_frozen_Mbps  {float(d.get('delta_vs_frozen_Mbps') or 0.0):+.6g}")
    print(f"true_sum_rate_Mbps    {ev.sum_rate_mbps:.6g}")
    print(f"feasible              {ev.feasible}")
    print(f"stop_reason           {d.get('stop_reason')}")
    print(f"n_iterations          {result.n_iterations}")
    print(
        "violations            "
        f"qos={c.qos_violations} aodt={c.aodt_violations} "
        f"sep={c.sep_violations} cpu={c.cpu_unstable_count} "
        f"bw_excess_Hz={c.bw_excess_hz:.4g}"
    )
    print("uav_xyz_m")
    print(result.uav_xyz_m)
    print(f"history_json          {args.history_json}")
    return 0


def cmd_sca_anchor(args: argparse.Namespace) -> int:
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
    anc = AnchorSettings(
        top_k=int(args.anchor_top_k),
        max_enumerate=int(args.anchor_max_enumerate),
        beam_width=int(args.anchor_beam_width),
        include_frozen=not bool(args.no_frozen_start),
        process_cohesive_candidate=bool(
            getattr(args, "anchor_process_cohesive", False)
        ),
    )
    result = solve_sca_anchor(
        scenario, args.seed, settings=settings, anchor=anc
    )
    write_history(result, args.history_json)
    write_history(result, args.history_csv)
    ev = result.true_eval
    c = ev.constraints
    d = result.diagnostics
    print("method                sca_anchor (zenith-subset + SCA polish; not headline SCA)")
    print(f"B_sys                 {_fmt_hz(cfg.b_sys_hz)}")
    print(f"enum_mode             {d.get('enum_mode')}")
    print(f"n_lp                  {d.get('n_lp')}")
    print(f"winner_kind           {d.get('winner_kind')}")
    print(f"winner_combo          {d.get('winner_combo')}")
    frozen = d.get("frozen_Mbps")
    if frozen is not None:
        print(f"frozen_SCA_Mbps       {float(frozen):.6g}")
        print(f"delta_vs_frozen_Mbps  {float(d.get('delta_vs_frozen_Mbps') or 0.0):+.6g}")
    lp_best = d.get("lp_best_Mbps")
    if lp_best is not None:
        print(f"lp_best_Mbps          {float(lp_best):.6g}")
    print(f"true_sum_rate_Mbps    {ev.sum_rate_mbps:.6g}")
    print(f"feasible              {ev.feasible}")
    print(f"stop_reason           {d.get('stop_reason')}")
    print(f"n_iterations          {result.n_iterations}")
    print(
        "violations            "
        f"qos={c.qos_violations} aodt={c.aodt_violations} "
        f"sep={c.sep_violations} cpu={c.cpu_unstable_count} "
        f"bw_excess_Hz={c.bw_excess_hz:.4g}"
    )
    print("uav_xyz_m")
    print(result.uav_xyz_m)
    print(f"history_json          {args.history_json}")
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
        if n not in KNOWN_METHODS:
            raise ValueError(f"unknown method {n!r}; expected {KNOWN_METHODS}")
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
        td3_settings=_td3_settings_from_args(args),
        multistart_settings=_multistart_from_args(args),
        anchor_settings=_anchor_from_args(args),
    )
    out_path = Path(args.out)
    ckpt = out_path.with_name(out_path.stem + ".checkpoint.json")
    payload = run_campaign(
        _parse_axes(args.axis), cfg, settings, checkpoint_path=ckpt
    )
    out = write_campaign(payload, out_path)
    if ckpt.exists():
        ckpt.unlink()
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


def cmd_n100(args: argparse.Namespace) -> int:
    from uavdt.experiments.n100 import evaluate_bank, write_eval
    from uavdt.experiments.n100_plot import plot_n100_figures
    from uavdt.experiments.scenario_bank import generate_bank, load_bank, write_bank

    if args.bandwidth_preset is None:
        args.bandwidth_preset = "8.8mhz"
    if args.max_bw_share is None:
        args.max_bw_share = PRIMARY_MAX_BW_SHARE
    cfg = config_for_counts(int(args.num_iot), int(args.num_uav), _cfg_from_args(args))
    solver = args.solver
    if solver in {None, "cvxpy", "python", "none"}:
        solver = None

    bank_path = Path(args.bank)
    if args.force_generate or not bank_path.exists():
        bank = generate_bank(
            int(args.n_scenarios),
            cfg,
            seed_start=int(args.seed_start),
            num_iot=int(args.num_iot),
            num_uav=int(args.num_uav),
        )
        write_bank(bank, bank_path)
        print(f"wrote {bank_path}")
        print(f"n_scenarios          {bank['n_scenarios']}")
        geo = bank["geometry"]
        print(
            f"geometry             {geo['area_x_m']:g}x{geo['area_y_m']:g} m  "
            f"I={geo['num_iot']}  J={geo['num_uav']}"
        )
    else:
        bank = load_bank(bank_path)
        print(f"loaded {bank_path}")
        if int(bank["n_scenarios"]) != int(args.n_scenarios):
            print(
                f"note                 bank has {bank['n_scenarios']} scenarios; "
                f"--n-scenarios {args.n_scenarios} ignored"
            )

    if args.generate_only:
        return 0

    payload = None
    out_path = Path(args.out)
    if not args.skip_eval:
        payload = evaluate_bank(
            bank,
            cfg,
            _parse_methods(args.methods),
            sca_settings=SCASettings(
                max_iterations=int(args.max_iterations),
                epsilon=float(args.epsilon),
                step_size_m=float(args.step_size),
                solver=solver,
            ),
            pso_settings=PSOSettings(),
            td3_settings=_td3_settings_from_args(args),
            multistart_settings=_multistart_from_args(args),
            anchor_settings=_anchor_from_args(args),
            checkpoint_path=args.checkpoint,
            resume=not args.no_resume,
            bank_path=bank_path,
        )
        wrote = write_eval(payload, out_path)
        print(f"wrote {wrote}")
        print(f"wrote {wrote.with_suffix('.csv')}")
        print(f"wrote {wrote.with_name(wrote.stem + '_summary.csv')}")
        for method, stats in payload["by_method"].items():
            print(
                f"{method:12s}  mean={stats['mean_sum_rate_Mbps']:.4f} Mbps  "
                f"std={stats['std_sum_rate_Mbps']:.4f}  "
                f"feas={100.0 * stats['feasible_fraction']:.1f}%  "
                f"n={stats['n']}"
            )

    if args.skip_plot:
        return 0
    if payload is None:
        if not out_path.exists():
            print(f"missing eval JSON {out_path}; run without --skip-eval")
            return 2
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    try:
        paths = plot_n100_figures(payload, bank, args.fig_dir)
    except ImportError:
        print("matplotlib is not installed; skip plots (pip install matplotlib)")
        return 0
    for path in paths:
        print(f"wrote {path}")
    return 0


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

    sj = sub.add_parser(
        "sca-joint",
        help="SCA-joint probe: Algorithm 1 plus discrete a_ij/b_ij re-match",
    )
    _add_shared(sj)
    sj.add_argument("--seed", type=int, default=1)
    sj.add_argument("--max-iterations", type=int, default=30)
    sj.add_argument("--epsilon", type=float, default=1e-4)
    sj.add_argument("--step-size", type=float, default=20.0)
    sj.add_argument(
        "--solver",
        type=str,
        default="cvxpy",
        help="CVXPY only (sca_joint has no MATLAB path)",
    )
    sj.add_argument("--history-json", type=str, default="results/sca_joint_history.json")
    sj.add_argument("--history-csv", type=str, default="results/sca_joint_history.csv")
    sj.add_argument(
        "--process-cohesive-candidate",
        action="store_true",
        help=(
            "Also try process-cohesive a_ij as a rematch candidate. "
            "Default off keeps recorded best-SE-only SCA-joint runs."
        ),
    )
    sj.set_defaults(func=cmd_sca_joint)

    sm = sub.add_parser(
        "sca-multistart",
        help="Keep-best extra SCA inits (Experiment A). Does not replace frozen SCA.",
    )
    _add_shared(sm)
    _add_multistart_args(sm)
    sm.add_argument("--seed", type=int, default=1)
    sm.add_argument("--max-iterations", type=int, default=30)
    sm.add_argument("--epsilon", type=float, default=1e-4)
    sm.add_argument("--step-size", type=float, default=20.0)
    sm.add_argument(
        "--solver",
        type=str,
        default="cvxpy",
        help="SCA backend for each start: cvxpy (default) or matlab/MOSEK",
    )
    sm.add_argument(
        "--no-frozen-start",
        action="store_true",
        help="Skip the k-means-seed frozen SCA start (extras only).",
    )
    sm.add_argument(
        "--history-json",
        type=str,
        default="results/sca_multistart_history.json",
    )
    sm.add_argument(
        "--history-csv",
        type=str,
        default="results/sca_multistart_history.csv",
    )
    sm.set_defaults(func=cmd_sca_multistart)

    sa = sub.add_parser(
        "sca-anchor",
        help="Zenith-subset enumeration + SCA polish. Does not replace frozen SCA.",
    )
    _add_shared(sa)
    _add_anchor_args(sa)
    sa.add_argument("--seed", type=int, default=1)
    sa.add_argument("--max-iterations", type=int, default=30)
    sa.add_argument("--epsilon", type=float, default=1e-4)
    sa.add_argument("--step-size", type=float, default=20.0)
    sa.add_argument(
        "--solver",
        type=str,
        default="cvxpy",
        help="SCA backend for frozen start and polish: cvxpy (default) or matlab/MOSEK",
    )
    sa.add_argument(
        "--no-frozen-start",
        action="store_true",
        help="Skip the k-means-seed frozen SCA start (anchors only).",
    )
    sa.add_argument(
        "--history-json",
        type=str,
        default="results/sca_anchor_history.json",
    )
    sa.add_argument(
        "--history-csv",
        type=str,
        default="results/sca_anchor_history.csv",
    )
    sa.set_defaults(func=cmd_sca_anchor)

    td = sub.add_parser(
        "td3",
        help="TD3 (opt-in). Default preset is Algorithm 2; residual-on-sca is the proposed (P) interface.",
    )
    _add_shared(td)
    _add_td3_preset_args(td)
    td.add_argument("--seed", type=int, default=1)
    td.add_argument("--total-steps", type=int, default=7000)
    td.add_argument("--horizon", type=int, default=50)
    td.add_argument("--hidden", type=int, default=256)
    td.add_argument("--batch-size", type=int, default=256)
    td.add_argument("--warmup-steps", type=int, default=256)
    td.add_argument("--buffer-size", type=int, default=100_000)
    td.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Print a TD3 train progress line every N env steps (0=silent)",
    )
    td.add_argument(
        "--export-avg-steps",
        type=int,
        default=10,
        help="Average last N UAV xy of the deterministic eval episode",
    )
    td.set_defaults(func=cmd_td3)

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
    _add_td3_preset_args(camp)
    _add_multistart_args(camp)
    _add_anchor_args(camp)
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
        help="Comma-separated: random,kmeans,pso,sca[,sca_joint][,sca_multistart][,sca_anchor][,td3]. Opt-in: sca_joint, sca_multistart, sca_anchor, td3.",
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

    n100 = sub.add_parser(
        "n100",
        help="Generate 100 area+IoT+UAV scenarios, run SCA/baselines, plot averages",
    )
    _add_shared(n100)
    _add_td3_preset_args(n100)
    _add_multistart_args(n100)
    _add_anchor_args(n100)
    n100.add_argument("--n-scenarios", type=int, default=100)
    n100.add_argument("--seed-start", type=int, default=1)
    n100.add_argument("--num-iot", type=int, default=10)
    n100.add_argument("--num-uav", type=int, default=3)
    n100.add_argument(
        "--bank",
        type=str,
        default="data/scenario_bank/n100_i10_j3_100m.json",
        help="Saved area+IoT+UAV layouts",
    )
    n100.add_argument("--out", type=str, default="results/n100/eval.json")
    n100.add_argument("--fig-dir", type=str, default="results/figures/n100")
    n100.add_argument(
        "--checkpoint",
        type=str,
        default="results/n100/eval.checkpoint.json",
    )
    n100.add_argument(
        "--methods",
        type=str,
        default="random,kmeans,pso,sca",
        help="Comma-separated: random,kmeans,pso,sca[,sca_joint][,sca_multistart][,sca_anchor][,td3]",
    )
    n100.add_argument("--max-iterations", type=int, default=30)
    n100.add_argument("--epsilon", type=float, default=1e-4)
    n100.add_argument("--step-size", type=float, default=20.0)
    n100.add_argument(
        "--solver",
        type=str,
        default="cvxpy",
        help="SCA backend: cvxpy (default) or matlab/MOSEK",
    )
    n100.add_argument("--generate-only", action="store_true")
    n100.add_argument("--skip-eval", action="store_true")
    n100.add_argument("--skip-plot", action="store_true")
    n100.add_argument("--force-generate", action="store_true")
    n100.add_argument("--no-resume", action="store_true")
    n100.set_defaults(func=cmd_n100)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
