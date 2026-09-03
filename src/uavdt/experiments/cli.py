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
    SimConfig,
)
from uavdt.experiments import run_one, run_seeds


def _cfg_from_args(args: argparse.Namespace) -> SimConfig:
    b_hz = args.bandwidth
    if args.bandwidth_preset is not None:
        b_hz = BANDWIDTH_PRESETS[args.bandwidth_preset]
    task_bits = args.task_size_bits
    if args.task_size_bytes is not None:
        task_bits = args.task_size_bytes * 8.0
    return SimConfig(
        b_sys_hz=float(b_hz),
        task_size_bits=float(task_bits),
        task_cycles=float(args.task_cycles),
        lambda_i_per_s=float(args.lambda_i),
        aodt_threshold_s=float(args.aodt_threshold),
        los_angle_unit=args.los_angle_unit,
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
    p.add_argument("--los-angle-unit", choices=("rad", "deg"), default="rad")
    p.add_argument("--placement", choices=("random", "kmeans"), default="random")


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
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
