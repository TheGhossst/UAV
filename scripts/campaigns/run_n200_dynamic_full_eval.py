"""Full n200 bank eval: baselines + frozen SCA + dynamic SCA (from scratch).

Usage:
  python scripts/campaigns/run_n200_dynamic_full_eval.py
  python scripts/campaigns/run_n200_dynamic_full_eval.py --only 500m
  python scripts/campaigns/run_n200_dynamic_full_eval.py --from-scratch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.grids import config_for_counts
from uavdt.experiments.n100 import evaluate_bank, write_eval
from uavdt.experiments.scenario_bank import load_bank
from uavdt.placement.pso import PSOSettings
from uavdt.sca.settings import SCASettings

METHODS = ("random", "kmeans", "pso", "frozen_sca", "sca")
N_SCENARIOS = 200

CASES = {
    "500m": {
        "bank": ROOT / "data" / "scenario_bank" / "n200_i10_j3_500m.json",
        "area_m": 500.0,
        "out": ROOT / "results" / "n200_500m_cap25" / "eval_all_methods_dynamic.json",
        "checkpoint": ROOT
        / "results"
        / "n200_500m_cap25"
        / "eval_all_methods_dynamic.checkpoint.json",
    },
    "100m": {
        "bank": ROOT / "data" / "scenario_bank" / "n200_i10_j3_100m.json",
        "area_m": 100.0,
        "out": ROOT / "results" / "n200" / "eval_all_methods_dynamic.json",
        "checkpoint": ROOT / "results" / "n200" / "eval_all_methods_dynamic.checkpoint.json",
    },
}


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _print_summary(payload: dict) -> None:
    by = payload.get("by_method") or {}
    _log("")
    _log("=== mean sum rate (Mbps) ===")
    order = [m for m in METHODS if m in by] + [m for m in by if m not in METHODS]
    for method in order:
        stats = by[method]
        _log(
            f"  {method:14s}  {stats['mean_sum_rate_Mbps']:.4f} "
            f"+/- {stats['std_sum_rate_Mbps']:.4f}  "
            f"feas={100.0 * stats['feasible_fraction']:.1f}%  n={stats['n']}"
        )
    pair = payload.get("sca_minus_frozen_sca_Mbps") or {}
    if pair:
        _log("")
        _log("=== SCA - frozen-SCA (paired) ===")
        for k in (
            "mean_delta_Mbps",
            "median_delta_Mbps",
            "std_delta_Mbps",
            "win_fraction",
            "loss_fraction",
            "practical_win_fraction",
            "practical_loss_fraction",
        ):
            if k in pair:
                _log(f"  {k}: {pair[k]}")
    vs = payload.get("sca_minus_baseline_Mbps") or {}
    if vs:
        _log("")
        _log("=== SCA minus each baseline (mean Mbps) ===")
        for name, row in vs.items():
            _log(f"  vs {name:12s}  {row['mean_delta_Mbps']:+.4f}  win={row['win_fraction']:.2f}")


def run_case(key: str, *, from_scratch: bool) -> dict:
    case = CASES[key]
    bank_path = case["bank"]
    if not bank_path.exists():
        raise SystemExit(f"missing bank {bank_path}")
    bank = load_bank(bank_path)
    out = case["out"]
    ckpt = case["checkpoint"]
    if from_scratch:
        for p in (out, ckpt, out.with_suffix(".csv"), out.with_name(out.stem + "_summary.csv")):
            if p.exists():
                p.unlink()
                _log(f"removed {p.relative_to(ROOT)}")
    overlay = config_for_counts(
        10,
        3,
        SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE),
    ).with_square_area_m(float(case["area_m"]))
    _log(f"case {key}  bank={bank_path.name}  methods={list(METHODS)}")
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        overlay,
        METHODS,
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
        checkpoint_path=ckpt,
        resume=not from_scratch,
        bank_path=bank_path,
    )
    payload["eval_tag"] = f"n200_{key}_all_methods_dynamic"
    write_eval(payload, out)
    _log(f"wrote {out.relative_to(ROOT)}  ({perf_counter() - t0:.1f}s)")
    _print_summary(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default="500m",
        help="Comma list: 100m,500m (default 500m)",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Delete checkpoint/out and rerun all method x seed jobs",
    )
    args = parser.parse_args(argv)
    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    for key in keys:
        if key not in CASES:
            raise SystemExit(f"unknown case {key!r}; expected {list(CASES)}")
        run_case(key, from_scratch=bool(args.from_scratch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
