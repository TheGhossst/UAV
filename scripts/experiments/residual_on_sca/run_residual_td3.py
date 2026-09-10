"""Experiment B: residual policy on SCA vs frozen SCA.

Held-out seeds 21–40 at primary 8.8 MHz / 25% cap. Same evaluate().
Official TD3 score is the incumbent best snapshot (feasibility then rate)
with inner frozen-q LP and frozen SCA a,b.

Usage:
  python scripts/experiments/residual_on_sca/run_residual_td3.py
  python scripts/experiments/residual_on_sca/run_residual_td3.py --n-runs 1 --total-steps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from _paths import OUT_RESIDUAL_TD3, PROTECTED, bootstrap  # noqa: E402

bootstrap()

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402
from uavdt.td3.settings import TD3Settings  # noqa: E402

OUT_DEFAULT = OUT_RESIDUAL_TD3
PRACTICAL_MBPS = 0.05


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _cfg() -> SimConfig:
    return SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _run_record(run) -> dict:
    ev = run.true_eval
    return {
        "method": run.method,
        "seed": int(run.seed),
        "sum_rate_Mbps": float(ev.sum_rate_mbps),
        "feasible": bool(ev.feasible),
        "uav_xyz_m": np.asarray(run.uav_xyz_m, dtype=float).tolist(),
        "diagnostics": _jsonable(run.diagnostics),
    }


def summarize(rows: list[dict]) -> dict:
    sca = np.array([r["sca"]["sum_rate_Mbps"] for r in rows], dtype=float)
    td3 = np.array([r["td3"]["sum_rate_Mbps"] for r in rows], dtype=float)
    deltas = td3 - sca
    seeds = [int(r["seed"]) for r in rows]
    wins = [s for s, d in zip(seeds, deltas) if d > 1e-9]
    losses = [s for s, d in zip(seeds, deltas) if d < -1e-9]
    worse_bar = [int(s) for s, d in zip(seeds, deltas) if d < -PRACTICAL_MBPS]
    w = wilcoxon_signed_rank(deltas)
    tstat = paired_t(deltas)
    mean_d = float(np.mean(deltas))
    if mean_d < 0.0:
        readout = "failure_mean_negative"
    elif w["p_greater"] < 0.05 and mean_d > 0.0:
        readout = "real_win"
    else:
        readout = "neck_and_neck"
    return {
        "n": len(rows),
        "seeds": seeds,
        "mean_sca_Mbps": float(np.mean(sca)),
        "mean_td3_Mbps": float(np.mean(td3)),
        "mean_delta_Mbps": mean_d,
        "std_delta_Mbps": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
        "min_delta_Mbps": float(np.min(deltas)) if deltas.size else 0.0,
        "max_delta_Mbps": float(np.max(deltas)) if deltas.size else 0.0,
        "n_td3_wins": len(wins),
        "n_td3_losses": len(losses),
        "win_rate": float(len(wins) / max(len(rows), 1)),
        "win_seeds": wins,
        "loss_seeds": losses,
        "seeds_worse_than_0p05": worse_bar,
        "wilcoxon": w,
        "paired_t": tstat,
        "readout": readout,
        "construction_broken": bool(len(worse_bar) > 0 or mean_d < -1e-6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment B: residual-on-SCA vs frozen SCA")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=21)
    parser.add_argument("--total-steps", type=int, default=7000)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.resolve() == PROTECTED.resolve():
        raise SystemExit(f"refusing to overwrite protected {PROTECTED}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _cfg()
    td3_settings = TD3Settings.residual_on_sca(
        total_steps=int(args.total_steps),
        log_every=int(args.log_every),
    )
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.n_runs)))
    _log(
        f"Experiment B  seeds={seeds[0]}..{seeds[-1]}  "
        f"B_sys=8.8 MHz  cap=25%  td3_steps={args.total_steps}  "
        f"preset=residual-on-sca"
    )
    rows = []
    for seed in seeds:
        sc = generate_scenario(seed, cfg)
        t0 = perf_counter()
        sca_run = run_method(sc, "sca", seed)
        _log(
            f"  seed {seed}  sca={sca_run.sum_rate_mbps:.4f} Mbps  "
            f"feas={sca_run.feasible}  {perf_counter() - t0:.1f}s"
        )
        t1 = perf_counter()
        td3_run = run_method(sc, "td3", seed, td3_settings=td3_settings)
        d = td3_run.sum_rate_mbps - sca_run.sum_rate_mbps
        origin = td3_run.diagnostics.get("origin_sum_rate_Mbps")
        _log(
            f"    td3={td3_run.sum_rate_mbps:.4f}  origin={origin}  "
            f"d={d:+.4f} Mbps  {perf_counter() - t1:.1f}s  "
            f"export={td3_run.diagnostics.get('export_mode')}"
        )
        rows.append(
            {
                "seed": int(seed),
                "delta_td3_minus_sca_Mbps": float(d),
                "sca": _run_record(sca_run),
                "td3": _run_record(td3_run),
            }
        )
    summary = summarize(rows)
    payload = {
        "experiment": "B",
        "label": "residual-on-SCA vs frozen SCA (same evaluate, incumbent export)",
        "held_out": bool(int(args.seed_start) == 21 and int(args.n_runs) == 20),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "n_runs": int(args.n_runs),
        "seed_start": int(args.seed_start),
        "total_steps": int(args.total_steps),
        "td3_preset": "residual-on-sca",
        "per_seed": rows,
        "summary": summary,
        "notes": [
            "Does not overwrite campaign_8.8mhz_cap25_si12k.json.",
            "Will not produce a 1 Mbps win at 100 m / 25% cap.",
            "Will not fix T_k=0.8 s (same frozen nearest-a).",
            "Seeds 1–20 are Experiment A / the published SCA table; 21–40 is the claim split.",
        ],
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    _log(f"wrote {out}")
    s = summary
    _log(
        f"mean d TD3-SCA = {s['mean_delta_Mbps']:+.4f} +/- {s['std_delta_Mbps']:.4f} Mbps  "
        f"min={s['min_delta_Mbps']:+.4f} max={s['max_delta_Mbps']:+.4f}  "
        f"wins={s['n_td3_wins']}/{s['n']}  "
        f"Wilcoxon p_greater={s['wilcoxon']['p_greater']:.4g}  "
        f"readout={s['readout']}"
    )
    if s["seeds_worse_than_0p05"]:
        _log(f"CONSTRUCTION BROKEN: seeds >0.05 Mbps worse: {s['seeds_worse_than_0p05']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
