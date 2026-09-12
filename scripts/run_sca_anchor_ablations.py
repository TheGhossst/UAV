"""Cheap zenith-anchor ablations: bound, random-anchor, beam, K, cap, deg.

Does not overwrite headline 25% campaigns or n100/eval.json.

Usage:
  python scripts/run_sca_anchor_ablations.py
  python scripts/run_sca_anchor_ablations.py --only bound,random,beam
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sca_anchor_eval import run_n20_eval  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.n100 import evaluate_bank, write_eval  # noqa: E402
from uavdt.experiments.scenario_bank import load_bank, scenario_from_record  # noqa: E402
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings, leftover_dump_upper_bound  # noqa: E402

OUT = ROOT / "results" / "sca_anchor_ablations"
PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
}


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _refuse(path: Path) -> None:
    resolved = path.resolve()
    for p in PROTECTED:
        if resolved == p.resolve():
            raise SystemExit(f"refusing to overwrite protected {p}")


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


def _write(path: Path, payload: dict) -> None:
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    _log(f"wrote {path}")


def _complete_n20(path: Path, n: int = 20) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("n_runs") or 0) == n and "sca_anchor" in (
        payload.get("mean_Mbps") or {}
    )


def _complete_n100(path: Path, n: int, method: str = "sca_anchor") -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    ms = (payload.get("by_method") or {}).get(method) or {}
    return int(ms.get("n") or 0) == int(n)


def run_bound() -> dict:
    out = OUT / "bound_n100_500m.json"
    if out.exists():
        _log(f"skip complete bound {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    bank = load_bank(ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json")
    overlay = SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)
    from uavdt.experiments.scenario_bank import eval_cfg_from_bank

    cfg = eval_cfg_from_bank(bank, overlay)
    eval_path = ROOT / "results" / "n100_500m_cap25" / "eval_anchor.json"
    rates: dict[int, dict] = {}
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        by = ev.get("by_method") or {}
        seeds = [int(s) for s in (by.get("sca_anchor") or {}).get("seeds") or []]
        for i, seed in enumerate(seeds):
            rates[seed] = {
                "anchor": float(by["sca_anchor"]["per_seed_Mbps"][i]),
                "sca": float(by["sca"]["per_seed_Mbps"][i]) if "sca" in by else None,
            }
    rows = []
    t0 = perf_counter()
    for rec in bank["scenarios"]:
        sc = scenario_from_record(rec, cfg)
        bound = leftover_dump_upper_bound(sc)
        seed = int(rec["seed"])
        got = rates.get(seed) or {}
        row = {
            "seed": seed,
            "bound_Mbps": bound["bound_Mbps"],
            "radio_ceiling_Mbps": bound["radio_ceiling_Mbps"],
            "se_zenith": bound["se_zenith"],
            "k_dump": bound["k_dump"],
            "combo": list(bound["combo"]) if bound["combo"] is not None else None,
            "anchor_Mbps": got.get("anchor"),
            "sca_Mbps": got.get("sca"),
        }
        if row["anchor_Mbps"] is not None:
            row["gap_anchor_Mbps"] = bound["bound_Mbps"] - float(row["anchor_Mbps"])
            row["gap_anchor_pct"] = 100.0 * row["gap_anchor_Mbps"] / bound["bound_Mbps"]
            row["closed_of_sca_gap"] = None
            if row["sca_Mbps"] is not None:
                denom = bound["bound_Mbps"] - float(row["sca_Mbps"])
                row["closed_of_sca_gap"] = (
                    None
                    if abs(denom) < 1e-12
                    else (float(row["anchor_Mbps"]) - float(row["sca_Mbps"])) / denom
                )
        rows.append(row)
    gaps = [r["gap_anchor_Mbps"] for r in rows if r.get("gap_anchor_Mbps") is not None]
    pcts = [r["gap_anchor_pct"] for r in rows if r.get("gap_anchor_pct") is not None]
    closed = [
        r["closed_of_sca_gap"]
        for r in rows
        if r.get("closed_of_sca_gap") is not None
    ]
    payload = {
        "label": "leftover-dump upper bound on n100 500 m / 25%",
        "n": len(rows),
        "mean_bound_Mbps": float(np.mean([r["bound_Mbps"] for r in rows])),
        "mean_anchor_Mbps": (
            float(np.mean([r["anchor_Mbps"] for r in rows if r["anchor_Mbps"] is not None]))
            if any(r["anchor_Mbps"] is not None for r in rows)
            else None
        ),
        "mean_sca_Mbps": (
            float(np.mean([r["sca_Mbps"] for r in rows if r["sca_Mbps"] is not None]))
            if any(r["sca_Mbps"] is not None for r in rows)
            else None
        ),
        "mean_gap_anchor_Mbps": float(np.mean(gaps)) if gaps else None,
        "mean_gap_anchor_pct": float(np.mean(pcts)) if pcts else None,
        "median_gap_anchor_pct": float(np.median(pcts)) if pcts else None,
        "mean_closed_of_sca_gap": float(np.mean(closed)) if closed else None,
        "n_anchor_above_bound": int(
            sum(1 for r in rows if r.get("gap_anchor_Mbps") is not None and r["gap_anchor_Mbps"] < -1e-6)
        ),
        "wall_s": perf_counter() - t0,
        "per_seed": rows,
    }
    _write(out, payload)
    _log(
        f"  bound {payload['mean_bound_Mbps']:.4f}  "
        f"anchor {payload['mean_anchor_Mbps']}  "
        f"gap {payload['mean_gap_anchor_pct']} %  "
        f"closed {payload['mean_closed_of_sca_gap']}"
    )
    return payload


def run_random_n20() -> dict:
    out = OUT / "random_n20_500m.json"
    return run_n20_eval(
        out=out,
        area_m=500.0,
        skip_if_complete=True,
        campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
        multistart_path=ROOT / "results" / "sca_multistart_n20_500m.json",
        anchor_settings=AnchorSettings(
            selection="random", n_random=1, top_k=1, include_frozen=True
        ),
    )


def run_random_n20_100m() -> dict:
    out = OUT / "random_n20_100m.json"
    return run_n20_eval(
        out=out,
        area_m=100.0,
        skip_if_complete=True,
        campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
        multistart_path=ROOT / "results" / "sca_multistart_n20.json",
        anchor_settings=AnchorSettings(
            selection="random", n_random=1, top_k=1, include_frozen=True
        ),
    )


def _n100_anchor_only(
    *,
    bank_path: Path,
    src_ckpt: Path | None,
    out: Path,
    checkpoint: Path,
    overlay: SimConfig,
    anchor: AnchorSettings,
    methods: tuple[str, ...] = ("sca_anchor",),
) -> dict:
    _refuse(out)
    bank = load_bank(bank_path)
    n = int(bank["n_scenarios"])
    if _complete_n100(out, n):
        _log(f"skip complete n100 {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if (
        src_ckpt is not None
        and src_ckpt.exists()
        and not checkpoint.exists()
    ):
        shutil.copy2(src_ckpt, checkpoint)
        _log(f"copied checkpoint {src_ckpt.name} -> {checkpoint}")
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        overlay,
        methods,
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
        anchor_settings=anchor,
        checkpoint_path=checkpoint,
        resume=True,
        bank_path=bank_path,
    )
    if src_ckpt is not None and src_ckpt.exists():
        src = json.loads(src_ckpt.read_text(encoding="utf-8"))
        for m, stats in (src.get("by_method") or {}).items():
            payload.setdefault("by_method", {}).setdefault(m, stats)
        payload["methods"] = list(payload.get("by_method") or payload.get("methods") or [])
    write_eval(payload, out)
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")
    for method, stats in payload["by_method"].items():
        _log(
            f"  {method:16s}  mean={stats['mean_sum_rate_Mbps']:.4f} Mbps  "
            f"n={stats['n']}"
        )
    return payload


def run_random_n100() -> dict:
    # Baselines already in eval.json; only run random-anchor.
    src = ROOT / "results" / "n100_500m_cap25" / "eval.json"
    return _n100_anchor_only(
        bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
        src_ckpt=src,
        out=OUT / "random_n100_500m.json",
        checkpoint=OUT / "random_n100_500m.checkpoint.json",
        overlay=SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE),
        anchor=AnchorSettings(
            selection="random", n_random=1, top_k=1, include_frozen=True
        ),
    )


def run_beam_n20(*, area_m: float) -> dict:
    tag = "100m" if area_m < 200 else "500m"
    camp = (
        ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"
        if area_m < 200
        else ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json"
    )
    ms = (
        ROOT / "results" / "sca_multistart_n20.json"
        if area_m < 200
        else ROOT / "results" / "sca_multistart_n20_500m.json"
    )
    return run_n20_eval(
        out=OUT / f"beam_n20_{tag}.json",
        area_m=area_m,
        skip_if_complete=True,
        campaign_path=camp,
        multistart_path=ms,
        anchor_settings=AnchorSettings(max_enumerate=0, beam_width=10, top_k=3),
    )


def run_k10_n20() -> dict:
    return run_n20_eval(
        out=OUT / "k10_n20_500m.json",
        area_m=500.0,
        skip_if_complete=True,
        campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
        multistart_path=ROOT / "results" / "sca_multistart_n20_500m.json",
        anchor_settings=AnchorSettings(top_k=10),
    )


def run_cap_cell(
    *,
    area_m: float,
    share: float | None,
    campaign: Path | None,
) -> dict:
    cap = "none" if share is None else f"{share:.0%}"
    tag = "100m" if area_m < 200 else "500m"
    out = OUT / f"cap_{cap}_{tag}.json"
    return run_n20_eval(
        out=out,
        area_m=area_m,
        max_bw_share=share,
        skip_if_complete=True,
        campaign_path=campaign if campaign and campaign.exists() else None,
        anchor_settings=AnchorSettings(),
        run_baselines=True,
    )


def run_cap_family() -> dict:
    cells = []
    specs = [
        (100.0, 0.15, ROOT / "results" / "campaign_8.8mhz_cap15_n20.json"),
        (100.0, 0.12, ROOT / "results" / "campaign_8.8mhz_cap12_n20.json"),
        (100.0, None, ROOT / "results" / "campaign_8.8mhz_n20.json"),
        (500.0, 0.15, ROOT / "results" / "campaign_8.8mhz_cap15_n20_500m.json"),
        (500.0, 0.12, ROOT / "results" / "campaign_8.8mhz_cap12_n20_500m.json"),
        (500.0, None, ROOT / "results" / "campaign_8.8mhz_nocap_n20_500m.json"),
    ]
    for area, share, camp in specs:
        _log(f"=== cap family area={area:g} share={share} ===")
        payload = run_cap_cell(area_m=area, share=share, campaign=camp)
        cells.append(
            {
                "area_m": area,
                "max_bw_share": share,
                "means_Mbps": payload.get("mean_Mbps"),
                "path": str(
                    OUT
                    / f"cap_{'none' if share is None else f'{share:.0%}'}_{'100m' if area < 200 else '500m'}.json"
                ),
            }
        )
    # n100 500 m at 12% (the PSO-beats-SCA cell) if a baseline eval exists
    n100_12 = ROOT / "results" / "n100_500m_cap12" / "eval.json"
    if n100_12.exists():
        _log("=== n100 500 m cap 12% sca_anchor ===")
        payload = _n100_anchor_only(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            src_ckpt=n100_12,
            out=OUT / "cap12_n100_500m.json",
            checkpoint=OUT / "cap12_n100_500m.checkpoint.json",
            overlay=SimConfig(b_sys_hz=8.8e6, max_bw_share=0.12),
            anchor=AnchorSettings(),
        )
        cells.append(
            {
                "area_m": 500.0,
                "max_bw_share": 0.12,
                "n": 100,
                "means_Mbps": {
                    m: s["mean_sum_rate_Mbps"]
                    for m, s in payload["by_method"].items()
                },
                "path": str(OUT / "cap12_n100_500m.json"),
            }
        )
    n100_15 = ROOT / "results" / "n100_500m_cap15" / "eval.json"
    if n100_15.exists():
        _log("=== n100 500 m cap 15% sca_anchor ===")
        payload = _n100_anchor_only(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            src_ckpt=n100_15,
            out=OUT / "cap15_n100_500m.json",
            checkpoint=OUT / "cap15_n100_500m.checkpoint.json",
            overlay=SimConfig(b_sys_hz=8.8e6, max_bw_share=0.15),
            anchor=AnchorSettings(),
        )
        cells.append(
            {
                "area_m": 500.0,
                "max_bw_share": 0.15,
                "n": 100,
                "means_Mbps": {
                    m: s["mean_sum_rate_Mbps"]
                    for m, s in payload["by_method"].items()
                },
                "path": str(OUT / "cap15_n100_500m.json"),
            }
        )
    summary = {"label": "cap family (anchor + reused baselines)", "cells": cells}
    _write(OUT / "cap_family.json", summary)
    return summary


def run_deg_n20() -> dict:
    return run_n20_eval(
        out=OUT / "deg_n20_500m.json",
        area_m=500.0,
        los_angle_unit="deg",
        skip_if_complete=True,
        anchor_settings=AnchorSettings(),
        run_baselines=True,
    )


def run_deg_n100() -> dict:
    return _n100_anchor_only(
        bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
        src_ckpt=None,
        out=OUT / "deg_n100_500m.json",
        checkpoint=OUT / "deg_n100_500m.checkpoint.json",
        overlay=SimConfig(
            b_sys_hz=8.8e6,
            max_bw_share=PRIMARY_MAX_BW_SHARE,
            los_angle_unit="deg",
        ),
        anchor=AnchorSettings(),
        methods=("random", "kmeans", "pso", "sca", "sca_anchor"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma list: bound,random,beam,k,cap,deg",
    )
    args = parser.parse_args(argv)
    wanted = {x.strip() for x in str(args.only).split(",") if x.strip()} or {
        "bound",
        "random",
        "beam",
        "k",
        "cap",
        "deg",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    if "bound" in wanted:
        _log("=== leftover-dump bound ===")
        run_bound()
    if "random" in wanted:
        _log("=== random-anchor control ===")
        run_random_n20_100m()
        run_random_n20()
        run_random_n100()
    if "beam" in wanted:
        _log("=== beam vs exhaustive (I=10) ===")
        run_beam_n20(area_m=100.0)
        run_beam_n20(area_m=500.0)
    if "k" in wanted:
        _log("=== K=10 polish ===")
        run_k10_n20()
    if "cap" in wanted:
        _log("=== cap family ===")
        run_cap_family()
    if "deg" in wanted:
        _log("=== degrees LoS ===")
        run_deg_n20()
        run_deg_n100()
    _log(f"ablations done in {perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
