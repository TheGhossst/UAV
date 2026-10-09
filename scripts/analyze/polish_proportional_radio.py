"""Polish UAV positions under bandwidth-proportional noise.

Nelder–Mead, about 200 inner solves each, started once from k-means and
once from Euclidean p-median. The inner solver is
``solve_bandwidth_proportional``. Frozen Algorithm 1 is not used.

If the polished p-median start still beats the polished k-means start,
the starting point matters under this radio as well as under constant
sigma^2.

    python scripts/analyze/polish_proportional_radio.py --n-runs 8 --areas 500 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402
from uavdt.analysis.bw_proportional import (  # noqa: E402
    calibrated_noise_density_w_per_hz,
    equal_share_reference_hz,
)
from uavdt.analysis.prop_polish import (  # noqa: E402
    kmeans_start,
    pmedian_start,
    polish_uav_positions,
)
from uavdt.config import headline_sim_config  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PRACTICAL_MBPS = 0.05


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _pair(rows: list[dict], left: str, right: str, key: str) -> dict:
    deltas = []
    wins = losses = ties = practical = 0
    for row in rows:
        a = row[left]
        b = row[right]
        if not (a["feasible"] and b["feasible"]):
            continue
        if not (np.isfinite(a[key]) and np.isfinite(b[key])):
            continue
        d = float(a[key]) - float(b[key])
        deltas.append(d)
        if d > 1e-6:
            wins += 1
        elif d < -1e-6:
            losses += 1
        else:
            ties += 1
        if d >= PRACTICAL_MBPS:
            practical += 1
    arr = np.asarray(deltas, dtype=float)
    wil = wilcoxon_signed_rank(arr) if arr.size else {
        "p_two_sided": 1.0,
        "p_greater": 1.0,
    }
    return {
        "n_paired": int(arr.size),
        "mean_delta_mbps": float(arr.mean()) if arr.size else float("nan"),
        "std_delta_mbps": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "practical_wins_0.05": practical,
        "wilcoxon_p_two_sided": wil["p_two_sided"],
        "wilcoxon_p_greater": wil["p_greater"],
    }


def _verdict(polished: dict) -> dict:
    mean = polished["mean_delta_mbps"]
    if polished["n_paired"] == 0 or not np.isfinite(mean):
        code = "no_paired_feasible"
        text = "No seed had both polished starts feasible."
    elif mean >= PRACTICAL_MBPS and polished["wins"] > polished["losses"]:
        code = "start_still_matters"
        text = (
            "After local polish, p-median still leads k-means by at least "
            f"{PRACTICAL_MBPS:.2f} Mbps. The starting point matters under "
            "proportional noise."
        )
    elif mean > 0.0 and polished["wins"] > polished["losses"]:
        code = "start_ahead_but_small"
        text = (
            "Polished p-median still leads polished k-means, but the mean "
            f"gap is under {PRACTICAL_MBPS:.2f} Mbps."
        )
    else:
        code = "start_gap_closed"
        text = (
            "Local polish removes the p-median lead over k-means. Under this "
            "radio the starting point does not keep a practical gain."
        )
    return {"code": code, "text": text, **polished}


def _one(scenario, seed: int, max_evals: int, step_m: float) -> dict:
    out = {"seed": int(seed)}
    for name, starter in (("kmeans", kmeans_start), ("pmedian", pmedian_start)):
        t0 = time.perf_counter()
        origin = starter(scenario, seed)
        polished = polish_uav_positions(
            scenario,
            origin,
            max_evals=max_evals,
            step_m=step_m,
        )
        out[name] = {
            "start_mbps": polished.start_rate_mbps,
            "start_feasible": polished.start_feasible,
            "polished_mbps": polished.rate_mbps,
            "feasible": polished.feasible,
            "n_evals": polished.n_evals,
            "moved_rms_m": polished.moved_rms_m,
            "seconds": time.perf_counter() - t0,
        }
        print(
            f"  {name:7} {polished.start_rate_mbps:.3f} -> {polished.rate_mbps:.3f} Mbps"
            f"  move {polished.moved_rms_m:.1f} m  evals {polished.n_evals}",
            flush=True,
        )
    return out


def _mean(rows: list[dict], method: str, key: str, feas: str = "feasible") -> float:
    vals = [
        float(row[method][key])
        for row in rows
        if row[method][feas] and np.isfinite(row[method][key])
    ]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def run_area(area_m: float, seeds: list[int], max_evals: int, step_m: float) -> dict:
    cfg = headline_sim_config(area_m=area_m)
    rows = []
    for k, seed in enumerate(seeds, start=1):
        print(f"area {area_m:.0f} m  seed {seed} ({k}/{len(seeds)})", flush=True)
        rows.append(_one(generate_scenario(seed, cfg), seed, max_evals, step_m))
    # Pair on polished rates. The row shape expected by _pair uses nested dicts
    # with feasible + the rate key.
    paired_rows = []
    for row in rows:
        paired_rows.append(
            {
                "pmedian": {
                    "feasible": row["pmedian"]["feasible"],
                    "rate_mbps": row["pmedian"]["polished_mbps"],
                },
                "kmeans": {
                    "feasible": row["kmeans"]["feasible"],
                    "rate_mbps": row["kmeans"]["polished_mbps"],
                },
                "pmedian_start": {
                    "feasible": row["pmedian"]["start_feasible"],
                    "rate_mbps": row["pmedian"]["start_mbps"],
                },
                "kmeans_start": {
                    "feasible": row["kmeans"]["start_feasible"],
                    "rate_mbps": row["kmeans"]["start_mbps"],
                },
            }
        )
    polished = _verdict(_pair(paired_rows, "pmedian", "kmeans", "rate_mbps"))
    raw = _pair(paired_rows, "pmedian_start", "kmeans_start", "rate_mbps")
    print(
        f"  polished p-median - k-means {polished['mean_delta_mbps']:+.3f} Mbps"
        f"  {polished['wins']}/{polished['n_paired']}  {polished['text']}",
        flush=True,
    )
    return {
        "area_m": float(area_m),
        "b_eq_hz": equal_share_reference_hz(cfg),
        "n0_w_per_hz": calibrated_noise_density_w_per_hz(cfg),
        "seeds": rows,
        "mean_polished_mbps": {
            "pmedian": _mean(rows, "pmedian", "polished_mbps"),
            "kmeans": _mean(rows, "kmeans", "polished_mbps"),
        },
        "mean_start_mbps": {
            "pmedian": _mean(rows, "pmedian", "start_mbps", "start_feasible"),
            "kmeans": _mean(rows, "kmeans", "start_mbps", "start_feasible"),
        },
        "polished_pmedian_minus_kmeans": polished,
        "raw_pmedian_minus_kmeans": raw,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--areas", type=float, nargs="+", default=[500.0, 100.0])
    p.add_argument("--max-evals", type=int, default=200)
    p.add_argument("--step-m", type=float, default=20.0)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "polish_proportional.json",
    )
    args = p.parse_args(argv)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_runs)))
    areas = [
        run_area(float(area), seeds, int(args.max_evals), float(args.step_m))
        for area in args.areas
    ]
    payload = {
        "optimizer": "Nelder-Mead",
        "max_evals": int(args.max_evals),
        "step_m": float(args.step_m),
        "inner": "solve_bandwidth_proportional",
        "starts": ["kmeans", "pmedian"],
        "areas": areas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
