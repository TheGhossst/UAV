"""P-median minus k-means SCA versus field side / H.

Two families go on one axis:

- vary H in {30, 50, 100, 200, 300} m at a 500 m field
- vary the field side in {100, 200, 300, 500, 800} m at H = 100 m

K-means SCA is frozen Algorithm 1 (constant sigma^2). Its bandwidth, and
the p-median layout's bandwidth, are then re-solved with
``solve_bandwidth_proportional``. The same positions are re-scored at
B_eq scales 0.1, 1, and 10 (N0 moves as 1/B_eq).

The prediction is that the gain rises once the field side passes about
2H. A vertical line at side/H = 2 marks that threshold.

    python scripts/analyze/mechanism_side_over_h.py --n-runs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.analysis.prop_polish import (  # noqa: E402
    nominal_b_eq_hz,
    pmedian_start,
    score_layout,
)
from uavdt.config import headline_sim_config  # noqa: E402
from uavdt.sca.algorithm import solve_sca  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _mean_std(vals: list[float]) -> tuple[float, float, int]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), sd, int(arr.size)


def _sca_positions(scenario, seed: int, settings: SCASettings):
    try:
        result = solve_sca(scenario, seed, settings=settings)
    except RuntimeError:
        return None
    return (
        result.uav_xyz_m,
        result.allocation.hard_association(),
        result.allocation.hard_processing(),
    )


def _gain(p_rate: float, p_ok: bool, s_rate: float, s_ok: bool) -> float:
    if not (p_ok and s_ok and np.isfinite(p_rate) and np.isfinite(s_rate)):
        return float("nan")
    return float(p_rate) - float(s_rate)


def run_geometry(
    *,
    side_m: float,
    height_m: float,
    family: str,
    seeds: list[int],
    scales: list[float],
    settings: SCASettings,
) -> dict:
    cfg = replace(
        headline_sim_config(area_m=side_m),
        uav_height_m=float(height_m),
    )
    per_scale: dict[float, list[float]] = {float(s): [] for s in scales}
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        scenario = generate_scenario(seed, cfg)
        p_uav = pmedian_start(scenario, seed)
        sca = _sca_positions(scenario, seed, settings)
        seed_row: dict = {"seed": int(seed), "scales": {}}
        if sca is None:
            for scale in scales:
                per_scale[float(scale)].append(float("nan"))
                seed_row["scales"][str(scale)] = {"gain_mbps": None, "feasible": False}
            rows.append(seed_row)
            continue
        s_uav, s_a, s_b = sca
        for scale in scales:
            b_eq = nominal_b_eq_hz(scenario, scale)
            p_score = score_layout(scenario, p_uav, b_eq_hz=b_eq)
            s_score = score_layout(scenario, s_uav, s_a, s_b, b_eq_hz=b_eq)
            gain = _gain(
                p_score.rate_mbps,
                p_score.feasible,
                s_score.rate_mbps,
                s_score.feasible,
            )
            per_scale[float(scale)].append(gain)
            seed_row["scales"][str(scale)] = {
                "gain_mbps": gain,
                "pmedian_mbps": p_score.rate_mbps,
                "sca_mbps": s_score.rate_mbps,
                "feasible": bool(p_score.feasible and s_score.feasible),
            }
        rows.append(seed_row)
        print(
            f"  {family} side {side_m:.0f} H {height_m:.0f} seed {seed}"
            f"  side/H {side_m / height_m:.2f}"
            f"  gain@1 {per_scale[1.0][-1]:+.3f} Mbps"
            f"  {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    summary = {}
    for scale, vals in per_scale.items():
        mean, sd, n = _mean_std(vals)
        summary[str(scale)] = {
            "mean_gain_mbps": mean,
            "std_gain_mbps": sd,
            "n": n,
        }
    return {
        "family": family,
        "side_m": float(side_m),
        "height_m": float(height_m),
        "side_over_h": float(side_m) / float(height_m),
        "scales": summary,
        "seeds": rows,
    }


def _plot(points: list[dict], scales: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0.1: "#4c78a8", 1.0: "#f58518", 10.0: "#54a24b"}
    markers = {"vary_h": "o", "vary_side": "s"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for scale in scales:
        for family, marker in markers.items():
            xs = []
            ys = []
            yerr = []
            for pt in points:
                if pt["family"] != family:
                    continue
                cell = pt["scales"][str(scale)]
                if not np.isfinite(cell["mean_gain_mbps"]):
                    continue
                xs.append(pt["side_over_h"])
                ys.append(cell["mean_gain_mbps"])
                yerr.append(cell["std_gain_mbps"])
            if not xs:
                continue
            order = np.argsort(xs)
            ax.errorbar(
                np.asarray(xs)[order],
                np.asarray(ys)[order],
                yerr=np.asarray(yerr)[order],
                marker=marker,
                color=colors.get(float(scale), "0.3"),
                linestyle="-" if family == "vary_h" else "--",
                capsize=3,
                label=f"B_eq x{scale:g}, {family.replace('_', ' ')}",
            )
    ax.axvline(2.0, color="0.45", linewidth=1.0, linestyle=":")
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.annotate(
        "side = 2H",
        xy=(2.0, 1.0),
        xytext=(4, -2),
        textcoords="offset points",
        xycoords=("data", "axes fraction"),
        color="0.35",
        fontsize=8,
        va="top",
    )
    ax.set_xlabel("field side / H")
    ax.set_ylabel("p-median minus k-means SCA (Mbps)")
    ax.set_title("Proportional noise: placement gain versus side / H")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--heights", type=float, nargs="+", default=[30, 50, 100, 200, 300])
    p.add_argument("--fixed-side", type=float, default=500.0)
    p.add_argument("--sides", type=float, nargs="+", default=[100, 200, 300, 500, 800])
    p.add_argument("--fixed-height", type=float, default=100.0)
    p.add_argument("--b-eq-scales", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    p.add_argument("--sca-iterations", type=int, default=30)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "mechanism_side_over_h.json",
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "results" / "mechanism_side_over_h.png",
    )
    args = p.parse_args(argv)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_runs)))
    scales = [float(s) for s in args.b_eq_scales]
    if 1.0 not in scales:
        raise SystemExit("--b-eq-scales must include 1")
    settings = SCASettings(
        solver=None,
        max_iterations=int(args.sca_iterations),
        dynamic_assignment=True,
    )
    points = []
    for height in args.heights:
        points.append(
            run_geometry(
                side_m=float(args.fixed_side),
                height_m=float(height),
                family="vary_h",
                seeds=seeds,
                scales=scales,
                settings=settings,
            )
        )
    for side in args.sides:
        # The (500 m, 100 m) cell is already in the H sweep.
        if any(
            abs(pt["side_m"] - float(side)) < 1e-6
            and abs(pt["height_m"] - float(args.fixed_height)) < 1e-6
            for pt in points
        ):
            continue
        points.append(
            run_geometry(
                side_m=float(side),
                height_m=float(args.fixed_height),
                family="vary_side",
                seeds=seeds,
                scales=scales,
                settings=settings,
            )
        )
    _plot(points, scales, args.plot)
    payload = {
        "prediction": "gain rises once field side / H passes about 2",
        "sca": "frozen solve_sca, dynamic assignment, constant sigma^2 positions",
        "inner": "solve_bandwidth_proportional at each B_eq scale",
        "b_eq_scales": scales,
        "n_runs": int(args.n_runs),
        "points": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
