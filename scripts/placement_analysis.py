"""Placement-analysis metrics from results/aodt/positions_default.csv."""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    def match_cost(a: np.ndarray, b: np.ndarray) -> float:
        cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].mean())

except Exception:

    def match_cost(a: np.ndarray, b: np.ndarray) -> float:
        cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        best = float("inf")
        n = len(a)
        for perm in itertools.permutations(range(n)):
            best = min(best, float(cost[np.arange(n), perm].mean()))
        return best


def pairwise_spread(xy: np.ndarray) -> float:
    n = len(xy)
    if n < 2:
        return 0.0
    dists = [
        float(np.linalg.norm(xy[i] - xy[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(dists))


def main() -> None:
    path = Path("results/aodt/positions_default.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    meta: dict[tuple[int, str], tuple[int, float]] = {}
    for r in rows:
        seed, method = int(r["seed"]), r["method"]
        by[seed][method].append((int(r["uav_id"]), float(r["x"]), float(r["y"])))
        meta[(seed, method)] = (int(r["feasible"]), float(r["sum_rate"]))

    methods = ["random", "kmeans", "pso", "sca", "td3"]
    stats = {m: {"spread": [], "to_cent": [], "feas": [], "rate": []} for m in methods}
    for seed, mm in sorted(by.items()):
        cents = np.array(sorted(mm["kmeans"], key=lambda t: t[0]))[:, 1:3].astype(float)
        for m in methods:
            xy = np.array(sorted(mm[m], key=lambda t: t[0]))[:, 1:3].astype(float)
            stats[m]["spread"].append(pairwise_spread(xy))
            stats[m]["to_cent"].append(match_cost(xy, cents))
            feas, rate = meta[(seed, m)]
            stats[m]["feas"].append(feas)
            stats[m]["rate"].append(rate)

    out = Path("results/aodt/placement_analysis.md")
    lines = [
        "# Placement analysis (I=10, J=3, seeds 100-119)",
        "",
        "Mean UAV spread is the mean pairwise horizontal distance among the three UAVs, averaged over 20 seeds.",
        "Distance to IoT centroids is the mean Hungarian-matched distance from that method's UAVs to the K-means UAV positions on the same seed (those positions are the IoT cluster centroids after the shared 10 m separation repair).",
        "Feasible-placement rate is the fraction of seeds with a fully feasible solution. Sum rate is the mean associated uplink sum rate.",
        "Source: `positions_default.csv`.",
        "",
        "| Method | Mean UAV spread (m) | Distance to IoT centroids (m) | Feasible-placement rate | Sum rate (bit/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "random": "Random",
        "kmeans": "K-means",
        "pso": "PSO",
        "sca": "SCA",
        "td3": "TD3",
    }
    print("n_seeds", len(by))
    for m in methods:
        s = stats[m]
        spread = float(np.mean(s["spread"]))
        dist = float(np.mean(s["to_cent"]))
        feas = float(np.mean(s["feas"]))
        rate = float(np.mean(s["rate"]))
        line = (
            f"| {labels[m]} | {spread:.1f} | {dist:.1f} | {feas:.2f} | {rate:.1f} |"
        )
        lines.append(line)
        print(line)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
