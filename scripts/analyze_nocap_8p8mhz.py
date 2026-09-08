"""Analyze 8.8 MHz no-cap campaign vs 25% primary and 15% tighter-cap sensitivity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NOCAP = ROOT / "results" / "campaign_8.8mhz_n20.json"
CAP15 = ROOT / "results" / "campaign_8.8mhz_cap15_n20.json"
CAP25 = ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pt(campaign: dict, axis: str, x: float) -> dict:
    for p in campaign["points"]:
        if p["axis"] == axis and abs(float(p["x"]) - x) < 1e-9:
            return p
    raise KeyError((axis, x))


def method_row(pt: dict, methods: tuple[str, ...] = ("sca", "random", "kmeans", "pso")) -> dict:
    return {
        m: {
            "mbps": pt["by_method"][m]["mean_sum_rate_Mbps"],
            "std": pt["by_method"][m]["std_sum_rate_Mbps"],
            "feas": pt["by_method"][m]["feasible_fraction"],
        }
        for m in methods
    }


def spread_mbps(row: dict) -> float:
    vals = [row[m]["mbps"] for m in row]
    return max(vals) - min(vals)


def per_seed_spread(pt: dict, methods: tuple[str, ...] = ("sca", "random", "kmeans", "pso")) -> list[float]:
    out = []
    for i in range(len(pt["by_method"]["sca"]["seeds"])):
        vals = [
            pt["by_method"][m]["per_seed_Mbps"][i]
            for m in methods
        ]
        out.append(max(vals) - min(vals))
    return out


def main() -> int:
    nc = load(NOCAP)
    c15 = load(CAP15)
    c25 = load(CAP25)

    print("=" * 72)
    print("8.8 MHz NO CAP vs CAP 25% (primary) vs CAP 15% (sensitivity)")
    print("=" * 72)
    print(f"No-cap file : {NOCAP.name}  n_runs={nc['n_runs']}")
    print(f"Cap25 file  : {CAP25.name}  (primary)")
    print(f"Cap15 file  : {CAP15.name}  (tighter-cap sensitivity)")

    pt3_nc = pt(nc, "uavs", 3)
    pt3_15 = pt(c15, "uavs", 3)
    pt3_c = pt(c25, "uavs", 3)
    r_nc = method_row(pt3_nc)
    r_15 = method_row(pt3_15)
    r_c = method_row(pt3_c)

    print("\n--- J=3, I=10 (default point) ---")
    print(
        f"{'Method':<8} {'No-cap':>10} {'Cap15':>10} {'Cap25':>10} "
        f"{'15-nc':>8} {'25-nc':>8}"
    )
    for m in ("sca", "random", "kmeans", "pso"):
        print(
            f"{m:<8} {r_nc[m]['mbps']:10.3f} {r_15[m]['mbps']:10.3f} "
            f"{r_c[m]['mbps']:10.3f} "
            f"{r_15[m]['mbps'] - r_nc[m]['mbps']:+8.3f} "
            f"{r_c[m]['mbps'] - r_nc[m]['mbps']:+8.3f}"
        )
    print(
        f"\nMethod spread (max-min mean Mbps)  no-cap: {spread_mbps(r_nc):.4f}  "
        f"cap15: {spread_mbps(r_15):.4f}  cap25: {spread_mbps(r_c):.4f}"
    )

    seeds_spread = per_seed_spread(pt3_nc)
    import statistics

    print(
        f"Per-seed spread on no-cap J=3: mean={statistics.mean(seeds_spread):.4f} Mbps  "
        f"max={max(seeds_spread):.4f}  "
        f"seeds with spread>0.01 Mbps: {sum(1 for s in seeds_spread if s > 0.01)}/20"
    )

    # Check bit-identical seeds across methods
    sca = pt3_nc["by_method"]["sca"]["per_seed_Mbps"]
    identical = 0
    for m in ("random", "kmeans", "pso"):
        other = pt3_nc["by_method"][m]["per_seed_Mbps"]
        for i, (a, b) in enumerate(zip(sca, other)):
            if abs(a - b) < 1e-9:
                identical += 1
    print(f"Exact per-seed matches SCA vs each baseline (J=3): up to {identical}/60 pairs checked")

    print("\n--- UAV sweep (no cap) ---")
    print(f"{'J':>3} {'SCA':>8} {'Random':>8} {'K':>8} {'PSO':>8} {'Spread':>8} {'SCA feas':>8}")
    for j in (1, 2, 3, 4, 5):
        p = pt(nc, "uavs", j)
        r = method_row(p)
        print(
            f"{j:3d} {r['sca']['mbps']:8.3f} {r['random']['mbps']:8.3f} "
            f"{r['kmeans']['mbps']:8.3f} {r['pso']['mbps']:8.3f} "
            f"{spread_mbps(r):8.4f} {r['sca']['feas']:8.0%}"
        )

    print("\n--- Axes at default (no cap): identical rates? ---")
    for axis, x in [("iots", 10), ("lambda", 2.0), ("aodt", 2.8), ("cpu", 2e8)]:
        try:
            p = pt(nc, axis, x)
        except KeyError:
            continue
        r = method_row(p)
        sp = spread_mbps(r)
        tag = "COLLAPSED" if sp < 0.005 else "some spread"
        print(f"  {axis} x={x:g}: spread={sp:.4f} Mbps  SCA={r['sca']['mbps']:.3f}  [{tag}]")

    print("\n--- Infeasible points (SCA feas < 100%) ---")
    bad = []
    for p in nc["points"]:
        sf = p["by_method"]["sca"]["feasible_fraction"]
        if sf < 1.0:
            bad.append((p["label"], sf, p["by_method"]["sca"]["mean_sum_rate_Mbps"]))
    if not bad:
        print("  none except possibly T_k=0.8")
    for label, sf, mbps in bad:
        print(f"  {label}: SCA feas={sf:.0%}  mean={mbps:.3f} Mbps")

    print("\n--- Verdict ---")
    j3_sp = spread_mbps(r_nc)
    if j3_sp < 0.04:
        print(
            f"At J=3 no-cap, methods differ by only {j3_sp:.3f} Mbps on average means — "
            "leftover spectrum piles on best SE links; LP hits similar sum rates."
        )
    print(
        f"Cap25 (primary) removes {r_nc['sca']['mbps'] - r_c['sca']['mbps']:.3f} Mbps "
        f"from SCA vs no-cap; spread {spread_mbps(r_c):.3f} Mbps."
    )
    print(
        f"Cap15 (sensitivity) removes {r_nc['sca']['mbps'] - r_15['sca']['mbps']:.3f} Mbps "
        f"from SCA vs no-cap but widens method spread from {j3_sp:.3f} to "
        f"{spread_mbps(r_15):.3f} Mbps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
