"""Print campaign summary and sanity checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/campaign_8.8mhz_cap25_si12k.json")
    p = json.loads(path.read_text(encoding="utf-8"))
    methods = p["methods"]
    print(f"file: {path}")
    print(f"n_runs={p['n_runs']}  B_sys={p['b_sys_hz']/1e6:g} MHz  max_bw_share={p['max_bw_share']}")
    print(f"area={p['area_m']} m  methods={methods}")
    print()

    cur = None
    for pt in p["points"]:
        if pt["axis"] != cur:
            cur = pt["axis"]
            print(f"=== {cur} ===")
        bits = []
        for m in methods:
            s = pt["by_method"][m]
            bits.append(
                f"{m}={s['mean_sum_rate_Mbps']:.2f}Mbps feas={s['feasible_fraction']:.0%}"
            )
        print(f"  {pt['x_name']}={pt['x']:g}  " + "  ".join(bits))

    # Sanity checks on default-ish point I=10 J=3 lambda=2
    default = None
    for pt in p["points"]:
        if pt["axis"] == "lambda" and pt["x"] == 2.0:
            default = pt
            break
    if default:
        print()
        print("=== default scenario (lambda=2, I=10, J=3) ranking ===")
        ranked = sorted(
            ((m, default["by_method"][m]["mean_sum_rate_Mbps"]) for m in methods),
            key=lambda t: -t[1],
        )
        for m, mbps in ranked:
            s = default["by_method"][m]
            print(
                f"  {m:8} {mbps:.3f} Mbps  feasible={s['feasible_fraction']:.0%}  "
                f"std={s['std_sum_rate_Mbps']:.3f}  max_AoDT={s['mean_max_AoDT_s']:.2f}s"
            )

    # Monotonicity checks
    print()
    print("=== sanity flags ===")
    uav_pts = [pt for pt in p["points"] if pt["axis"] == "uavs"]
    for m in methods:
        rates = [pt["by_method"][m]["mean_sum_rate_Mbps"] for pt in uav_pts]
        js = [pt["x"] for pt in uav_pts]
        mono = all(rates[i] <= rates[i + 1] + 0.05 for i in range(len(rates) - 1))
        print(f"  J sweep {m}: rates={[f'{r:.2f}' for r in rates]}  non-decreasing~={mono}")

    aodt_pts = [pt for pt in p["points"] if pt["axis"] == "aodt"]
    tk08 = next(pt for pt in aodt_pts if pt["x"] == 0.8)
    print(f"  T_k=0.8s feasible: " + ", ".join(
        f"{m}={tk08['by_method'][m]['feasible_fraction']:.0%}" for m in methods
    ))

    lam_pts = [pt for pt in p["points"] if pt["axis"] == "lambda"]
    lam_rates = {m: [pt["by_method"][m]["mean_sum_rate_Mbps"] for pt in lam_pts] for m in methods}
    flat = all(max(v) - min(v) < 0.15 for v in lam_rates.values())
    print(f"  lambda axis flat (<0.15 Mbps spread): {flat}  (expected when comm-limited)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
