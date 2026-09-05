"""20 kHz Table II check: model-free ceiling + 100 m vs 500 m control.

    python scripts/check_bsys_20khz.py
    python scripts/check_bsys_20khz.py --n-runs 5 --methods random,kmeans
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.experiments.bsys_check import run_bsys_20khz_check, write_bsys_check


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--methods", type=str, default="random,kmeans")
    p.add_argument("--out", type=str, default="results/check_bsys_20khz.json")
    args = p.parse_args()
    methods = tuple(x.strip().lower() for x in args.methods.split(",") if x.strip())
    print("model-free Eq. (6)+(27) ceilings, then 20 kHz at 100 m and 500 m", flush=True)
    payload = run_bsys_20khz_check(
        n_runs=int(args.n_runs),
        seed_start=int(args.seed_start),
        methods=methods,
    )
    mf = payload["model_free"]
    print(f"B_sys = {mf['b_sys_hz']:g} Hz")
    print(f"formula: {mf['formula']}")
    for row in mf["grid"]:
        print(
            f"  SNR_max={row['snr_max']:.3g}  "
            f"SE={row['spectral_efficiency_bit_per_s_per_hz']:.4g}  "
            f"ceiling={row['ceiling_Mbps']:.4g} Mbps"
        )
    print(
        f"generous SNR_max={mf['generous_snr_max']:.3g} -> "
        f"{mf['generous_ceiling_Mbps']:.4g} Mbps  (not 7-14 Mbps)"
    )
    print(f"\nFig. 6-10 vs readings: {payload['fig_6_10_rules_out']}")
    print("\nReading B per-link B_i to match published Mbps (written SNR):")
    for row in payload["reading_b_written_snr"]["anchors"]:
        print(
            f"  Fig {row['fig']}: {row['target_Mbps']:g} Mbps, I={row['n_links']}  "
            f"B_i={row['per_link_kHz']:.1f} kHz  "
            f"({row['vs_stated_20kHz']:.1f}x the stated 20 kHz)"
        )
    print("Reading B at SNR_max=1e15 (fantasy; 20 kHz floor can suffice):")
    for row in payload["reading_b_generous_snr"]["anchors"]:
        print(
            f"  Fig {row['fig']}: B_i={row['per_link_kHz']:.2f} kHz  "
            f"({row['vs_stated_20kHz']:.2f}x)"
        )
    for area in payload["areas"]:
        side = area["area_m"][0]
        print(
            f"\n=== {side:g}x{side:g} m  max_SNR={area['observed_max_snr']:.4g}  "
            f"mean_SE={area['mean_associated_se']:.4g}  "
            f"dump={area['best_link_dump_mean_Mbps']:.4g} Mbps  "
            f"equal-share pred={area['equal_share_pred_mean_Mbps']:.4g} Mbps ==="
        )
        for method, stats in area["by_method"].items():
            print(
                f"  {method:8s}  mean={stats['mean_sum_rate_Mbps']:.4g} Mbps  "
                f"SE={stats['mean_associated_se']:.4g}  "
                f"feasible={stats['feasible_fraction']:.0%}"
            )
    out = write_bsys_check(payload, args.out)
    print(f"\nwrote {out}")
    print(f"all_areas_zero_feasible={payload['all_areas_zero_feasible']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
