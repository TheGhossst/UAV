"""Analyze all bandwidth campaign results for sensibility checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def label_campaign(p: dict) -> str:
    b_mhz = p["b_sys_hz"] / 1e6
    cap = p.get("max_bw_share")
    cap_s = "no cap" if cap is None else f"cap {cap:.0%}"
    return f"{b_mhz:g} MHz ({cap_s}), n={p['n_runs']}"


def summarize_point(pt: dict, methods: list[str]) -> dict:
    out = {}
    for m in methods:
        s = pt["by_method"][m]
        out[m] = {
            "mbps": s["mean_sum_rate_Mbps"],
            "feas": s["feasible_fraction"],
            "std": s["std_sum_rate_Mbps"],
        }
    return out


def method_ranking(by_method: dict) -> list[str]:
    return sorted(
        by_method.keys(),
        key=lambda m: by_method[m]["mean_sum_rate_Mbps"],
        reverse=True,
    )


def analyze_file(path: Path) -> dict:
    p = load(path)
    methods = p["methods"]
    issues: list[str] = []
    notes: list[str] = []

    # Group by axis
    by_axis: dict[str, list] = {}
    for pt in p["points"]:
        by_axis.setdefault(pt["axis"], []).append(pt)

    # Check 1: SCA should generally be best or tied at feasible points
    sca_not_best = 0
    sca_best = 0
    feasible_points = 0
    for pt in p["points"]:
        ranks = method_ranking(pt["by_method"])
        top = ranks[0]
        sca_mbps = pt["by_method"]["sca"]["mean_sum_rate_Mbps"]
        sca_feas = pt["by_method"]["sca"]["feasible_fraction"]
        if sca_feas < 1.0:
            continue
        feasible_points += 1
        if top == "sca":
            sca_best += 1
        elif sca_mbps < pt["by_method"][top]["mean_sum_rate_Mbps"] * 0.99:
            sca_not_best += 1

    if feasible_points and sca_not_best > feasible_points * 0.3:
        issues.append(
            f"SCA not top method on {sca_not_best}/{feasible_points} "
            "fully-feasible points (>30% threshold)"
        )

    # Check 2: Rates should not exceed B_sys * log2(1+SNR) rough ceiling
    # At 8.8 MHz, max ~9 Mbps is expected (docs say ~8.96-8.99)
    b_mhz = p["b_sys_hz"] / 1e6
    max_rate = max(
        pt["by_method"][m]["mean_sum_rate_Mbps"]
        for pt in p["points"]
        for m in methods
    )
    if b_mhz <= 0.1 and max_rate > 0.5:
        issues.append(f"20 kHz but max rate {max_rate:.3f} Mbps seems too high")
    if b_mhz >= 8 and max_rate > 10:
        issues.append(f"8.8 MHz but max rate {max_rate:.3f} Mbps exceeds ~9 Mbps ceiling")

    # Check 3: No-cap 8.8 should saturate near B_sys Shannon-ish bound
    if p["b_sys_hz"] == 8_800_000 and p.get("max_bw_share") is None:
        uav3 = next(pt for pt in p["points"] if pt["axis"] == "uavs" and pt["x"] == 3)
        rates = [uav3["by_method"][m]["mean_sum_rate_Mbps"] for m in methods]
        spread = max(rates) - min(rates)
        if spread > 0.5:
            notes.append(
                f"No-cap 8.8 MHz: methods differ by {spread:.3f} Mbps at J=3 "
                "(expected near-saturation; small spread OK)"
            )

    # Check 4: Cap should reduce rates vs no-cap (same B_sys)
    # Check 5: lambda/cpu axes should not move comm objective much
    lam_pts = by_axis.get("lambda", [])
    if lam_pts:
        sca_rates = [pt["by_method"]["sca"]["mean_sum_rate_Mbps"] for pt in lam_pts]
        lam_spread = max(sca_rates) - min(sca_rates)
        if lam_spread > 0.1:
            notes.append(f"lambda axis moves SCA rate by {lam_spread:.3f} Mbps (unexpected?)")

    cpu_pts = by_axis.get("cpu", [])
    if cpu_pts:
        sca_rates = [pt["by_method"]["sca"]["mean_sum_rate_Mbps"] for pt in cpu_pts]
        cpu_spread = max(sca_rates) - min(sca_rates)
        if cpu_spread > 0.1:
            notes.append(f"CPU axis moves SCA rate by {cpu_spread:.3f} Mbps (unexpected?)")

    # Check 6: More UAVs should help (generally)
    uav_pts = sorted(by_axis.get("uavs", []), key=lambda x: x["x"])
    if len(uav_pts) >= 2:
        sca_j1 = next(pt for pt in uav_pts if pt["x"] == 1)["by_method"]["sca"]
        sca_j5 = next(pt for pt in uav_pts if pt["x"] == 5)["by_method"]["sca"]
        if (
            sca_j5["feasible_fraction"] >= 0.8
            and sca_j1["feasible_fraction"] >= 0.8
            and sca_j5["mean_sum_rate_Mbps"] < sca_j1["mean_sum_rate_Mbps"] * 0.95
        ):
            issues.append("J=5 SCA rate lower than J=1 (unexpected when both feasible)")

    # Check 7: T_k=0.8 should be hard
    aodt_pts = by_axis.get("aodt", [])
    tk08 = next((pt for pt in aodt_pts if pt["x"] == 0.8), None)
    if tk08:
        feas_all = [tk08["by_method"][m]["feasible_fraction"] for m in methods]
        if any(f > 0.5 for f in feas_all):
            notes.append(f"T_k=0.8: some methods >50% feasible: {dict(zip(methods, feas_all))}")

    return {
        "path": str(path),
        "label": label_campaign(p),
        "sca_best": sca_best,
        "feasible_points": feasible_points,
        "sca_not_best": sca_not_best,
        "max_rate_mbps": max_rate,
        "issues": issues,
        "notes": notes,
        "by_axis": by_axis,
        "methods": methods,
    }


def print_report(analyses: list[dict]) -> None:
    print("=" * 72)
    print("BANDWIDTH CAMPAIGN ANALYSIS")
    print("=" * 72)

    for a in analyses:
        print(f"\n--- {a['label']} ---")
        print(f"  File: {a['path']}")
        print(f"  Max rate: {a['max_rate_mbps']:.3f} Mbps")
        print(f"  SCA best on {a['sca_best']}/{a['feasible_points']} feasible points")

        # Key reference point: J=3, default
        for axis_name, x_val, title in [
            ("uavs", 3, "Fig.6 J=3"),
            ("iots", 10, "Fig.7 I=10"),
            ("lambda", 2.0, "Fig.8 lambda=2"),
            ("aodt", 2.8, "Fig.9 Tk=2.8"),
            ("cpu", 2e8, "Fig.10 f=2e8"),
        ]:
            pts = a["by_axis"].get(axis_name, [])
            pt = next((p for p in pts if abs(p["x"] - x_val) < 1e-6), None)
            if not pt:
                continue
            ranks = method_ranking(pt["by_method"])
            parts = []
            for m in ranks:
                s = pt["by_method"][m]
                parts.append(f"{m}={s['mean_sum_rate_Mbps']:.3f}(feas={s['feasible_fraction']:.0%})")
            print(f"  {title}: {' < '.join(parts)}")

        if a["issues"]:
            print("  ISSUES:")
            for i in a["issues"]:
                print(f"    ! {i}")
        if a["notes"]:
            print("  Notes:")
            for n in a["notes"]:
                print(f"    - {n}")

    # Cross-bandwidth comparison at J=3
    print("\n" + "=" * 72)
    print("CROSS-BANDWIDTH (SCA at J=3, I=10)")
    print("=" * 72)
    for a in analyses:
        pts = a["by_axis"].get("uavs", [])
        pt = next((p for p in pts if p["x"] == 3), None)
        if pt:
            s = pt["by_method"]["sca"]
            print(
                f"  {a['label']:40s}  SCA={s['mean_sum_rate_Mbps']:.3f} Mbps  "
                f"feas={s['feasible_fraction']:.0%}"
            )

    # Cap vs no-cap deltas
    print("\n" + "=" * 72)
    print("CAP vs NO-CAP (SCA mean Mbps at J=3)")
    print("=" * 72)
    by_bw: dict[float, dict] = {}
    for a in analyses:
        p = load(Path(a["path"]))
        bw = p["b_sys_hz"]
        cap = p.get("max_bw_share")
        pts = a["by_axis"].get("uavs", [])
        pt = next((p for p in pts if p["x"] == 3), None)
        if pt:
            by_bw.setdefault(bw, {})[cap] = pt["by_method"]["sca"]["mean_sum_rate_Mbps"]

    for bw, caps in sorted(by_bw.items()):
        no_cap = caps.get(None)
        for share, tag in ((0.25, "cap25"), (0.15, "cap15")):
            capped = caps.get(share)
            if no_cap is not None and capped is not None:
                delta = no_cap - capped
                pct = 100 * delta / no_cap if no_cap else 0
                print(
                    f"  {bw/1e6:g} MHz: no_cap={no_cap:.3f}, {tag}={capped:.3f}, "
                    f"delta={delta:.3f} Mbps ({pct:.1f}% reduction)"
                )


def main() -> int:
    patterns = [
        "results/campaign_20khz.json",
        "results/campaign_20khz_cap25.json",
        "results/campaign_2.4mhz.json",
        "results/campaign_2.4mhz_cap25.json",
        "results/campaign_8.8mhz_n20.json",
        "results/campaign_8.8mhz_cap25_si12k.json",
        "results/campaign_8.8mhz_cap25_n20.json",
        "results/campaign_8.8mhz_cap15_n20.json",
        # fallbacks
        "results/campaign_8.8mhz.json",
        "results/campaign_20260904_cap25.json",
    ]
    seen = set()
    analyses = []
    for pat in patterns:
        path = Path(pat)
        if path.exists() and str(path) not in seen:
            seen.add(str(path))
            analyses.append(analyze_file(path))

    if not analyses:
        print("No campaign files found.", file=sys.stderr)
        return 1

    print_report(analyses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
