"""Analyze 7 MHz campaigns: no cap vs 15% vs 25%. No TD3."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402

NOCAP = ROOT / "results" / "campaign_7mhz_n20.json"
CAP15 = ROOT / "results" / "campaign_7mhz_cap15_n20.json"
CAP25 = ROOT / "results" / "campaign_7mhz_cap25_n20.json"
REF88 = ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"
REF88_NC = ROOT / "results" / "campaign_8.8mhz_n20.json"
REF88_15 = ROOT / "results" / "campaign_8.8mhz_cap15_n20.json"
OUT = ROOT / "results" / "analyze_7mhz.json"
METHODS = ("sca", "random", "kmeans", "pso")


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pt(campaign: dict, axis: str, x: float) -> dict:
    for p in campaign["points"]:
        if p["axis"] == axis and abs(float(p["x"]) - x) < 1e-9:
            return p
    raise KeyError((axis, x))


def method_row(row: dict) -> dict:
    return {
        m: {
            "mbps": row["by_method"][m]["mean_sum_rate_Mbps"],
            "std": row["by_method"][m]["std_sum_rate_Mbps"],
            "feas": row["by_method"][m]["feasible_fraction"],
        }
        for m in METHODS
    }


def spread_mbps(row: dict) -> float:
    vals = [row[m]["mbps"] for m in row]
    return max(vals) - min(vals)


def paired_sca(row: dict, baseline: str) -> dict:
    sca = row["by_method"]["sca"]["per_seed_Mbps"]
    base = row["by_method"][baseline]["per_seed_Mbps"]
    deltas = [a - b for a, b in zip(sca, base)]
    w = wilcoxon_signed_rank(deltas)
    n = len(deltas)
    return {
        "delta_mbps": sum(deltas) / n,
        "wins": sum(1 for d in deltas if d > 1e-12),
        "ties": sum(1 for d in deltas if abs(d) <= 1e-12),
        "n": n,
        "p": float(w["p_two_sided"]),
    }


def print_method_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n--- {title} ---")
    header = f"{'Method':<8}" + "".join(f"{name:>10}" for name in rows)
    print(header)
    for m in METHODS:
        print(f"{m:<8}" + "".join(f"{rows[name][m]['mbps']:10.3f}" for name in rows))
    print(f"{'feas SCA':<8}" + "".join(f"{rows[name]['sca']['feas']:10.0%}" for name in rows))
    print(f"{'spread':<8}" + "".join(f"{spread_mbps(rows[name]):10.4f}" for name in rows))


def print_axis(name: str, campaign: dict, xs: list[float]) -> None:
    print(f"\n--- {name} ---")
    print(f"{'x':>10} {'SCA':>8} {'Random':>8} {'K':>8} {'PSO':>8} {'Spread':>8} {'feas':>6}")
    for x in xs:
        try:
            row = method_row(pt(campaign, name if name != "UAV J" else "uavs", x))
        except KeyError:
            continue
        print(
            f"{x:10g} {row['sca']['mbps']:8.3f} {row['random']['mbps']:8.3f} "
            f"{row['kmeans']['mbps']:8.3f} {row['pso']['mbps']:8.3f} "
            f"{spread_mbps(row):8.4f} {row['sca']['feas']:6.0%}"
        )


def infeasible(campaign: dict) -> list[tuple[str, float, float]]:
    bad = []
    for p in campaign["points"]:
        sf = p["by_method"]["sca"]["feasible_fraction"]
        if sf < 1.0:
            bad.append((p["label"], sf, p["by_method"]["sca"]["mean_sum_rate_Mbps"]))
    return bad


def main() -> int:
    missing = [p for p in (NOCAP, CAP15, CAP25) if not p.exists()]
    if missing:
        print("Missing:", ", ".join(str(p) for p in missing), file=sys.stderr)
        return 1

    nc, c15, c25 = load(NOCAP), load(CAP15), load(CAP25)
    print("=" * 72)
    print("7 MHz  NO CAP vs CAP 15% vs CAP 25%   (random, k-means, PSO, SCA)")
    print("=" * 72)
    print(f"No-cap : {NOCAP.name}  n={nc['n_runs']}  B={nc['b_sys_hz']/1e6:g} MHz")
    print(f"Cap15  : {CAP15.name}  n={c15['n_runs']}")
    print(f"Cap25  : {CAP25.name}  n={c25['n_runs']}")
    print(f"Area   : {nc['area_m']} m   methods={nc['methods']}")

    r_nc = method_row(pt(nc, "uavs", 3))
    r_15 = method_row(pt(c15, "uavs", 3))
    r_25 = method_row(pt(c25, "uavs", 3))
    print_method_table(
        "J=3, I=10 default (mean Mbps)",
        {"no-cap": r_nc, "cap15": r_15, "cap25": r_25},
    )
    print("\nDeltas vs no-cap (SCA):")
    print(
        f"  cap15: {r_15['sca']['mbps'] - r_nc['sca']['mbps']:+.4f} Mbps  "
        f"({100*(r_nc['sca']['mbps']-r_15['sca']['mbps'])/r_nc['sca']['mbps']:.2f}% drop)"
    )
    print(
        f"  cap25: {r_25['sca']['mbps'] - r_nc['sca']['mbps']:+.4f} Mbps  "
        f"({100*(r_nc['sca']['mbps']-r_25['sca']['mbps'])/r_nc['sca']['mbps']:.2f}% drop)"
    )

    print("\n--- Paired SCA vs baselines at J=3 ---")
    for label, camp in (("no-cap", nc), ("cap15", c15), ("cap25", c25)):
        row = pt(camp, "uavs", 3)
        print(f"  {label}:")
        for b in ("random", "kmeans", "pso"):
            s = paired_sca(row, b)
            print(
                f"    vs {b:<7} d={s['delta_mbps']:+.4f} Mbps  "
                f"{s['wins']}/{s['n']} wins  p={s['p']:.4g}"
            )

    for tag, camp in (("no-cap", nc), ("cap15", c15), ("cap25", c25)):
        print(f"\n======== {tag} ========")
        print_axis("uavs", camp, [1, 2, 3, 4, 5])
        print_axis("iots", camp, [10, 16, 20, 24, 28, 32])
        print_axis("lambda", camp, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        print_axis("aodt", camp, [0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0])
        print_axis("cpu", camp, [0.5e8, 1.0e8, 1.5e8, 2.0e8, 2.5e8])
        bad = infeasible(camp)
        print("\nSCA feas < 100%:")
        if not bad:
            print("  none")
        for label, sf, mbps in bad:
            print(f"  {label}: feas={sf:.0%}  mean={mbps:.3f} Mbps")

    payload: dict = {
        "b_sys_hz": 7_000_000.0,
        "n_runs": nc["n_runs"],
        "j3": {
            "nocap": r_nc,
            "cap15": r_15,
            "cap25": r_25,
            "spread": {
                "nocap": spread_mbps(r_nc),
                "cap15": spread_mbps(r_15),
                "cap25": spread_mbps(r_25),
            },
        },
    }

    print("\n======== vs 8.8 MHz (if present) ========")
    refs = []
    if REF88_NC.exists():
        refs.append(("8.8 no-cap", load(REF88_NC), r_nc, "nocap"))
    if REF88_15.exists():
        refs.append(("8.8 cap15", load(REF88_15), r_15, "cap15"))
    if REF88.exists():
        refs.append(("8.8 cap25", load(REF88), r_25, "cap25"))
    scale = 7.0 / 8.8
    for name, camp, seven, key in refs:
        eight = method_row(pt(camp, "uavs", 3))
        sca7 = seven["sca"]["mbps"]
        sca8 = eight["sca"]["mbps"]
        pred = sca8 * scale
        print(
            f"  {name}: 8.8 SCA={sca8:.3f}  7.0 SCA={sca7:.3f}  "
            f"linear 7/8.8 pred={pred:.3f}  residual={sca7 - pred:+.4f} Mbps"
        )
        payload.setdefault("vs_8p8", {})[key] = {
            "sca_8p8": sca8,
            "sca_7": sca7,
            "linear_pred": pred,
            "residual": sca7 - pred,
        }

    print("\n--- Verdict ---")
    print(
        f"No-cap J=3 spread {spread_mbps(r_nc):.4f} Mbps; "
        f"cap25 {spread_mbps(r_25):.4f}; cap15 {spread_mbps(r_15):.4f}."
    )
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    _ = statistics.mean
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
