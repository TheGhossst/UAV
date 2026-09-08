"""Compare bandwidth campaign configs side by side."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402


def load(fp: str) -> dict:
    return json.loads(Path(fp).read_text(encoding="utf-8"))


def j3_point(d: dict) -> dict:
    return next(x for x in d["points"] if x["axis"] == "uavs" and x["x"] == 3)


def paired_j3(d: dict, baseline: str) -> dict:
    pt = j3_point(d)
    sca = pt["by_method"]["sca"]["per_seed_Mbps"]
    base = pt["by_method"][baseline]["per_seed_Mbps"]
    deltas = [s - b for s, b in zip(sca, base, strict=True)]
    w = wilcoxon_signed_rank(deltas)
    return {
        "delta": sum(deltas) / len(deltas),
        "wins": sum(1 for x in deltas if x > 0),
        "p": w["p_two_sided"],
    }


def summarize(name: str, fp: str) -> dict:
    d = load(fp)
    pt = j3_point(d)
    means = {m: pt["by_method"][m]["mean_sum_rate_Mbps"] for m in d["methods"]}
    spread = max(means.values()) - min(means.values())
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    sca_rank = [m for m, _ in ranked].index("sca") + 1
    sca_lead = means["sca"] - ranked[1][1] if ranked[0][0] == "sca" else means["sca"] - ranked[0][1]
    feas_pts = [p for p in d["points"] if p["by_method"]["sca"]["feasible_fraction"] >= 1.0]
    sca_best = sum(
        1
        for p in feas_pts
        if max(p["by_method"], key=lambda m: p["by_method"][m]["mean_sum_rate_Mbps"]) == "sca"
    )
    min_feas = min(p["by_method"]["sca"]["feasible_fraction"] for p in d["points"])
    return {
        "name": name,
        "file": fp,
        "sca_mbps": means["sca"],
        "spread": spread,
        "sca_rank": sca_rank,
        "sca_lead": sca_lead,
        "means": means,
        "sca_best": sca_best,
        "feas_pts": len(feas_pts),
        "min_feas": min_feas,
        "vs_random": paired_j3(d, "random"),
        "vs_kmeans": paired_j3(d, "kmeans"),
        "vs_pso": paired_j3(d, "pso"),
    }


def main() -> int:
    configs = [
        ("8.8 MHz cap=25% (primary)", "results/campaign_8.8mhz_cap25_si12k.json"),
        ("8.8 MHz cap=15% (sensitivity)", "results/campaign_8.8mhz_cap15_n20.json"),
        ("8.8 MHz cap=20%", "results/campaign_8.8mhz_cap20_n20.json"),
        ("8.8 MHz no cap", "results/campaign_8.8mhz_n20.json"),
        ("2.4 MHz cap=25%", "results/campaign_2.4mhz_cap25.json"),
    ]
    rows = [summarize(n, fp) for n, fp in configs if Path(fp).exists()]

    print("=" * 96)
    print("FULL CAMPAIGN COMPARISON @ J=3 (I=10, 20 seeds)")
    print("=" * 96)
    hdr = (
        f"{'Config':28s} {'SCA':>7s} {'Spread':>7s} {'Rank':>5s} "
        f"{'Lead':>7s} {'SCA top':>8s} {'d_rand':>7s} {'p_rand':>8s}"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r['name']:28s} {r['sca_mbps']:7.3f} {r['spread']:7.3f} {r['sca_rank']:5d} "
            f"{r['sca_lead']:7.3f} {r['sca_best']:3d}/{r['feas_pts']:<3d} "
            f"{r['vs_random']['delta']:+7.3f} {r['vs_random']['p']:8.4f}"
        )

    print("\nMethod means @ J=3:")
    for r in rows[:3]:
        m = r["means"]
        print(
            f"  {r['name']:28s}  sca={m['sca']:.3f}  random={m['random']:.3f}  "
            f"kmeans={m['kmeans']:.3f}  pso={m['pso']:.3f}"
        )

    ref = next(r for r in rows if "cap=25%" in r["name"])
    print("\nVs 25% primary (SCA-best + all three paired p<0.05):")
    for r in rows:
        if r["name"] == ref["name"]:
            continue
        ok = (
            r["sca_rank"] == 1
            and r["vs_random"]["p"] < 0.05
            and r["vs_kmeans"]["p"] < 0.05
            and r["vs_pso"]["p"] < 0.05
        )
        print(
            f"  {r['name']}: ranking+stats={ok}  "
            f"spread={r['spread']:.3f} (primary {ref['spread']:.3f})  "
            f"p_rand={r['vs_random']['p']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
