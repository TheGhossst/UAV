"""Dump all post-fix 12%/15% PSO bank and campaign numbers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from paired_winrate import wilcoxon_signed_rank  # noqa: E402


def pair(a: np.ndarray, b: np.ndarray, name: str) -> None:
    d = a - b
    w = wilcoxon_signed_rank(d)
    print(
        f"  {name} {d.mean():+.4f}  wins {(d > 1e-4).sum()}/{len(d)}  "
        f"p={w['p_two_sided']:.4g}  a={a.mean():.4f} b={b.mean():.4f}"
    )


def bank(path: Path, lab: str) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    bm = d["by_method"]
    print(lab, "pso_tag", "pso method replayed" in str(d.get("note")))
    for m in ("sca", "pso", "random", "kmeans"):
        print(
            f"  {m} {bm[m]['mean_sum_rate_Mbps']:.4f} "
            f"+/- {bm[m].get('std_sum_rate_Mbps', float('nan')):.4f}"
        )
    sca = np.array(bm["sca"]["per_seed_Mbps"])
    pso = np.array(bm["pso"]["per_seed_Mbps"])
    rnd = np.array(bm["random"]["per_seed_Mbps"])
    km = np.array(bm["kmeans"]["per_seed_Mbps"])
    pair(sca, pso, "SCA-PSO")
    pair(sca, rnd, "SCA-rnd")
    pair(pso, rnd, "PSO-rnd")
    spread = max(sca.mean(), pso.mean(), rnd.mean(), km.mean()) - min(
        sca.mean(), pso.mean(), rnd.mean(), km.mean()
    )
    print(f"  spread {spread:.3f}")


def camp_axis(path: Path, axis: str) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    print(path.name, axis)
    for pt in d["points"]:
        if pt.get("axis") != axis:
            continue
        bm = pt["by_method"]
        sca = np.array(bm["sca"]["per_seed_Mbps"])
        pso = np.array(bm["pso"]["per_seed_Mbps"])
        rnd = np.array(bm["random"]["per_seed_Mbps"])
        dlt = sca - pso
        w = wilcoxon_signed_rank(dlt)
        uniq = "SCA" if sca.mean() > pso.mean() else "PSO"
        print(
            f"  {axis}={pt['x']} SCA {sca.mean():.4f} PSO {pso.mean():.4f} "
            f"rnd {rnd.mean():.4f} {uniq}  SCA-PSO {dlt.mean():+.4f} "
            f"{int((dlt > 1e-4).sum())}/20 p={w['p_two_sided']:.3g}"
        )


def main() -> None:
    bank(ROOT / "results" / "n100_cap12" / "eval.json", "n100 12 100m")
    bank(ROOT / "results" / "n100_500m_cap12" / "eval.json", "n100 12 500m")
    bank(ROOT / "results" / "n100_cap15" / "eval.json", "n100 15 100m")
    bank(ROOT / "results" / "n100_500m_cap15" / "eval.json", "n100 15 500m")
    for axis in ("uavs", "devices", "arrival", "aodt", "cpu"):
        camp_axis(ROOT / "results" / "campaign_8.8mhz_cap15_n20.json", axis)
    camp_axis(ROOT / "results" / "campaign_8.8mhz_cap12_n20.json", "uavs")
    camp_axis(ROOT / "results" / "campaign_8.8mhz_cap12_n20_500m.json", "uavs")


if __name__ == "__main__":
    main()
