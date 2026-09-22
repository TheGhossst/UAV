"""Print J=3 PSO means after the tight-cap replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from paired_winrate import wilcoxon_signed_rank  # noqa: E402


def camp_j3(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    print(path.name, "pso_tag", "pso method replayed" in str(d.get("note")))
    for pt in d["points"]:
        if pt.get("axis") != "uavs" or int(pt.get("x", -1)) != 3:
            continue
        bm = pt["by_method"]
        sca = np.array(bm["sca"]["per_seed_Mbps"])
        pso = np.array(bm["pso"]["per_seed_Mbps"])
        rnd = np.array(bm["random"]["per_seed_Mbps"])
        km = np.array(bm["kmeans"]["per_seed_Mbps"])
        dlt = sca - pso
        w = wilcoxon_signed_rank(dlt)
        spread = float(
            max(sca.mean(), pso.mean(), rnd.mean(), km.mean())
            - min(sca.mean(), pso.mean(), rnd.mean(), km.mean())
        )
        print(
            f"  SCA {sca.mean():.4f} PSO {pso.mean():.4f} "
            f"rnd {rnd.mean():.4f} km {km.mean():.4f} spread {spread:.3f}"
        )
        print(
            f"  SCA-PSO {dlt.mean():+.4f} {int((dlt > 1e-4).sum())}/{len(dlt)} "
            f"p={w['p_two_sided']:.4g}"
        )
        return
    print("  no J=3")


def jsweep(path: Path, lab: str) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    print("J-sweep", lab)
    for pt in d["points"]:
        if pt.get("axis") != "uavs":
            continue
        bm = pt["by_method"]
        print(
            f"  J={int(pt['x'])} SCA {bm['sca']['mean_sum_rate_Mbps']:.4f} "
            f"PSO {bm['pso']['mean_sum_rate_Mbps']:.4f} "
            f"rnd {bm['random']['mean_sum_rate_Mbps']:.4f}"
        )


def main() -> None:
    for p in (
        ROOT / "results" / "campaign_8.8mhz_cap12_n20.json",
        ROOT / "results" / "campaign_8.8mhz_cap12_n20_500m.json",
        ROOT / "results" / "campaign_8.8mhz_cap15_n20.json",
    ):
        camp_j3(p)
    jsweep(ROOT / "results" / "campaign_8.8mhz_cap12_n20.json", "12 100m")
    jsweep(ROOT / "results" / "campaign_8.8mhz_cap12_n20_500m.json", "12 500m")
    p = ROOT / "results" / "n100_500m_cap15" / "eval.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    print("n100 15 500 tag", "pso method replayed" in str(d.get("note")))
    bm = d["by_method"]
    sca = np.array(bm["sca"]["per_seed_Mbps"])
    pso = np.array(bm["pso"]["per_seed_Mbps"])
    w = wilcoxon_signed_rank(sca - pso)
    print(
        "  SCA",
        round(float(sca.mean()), 4),
        "PSO",
        round(float(pso.mean()), 4),
        "SCA-PSO",
        round(float((sca - pso).mean()), 4),
        int(((sca - pso) > 1e-4).sum()),
        "p",
        w["p_two_sided"],
    )
    cap15 = ROOT / "results" / "sca_anchor_ablations" / "cap_15%_500m.json"
    print("n20 15 500", json.loads(cap15.read_text(encoding="utf-8")).get("mean_Mbps"))


if __name__ == "__main__":
    main()
