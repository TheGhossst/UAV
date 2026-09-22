"""Dump remaining 15% axes, T_k feas, and 15% anchor vs PSO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from paired_winrate import wilcoxon_signed_rank  # noqa: E402

PRACTICAL = 0.05


def axes(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    print(path.name, "axes", sorted({pt["axis"] for pt in d["points"]}))
    for pt in d["points"]:
        bm = pt["by_method"]
        print(
            f"  {pt['axis']}={pt['x']}  SCA {bm['sca']['mean_sum_rate_Mbps']:.4f} "
            f"PSO {bm['pso']['mean_sum_rate_Mbps']:.4f} "
            f"rnd {bm['random']['mean_sum_rate_Mbps']:.4f} "
            f"km {bm['kmeans']['mean_sum_rate_Mbps']:.4f} "
            f"pso_feas {bm['pso'].get('feasible_frac', bm['pso'].get('frac_feasible', '?'))}"
        )


def feas_tk08(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    for pt in d["points"]:
        if pt.get("axis") not in ("aodt", "deadline") and "0.8" not in str(pt.get("x")):
            if not (
                pt.get("axis") in ("aodt",) and abs(float(pt.get("x", 0)) - 0.8) < 1e-9
            ):
                continue
        if abs(float(pt.get("x", 0)) - 0.8) > 1e-9:
            continue
        print(path.name, "T_k=0.8")
        for m in ("sca", "random", "pso", "kmeans"):
            bm = pt["by_method"][m]
            frac = bm.get("feasible_frac", bm.get("frac_feasible"))
            print(f"  {m} feas={frac} mean={bm['mean_sum_rate_Mbps']:.4f}")


def pair_files(lab: str, a_path: Path, a_key: str, b_path: Path, b_key: str) -> None:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    if "per_seed" in a:
        xs = np.array([float(r[a_key]["sum_rate_Mbps"]) for r in a["per_seed"]])
    else:
        xs = np.array(a["by_method"][a_key]["per_seed_Mbps"])
    if "per_seed" in b:
        ys = np.array([float(r[b_key]["sum_rate_Mbps"]) for r in b["per_seed"]])
    else:
        ys = np.array(b["by_method"][b_key]["per_seed_Mbps"])
    d = xs - ys
    w = wilcoxon_signed_rank(d)
    print(
        f"{lab} a={xs.mean():.4f} b={ys.mean():.4f} d={d.mean():+.4f} "
        f"wins={(d > 1e-4).sum()}/{len(d)} prac={(d > PRACTICAL).sum()}/{len(d)} "
        f"p={w['p_two_sided']:.4g}"
    )


def main() -> None:
    axes(ROOT / "results" / "campaign_8.8mhz_cap15_n20.json")
    for p in (
        ROOT / "results" / "campaign_8.8mhz_cap15_n20.json",
        ROOT / "results" / "campaign_8.8mhz_cap12_n20.json",
        ROOT / "results" / "campaign_8.8mhz_cap12_n20_500m.json",
    ):
        feas_tk08(p)
    pair_files(
        "n20 15 500 anchor-pso",
        ROOT / "results" / "sca_anchor_ablations" / "cap_15%_500m.json",
        "sca_anchor",
        ROOT / "results" / "sca_anchor_ablations" / "cap_15%_500m.json",
        "pso",
    )
    pair_files(
        "n20 15 500 sca-pso",
        ROOT / "results" / "sca_anchor_ablations" / "cap_15%_500m.json",
        "sca",
        ROOT / "results" / "sca_anchor_ablations" / "cap_15%_500m.json",
        "pso",
    )
    pair_files(
        "n100 15 500 anchor-pso",
        ROOT / "results" / "n100_500m_cap15" / "eval_anchor.json",
        "sca_anchor",
        ROOT / "results" / "n100_500m_cap15" / "eval.json",
        "pso",
    )


if __name__ == "__main__":
    main()
