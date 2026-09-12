"""Analyze the 7.1-8.8 MHz x cap fine search.

Does not overwrite campaign JSON. Writes:
  results/bw_fine_7p1_8p8/analysis.json
  results/bw_fine_7p1_8p8/analysis.txt

Usage:
  python scripts/analyze_bw_fine_search.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "results" / "bw_fine_7p1_8p8" / "index.json"
OUT_JSON = ROOT / "results" / "bw_fine_7p1_8p8" / "analysis.json"
OUT_TXT = ROOT / "results" / "bw_fine_7p1_8p8" / "analysis.txt"

MHZ = tuple(round(7.1 + 0.1 * i, 1) for i in range(18))
CAP_ORDER = (None, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25)


def _cap_label(share: float | None) -> str:
    if share is None:
        return "none"
    return f"{int(round(share * 100))}%"


def _cap_key(share: float | None) -> str:
    if share is None:
        return "none"
    return f"{share:.2f}"


def linreg(x: np.ndarray, y: np.ndarray) -> dict:
    if x.size < 3:
        return {"slope": None, "r2": None, "intercept": None}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "y_at_7p1": float(slope * 7.1 + intercept),
        "y_at_8p8": float(slope * 8.8 + intercept),
    }


def main() -> int:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    cells = list(index.get("cells", {}).values())
    meta = index.get("meta") or {}

    by_key: dict[tuple[float, str], dict] = {}
    for c in cells:
        share = c.get("max_bw_share")
        by_key[(round(float(c["b_mhz"]), 1), _cap_key(share))] = c

    missing: list[str] = []
    for mhz in MHZ:
        for cap in CAP_ORDER:
            if (mhz, _cap_key(cap)) not in by_key:
                missing.append(f"{mhz:.1f}:{_cap_label(cap)}")

    rows: list[dict] = []
    for c in cells:
        s = c["summary"]
        share = c.get("max_bw_share")
        means = s["j3_means_mbps"]
        rows.append(
            {
                "mhz": round(float(c["b_mhz"]), 1),
                "cap": share,
                "cap_label": _cap_label(share),
                "reused": bool(c.get("reused")),
                "sca": float(means["sca"]),
                "random": float(means["random"]),
                "kmeans": float(means["kmeans"]),
                "pso": float(means["pso"]),
                "spread": float(s["j3_spread_mbps"]),
                "d_rand": float(s["sca_minus_random_mbps"]),
                "p_rand": float(s["j3_vs_random"]["p"]) if s.get("j3_vs_random") else None,
                "wins_rand": int(s["j3_vs_random"]["wins"]) if s.get("j3_vs_random") else None,
                "sca_best": bool(s["sca_best"]),
                "rank": list(s["j3_rank"]),
                "perfect": bool(s["perfect"]),
                "score": float(s["score"]),
                "feas_sca": float(s["j3_feas"]["sca"]),
                "j1_spread": s.get("j1_spread_mbps"),
                "i32_spread": s.get("i32_spread_mbps"),
                "tk08_sca_feas": s.get("tk08_sca_feas"),
                "sig": bool(s.get("sig_vs_random")),
            }
        )

    perfect = [r for r in rows if r["perfect"]]
    not_sca_best = [r for r in rows if not r["sca_best"]]
    feas_fail = [r for r in rows if r["feas_sca"] < 1.0]

    by_cap: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cap[r["cap_label"]].append(r)
    for lab in by_cap:
        by_cap[lab].sort(key=lambda x: x["mhz"])

    cap_stats: dict[str, dict] = {}
    for lab, grp in by_cap.items():
        mhz = np.array([g["mhz"] for g in grp], dtype=float)
        spread = np.array([g["spread"] for g in grp], dtype=float)
        sca = np.array([g["sca"] for g in grp], dtype=float)
        d_rand = np.array([g["d_rand"] for g in grp], dtype=float)
        spread_per_mhz = spread / mhz
        cap_stats[lab] = {
            "n": len(grp),
            "spread_mean": float(np.mean(spread)),
            "spread_min": float(np.min(spread)),
            "spread_max": float(np.max(spread)),
            "d_rand_mean": float(np.mean(d_rand)),
            "d_rand_min": float(np.min(d_rand)),
            "d_rand_max": float(np.max(d_rand)),
            "sca_best_n": int(sum(g["sca_best"] for g in grp)),
            "perfect_n": int(sum(g["perfect"] for g in grp)),
            "sig_n": int(sum(g["sig"] for g in grp)),
            "spread_per_mhz_mean": float(np.mean(spread_per_mhz)),
            "spread_per_mhz_cv": float(np.std(spread_per_mhz) / np.mean(spread_per_mhz))
            if float(np.mean(spread_per_mhz)) > 0
            else None,
            "spread_vs_mhz": linreg(mhz, spread),
            "sca_vs_mhz": linreg(mhz, sca),
            "d_rand_vs_mhz": linreg(mhz, d_rand),
            "rank_at_8p8": grp[-1]["rank"] if grp else None,
            "spread_at_7p1": float(grp[0]["spread"]) if grp else None,
            "spread_at_8p8": float(grp[-1]["spread"]) if grp else None,
            "sca_at_8p8": float(grp[-1]["sca"]) if grp else None,
            "p_at_8p8": grp[-1]["p_rand"] if grp else None,
            "not_sca_best": [
                {"mhz": g["mhz"], "rank": g["rank"]}
                for g in grp
                if not g["sca_best"]
            ],
        }

    tk08 = [r["tk08_sca_feas"] for r in rows if r["tk08_sca_feas"] is not None]
    tk08_all_zero = all(float(x) == 0.0 for x in tk08) if tk08 else None

    # Null: at fixed cap, spread(B) / B is constant.
    # Report mean |spread - k*mhz| / spread as relative residual.
    null_ok: dict[str, dict] = {}
    for lab, st in cap_stats.items():
        grp = by_cap[lab]
        k = st["spread_per_mhz_mean"]
        rel = [
            abs(g["spread"] - k * g["mhz"]) / g["spread"] if g["spread"] > 0 else 0.0
            for g in grp
        ]
        null_ok[lab] = {
            "k_spread_per_mhz": k,
            "max_rel_residual": float(max(rel)) if rel else None,
            "mean_rel_residual": float(np.mean(rel)) if rel else None,
            "r2": st["spread_vs_mhz"]["r2"],
        }

    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    perfect_ranked = [r for r in ranked if r["perfect"]]
    headline = next(
        (r for r in rows if r["mhz"] == 8.8 and r["cap_label"] == "25%"),
        None,
    )
    best_perfect = perfect_ranked[0] if perfect_ranked else None

    # Cap 10%: often PSO best
    pso_best = [r for r in rows if r["rank"][0] == "pso"]
    kmeans_best = [r for r in rows if r["rank"][0] == "kmeans"]
    random_best = [r for r in rows if r["rank"][0] == "random"]

    payload = {
        "n_cells": len(rows),
        "n_expected": 18 * 8,
        "missing": missing,
        "n_reused": int(sum(r["reused"] for r in rows)),
        "n_perfect": len(perfect),
        "perfect_by_cap": dict(Counter(r["cap_label"] for r in perfect)),
        "perfect_mhz_min": min((r["mhz"] for r in perfect), default=None),
        "perfect_mhz_max": max((r["mhz"] for r in perfect), default=None),
        "n_not_sca_best": len(not_sca_best),
        "not_sca_best_by_cap": dict(Counter(r["cap_label"] for r in not_sca_best)),
        "n_feas_fail_j3": len(feas_fail),
        "tk08_n": len(tk08),
        "tk08_all_zero": tk08_all_zero,
        "tk08_max": max((float(x) for x in tk08), default=None),
        "pso_best_n": len(pso_best),
        "pso_best_by_cap": dict(Counter(r["cap_label"] for r in pso_best)),
        "kmeans_best_n": len(kmeans_best),
        "random_best_n": len(random_best),
        "cap_stats": cap_stats,
        "null_linearity": null_ok,
        "headline_8p8_cap25": headline,
        "best_perfect": (
            None
            if best_perfect is None
            else {
                "mhz": best_perfect["mhz"],
                "cap": best_perfect["cap_label"],
                "sca": best_perfect["sca"],
                "spread": best_perfect["spread"],
                "d_rand": best_perfect["d_rand"],
                "score": best_perfect["score"],
            }
        ),
        "top10_score": [
            {
                "mhz": r["mhz"],
                "cap": r["cap_label"],
                "sca": r["sca"],
                "spread": r["spread"],
                "d_rand": r["d_rand"],
                "perfect": r["perfect"],
                "score": r["score"],
                "rank": r["rank"],
            }
            for r in ranked[:10]
        ],
        "elapsed_s": meta.get("elapsed_s"),
        "n_run": meta.get("n_run"),
        "n_skip": meta.get("n_skip"),
        "rows": rows,
    }

    lines: list[str] = []
    lines.append("Fine B_sys x cap search  7.1-8.8 MHz, 100x100 m, 20 seeds, no TD3")
    lines.append(
        f"cells {len(rows)}/{18 * 8}  missing {len(missing)}  reused {payload['n_reused']}  "
        f"perfect {len(perfect)}"
    )
    if missing:
        lines.append("MISSING: " + ", ".join(missing))
    lines.append(
        f"T_k=0.8 SCA feas: n={len(tk08)}  all_zero={tk08_all_zero}  max={payload['tk08_max']}"
    )
    lines.append(f"J=3 100% feas fails: {len(feas_fail)}")
    lines.append("")
    lines.append("Null: at fixed cap, J=3 spread ~ k * B_sys (Mbps/MHz).")
    lines.append(
        f"{'cap':>6} {'n':>3} {'spread':>7} {'d-rnd':>7} {'k/MHz':>8} {'r2':>6} "
        f"{'relmax':>7} {'SCA#1':>5} {'perf':>4} {'8.8 spr':>8}"
    )
    for lab in [_cap_label(c) for c in CAP_ORDER]:
        st = cap_stats[lab]
        nl = null_ok[lab]
        lines.append(
            f"{lab:>6} {st['n']:3d} {st['spread_mean']:7.3f} {st['d_rand_mean']:7.3f} "
            f"{st['spread_per_mhz_mean']:8.4f} {st['spread_vs_mhz']['r2']:6.4f} "
            f"{nl['max_rel_residual']:7.3f} {st['sca_best_n']:5d} {st['perfect_n']:4d} "
            f"{st['spread_at_8p8']:8.3f}"
        )
    lines.append("")
    lines.append(
        f"SCA uniquely best: {len(rows) - len(not_sca_best)}/{len(rows)}  "
        f"not-best by cap {payload['not_sca_best_by_cap']}"
    )
    lines.append(
        f"PSO best: {len(pso_best)}  by cap {payload['pso_best_by_cap']}  "
        f"k-means best {len(kmeans_best)}  random best {len(random_best)}"
    )
    lines.append(f"Perfect by cap: {payload['perfect_by_cap']}")
    if best_perfect:
        lines.append(
            f"Best-perfect (score artifact): {best_perfect['mhz']:.1f} MHz "
            f"cap={best_perfect['cap_label']}  spread={best_perfect['spread']:.3f}  "
            f"SCA={best_perfect['sca']:.3f}"
        )
    if headline:
        lines.append(
            f"Headline 8.8/25%: SCA={headline['sca']:.3f}  rnd={headline['random']:.3f}  "
            f"spread={headline['spread']:.3f}  d-rnd={headline['d_rand']:+.3f}  "
            f"p={headline['p_rand']:.3g}  perfect={headline['perfect']}  "
            f"rank={headline['rank']}"
        )
    lines.append("")
    lines.append("Top 10 by search score (do not promote to headline B_sys):")
    for r in ranked[:10]:
        lines.append(
            f"  {r['mhz']:.1f} {r['cap_label']:>4}  SCA {r['sca']:.3f}  "
            f"spr {r['spread']:.3f}  d-rnd {r['d_rand']:+.3f}  "
            f"{'PERF' if r['perfect'] else '    '}  {r['rank']}"
        )
    lines.append("")
    lines.append(
        "Verdict: cap is the lever; B_sys only scales leftover-dump Mbps. "
        "Do not replace 8.8 MHz / 25%."
    )

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    try:
        print(text, end="")
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), end="")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TXT}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
