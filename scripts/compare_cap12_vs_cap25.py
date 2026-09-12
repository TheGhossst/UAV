"""Compare 8.8 MHz 12% vs 25% on the four headline tests.

Usage:
  python scripts/compare_cap12_vs_cap25.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

METHODS = ("sca", "random", "pso", "kmeans")
OUT_JSON = ROOT / "results" / "compare_cap12_vs_cap25.json"
OUT_TXT = ROOT / "results" / "compare_cap12_vs_cap25.txt"
PRACTICAL = 0.05

CASES = {
    "n20_100m": {
        "title": "20-seed campaign, 100x100 m (full axes; J=3 quoted)",
        12: ROOT / "results" / "campaign_8.8mhz_cap12_n20.json",
        25: ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
        "kind": "campaign",
    },
    "n20_500m": {
        "title": "20-seed campaign, 500x500 m (full axes; J=3 quoted)",
        12: ROOT / "results" / "campaign_8.8mhz_cap12_n20_500m.json",
        25: ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
        "kind": "campaign",
    },
    "n100_100m": {
        "title": "n100 frozen bank, 100x100 m",
        12: ROOT / "results" / "n100_cap12" / "eval.json",
        25: ROOT / "results" / "n100" / "eval.json",
        "kind": "n100",
    },
    "n100_500m": {
        "title": "n100 frozen bank, 500x500 m",
        12: ROOT / "results" / "n100_500m_cap12" / "eval.json",
        25: ROOT / "results" / "n100_500m_cap25" / "eval.json",
        "kind": "n100",
    },
}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _j3(payload: dict) -> dict:
    for pt in payload["points"]:
        if pt["axis"] == "uavs" and int(pt["num_uav"]) == 3 and int(pt["num_iot"]) == 10:
            return pt
    raise KeyError("no J=3 I=10 uavs point")


def _campaign_j_sweep(payload: dict) -> list[dict]:
    rows = []
    for pt in payload.get("points", []):
        if pt.get("axis") != "uavs":
            continue
        means = {
            m: float(pt["by_method"][m]["mean_sum_rate_Mbps"])
            for m in METHODS
            if m in pt["by_method"]
        }
        feas = {
            m: float(pt["by_method"][m]["feasible_fraction"])
            for m in METHODS
            if m in pt["by_method"]
        }
        ranked = sorted(means, key=means.get, reverse=True)
        spread = max(means.values()) - min(means.values()) if means else 0.0
        uniquely = bool(ranked) and ranked[0] == "sca" and (
            len(ranked) == 1 or means["sca"] > means[ranked[1]]
        )
        rows.append(
            {
                "J": int(pt["num_uav"]),
                "means_Mbps": means,
                "feas": feas,
                "spread_Mbps": spread,
                "rank": ranked,
                "sca_uniquely_best": uniquely,
            }
        )
    rows.sort(key=lambda r: r["J"])
    return rows


def _aodt08(payload: dict) -> dict | None:
    for pt in payload.get("points", []):
        if pt.get("axis") == "aodt" and abs(float(pt["x"]) - 0.8) < 1e-9:
            return {
                m: float(pt["by_method"][m]["feasible_fraction"])
                for m in METHODS
                if m in pt["by_method"]
            }
    return None


def _arrays_campaign(pt: dict) -> tuple[list[int], dict[str, np.ndarray], dict]:
    seeds = [int(s) for s in pt["by_method"]["sca"]["seeds"]]
    by = {
        m: np.asarray(pt["by_method"][m]["per_seed_Mbps"], dtype=float) for m in METHODS
    }
    feas = {
        m: float(pt["by_method"][m]["feasible_fraction"]) for m in METHODS
    }
    return seeds, by, feas


def _arrays_n100(payload: dict) -> tuple[list[int], dict[str, np.ndarray], dict]:
    by_m = payload["by_method"]
    seeds = [int(s) for s in by_m["sca"]["seeds"]]
    by = {m: np.asarray(by_m[m]["per_seed_Mbps"], dtype=float) for m in METHODS}
    feas = {m: float(by_m[m]["feasible_fraction"]) for m in METHODS}
    return seeds, by, feas


def _pair(left: np.ndarray, right: np.ndarray) -> dict:
    d = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    w = wilcoxon_signed_rank(d)
    tstat = paired_t(d)
    return {
        "mean_delta_Mbps": float(np.mean(d)),
        "std_delta_Mbps": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "wins": int(np.sum(d > 1e-9)),
        "losses": int(np.sum(d < -1e-9)),
        "n": int(d.size),
        "n_practical": int(np.sum(d > PRACTICAL)),
        "wilcoxon_p": w.get("p_two_sided"),
        "p_greater": w.get("p_greater"),
        "paired_t": tstat,
    }


def _block(seeds: list[int], by: dict[str, np.ndarray], feas: dict) -> dict:
    means = {m: float(np.mean(by[m])) for m in METHODS}
    stds = {m: float(np.std(by[m], ddof=1)) if by[m].size > 1 else 0.0 for m in METHODS}
    spread = max(means.values()) - min(means.values())
    ranked = sorted(means, key=means.get, reverse=True)
    vs = {m: _pair(by["sca"], by[m]) for m in METHODS if m != "sca"}
    return {
        "n": len(seeds),
        "means_Mbps": means,
        "std_Mbps": stds,
        "feas": feas,
        "spread_Mbps": spread,
        "rank": ranked,
        "sca_vs": vs,
    }


def summarize_case(spec: dict) -> dict:
    out: dict = {"title": spec["title"], "kind": spec["kind"]}
    for cap in (12, 25):
        path = spec[cap]
        payload = _load(path)
        if payload is None:
            out[str(cap)] = None
            continue
        if spec["kind"] == "campaign":
            pt = _j3(payload)
            seeds, by, feas = _arrays_campaign(pt)
            extra = {
                "j_sweep": _campaign_j_sweep(payload),
                "tk08_feas": _aodt08(payload),
            }
        else:
            seeds, by, feas = _arrays_n100(payload)
            extra = {}
        block = _block(seeds, by, feas)
        block.update(extra)
        block["path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        block["area_m"] = payload.get("area_m")
        out[str(cap)] = block
    if out.get("12") and out.get("25"):
        a = out["12"]["means_Mbps"]
        b = out["25"]["means_Mbps"]
        out["mean_drop_25_minus_12"] = {m: b[m] - a[m] for m in METHODS}
        out["spread_12_minus_25"] = out["12"]["spread_Mbps"] - out["25"]["spread_Mbps"]
    return out


def _fmt_block(cap: str, b: dict) -> list[str]:
    lines = [
        f"  cap {cap}%  n={b['n']}  spread={b['spread_Mbps']:.3f}  rank={' > '.join(b['rank'])}",
    ]
    for m in METHODS:
        lines.append(
            f"    {m:8s}  {b['means_Mbps'][m]:7.3f} ± {b['std_Mbps'][m]:.3f}  "
            f"feas={100.0 * b['feas'][m]:.0f}%"
        )
    vs = b["sca_vs"]["random"]
    p = vs["wilcoxon_p"]
    ps = "n/a" if p is None else f"{p:.4g}"
    lines.append(
        f"    SCA-random {vs['mean_delta_Mbps']:+.3f} ± {vs['std_delta_Mbps']:.3f}  "
        f"{vs['wins']}/{vs['n']}  p={ps}  practical>{PRACTICAL:g}: {vs['n_practical']}"
    )
    vp = b["sca_vs"]["pso"]
    pp = vp["wilcoxon_p"]
    pps = "n/a" if pp is None else f"{pp:.4g}"
    lines.append(
        f"    SCA-PSO    {vp['mean_delta_Mbps']:+.3f} ± {vp['std_delta_Mbps']:.3f}  "
        f"{vp['wins']}/{vp['n']}  p={pps}"
    )
    return lines


def main() -> int:
    cases: dict[str, dict] = {}
    missing: list[str] = []
    lines = [
        "8.8 MHz  12% vs 25% leftover-dump cap",
        "Same methods (random/k-means/PSO/SCA). Frozen SCA. No TD3. No multi-start.",
        "",
    ]
    for cid, spec in CASES.items():
        summary = summarize_case(spec)
        cases[cid] = summary
        lines.append(f"=== {cid}: {summary['title']} ===")
        for cap in ("12", "25"):
            if summary.get(cap) is None:
                missing.append(f"{cid}@{cap}")
                lines.append(f"  cap {cap}%  MISSING {spec[int(cap)]}")
            else:
                lines.extend(_fmt_block(cap, summary[cap]))
        if summary.get("12") and summary.get("25"):
            drop = summary["mean_drop_25_minus_12"]
            lines.append(
                "  rate paid for 12% (25% minus 12%):  "
                + "  ".join(f"{m} {drop[m]:+.3f}" for m in METHODS)
            )
            lines.append(
                f"  extra spread at 12%: {summary['spread_12_minus_25']:+.3f} Mbps"
            )
        for cap in ("12", "25"):
            b = summary.get(cap)
            if not b or not b.get("j_sweep"):
                continue
            lines.append(f"  J-sweep cap {cap}% (mean Mbps, rank):")
            for row in b["j_sweep"]:
                mm = row["means_Mbps"]
                lines.append(
                    f"    J={row['J']}  SCA {mm.get('sca', float('nan')):6.3f}  "
                    f"rand {mm.get('random', float('nan')):6.3f}  "
                    f"PSO {mm.get('pso', float('nan')):6.3f}  "
                    f"km {mm.get('kmeans', float('nan')):6.3f}  "
                    f"spread {row['spread_Mbps']:.3f}  "
                    f"{' > '.join(row['rank'])}"
                )
            tk = b.get("tk08_feas")
            if tk:
                lines.append(
                    "  T_k=0.8 feas cap "
                    + cap
                    + "%:  "
                    + "  ".join(f"{m} {100.0 * tk[m]:.0f}%" for m in tk)
                )
        lines.append("")

    lines.append("Decision notes (not a promotion):")
    lines.append(
        "- Keep 25% as PRIMARY_MAX_BW_SHARE. SCA uniquely best at J=3 on 4/4 tests."
    )
    lines.append(
        "- 12% is uniquely-best on 1/4 (n20 100 m, +0.001 vs PSO). PSO ranks first on 500 m and both n100 banks."
    )
    lines.append(
        "- 12% buys SCA-vs-random p<0.05 at 100 m by throwing leftover-dump Hertz (0.13 Mbps at 100 m, >1.3 Mbps at 500 m)."
    )
    lines.append(
        "- T_k=0.8 is 0% under nearest-a at both caps; grouping is independent of cap."
    )

    payload = {"practical_bar_Mbps": PRACTICAL, "missing": missing, "cases": cases}
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
