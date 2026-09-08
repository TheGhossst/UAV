"""Side-by-side 8.8 MHz cap=15% vs cap=25% across all campaign axes.

Uses the existing 20-seed full-axis campaigns. Prints J=3 default, every
sweep point, and SCA-vs-baseline Wilcoxon p at the default point.

Output: results/compare_cap15_vs_cap25.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402

CAP15 = ROOT / "results" / "campaign_8.8mhz_cap15_n20.json"
CAP25 = ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"
OUT = ROOT / "results" / "compare_cap15_vs_cap25.json"
METHODS = ("sca", "random", "kmeans", "pso")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def point(d: dict, axis: str, x: float) -> dict:
    return next(p for p in d["points"] if p["axis"] == axis and abs(float(p["x"]) - x) < 1e-9)


def means(pt: dict) -> dict[str, float]:
    return {m: pt["by_method"][m]["mean_sum_rate_Mbps"] for m in METHODS}


def feas(pt: dict) -> dict[str, float]:
    return {m: pt["by_method"][m]["feasible_fraction"] for m in METHODS}


def paired(pt: dict, baseline: str) -> dict[str, Any]:
    sca = np.array(pt["by_method"]["sca"]["per_seed_Mbps"], dtype=float)
    base = np.array(pt["by_method"][baseline]["per_seed_Mbps"], dtype=float)
    deltas = sca - base
    w = wilcoxon_signed_rank(deltas.tolist())
    std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan")
    return {
        "delta_mbps": float(np.mean(deltas)),
        "std_mbps": std,
        "wins": int(np.sum(deltas > 0)),
        "n": int(deltas.size),
        "p": float(w["p_two_sided"]),
    }


def summarize_point(pt15: dict, pt25: dict) -> dict[str, Any]:
    m15, m25 = means(pt15), means(pt25)
    spread15 = max(m15.values()) - min(m15.values())
    spread25 = max(m25.values()) - min(m25.values())
    return {
        "axis": pt15["axis"],
        "x": pt15["x"],
        "label": pt15["label"],
        "cap15": {
            "means_mbps": m15,
            "feasible": feas(pt15),
            "spread_mbps": spread15,
            "vs_random": paired(pt15, "random"),
            "vs_kmeans": paired(pt15, "kmeans"),
            "vs_pso": paired(pt15, "pso"),
        },
        "cap25": {
            "means_mbps": m25,
            "feasible": feas(pt25),
            "spread_mbps": spread25,
            "vs_random": paired(pt25, "random"),
            "vs_kmeans": paired(pt25, "kmeans"),
            "vs_pso": paired(pt25, "pso"),
        },
        "delta_cap15_minus_cap25_mbps": {m: m15[m] - m25[m] for m in METHODS},
    }


def _out(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def print_axis(rows: list[dict[str, Any]], axis: str) -> None:
    axis_rows = [r for r in rows if r["axis"] == axis]
    _out("")
    _out(f"=== {axis} ===")
    _out(
        f"{'x':>10s} {'SCA15':>7s} {'SCA25':>7s} {'dSCA':>7s} "
        f"{'rnd15':>7s} {'rnd25':>7s} {'spr15':>6s} {'spr25':>6s} "
        f"{'p_r15':>8s} {'p_r25':>8s} {'feas15':>6s}"
    )
    for r in axis_rows:
        a, b = r["cap15"], r["cap25"]
        _out(
            f"{r['x']:10g} {a['means_mbps']['sca']:7.3f} {b['means_mbps']['sca']:7.3f} "
            f"{r['delta_cap15_minus_cap25_mbps']['sca']:+7.3f} "
            f"{a['means_mbps']['random']:7.3f} {b['means_mbps']['random']:7.3f} "
            f"{a['spread_mbps']:6.3f} {b['spread_mbps']:6.3f} "
            f"{a['vs_random']['p']:8.4f} {b['vs_random']['p']:8.4f} "
            f"{a['feasible']['sca']:6.0%}"
        )


def main() -> int:
    if not CAP15.exists() or not CAP25.exists():
        print(f"need {CAP15} and {CAP25}", file=sys.stderr)
        return 1
    d15, d25 = load(CAP15), load(CAP25)
    keys15 = {(p["axis"], float(p["x"])) for p in d15["points"]}
    keys25 = {(p["axis"], float(p["x"])) for p in d25["points"]}
    if keys15 != keys25:
        print(f"axis/x mismatch: extra15={keys15-keys25} extra25={keys25-keys15}", file=sys.stderr)
        return 1

    rows = []
    for p in d15["points"]:
        rows.append(summarize_point(p, point(d25, p["axis"], float(p["x"]))))

    j3 = next(r for r in rows if r["axis"] == "uavs" and r["x"] == 3.0)
    payload = {
        "cap15_file": CAP15.name,
        "cap25_file": CAP25.name,
        "b_sys_hz": 8_800_000.0,
        "n_runs": 20,
        "default_j3": j3,
        "points": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _out("=" * 96)
    _out("8.8 MHz  cap=25% (primary) vs cap=15% (sensitivity)  (20 seeds, I=10, all axes)")
    _out("=" * 96)
    _out("DEFAULT J=3:")
    for tag, block in (("25%", j3["cap25"]), ("15%", j3["cap15"])):
        m = block["means_mbps"]
        _out(
            f"  cap={tag}  SCA={m['sca']:.3f}  random={m['random']:.3f}  "
            f"kmeans={m['kmeans']:.3f}  pso={m['pso']:.3f}  "
            f"spread={block['spread_mbps']:.3f}  "
            f"vs_rand d={block['vs_random']['delta_mbps']:+.4f} p={block['vs_random']['p']:.4f}  "
            f"vs_kmeans d={block['vs_kmeans']['delta_mbps']:+.4f} p={block['vs_kmeans']['p']:.4f}  "
            f"vs_pso d={block['vs_pso']['delta_mbps']:+.4f} p={block['vs_pso']['p']:.4f}"
        )
    dlt = j3["delta_cap15_minus_cap25_mbps"]
    _out(
        f"  SCA@15 minus SCA@25 = {dlt['sca']:+.4f} Mbps  "
        f"(random {dlt['random']:+.4f}, kmeans {dlt['kmeans']:+.4f}, pso {dlt['pso']:+.4f})"
    )

    for axis in ("uavs", "iots", "lambda", "aodt", "cpu"):
        print_axis(rows, axis)

    _out("")
    _out(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
