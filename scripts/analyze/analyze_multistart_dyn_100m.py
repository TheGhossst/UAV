"""Paired readout: dynamic vs frozen multi-start — default n100 + §VII campaign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

EVAL_ROOT = ROOT / "results" / "run_10mhz_open" / "multistart_dyn_100m"
REF_ROOT = ROOT / "results" / "run_10mhz_cap25" / "100m"
OUT_JSON = EVAL_ROOT / "analysis.json"
OUT_TXT = EVAL_ROOT / "analysis.txt"
PRACTICAL_MBPS = 0.05

METHODS = ("sca", "sca_multistart", "sca_multistart_dyn")
REF_METHOD_FILES = {
    "sca": REF_ROOT / "campaign_sca.json",
    "sca_multistart": REF_ROOT / "campaign_sca_multistart.json",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _point_key(pt: dict) -> tuple[str, float]:
    return (str(pt.get("axis")), float(pt.get("x")))


def _index_campaign(payload: dict) -> dict[tuple[str, float], dict]:
    return {_point_key(pt): pt for pt in payload.get("points") or []}


def _per_seed(by: dict, method: str) -> np.ndarray | None:
    block = (by or {}).get(method) or {}
    arr = block.get("per_seed_Mbps")
    if not arr:
        return None
    return np.asarray(arr, dtype=float)


def _pair(left: np.ndarray, right: np.ndarray, *, label: str) -> dict:
    d = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    wins = int(np.sum(d > 1e-9))
    losses = int(np.sum(d < -1e-9))
    ties = int(d.size - wins - losses)
    w = wilcoxon_signed_rank(d)
    return {
        "label": label,
        "mean_delta_Mbps": float(np.mean(d)),
        "std_delta_Mbps": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_practical": int(np.sum(d > PRACTICAL_MBPS)),
        "wilcoxon": w,
        "paired_t": paired_t(d),
    }


def _analyze_n100(payload: dict) -> dict:
    by = payload["by_method"]
    dyn = np.asarray(by["sca_multistart_dyn"]["per_seed_Mbps"], dtype=float)
    out = {
        "kind": "default_n100",
        "mean_Mbps": {k: float(by[k]["mean_sum_rate_Mbps"]) for k in by if k in METHODS},
        "pairs": [],
    }
    if "sca" in by:
        out["pairs"].append(
            _pair(dyn, np.asarray(by["sca"]["per_seed_Mbps"], dtype=float), label="dyn_ms − sca")
        )
    if "sca_multistart" in by:
        out["pairs"].append(
            _pair(
                dyn,
                np.asarray(by["sca_multistart"]["per_seed_Mbps"], dtype=float),
                label="dyn_ms − frozen_ms",
            )
        )
    return out


def _load_ref_index() -> dict[str, dict[tuple[str, float], dict]]:
    out: dict[str, dict[tuple[str, float], dict]] = {}
    for method, path in REF_METHOD_FILES.items():
        if path.is_file():
            out[method] = _index_campaign(_load(path))
    return out


def _analyze_campaign(payload: dict, ref_index: dict[str, dict[tuple[str, float], dict]]) -> dict:
    rows: list[dict] = []
    by_axis: dict[str, list[dict]] = {}
    for pt in payload.get("points") or []:
        by = pt.get("by_method") or {}
        dyn = _per_seed(by, "sca_multistart_dyn")
        if dyn is None:
            continue
        key = _point_key(pt)
        row = {
            "axis": pt.get("axis"),
            "x": pt.get("x"),
            "label": pt.get("label"),
            "mean_dyn_Mbps": float(by["sca_multistart_dyn"]["mean_sum_rate_Mbps"]),
            "feasible_fraction": float(by["sca_multistart_dyn"].get("feasible_fraction") or 0.0),
            "ref_mean_Mbps": {},
            "pairs": [],
        }
        for method, idx in ref_index.items():
            ref_pt = idx.get(key)
            if ref_pt is None:
                continue
            ref_by = ref_pt.get("by_method") or {}
            ref_seed = _per_seed(ref_by, method)
            if ref_seed is None:
                continue
            row["ref_mean_Mbps"][method] = float(ref_by[method]["mean_sum_rate_Mbps"])
            if method == "sca":
                row["pairs"].append(_pair(dyn, ref_seed, label="dyn_ms − sca (cap25 ref)"))
            elif method == "sca_multistart":
                row["pairs"].append(
                    _pair(dyn, ref_seed, label="dyn_ms − frozen_ms (cap25 ref)")
                )
        rows.append(row)
        by_axis.setdefault(str(row["axis"]), []).append(row)

    axis_order = ("uavs", "iots", "lambda", "aodt", "cpu")
    summary_axes: list[dict] = []
    for axis in axis_order:
        pts = by_axis.get(axis) or []
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: float(r["x"]))
        frozen_pairs = [
            p for r in pts for p in r["pairs"] if p["label"].startswith("dyn_ms − frozen_ms")
        ]
        sca_pairs = [p for r in pts for p in r["pairs"] if p["label"].startswith("dyn_ms − sca")]
        summary_axes.append(
            {
                "axis": axis,
                "n_points": len(pts),
                "mean_dyn_Mbps": [round(r["mean_dyn_Mbps"], 4) for r in pts],
                "x": [float(r["x"]) for r in pts],
                "vs_frozen_ms": {
                    "mean_delta_range": [
                        min(p["mean_delta_Mbps"] for p in frozen_pairs),
                        max(p["mean_delta_Mbps"] for p in frozen_pairs),
                    ]
                    if frozen_pairs
                    else None,
                    "n_significant_wilcoxon": sum(
                        1
                        for p in frozen_pairs
                        if (p["wilcoxon"].get("p_greater") or 1.0) < 0.05
                    ),
                    "total_losses": sum(p["losses"] for p in frozen_pairs),
                },
                "vs_sca": {
                    "mean_delta_range": [
                        min(p["mean_delta_Mbps"] for p in sca_pairs),
                        max(p["mean_delta_Mbps"] for p in sca_pairs),
                    ]
                    if sca_pairs
                    else None,
                    "n_significant_wilcoxon": sum(
                        1 for p in sca_pairs if (p["wilcoxon"].get("p_greater") or 1.0) < 0.05
                    ),
                },
            }
        )

    return {"points": rows, "summary_by_axis": summary_axes, "n_points": len(rows)}


def _fmt_pair(p: dict) -> str:
    w = p["wilcoxon"]
    p_g = w.get("p_greater")
    p_s = "n/a" if p_g is None else f"{p_g:.4g}"
    n = p["wins"] + p["losses"] + p["ties"]
    return (
        f"    {p['label']:32s}  {p['mean_delta_Mbps']:+.4f} ± {p['std_delta_Mbps']:.4f}  "
        f"{p['wins']}/{n} wins  {p['losses']} losses  "
        f"practical>{PRACTICAL_MBPS:g}: {p['n_practical']}  p={p_s}"
    )


def main() -> int:
    lines: list[str] = [
        "Dynamic-assignment multi-start SCA (sca_multistart_dyn)",
        "100×100 m, 10 MHz, 25% cap, 100 seeds/point",
        f"Campaign: {EVAL_ROOT / 'campaign_eval.json'}",
        f"Reference paired baselines: {REF_ROOT} campaign_sca + campaign_sca_multistart",
        "",
    ]
    report: dict = {}

    for rel in ("default/n100_eval.json", "cap25/n100_eval.json"):
        path = EVAL_ROOT / rel
        if path.is_file():
            default = _analyze_n100(_load(path))
            report["default"] = default
            lines.append("--- Default (n100 bank, J=3 I=10 layouts) ---")
            for m, v in default["mean_Mbps"].items():
                lines.append(f"  {m:22s}  {v:.4f} Mbps")
            for p in default["pairs"]:
                lines.append(_fmt_pair(p))
            lines.append("")
            break

    camp_path = EVAL_ROOT / "campaign_eval.json"
    ref_index = _load_ref_index()
    if camp_path.is_file():
        camp = _analyze_campaign(_load(camp_path), ref_index)
        report["campaign"] = camp
        lines.append(f"--- §VII axis sweep ({camp['n_points']} points, 100 seeds each) ---")
        lines.append(
            "Paired deltas use the same seeds as results/run_10mhz_cap25/100m/ "
            "(dynamic SCA and frozen multi-start reference campaigns)."
        )
        lines.append("")
        cur_axis = None
        for row in camp["points"]:
            if row["axis"] != cur_axis:
                cur_axis = row["axis"]
                lines.append(f"  [{cur_axis}]")
            lines.append(
                f"    x={row['x']!s:6s}  dyn={row['mean_dyn_Mbps']:.4f} Mbps  "
                f"feas={row['feasible_fraction']:.0%}"
            )
            for method, mean in row.get("ref_mean_Mbps", {}).items():
                lines.append(f"      ref {method:18s}  {mean:.4f} Mbps")
            for p in row["pairs"]:
                lines.append(_fmt_pair(p))
        lines.append("")
        lines.append("--- Axis summary (dyn_ms − frozen_ms vs cap25 ref) ---")
        for ax in camp["summary_by_axis"]:
            vf = ax["vs_frozen_ms"]
            rng = vf["mean_delta_range"]
            rng_s = "n/a" if rng is None else f"[{rng[0]:+.4f}, {rng[1]:+.4f}]"
            lines.append(
                f"  {ax['axis']:6s}  n={ax['n_points']}  Δ range {rng_s}  "
                f"losses={vf['total_losses']}  sig(p<0.05)={vf['n_significant_wilcoxon']}/{ax['n_points']}"
            )

    if not report:
        print(f"no results under {EVAL_ROOT}", file=sys.stderr)
        return 1

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
