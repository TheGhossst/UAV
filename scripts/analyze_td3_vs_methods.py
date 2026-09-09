"""Compare TD3 to SCA / random / k-means / PSO after the full eval.

Expected qualitative story (paper Figs. 6–7, kept as the reading frame):
  - Default geometry (I=10, J=3): TD3 below SCA.
  - As IoT count grows (Fig. 7): TD3 should close the gap and can beat SCA.

This script reports measured numbers against that story. It does not change
SimConfig or freeze SCA. Not a claim that paper 7–14 Mbps is reproduced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N100 = ROOT / "results" / "n100" / "eval_td3_SNAPSHOT_INVALID_do_not_cite.json"
CAMP = ROOT / "results" / "campaign_8.8mhz_cap25_td3_SNAPSHOT_INVALID_do_not_cite.json"
CAMP_CKPT = (
    ROOT / "results" / "campaign_8.8mhz_cap25_td3_SNAPSHOT_INVALID_do_not_cite.checkpoint.json"
)
OUT = ROOT / "results" / "td3" / "full_eval_analysis_SNAPSHOT_INVALID_do_not_cite.txt"

METHODS = ("sca", "td3", "random", "kmeans", "pso")
DEFAULT_I = 10
DEFAULT_J = 3


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("invalid") or data.get("do_not_cite"):
        raise SystemExit(
            f"refusing to analyze {path}: {data.get('reason')} "
            f"(moved_to={data.get('moved_to')})"
        )
    return data


def _mbps(pt: dict, method: str) -> float | None:
    s = (pt.get("by_method") or {}).get(method)
    if not s:
        return None
    return float(s["mean_sum_rate_Mbps"])


def _feas(pt: dict, method: str) -> float | None:
    s = (pt.get("by_method") or {}).get(method)
    if not s:
        return None
    return float(s["feasible_fraction"])


def _std(pt: dict, method: str) -> float | None:
    s = (pt.get("by_method") or {}).get(method)
    if not s:
        return None
    return float(s["std_sum_rate_Mbps"])


def _paired(a: list[float], b: list[float]) -> dict:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    d = x - y
    return {
        "mean_delta": float(np.mean(d)),
        "std_delta": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "win": float(np.mean(d > 0.0)),
        "n": int(d.size),
    }


def _fmt_row(pt: dict) -> str:
    bits = []
    for m in METHODS:
        v = _mbps(pt, m)
        if v is None:
            continue
        f = _feas(pt, m)
        s = _std(pt, m)
        bits.append(f"{m}={v:.4f}+/-{s:.4f} feas={100.0 * f:.0f}%")
    d = None
    if _mbps(pt, "td3") is not None and _mbps(pt, "sca") is not None:
        d = _mbps(pt, "td3") - _mbps(pt, "sca")
    extra = f"  TD3-SCA={d:+.4f}" if d is not None else ""
    return f"  {pt['label']}  " + "  ".join(bits) + extra


def _axis(camp: dict, name: str) -> list[dict]:
    pts = [p for p in camp.get("points", []) if p.get("axis") == name]
    pts.sort(key=lambda p: float(p["x"]))
    return pts


def _default_points(camp: dict) -> list[dict]:
    out = []
    for pt in camp.get("points", []):
        if int(pt.get("num_iot", -1)) != DEFAULT_I:
            continue
        if int(pt.get("num_uav", -1)) != DEFAULT_J:
            continue
        out.append(pt)
    return out


def analyze_n100(payload: dict) -> list[str]:
    lines = [
        "=== n100 Monte Carlo (100 frozen layouts, I=10 J=3, 8.8 MHz, 25% cap) ===",
        "This is the default-device bank. Snapshot export only; not the trained policy.",
    ]
    bm = payload["by_method"]
    lines.append(
        f"{'method':10s}  {'mean Mbps':>10s}  {'std':>8s}  {'feas %':>7s}"
    )
    for name in METHODS:
        if name not in bm:
            continue
        s = bm[name]
        lines.append(
            f"{name:10s}  {s['mean_sum_rate_Mbps']:10.4f}  "
            f"{s['std_sum_rate_Mbps']:8.4f}  "
            f"{100.0 * s['feasible_fraction']:6.1f}%"
        )
    if "td3" in bm and "sca" in bm:
        td3 = np.asarray(bm["td3"]["per_seed_Mbps"], dtype=float)
        sca = np.asarray(bm["sca"]["per_seed_Mbps"], dtype=float)
        pr = _paired(td3, sca)
        lines.append(
            f"TD3 vs SCA  d={pr['mean_delta']:+.4f} Mbps  "
            f"TD3-win={100.0 * pr['win']:.1f}%  n={pr['n']}"
        )
        ok = pr["mean_delta"] < 0.0
        lines.append(
            "story check (default): NOT SCORED on best-snapshot export"
            + (" (snapshot TD3 was below SCA)" if ok else " (snapshot TD3 was not below SCA)")
        )
    for other in ("random", "kmeans", "pso"):
        if "td3" not in bm or other not in bm:
            continue
        pr = _paired(bm["td3"]["per_seed_Mbps"], bm[other]["per_seed_Mbps"])
        lines.append(
            f"TD3 vs {other:8s}  d={pr['mean_delta']:+.4f}  "
            f"TD3-win={100.0 * pr['win']:.1f}%"
        )
    return lines


def analyze_campaign(camp: dict) -> list[str]:
    lines = [
        "=== campaign (20 seeds/point, same 8.8 MHz / 25% cap) ===",
        "Story: default I=10 J=3 TD3 < SCA; as I grows TD3 should improve vs SCA.",
        "These campaign TD3 numbers are best-snapshot bookkeeping, not policy export.",
    ]
    iots = [p for p in _axis(camp, "iots") if _mbps(p, "td3") is not None]
    uavs = [p for p in _axis(camp, "uavs") if _mbps(p, "td3") is not None]
    defaults = [p for p in _default_points(camp) if _mbps(p, "td3") is not None]

    lines.append("")
    lines.append("--- default-device points (I=10, J=3) ---")
    if not defaults:
        lines.append("  (TD3 not finished on any I=10 J=3 campaign point yet)")
    else:
        ds = []
        for pt in defaults:
            lines.append(_fmt_row(pt))
            ds.append(_mbps(pt, "td3") - _mbps(pt, "sca"))
        mean_d = float(np.mean(ds))
        lines.append(f"  mean TD3-SCA on default-device points: {mean_d:+.4f} Mbps")
        lines.append(
            "  story check (default): NOT SCORED on best-snapshot export"
            + (
                " (snapshot mean TD3-SCA was negative)"
                if mean_d < 0.0
                else " (snapshot mean TD3-SCA was not negative)"
            )
        )

    lines.append("")
    lines.append("--- Fig. 7 IoT growth (J=3, vary I) ---")
    if not iots:
        lines.append("  (TD3 not finished on the IoT axis yet)")
    else:
        deltas = []
        for pt in iots:
            lines.append(_fmt_row(pt))
            deltas.append(
                (
                    float(pt["x"]),
                    _mbps(pt, "td3") - _mbps(pt, "sca"),
                    _mbps(pt, "td3") - _mbps(pt, "random"),
                )
            )
        i0, d0, _ = deltas[0]
        i1, d1, _ = deltas[-1]
        lines.append(
            f"  TD3-SCA at I={i0:g}: {d0:+.4f} Mbps;  "
            f"at I={i1:g}: {d1:+.4f} Mbps;  shift={d1 - d0:+.4f}"
        )
        improved = d1 > d0
        beats = d1 > 0.0
        lines.append(
            "  story check (devices grow): NOT SCORED on best-snapshot export"
            + (" (snapshot gap did shrink)" if improved else " (snapshot gap did not shrink)")
        )
        lines.append(
            "  story check (large I): NOT SCORED on best-snapshot export"
            + (
                " (snapshot TD3 was above SCA at largest I)"
                if beats
                else " (snapshot TD3 was still at or below SCA)"
            )
        )

    lines.append("")
    lines.append("--- Fig. 6 UAV count (I=10, vary J) ---")
    if not uavs:
        lines.append("  (TD3 not finished on the UAV axis yet)")
    else:
        for pt in uavs:
            lines.append(_fmt_row(pt))

    for axis in ("lambda", "aodt", "cpu"):
        pts = [p for p in _axis(camp, axis) if _mbps(p, "td3") is not None]
        lines.append("")
        lines.append(f"--- axis {axis} ---")
        if not pts:
            lines.append("  (TD3 not finished)")
            continue
        for pt in pts:
            lines.append(_fmt_row(pt))
    return lines


def _progress(camp: dict) -> str:
    n_pts = 0
    n_done = 0
    for pt in camp.get("points", []):
        n_pts += 1
        s = (pt.get("by_method") or {}).get("td3") or {}
        if int(s.get("n", 0)) >= int(camp.get("n_runs") or 20):
            n_done += 1
    partial = camp.get("td3_partial") or {}
    n_part = sum(len(v) for v in partial.values())
    return (
        f"TD3 campaign progress: {n_done}/{n_pts} points complete, "
        f"{n_part} in-progress seeds"
    )


def main() -> int:
    lines = [
        "TD3 vs other methods — Khalaf et al. IEEE TNSM 2026 Alg. 2 fill-in",
        "Reproduction: 100x100 m, B_sys=8.8 MHz, 25% per-link cap. Not Table II 20 kHz.",
        "DO NOT CITE. Files are SNAPSHOT_INVALID. Best-snapshot export, not policy.",
        "No per-step trajectories or actor weights were saved. NOT SCORED.",
        "Policy-export numbers: results/td3/policy_export_*.txt",
        "",
    ]
    if N100.exists():
        lines.extend(analyze_n100(_load(N100)))
    else:
        lines.append(f"missing {N100}")
    lines.append("")
    camp_path = CAMP if CAMP.exists() else CAMP_CKPT
    if camp_path.exists():
        camp = _load(camp_path)
        lines.append(_progress(camp))
        lines.extend(analyze_campaign(camp))
    else:
        lines.append(f"missing {CAMP} and {CAMP_CKPT}")
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
