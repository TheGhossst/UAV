"""Compare TD3 (Algorithm 2, policy export) to SCA / random / k-means / PSO.

Expected qualitative story (paper Figs. 6–7, kept as the reading frame):
  - Default geometry (I=10, J=3): TD3 below SCA.
  - As IoT count grows (Fig. 7): TD3 should close the gap and can beat SCA.

This script reports measured numbers against that story. Official TD3 score
is the trained policy (last-N xy, last-step a/b, frozen-q LP). Same-train
best-snapshot is A/B only. Does not change SimConfig. Not a claim that
paper 7–14 Mbps is reproduced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_winrate import wilcoxon_signed_rank  # noqa: E402

N100 = ROOT / "results" / "n100" / "eval_td3.json"
CAMP = ROOT / "results" / "campaign_8.8mhz_cap25_td3.json"
OUT = ROOT / "results" / "td3" / "full_eval_analysis.txt"
SUMMARY = ROOT / "results" / "td3" / "full_eval_summary.json"

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
    w = wilcoxon_signed_rank(d)
    return {
        "mean_delta": float(np.mean(d)),
        "std_delta": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "win": float(np.mean(d > 1e-12)),
        "loss": float(np.mean(d < -1e-12)),
        "tie": float(np.mean(np.abs(d) <= 1e-12)),
        "n": int(d.size),
        "wilcoxon_p_two_sided": float(w["p_two_sided"]),
        "wilcoxon_p_greater": float(w["p_greater"]),
        "wilcoxon_n_nonzero": int(w["n_nonzero"]),
    }


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3g}"


def _fmt_row(pt: dict) -> str:
    bits = []
    for m in METHODS:
        v = _mbps(pt, m)
        if v is None:
            continue
        f = _feas(pt, m)
        s = _std(pt, m)
        bits.append(f"{m}={v:.4f}+/-{s:.4f} feas={100.0 * f:.0f}%")
    extra = ""
    if _mbps(pt, "td3") is not None and _mbps(pt, "sca") is not None:
        extra = f"  TD3-SCA={_mbps(pt, 'td3') - _mbps(pt, 'sca'):+.4f}"
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


def _diag_field(stats: dict, key: str) -> list[float]:
    out = []
    for d in stats.get("diagnostics") or []:
        if not isinstance(d, dict) or key not in d:
            continue
        v = d[key]
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.append(float(v))
    return out


def _export_audit(stats: dict) -> dict:
    official = np.asarray(stats.get("per_seed_Mbps") or [], dtype=float)
    policy = np.asarray(_diag_field(stats, "policy_export_sum_rate_Mbps"), dtype=float)
    snap = np.asarray(_diag_field(stats, "snapshot_export_sum_rate_Mbps"), dtype=float)
    wall = np.asarray(_diag_field(stats, "wall_clock_s"), dtype=float)
    modes = {
        str(d.get("export_mode"))
        for d in (stats.get("diagnostics") or [])
        if isinstance(d, dict)
    }
    rules = {
        str(d.get("export_rule"))
        for d in (stats.get("diagnostics") or [])
        if isinstance(d, dict)
    }
    presets = {
        str(d.get("preset"))
        for d in (stats.get("diagnostics") or [])
        if isinstance(d, dict)
    }
    match = None
    if official.size and policy.size and official.size == policy.size:
        match = bool(np.max(np.abs(official - policy)) < 1e-9)
    return {
        "export_modes": sorted(modes),
        "export_rules": sorted(rules),
        "presets": sorted(presets),
        "official_equals_policy": match,
        "n_policy": int(policy.size),
        "n_snapshot": int(snap.size),
        "policy_mean": float(np.mean(policy)) if policy.size else None,
        "snapshot_mean": float(np.mean(snap)) if snap.size else None,
        "snapshot_minus_policy": (
            float(np.mean(snap - policy)) if snap.size and policy.size == snap.size else None
        ),
        "mean_wall_s": float(np.mean(wall)) if wall.size else None,
        "n_wall": int(wall.size),
    }


def analyze_n100(payload: dict) -> tuple[list[str], dict]:
    lines = [
        "=== n100 Monte Carlo (100 frozen layouts, I=10 J=3, 8.8 MHz, 25% cap) ===",
        "Official TD3 = policy export (last-10 xy, last-step a/b, frozen-q LP).",
    ]
    bm = payload["by_method"]
    lines.append(f"{'method':10s}  {'mean Mbps':>10s}  {'std':>8s}  {'feas %':>7s}")
    means = {}
    for name in METHODS:
        if name not in bm:
            continue
        s = bm[name]
        means[name] = {
            "mean": float(s["mean_sum_rate_Mbps"]),
            "std": float(s["std_sum_rate_Mbps"]),
            "feas": float(s["feasible_fraction"]),
        }
        lines.append(
            f"{name:10s}  {s['mean_sum_rate_Mbps']:10.4f}  "
            f"{s['std_sum_rate_Mbps']:8.4f}  "
            f"{100.0 * s['feasible_fraction']:6.1f}%"
        )
    paired = {}
    if "td3" in bm:
        for other in ("sca", "random", "kmeans", "pso"):
            if other not in bm:
                continue
            pr = _paired(bm["td3"]["per_seed_Mbps"], bm[other]["per_seed_Mbps"])
            paired[other] = pr
            lines.append(
                f"TD3 vs {other:8s}  d={pr['mean_delta']:+.4f} +/- {pr['std_delta']:.4f}  "
                f"win={100.0 * pr['win']:.1f}%  loss={100.0 * pr['loss']:.1f}%  "
                f"{_fmt_p(pr['wilcoxon_p_two_sided'])}  n={pr['n']}"
            )
    td3_runs = [
        r
        for r in payload.get("runs") or []
        if r.get("method") == "td3"
    ]
    if td3_runs:
        fake_stats = {
            "per_seed_Mbps": [float(r["sum_rate_Mbps"]) for r in td3_runs],
            "diagnostics": [r.get("diagnostics") or {} for r in td3_runs],
        }
        audit = _export_audit(fake_stats)
    else:
        audit = _export_audit(bm.get("td3") or {})
    if audit["official_equals_policy"] is True:
        lines.append(
            f"export audit: official Mbps == policy_export on all {audit['n_policy']} seeds"
        )
    elif audit["official_equals_policy"] is False:
        lines.append("export audit: WARNING official Mbps != policy_export")
    if audit["snapshot_minus_policy"] is not None:
        lines.append(
            f"same-train snapshot minus policy: {audit['snapshot_minus_policy']:+.4f} Mbps  "
            f"(snapshot={audit['snapshot_mean']:.4f}, policy={audit['policy_mean']:.4f})"
        )
    if audit["mean_wall_s"] is not None:
        lines.append(
            f"TD3 mean wall-clock: {audit['mean_wall_s']:.1f} s / scenario  "
            f"(n={audit['n_wall']})"
        )
    if "sca" in paired:
        ok = paired["sca"]["mean_delta"] < 0.0
        lines.append(
            "story check (default bank): TD3 below SCA: "
            + ("YES" if ok else "NO -- TD3 mean >= SCA")
        )
    summary = {"means": means, "paired": paired, "export": audit}
    return lines, summary


def analyze_campaign(camp: dict) -> tuple[list[str], dict]:
    lines = [
        "=== campaign (20 seeds/point, same 8.8 MHz / 25% cap) ===",
        "Story: default I=10 J=3 TD3 < SCA; as I grows TD3 should improve vs SCA.",
        "Official TD3 = policy export. Snapshot is A/B only.",
    ]
    iots = [p for p in _axis(camp, "iots") if _mbps(p, "td3") is not None]
    uavs = [p for p in _axis(camp, "uavs") if _mbps(p, "td3") is not None]
    defaults = [p for p in _default_points(camp) if _mbps(p, "td3") is not None]
    by_axis_out: dict[str, list[dict]] = {}
    n_td3_higher = 0
    n_pts = 0
    walls = []
    snap_minus_pol = []
    export_ok = True
    modes = set()
    for pt in camp.get("points", []):
        if "td3" not in pt.get("by_method", {}):
            continue
        n_pts += 1
        if _mbps(pt, "td3") is not None and _mbps(pt, "sca") is not None:
            if _mbps(pt, "td3") > _mbps(pt, "sca"):
                n_td3_higher += 1
        audit = _export_audit(pt["by_method"]["td3"])
        modes.update(audit["export_modes"])
        if audit["official_equals_policy"] is False:
            export_ok = False
        if audit["mean_wall_s"] is not None:
            walls.append(audit["mean_wall_s"])
        if audit["snapshot_minus_policy"] is not None:
            snap_minus_pol.append(audit["snapshot_minus_policy"])

    lines.append("")
    lines.append("--- default-device points (I=10, J=3) ---")
    default_ds = []
    j3_paired = None
    if not defaults:
        lines.append("  (TD3 not finished on any I=10 J=3 campaign point yet)")
    else:
        for pt in defaults:
            lines.append(_fmt_row(pt))
            default_ds.append(_mbps(pt, "td3") - _mbps(pt, "sca"))
            if pt.get("axis") == "uavs" and abs(float(pt["x"]) - 3.0) < 1e-9:
                j3_paired = {
                    other: _paired(
                        pt["by_method"]["td3"]["per_seed_Mbps"],
                        pt["by_method"][other]["per_seed_Mbps"],
                    )
                    for other in ("sca", "random", "kmeans", "pso")
                    if other in pt["by_method"]
                }
        mean_d = float(np.mean(default_ds))
        lines.append(f"  mean TD3-SCA on default-device copies: {mean_d:+.4f} Mbps")
        lines.append(
            "  story check (default): TD3 below SCA: "
            + ("YES" if mean_d < 0.0 else "NO -- mean TD3-SCA was not negative")
        )
        if j3_paired:
            for other, pr in j3_paired.items():
                lines.append(
                    f"  J=3 paired TD3 vs {other:8s}  d={pr['mean_delta']:+.4f} +/- "
                    f"{pr['std_delta']:.4f}  win={int(round(pr['win'] * pr['n']))}/{pr['n']}  "
                    f"{_fmt_p(pr['wilcoxon_p_two_sided'])}"
                )

    lines.append("")
    lines.append("--- Fig. 7 IoT growth (J=3, vary I) ---")
    iot_rows = []
    if not iots:
        lines.append("  (TD3 not finished on the IoT axis yet)")
        d0 = d1 = None
    else:
        deltas = []
        for pt in iots:
            lines.append(_fmt_row(pt))
            d_sca = _mbps(pt, "td3") - _mbps(pt, "sca")
            d_rnd = _mbps(pt, "td3") - _mbps(pt, "random")
            d_km = _mbps(pt, "td3") - _mbps(pt, "kmeans")
            pr = _paired(
                pt["by_method"]["td3"]["per_seed_Mbps"],
                pt["by_method"]["sca"]["per_seed_Mbps"],
            )
            iot_rows.append(
                {
                    "I": float(pt["x"]),
                    "sca": _mbps(pt, "sca"),
                    "td3": _mbps(pt, "td3"),
                    "random": _mbps(pt, "random"),
                    "kmeans": _mbps(pt, "kmeans"),
                    "pso": _mbps(pt, "pso"),
                    "td3_minus_sca": d_sca,
                    "td3_minus_random": d_rnd,
                    "td3_minus_kmeans": d_km,
                    "paired_vs_sca": pr,
                }
            )
            deltas.append((float(pt["x"]), d_sca, d_rnd))
        i0, d0, _ = deltas[0]
        i1, d1, _ = deltas[-1]
        lines.append(
            f"  TD3-SCA at I={i0:g}: {d0:+.4f} Mbps;  "
            f"at I={i1:g}: {d1:+.4f} Mbps;  shift={d1 - d0:+.4f}"
        )
        lines.append(
            "  story check (devices grow): TD3-SCA improves as I grows: "
            + ("YES" if d1 > d0 else "NO")
        )
        lines.append(
            "  story check (large I): TD3 > SCA at largest I: "
            + ("YES" if d1 > 0.0 else "NO -- policy TD3 still at or below SCA")
        )

    uav_rows = []
    lines.append("")
    lines.append("--- Fig. 6 UAV count (I=10, vary J) ---")
    if not uavs:
        lines.append("  (TD3 not finished on the UAV axis yet)")
    else:
        for pt in uavs:
            lines.append(_fmt_row(pt))
            uav_rows.append(
                {
                    "J": float(pt["x"]),
                    "sca": _mbps(pt, "sca"),
                    "td3": _mbps(pt, "td3"),
                    "random": _mbps(pt, "random"),
                    "kmeans": _mbps(pt, "kmeans"),
                    "pso": _mbps(pt, "pso"),
                    "td3_minus_sca": _mbps(pt, "td3") - _mbps(pt, "sca"),
                    "feas_td3": _feas(pt, "td3"),
                    "feas_sca": _feas(pt, "sca"),
                }
            )

    other_axes = {}
    for axis in ("lambda", "aodt", "cpu"):
        pts = [p for p in _axis(camp, axis) if _mbps(p, "td3") is not None]
        lines.append("")
        lines.append(f"--- axis {axis} ---")
        rows = []
        if not pts:
            lines.append("  (TD3 not finished)")
            other_axes[axis] = rows
            continue
        for pt in pts:
            lines.append(_fmt_row(pt))
            rows.append(
                {
                    "x": float(pt["x"]),
                    "label": pt["label"],
                    "sca": _mbps(pt, "sca"),
                    "td3": _mbps(pt, "td3"),
                    "random": _mbps(pt, "random"),
                    "kmeans": _mbps(pt, "kmeans"),
                    "pso": _mbps(pt, "pso"),
                    "td3_minus_sca": _mbps(pt, "td3") - _mbps(pt, "sca"),
                    "feas_td3": _feas(pt, "td3"),
                    "feas_sca": _feas(pt, "sca"),
                    "feas_random": _feas(pt, "random"),
                }
            )
        other_axes[axis] = rows

    mean_of_means = {}
    for other in ("sca", "random", "kmeans", "pso"):
        deltas = []
        wins = 0
        for pt in camp.get("points", []):
            bm = pt.get("by_method") or {}
            if "td3" not in bm or other not in bm:
                continue
            d = bm["td3"]["mean_sum_rate_Mbps"] - bm[other]["mean_sum_rate_Mbps"]
            deltas.append(d)
            if d > 0:
                wins += 1
        if deltas:
            mean_of_means[other] = {
                "mean_of_means_d": float(np.mean(deltas)),
                "points_td3_higher": wins,
                "n_points": len(deltas),
            }
            lines.append(
                f"  TD3 vs {other:8s}  mean-of-means d={np.mean(deltas):+.4f}  "
                f"points TD3 higher={wins}/{len(deltas)}"
            )

    lines.append("")
    lines.append("--- export / runtime audit ---")
    lines.append(f"  export_modes={sorted(modes)}")
    lines.append(
        "  official Mbps == policy_export on every seed: "
        + ("YES" if export_ok else "NO")
    )
    if snap_minus_pol:
        lines.append(
            f"  campaign-mean snapshot minus policy: {float(np.mean(snap_minus_pol)):+.4f} Mbps"
        )
    if walls:
        lines.append(
            f"  TD3 mean wall-clock: {float(np.mean(walls)):.1f} s / seed  "
            f"({len(walls)} point-means; ~{float(np.mean(walls)) * 20 / 60.0:.1f} min / point)"
        )
    lines.append(f"  points complete: {n_pts}; TD3 mean > SCA mean: {n_td3_higher}/{n_pts}")

    summary = {
        "n_points": n_pts,
        "td3_mean_higher_than_sca": n_td3_higher,
        "j3_paired": j3_paired,
        "iots": iot_rows,
        "uavs": uav_rows,
        "other_axes": other_axes,
        "mean_of_means": mean_of_means,
        "export_ok": export_ok,
        "mean_wall_s": float(np.mean(walls)) if walls else None,
        "snapshot_minus_policy": float(np.mean(snap_minus_pol)) if snap_minus_pol else None,
        "iot_gap_shift": None if d0 is None or d1 is None else float(d1 - d0),
        "iot_d_small": d0,
        "iot_d_large": d1,
    }
    return lines, summary


def _progress(camp: dict) -> str:
    n_pts = 0
    n_done = 0
    for pt in camp.get("points", []):
        n_pts += 1
        s = (pt.get("by_method") or {}).get("td3") or {}
        if int(s.get("n", 0)) >= int(camp.get("n_runs") or 20):
            n_done += 1
    return f"TD3 campaign progress: {n_done}/{n_pts} points complete"


def main() -> int:
    lines = [
        "TD3 vs other methods — Khalaf et al. IEEE TNSM 2026 Alg. 2 fill-in",
        "Reproduction: 100x100 m, B_sys=8.8 MHz, 25% per-link cap. Not Table II 20 kHz.",
        "Official score: deterministic policy, last-10 UAV xy mean, last-step a/b, frozen-q LP.",
        "College-server full eval (CUDA). Filenames on the dump still said SNAPSHOT_INVALID;",
        "export_mode on every seed is policy. Cite results/campaign_8.8mhz_cap25_td3.json",
        "and results/n100/eval_td3.json.",
        "",
    ]
    summary: dict = {}
    if N100.exists():
        n100_lines, n100_sum = analyze_n100(_load(N100))
        lines.extend(n100_lines)
        summary["n100"] = n100_sum
    else:
        lines.append(f"missing {N100}")
    lines.append("")
    if CAMP.exists():
        camp = _load(CAMP)
        lines.append(_progress(camp))
        camp_lines, camp_sum = analyze_campaign(camp)
        lines.extend(camp_lines)
        summary["campaign"] = camp_sum
    else:
        lines.append(f"missing {CAMP}")
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text, end="")
    print(f"wrote {OUT}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
