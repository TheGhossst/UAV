"""Paired readout for zenith-anchor default-point cases and J-sweep.

Usage:
  python scripts/analyze_sca_anchor_cases.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

PRACTICAL_MBPS = 0.05
# Match SCASettings.improvement_tolerance (1 bit/s). Keep-best vs SCA is
# non-negative by construction; p-values against SCA are not reported.
TIE_EPS_MBPS = 1.0e-6
CONSTRUCTION_BASELINES = {"sca"}
OUT_JSON = ROOT / "results" / "sca_anchor_cases_analysis.json"
OUT_TXT = ROOT / "results" / "sca_anchor_cases_analysis.txt"
ABLATIONS = {
    "random_anchor_n20_500m": ROOT / "results" / "sca_anchor_ablations" / "random_n20_500m.json",
    "random_anchor_n100_500m": ROOT / "results" / "sca_anchor_ablations" / "random_n100_500m.json",
    "beam_n20_100m": ROOT / "results" / "sca_anchor_ablations" / "beam_n20_100m.json",
    "beam_n20_500m": ROOT / "results" / "sca_anchor_ablations" / "beam_n20_500m.json",
    "k10_n20_500m": ROOT / "results" / "sca_anchor_ablations" / "k10_n20_500m.json",
    "bound_n100_500m": ROOT / "results" / "sca_anchor_ablations" / "bound_n100_500m.json",
    "cap_family": ROOT / "results" / "sca_anchor_ablations" / "cap_family.json",
    "deg_n20_500m": ROOT / "results" / "sca_anchor_ablations" / "deg_n20_500m.json",
    "deg_n100_500m": ROOT / "results" / "sca_anchor_ablations" / "deg_n100_500m.json",
}

CASES = {
    "n20_100m": {
        "kind": "n20",
        "path": ROOT / "results" / "sca_anchor_n20.json",
        "title": "20-seed campaign point, 100×100 m",
    },
    "n20_500m": {
        "kind": "n20",
        "path": ROOT / "results" / "sca_anchor_n20_500m.json",
        "title": "20-seed campaign point, 500×500 m",
    },
    "n100_100m": {
        "kind": "n100",
        "path": ROOT / "results" / "n100" / "eval_anchor.json",
        "title": "n100 frozen bank, 100×100 m",
    },
    "n100_500m": {
        "kind": "n100",
        "path": ROOT / "results" / "n100_500m_cap25" / "eval_anchor.json",
        "title": "n100 frozen bank, 500×500 m",
    },
    "n20_500m_cap15": {
        "kind": "n20",
        "path": ROOT / "results" / "sca_anchor_n20_500m_cap15.json",
        "title": "20-seed campaign point, 500×500 m, 15% cap",
    },
    "n100_500m_cap15": {
        "kind": "n100",
        "path": ROOT / "results" / "n100_500m_cap15" / "eval_anchor.json",
        "title": "n100 frozen bank, 500×500 m, 15% cap",
    },
}
SWEEPS = {
    "j_sweep_100m": ROOT / "results" / "campaign_sca_anchor_uavs.json",
    "j_sweep_500m": ROOT / "results" / "campaign_sca_anchor_uavs_500m.json",
    "i_sweep_500m": ROOT / "results" / "campaign_sca_anchor_iots_500m.json",
}
TK08 = ROOT / "results" / "sca_anchor_tk08.json"
TK08_REF = ROOT / "results" / "tk08_scajoint_cohesive.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_from_arrays(left: np.ndarray, right: np.ndarray, seeds: list[int]) -> dict:
    d = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    wins = int(np.sum(d > TIE_EPS_MBPS))
    losses = int(np.sum(d < -TIE_EPS_MBPS))
    ties = int(d.size) - wins - losses
    w = wilcoxon_signed_rank(d, zero_eps=TIE_EPS_MBPS)
    tstat = paired_t(d)
    loss_seeds = [int(s) for s, delta in zip(seeds, d) if delta < -TIE_EPS_MBPS]
    finite = d[np.isfinite(d)]
    return {
        "mean_delta_Mbps": float(np.mean(d)),
        "std_delta_Mbps": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "median_delta_Mbps": float(np.median(finite)) if finite.size else 0.0,
        "p10_delta_Mbps": float(np.percentile(finite, 10)) if finite.size else 0.0,
        "p90_delta_Mbps": float(np.percentile(finite, 90)) if finite.size else 0.0,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n": int(d.size),
        "n_moved": wins,
        "loss_seeds": loss_seeds,
        "n_practical": int(np.sum(d > PRACTICAL_MBPS)),
        "wilcoxon": w,
        "paired_t": tstat,
    }


def _fmt_pair(label: str, p: dict, *, construction: bool = False) -> str:
    dist = (
        f"median {p['median_delta_Mbps']:+.4f}  "
        f"p10–p90 [{p['p10_delta_Mbps']:+.4f}, {p['p90_delta_Mbps']:+.4f}]"
    )
    counts = (
        f"{p['wins']}/{p['n']} moved  {p['losses']} losses  {p.get('ties', 0)} ties  "
        f"practical>{PRACTICAL_MBPS:g}: {p['n_practical']}"
    )
    if construction:
        return (
            f"  vs {label:16s}  {p['mean_delta_Mbps']:+.4f} ± {p['std_delta_Mbps']:.4f}  "
            f"{dist}  {counts}  (keep-best; no p)"
        )
    w = p["wilcoxon"]
    p_g = w.get("p_greater")
    p_s = "n/a" if p_g is None else f"{p_g:.4g}"
    return (
        f"  vs {label:16s}  {p['mean_delta_Mbps']:+.4f} ± {p['std_delta_Mbps']:.4f}  "
        f"{dist}  {counts}  p_greater={p_s}"
    )


def _n20_rows(payload: dict) -> tuple[list[int], dict[str, np.ndarray], Counter]:
    rows = payload["per_seed"]
    seeds = [int(r["seed"]) for r in rows]
    by: dict[str, np.ndarray] = {}
    for m in ("random", "kmeans", "pso", "sca", "sca_multistart", "sca_anchor"):
        if all(m in r for r in rows):
            by[m] = np.array([float(r[m]["sum_rate_Mbps"]) for r in rows], dtype=float)
    kinds = Counter(
        str(r["sca_anchor"]["diagnostics"].get("winner_kind") or "unknown")
        for r in rows
    )
    return seeds, by, kinds


def _n100_rows(payload: dict) -> tuple[list[int], dict[str, np.ndarray], Counter]:
    by_m = payload["by_method"]
    seeds = [int(s) for s in by_m["sca_anchor"]["seeds"]]
    by = {m: np.asarray(by_m[m]["per_seed_Mbps"], dtype=float) for m in by_m}
    kinds: Counter = Counter()
    for run in payload.get("runs") or []:
        if run.get("method") != "sca_anchor":
            continue
        kind = (run.get("diagnostics") or {}).get("winner_kind") or "unknown"
        kinds[str(kind)] += 1
    return seeds, by, kinds


def summarize_case(case_id: str, spec: dict) -> dict | None:
    path = spec["path"]
    if not path.exists():
        return None
    payload = _load(path)
    if spec["kind"] == "n20":
        seeds, by, kinds = _n20_rows(payload)
        walls = payload.get("mean_wall_clock_s") or {}
        lp_best = None
        if "per_seed" in payload:
            vals = [
                r["sca_anchor"]["diagnostics"].get("lp_best_Mbps")
                for r in payload["per_seed"]
            ]
            vals = [float(v) for v in vals if v is not None]
            lp_best = float(np.mean(vals)) if vals else None
    else:
        seeds, by, kinds = _n100_rows(payload)
        walls = {}
        lp_best = None
        lp_vals = []
        for run in payload.get("runs") or []:
            if run.get("method") != "sca_anchor":
                continue
            v = (run.get("diagnostics") or {}).get("lp_best_Mbps")
            if v is not None:
                lp_vals.append(float(v))
        lp_best = float(np.mean(lp_vals)) if lp_vals else None

    means = {m: float(np.mean(v)) for m, v in by.items()}
    pairs = {}
    if "sca_anchor" in by:
        for other in ("sca", "sca_multistart", "pso", "random", "kmeans"):
            if other in by:
                pairs[other] = _pair_from_arrays(by["sca_anchor"], by[other], seeds)
    random_beat_sca = None
    still_ms = None
    still_an = None
    closed_an = None
    if "sca" in by and "random" in by:
        random_beat_sca = [
            int(s)
            for s, a, b in zip(seeds, by["sca"], by["random"])
            if float(a) + 1e-9 < float(b)
        ]
        if "sca_multistart" in by:
            still_ms = [
                int(s)
                for s, a, b in zip(seeds, by["sca_multistart"], by["random"])
                if int(s) in random_beat_sca and float(a) + 1e-9 < float(b)
            ]
        if "sca_anchor" in by:
            still_an = [
                int(s)
                for s, a, b in zip(seeds, by["sca_anchor"], by["random"])
                if int(s) in random_beat_sca and float(a) + 1e-9 < float(b)
            ]
            closed_an = [s for s in random_beat_sca if s not in still_an]
    extra = _extra_from_payload(payload, spec["kind"])
    return {
        "id": case_id,
        "title": spec["title"],
        "path": str(path),
        "n": len(seeds),
        "means_Mbps": means,
        "lp_best_mean_Mbps": lp_best,
        "winner_kinds": dict(kinds),
        "pairs": pairs,
        "mean_wall_clock_s": walls,
        "random_beat_sca_seeds": random_beat_sca,
        "multistart_still_lose_to_random": still_ms,
        "anchor_still_lose_to_random": still_an,
        "anchor_closed_random_losses": closed_an,
        **extra,
    }


def _extra_from_payload(payload: dict, kind: str) -> dict:
    """Polish / bound / K=1 reconstruction from stored diagnostics."""
    diags: list[dict] = []
    if kind == "n20":
        for r in payload.get("per_seed") or []:
            d = (r.get("sca_anchor") or {}).get("diagnostics") or {}
            diags.append(d)
    else:
        for run in payload.get("runs") or []:
            if run.get("method") != "sca_anchor":
                continue
            diags.append(run.get("diagnostics") or {})
    k1 = []
    k3 = []
    n_k_gt1 = 0
    disps = []
    assoc_chg = []
    bounds = []
    gaps = []
    jitters = []
    skips = []
    for d in diags:
        best = d.get("best_Mbps")
        if best is None:
            continue
        k3.append(float(best))
        pol = d.get("polished") or []
        fr = d.get("frozen_Mbps")
        if pol:
            r1 = pol[0]
            cand = []
            if not r1.get("failed") and r1.get("sum_rate_Mbps") is not None:
                cand.append(float(r1["sum_rate_Mbps"]))
            if fr is not None:
                cand.append(float(fr))
            k1.append(max(cand) if cand else float(best))
            win_c = d.get("winner_combo")
            if d.get("winner_kind") == "anchor" and r1.get("combo") != win_c:
                n_k_gt1 += 1
        if d.get("winner_polish_mean_disp_m") is not None:
            disps.append(float(d["winner_polish_mean_disp_m"]))
        elif d.get("polish_mean_disp_m") is not None:
            disps.append(float(d["polish_mean_disp_m"]))
        if d.get("winner_polish_assoc_changed") is not None:
            assoc_chg.append(int(bool(d["winner_polish_assoc_changed"])))
        if d.get("bound_Mbps") is not None:
            bounds.append(float(d["bound_Mbps"]))
            gaps.append(float(d["bound_Mbps"]) - float(best))
        if d.get("n_jittered_sep") is not None:
            jitters.append(int(d["n_jittered_sep"]))
        if d.get("n_skipped_sep") is not None:
            skips.append(int(d["n_skipped_sep"]))
    out: dict = {}
    if k1 and k3 and len(k1) == len(k3):
        dlt = np.asarray(k3, dtype=float) - np.asarray(k1, dtype=float)
        out["k1_recon_mean_Mbps"] = float(np.mean(k1))
        out["k3_mean_Mbps"] = float(np.mean(k3))
        out["k3_minus_k1_Mbps"] = float(np.mean(dlt))
        out["n_k_gt1_identity"] = int(n_k_gt1)
        out["n_k3_gt_k1"] = int(np.sum(dlt > TIE_EPS_MBPS))
        out["n_k3_gt_k1_practical"] = int(np.sum(dlt > PRACTICAL_MBPS))
    if disps:
        out["winner_polish_mean_disp_m"] = float(np.mean(disps))
    if assoc_chg:
        out["winner_polish_assoc_changed_frac"] = float(np.mean(assoc_chg))
    if bounds:
        out["bound_mean_Mbps"] = float(np.mean(bounds))
        out["gap_vs_bound_mean_Mbps"] = float(np.mean(gaps))
        out["gap_vs_bound_pct"] = float(
            100.0 * np.mean(np.asarray(gaps) / np.asarray(bounds))
        )
    if jitters:
        out["n_jittered_sep_mean"] = float(np.mean(jitters))
    if skips:
        out["n_skipped_sep_mean"] = float(np.mean(skips))
        out["n_skipped_sep_any"] = int(sum(1 for s in skips if s > 0))
    return out


def summarize_tk08() -> dict | None:
    if not TK08.exists():
        return None
    payload = _load(TK08)
    rows = payload.get("per_seed") or []
    if not rows:
        return None
    seeds = [int(r["seed"]) for r in rows]
    left = np.array([float(r["sum_rate_Mbps"]) for r in rows], dtype=float)
    feas = [bool(r["feasible"]) for r in rows]
    kinds = Counter(str(r.get("winner_kind") or "unknown") for r in rows)
    assoc = Counter(str(r.get("winner_assoc_kind") or "unknown") for r in rows)
    ref_mean = payload.get("sca_joint_cohesive_ref_Mbps")
    pair = None
    ref_rates = None
    if TK08_REF.exists():
        ref = _load(TK08_REF)
        by_seed = {int(s["seed"]): float(s["sum_rate_Mbps"]) for s in ref.get("seeds") or []}
        if all(s in by_seed for s in seeds):
            ref_rates = np.array([by_seed[s] for s in seeds], dtype=float)
            pair = _pair_from_arrays(left, ref_rates, seeds)
            ref_mean = float(np.mean(ref_rates))
    return {
        "id": "tk08_100m",
        "title": "T_k=0.8 s, 100×100 m, process-cohesive candidate per zenith set",
        "path": str(TK08),
        "n": len(seeds),
        "n_feasible": int(sum(feas)),
        "feasible_fraction": float(sum(feas) / len(feas)),
        "mean_Mbps": float(np.mean(left)),
        "ref_sca_joint_cohesive_Mbps": None if ref_mean is None else float(ref_mean),
        "winner_kinds": dict(kinds),
        "winner_assoc_kinds": dict(assoc),
        "vs_sca_joint_cohesive": pair,
        "wall_s": payload.get("wall_s"),
    }


def summarize_sweep(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = _load(path)
    rows = []
    for pt in payload.get("points") or []:
        if pt.get("axis") not in {"uavs", "iots"}:
            continue
        by = pt.get("by_method") or {}
        if "sca_anchor" not in by:
            continue
        j = int(float(pt["x"]))
        seeds = [int(s) for s in by["sca_anchor"]["seeds"]]
        left = np.asarray(by["sca_anchor"]["per_seed_Mbps"], dtype=float)
        rec = {
            "axis": pt.get("axis"),
            "x": j,
            "means_Mbps": {
                m: float(stats["mean_sum_rate_Mbps"]) for m, stats in by.items()
            },
            "feasible": {
                m: float(stats["feasible_fraction"]) for m, stats in by.items()
            },
        }
        if "sca" in by:
            rec["vs_sca"] = _pair_from_arrays(
                left, np.asarray(by["sca"]["per_seed_Mbps"], dtype=float), seeds
            )
        if "pso" in by:
            rec["vs_pso"] = _pair_from_arrays(
                left, np.asarray(by["pso"]["per_seed_Mbps"], dtype=float), seeds
            )
        rows.append(rec)
    return {"path": str(path), "points": rows}


def _fmt_case(case: dict) -> list[str]:
    lines = [f"{case['id']}: {case['title']}  n={case['n']}"]
    means = case["means_Mbps"]
    order = [m for m in ("sca_anchor", "sca_multistart", "sca", "pso", "random", "kmeans") if m in means]
    lines.append("  means  " + "  ".join(f"{m}={means[m]:.4f}" for m in order))
    if case.get("lp_best_mean_Mbps") is not None:
        lines.append(f"  LP-only mean {case['lp_best_mean_Mbps']:.4f} Mbps")
    kinds = case.get("winner_kinds") or {}
    if kinds:
        lines.append("  winner kinds  " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    walls = case.get("mean_wall_clock_s") or {}
    if walls.get("sca_anchor"):
        lines.append(f"  wall sca_anchor {walls['sca_anchor']:.2f}s/seed")
    for other in ("sca", "sca_multistart", "pso", "random", "kmeans"):
        p = (case.get("pairs") or {}).get(other)
        if p:
            lines.append(
                _fmt_pair(other, p, construction=other in CONSTRUCTION_BASELINES)
            )
    if case.get("k1_recon_mean_Mbps") is not None:
        lines.append(
            f"  K=1 recon {case['k1_recon_mean_Mbps']:.4f}  "
            f"K=3 {case['k3_mean_Mbps']:.4f}  "
            f"delta {case['k3_minus_k1_Mbps']:+.4f}  "
            f"identity K>1 {case['n_k_gt1_identity']}/{case['n']}  "
            f"rate K>1 {case['n_k3_gt_k1']}/{case['n']}  "
            f"practical {case['n_k3_gt_k1_practical']}"
        )
    if case.get("winner_polish_mean_disp_m") is not None:
        frac = case.get("winner_polish_assoc_changed_frac")
        frac_s = "n/a" if frac is None else f"{100.0 * frac:.0f}%"
        lines.append(
            f"  polish winner mean disp {case['winner_polish_mean_disp_m']:.2f} m  "
            f"nearest-a changed {frac_s}"
        )
    if case.get("bound_mean_Mbps") is not None:
        lines.append(
            f"  bound {case['bound_mean_Mbps']:.4f}  "
            f"gap {case['gap_vs_bound_mean_Mbps']:+.4f}  "
            f"({case['gap_vs_bound_pct']:.2f}% of bound)"
        )
    if case.get("n_skipped_sep_mean") is not None:
        lines.append(
            f"  sep skip mean {case['n_skipped_sep_mean']:.2f}  "
            f"any {case.get('n_skipped_sep_any', 0)}/{case['n']}"
            + (
                f"  jitter mean {case['n_jittered_sep_mean']:.2f}"
                if case.get("n_jittered_sep_mean") is not None
                else ""
            )
        )
    beat = case.get("random_beat_sca_seeds")
    still_ms = case.get("multistart_still_lose_to_random")
    still_an = case.get("anchor_still_lose_to_random")
    if beat is not None:
        lines.append(
            f"  SCA losses to random: {len(beat)}/{case['n']}  {beat}"
        )
        denom = max(len(beat), 1)
        if still_ms is not None:
            lines.append(
                f"  multi-start still lose: {len(still_ms)}/{len(beat)}  {still_ms}"
            )
        if still_an is not None:
            lines.append(
                f"  anchor still lose: {len(still_an)}/{len(beat)}  {still_an}"
            )
    return lines


def _fmt_tk08(rec: dict) -> list[str]:
    lines = [
        f"{rec['id']}: {rec['title']}  n={rec['n']}",
        f"  feasible {rec['n_feasible']}/{rec['n']}  "
        f"mean {rec['mean_Mbps']:.4f} Mbps",
    ]
    ref = rec.get("ref_sca_joint_cohesive_Mbps")
    if ref is not None:
        lines.append(f"  cohesive SCA-joint ref {ref:.4f} Mbps")
    kinds = rec.get("winner_kinds") or {}
    if kinds:
        lines.append("  winner kinds  " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    assoc = rec.get("winner_assoc_kinds") or {}
    if assoc:
        lines.append("  assoc  " + ", ".join(f"{k}={v}" for k, v in sorted(assoc.items())))
    p = rec.get("vs_sca_joint_cohesive")
    if p:
        lines.append(_fmt_pair("sca_joint_cohesive", p, construction=False))
    wall = rec.get("wall_s")
    if wall:
        lines.append(f"  wall {float(wall):.1f}s total  ({float(wall)/rec['n']:.1f}s/seed)")
    return lines


def _fmt_sweep(name: str, sweep: dict) -> list[str]:
    lines = [f"{name}: {sweep['path']}"]
    for pt in sweep["points"]:
        means = pt["means_Mbps"]
        label = "J" if pt.get("axis") != "iots" else "I"
        line = f"  {label}={pt.get('x', pt.get('J'))}  " + "  ".join(
            f"{m}={means[m]:.4f}"
            for m in ("sca_anchor", "sca", "pso", "random", "kmeans")
            if m in means
        )
        vs = pt.get("vs_sca")
        if vs:
            line += (
                f"  vs SCA {vs['mean_delta_Mbps']:+.4f} "
                f"median {vs['median_delta_Mbps']:+.4f}  "
                f"{vs['wins']}/{vs['n']} moved  "
                f"practical={vs['n_practical']}"
            )
        vp = pt.get("vs_pso")
        if vp:
            p_g = (vp.get("wilcoxon") or {}).get("p_greater")
            p_s = "n/a" if p_g is None else f"{p_g:.4g}"
            line += (
                f"  vs PSO {vp['mean_delta_Mbps']:+.4f} "
                f"{vp['wins']}/{vp['n']}  p={p_s}"
            )
        lines.append(line)
    return lines


def _ablation_summary(name: str, payload: dict) -> dict:
    """Compact readout; keep the raw file as the artifact."""
    if "mean_Mbps" in payload:
        return {
            "name": name,
            "means_Mbps": payload.get("mean_Mbps"),
            "n": payload.get("n_runs") or payload.get("n"),
            "label": payload.get("label"),
        }
    if "by_method" in payload:
        return {
            "name": name,
            "means_Mbps": {
                m: float(s.get("mean_sum_rate_Mbps"))
                for m, s in (payload.get("by_method") or {}).items()
                if s.get("mean_sum_rate_Mbps") is not None
            },
            "n": payload.get("n_scenarios") or payload.get("n"),
            "label": payload.get("label") or payload.get("note"),
        }
    if "cells" in payload:
        return {"name": name, "cells": payload.get("cells"), "label": payload.get("label")}
    if "mean_bound_Mbps" in payload or "bound_mean_Mbps" in payload:
        return {"name": name, **{k: v for k, v in payload.items() if k != "per_seed"}}
    return {"name": name, "keys": sorted(payload.keys())}


def _fmt_ablation(name: str, rec: dict) -> list[str]:
    lines = [f"ablation {name}:"]
    means = rec.get("means_Mbps")
    if means:
        lines.append(
            "  means  "
            + "  ".join(f"{m}={means[m]:.4f}" for m in means)
        )
    if rec.get("cells"):
        for cell in rec["cells"]:
            lines.append(
                "  "
                + "  ".join(f"{k}={v}" for k, v in cell.items() if k != "per_seed")
            )
    extra = {
        k: v
        for k, v in rec.items()
        if k not in {"name", "means_Mbps", "cells", "keys", "label", "n"}
    }
    if extra:
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in extra.items()))
    return lines


def main(argv: list[str] | None = None) -> int:
    del argv
    cases = []
    missing = []
    for case_id, spec in CASES.items():
        rec = summarize_case(case_id, spec)
        if rec is None:
            missing.append(case_id)
        else:
            cases.append(rec)
    sweeps = {}
    for name, path in SWEEPS.items():
        rec = summarize_sweep(path)
        if rec is None:
            missing.append(name)
        else:
            sweeps[name] = rec

    ablations = {}
    for name, path in ABLATIONS.items():
        if path.exists():
            ablations[name] = _load(path)
        else:
            missing.append(name)

    tk08 = summarize_tk08()
    if tk08 is None:
        missing.append("tk08_100m")

    payload = {
        "practical_Mbps": PRACTICAL_MBPS,
        "tie_eps_Mbps": TIE_EPS_MBPS,
        "construction_baselines": sorted(CONSTRUCTION_BASELINES),
        "cases": cases,
        "sweeps": sweeps,
        "tk08": tk08,
        "ablations": {k: _ablation_summary(k, v) for k, v in ablations.items()},
        "missing": missing,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "Zenith-anchor SCA (sca_anchor) vs frozen SCA / multi-start / baselines",
        "8.8 MHz, I=10 unless noted. 25% cap on the four default cells; 15% on *_cap15.",
        "Keep-best includes frozen k-means SCA. Tie 1e-6 Mbps (1 bit/s). No p vs SCA.",
        "",
    ]
    for case in cases:
        lines.extend(_fmt_case(case))
        lines.append("")
    for name, sweep in sweeps.items():
        lines.extend(_fmt_sweep(name, sweep))
        lines.append("")
    if tk08:
        lines.extend(_fmt_tk08(tk08))
        lines.append("")
    for name, rec in (payload.get("ablations") or {}).items():
        lines.extend(_fmt_ablation(name, rec))
        lines.append("")
    if missing:
        lines.append("missing: " + ", ".join(missing))
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
