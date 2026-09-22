"""Paired Hertz readout for E1 / medoid / E1b.

Wording rules
-------------
- Lower Hertz is better. Deltas are ``other - champion`` in kHz, so a
  positive number means the champion uses fewer Hertz.
- ``ratio_of_means`` is (mean_other - mean_champ) / mean_other.
  ``mean_of_ratios`` is the mean of per-seed (other-champ)/other.
  Say which one you quote.
- Overprovision is 8.8 MHz / method mean, and is method-specific.
- Search abs_tol is 1 kHz; gaps of a few kHz at 100 m are not practical.

Usage:
  python scripts/analyze/analyze_min_spectrum.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

PLACE_HZ = 8.8e6
PRACTICAL_HZ = 50.0e3
TIE_EPS_HZ = 1.0e3
SEARCH_ABS_TOL_HZ = 1.0e3
CASES = {
    "n20_100m": ROOT / "results" / "min_spectrum_n20_100m.json",
    "n20_500m": ROOT / "results" / "min_spectrum_n20_500m.json",
    "n20_500m_j2": ROOT / "results" / "min_spectrum_n20_500m_j2.json",
}
MEDOID_MBPS = {
    "n20_100m": ROOT / "results" / "sca_medoid_n20_100m.json",
    "n20_500m": ROOT / "results" / "sca_medoid_n20_500m.json",
    "n20_500m_j2": ROOT / "results" / "sca_medoid_n20_500m_j2.json",
}
E1B = {
    "n20_500m": ROOT / "results" / "e1b_n20_500m.json",
    "n20_500m_j2": ROOT / "results" / "e1b_n20_500m_j2.json",
}
OUT_JSON = ROOT / "results" / "min_spectrum_analysis.json"
OUT_TXT = ROOT / "results" / "min_spectrum_analysis.txt"


def _hz_vector(payload: dict, method: str) -> np.ndarray:
    stats = (payload.get("by_method") or {}).get(method) or {}
    vals = stats.get("per_seed_hz") or []
    return np.asarray(
        [float("nan") if v is None else float(v) for v in vals], dtype=float
    )


def _t_crit(n: int) -> float:
    if n <= 1:
        return float("nan")
    if n == 20:
        return 2.093
    if n == 50:
        return 2.010
    return 1.96


def _ci95(d: np.ndarray) -> tuple[float, float]:
    n = int(d.size)
    mean = float(np.mean(d))
    if n < 2:
        return mean, float("nan")
    sem = float(np.std(d, ddof=1) / math.sqrt(n))
    return mean, _t_crit(n) * sem


def _ci95_khz(d_hz: np.ndarray) -> tuple[float, float]:
    mean, half = _ci95(d_hz)
    return mean / 1e3, (float("nan") if not np.isfinite(half) else half / 1e3)


def _pair(other: np.ndarray, champ: np.ndarray) -> dict:
    """Positive = champion uses fewer Hertz than other."""
    mask = np.isfinite(other) & np.isfinite(champ)
    d = other[mask] - champ[mask]
    if d.size == 0:
        return {"n": 0}
    mean_o = float(np.mean(other[mask]))
    mean_c = float(np.mean(champ[mask]))
    rel_seed = d / np.maximum(other[mask], 1.0)
    mean_khz, half = _ci95_khz(d)
    return {
        "n": int(d.size),
        "mean_delta_kHz": mean_khz,
        "ci95_half_kHz": half,
        "std_delta_kHz": float(np.std(d, ddof=1) / 1e3) if d.size > 1 else 0.0,
        "median_delta_kHz": float(np.median(d) / 1e3),
        "ratio_of_means": (mean_o - mean_c) / max(mean_o, 1.0),
        "mean_of_ratios": float(np.mean(rel_seed)),
        "n_champ_fewer": int(np.sum(d > TIE_EPS_HZ)),
        "n_champ_more": int(np.sum(d < -TIE_EPS_HZ)),
        "n_tie": int(np.sum(np.abs(d) <= TIE_EPS_HZ)),
        "n_practical": int(np.sum(d > PRACTICAL_HZ)),
        "wilcoxon": wilcoxon_signed_rank(d, alternative="greater"),
        "paired_t": paired_t(d),
    }


def _uses_phrase(stats: dict, champ: str, other: str) -> str:
    if not stats.get("n"):
        return f"  vs {other}: no paired finite Hertz"
    d = stats["mean_delta_kHz"]
    if d >= 0:
        verb = f"{champ} uses {d:.1f} kHz less than {other}"
    else:
        verb = f"{champ} uses {abs(d):.1f} kHz more than {other}"
    p = (stats.get("wilcoxon") or {}).get("p_greater")
    p_s = "n/a" if p is None else f"{p:.2e}"
    ci = stats.get("ci95_half_kHz")
    ci_s = "" if ci is None or not np.isfinite(ci) else f", 95% CI +/-{ci:.1f}"
    return (
        f"  vs {other}: {verb}{ci_s} "
        f"(ratio-of-means {100.0 * stats['ratio_of_means']:+.1f}%, "
        f"mean-of-ratios {100.0 * stats['mean_of_ratios']:+.1f}%; "
        f"{stats['n_champ_fewer']}/{stats['n']} fewer Hz, "
        f"{stats['n_practical']}/{stats['n']} practical>{PRACTICAL_HZ/1e3:.0f} kHz, "
        f"p_greater={p_s})"
    )


def analyze_hertz(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = list(payload.get("methods") or [])
    means = payload.get("mean_kHz") or {}
    over = {
        m: (None if v is None else PLACE_HZ / (v * 1e3))
        for m, v in means.items()
    }
    pairs = {}
    for champ in methods:
        pairs[champ] = {
            m: _pair(_hz_vector(payload, m), _hz_vector(payload, champ))
            for m in methods
            if m != champ
        }
    bindings = {}
    for method in methods:
        counts: dict[str, int] = {}
        for row in payload.get("per_seed") or []:
            cell = row.get(method) or {}
            key = str(cell.get("binding") or "unknown")
            counts[key] = counts.get(key, 0) + 1
        bindings[method] = counts
    return {
        "case": payload.get("case"),
        "path": str(path),
        "area_m": payload.get("area_m"),
        "num_uav": payload.get("num_uav"),
        "n_runs": payload.get("n_runs"),
        "mean_kHz": means,
        "overprovision_vs_8p8": over,
        "bindings": bindings,
        "pairs": pairs,
        "search_abs_tol_kHz": SEARCH_ABS_TOL_HZ / 1e3,
    }


def _mbps_vector(payload: dict, method: str) -> np.ndarray:
    vals = []
    for row in payload.get("per_seed") or []:
        v = (row.get(method) or {}).get("place_sum_rate_Mbps")
        vals.append(float("nan") if v is None else float(v))
    return np.asarray(vals, dtype=float)


def _pair_mbps(other: np.ndarray, champ: np.ndarray) -> dict:
    """Positive = champion has higher leftover-dump Mbps than other."""
    mask = np.isfinite(other) & np.isfinite(champ)
    d = champ[mask] - other[mask]
    if d.size == 0:
        return {"n": 0}
    mean_o = float(np.mean(other[mask]))
    mean_c = float(np.mean(champ[mask]))
    mean_d, half = _ci95(d)
    return {
        "n": int(d.size),
        "mean_delta_Mbps": mean_d,
        "ci95_half_Mbps": half,
        "std_delta_Mbps": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "mean_champ_Mbps": mean_c,
        "mean_other_Mbps": mean_o,
        "n_champ_better": int(np.sum(d > 1e-4)),
        "n_champ_worse": int(np.sum(d < -1e-4)),
        "n_tie": int(np.sum(np.abs(d) <= 1e-4)),
        "wilcoxon": wilcoxon_signed_rank(d, alternative="greater"),
        "paired_t": paired_t(d),
    }


def _mbps_phrase(stats: dict, champ: str, other: str) -> str:
    if not stats.get("n"):
        return f"  vs {other}: no paired Mbps"
    d = stats["mean_delta_Mbps"]
    sign = "+" if d >= 0 else ""
    p = (stats.get("wilcoxon") or {}).get("p_greater")
    p_s = "n/a" if p is None else f"{p:.2e}"
    ci = stats.get("ci95_half_Mbps")
    ci_s = "" if ci is None or not np.isfinite(ci) else f", 95% CI +/-{ci:.4f}"
    return (
        f"  vs {other}: {champ} {sign}{d:.4f} Mbps{ci_s} "
        f"({stats['n_champ_better']}/{stats['n']} higher, "
        f"{stats['n_champ_worse']}/{stats['n']} lower, p_greater={p_s})"
    )


def _e1b_decomp(payload: dict) -> dict:
    rows = payload.get("per_seed") or []
    km, pre, fin = [], [], []
    for row in rows:
        k = row.get("kmeans_bound_hz")
        p = row.get("pre_polish_bound_hz")
        e = row.get("min_b_sys_hz")
        if k is None or p is None or e is None:
            continue
        km.append(float(k))
        pre.append(float(p))
        fin.append(float(e))
    if not km:
        return {}
    km_a = np.asarray(km)
    pre_a = np.asarray(pre)
    fin_a = np.asarray(fin)
    sel = km_a - pre_a
    pol = pre_a - fin_a
    tot = km_a - fin_a
    mean_tot, half_tot = _ci95(tot)
    return {
        "n": int(tot.size),
        "mean_save_vs_kmeans_kHz": mean_tot / 1e3,
        "ci95_half_kHz": half_tot / 1e3 if np.isfinite(half_tot) else None,
        "mean_selection_save_kHz": float(np.mean(sel) / 1e3),
        "mean_polish_save_kHz": float(np.mean(pol) / 1e3),
        "n_practical": int(np.sum(tot > PRACTICAL_HZ)),
        "wilcoxon": wilcoxon_signed_rank(tot, alternative="greater"),
        "paired_t": paired_t(tot),
        "winner_pre_polish": payload.get("winner_pre_polish") or {},
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    reports: dict = {"hertz": [], "leftover_mbps": {}, "e1b": {}}
    lines = [
        "E1 family readout",
        f"Place radio 8.8 MHz / 25% cap. Search abs_tol {SEARCH_ABS_TOL_HZ/1e3:.0f} kHz.",
        "Overprovision = 8.8 MHz / method mean min-B_sys. Quote method-specific values;",
        "do not write a single 10-18x for every row.",
        "",
        "== Frozen-q min Hertz (E1) ==",
        "",
    ]
    for name, path in CASES.items():
        rep = analyze_hertz(path)
        if rep is None:
            lines.append(f"{name}: missing {path}")
            continue
        reports["hertz"].append(rep)
        lines.append(
            f"{name}  {rep['area_m']} m  J={rep['num_uav']}  n={rep['n_runs']}"
        )
        for method, khz in (rep["mean_kHz"] or {}).items():
            if khz is None:
                s = "n/a"
                ov = ""
            else:
                s = f"{khz:.1f} kHz"
                ov = f"  {rep['overprovision_vs_8p8'][method]:.1f}x vs 8.8 MHz"
            lines.append(
                f"  mean {method:<12} {s:<14}{ov}  bind={rep['bindings'].get(method)}"
            )
        for champ in ("sca_anchor", "kmeans", "sca_medoid"):
            if champ not in (rep["mean_kHz"] or {}):
                continue
            lines.append(f"  -- paired vs champion={champ} --")
            for other, stats in (rep["pairs"].get(champ) or {}).items():
                line = _uses_phrase(stats, champ, other)
                if champ == "sca_anchor" and other == "sca":
                    rom = stats.get("ratio_of_means")
                    mor = stats.get("mean_of_ratios")
                    if rom is not None and mor is not None:
                        line += (
                            f"  [quote ratio-of-means {100*rom:+.1f}% or "
                            f"mean-of-ratios {100*mor:+.1f}%, not both as one number]"
                        )
                lines.append(line)
        means = rep["mean_kHz"]
        if "sca_anchor" in means and "sca" in means:
            a = means["sca_anchor"]
            s = means["sca"]
            if a is not None and s is not None and abs(s - a) < 5.0:
                lines.append(
                    f"  note: |anchor-SCA|={abs(s-a):.1f} kHz is within a few "
                    f"search steps (tol {SEARCH_ABS_TOL_HZ/1e3:.0f} kHz); do not "
                    "treat the sign as a result."
                )
        lines.append("")

    lines += [
        "== Leftover-dump Mbps at the 8.8 MHz place radio (same geometries as E1 Hertz) ==",
        "",
    ]
    reports["leftover_mbps"] = {}
    for name, path in CASES.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        methods = list(payload.get("methods") or [])
        means = {}
        bits = []
        for method in methods:
            vals = _mbps_vector(payload, method)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                means[method] = float(np.mean(finite))
                bits.append(f"{method}={means[method]:.4f}")
        lines.append(f"{name}  " + "  ".join(bits))
        cell_pairs = {}
        for champ in ("sca_anchor", "sca_medoid", "kmeans"):
            if champ not in means:
                continue
            lines.append(f"  -- leftover-dump Mbps champion={champ} --")
            for other in methods:
                if other == champ:
                    continue
                stats = _pair_mbps(_mbps_vector(payload, other), _mbps_vector(payload, champ))
                cell_pairs[f"{champ}_vs_{other}"] = stats
                lines.append(_mbps_phrase(stats, champ, other))
        reports["leftover_mbps"][name] = {"mean_Mbps": means, "pairs": cell_pairs}
    lines.append("(E1 place_sum_rate_Mbps; same seeds as Hertz. Not a second campaign.)")
    lines.append("")

    lines += [
        "== E1b min-Hertz keep-best (falsifier: mean save vs one-shot k-means < 50 kHz => drop Path B) ==",
        "",
    ]
    path_b = True
    for name, path in E1B.items():
        if not path.exists():
            lines.append(f"{name}: missing {path}")
            path_b = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        decomp = _e1b_decomp(payload)
        reports["e1b"][name] = {
            "mean_e1b_kHz": payload.get("mean_e1b_kHz"),
            "mean_kmeans_kHz": payload.get("mean_kmeans_kHz"),
            "mean_save_vs_kmeans_kHz": payload.get("mean_save_vs_kmeans_kHz"),
            "n_practical": payload.get("n_practical"),
            "n_runs": payload.get("n_runs"),
            "path_b_survives": payload.get("path_b_survives"),
            "winner_pre_polish": payload.get("winner_pre_polish"),
            "decomp": decomp,
            "wilcoxon": payload.get("wilcoxon"),
            "paired_t": payload.get("paired_t"),
        }
        survives = bool(payload.get("path_b_survives"))
        path_b = path_b and survives
        ci = decomp.get("ci95_half_kHz")
        ci_s = "" if ci is None or not np.isfinite(ci) else f", 95% CI +/-{ci:.1f}"
        p = (decomp.get("wilcoxon") or {}).get("p_greater")
        p_s = "n/a" if p is None else f"{p:.2e}"
        lines.append(
            f"{name}  e1b={payload.get('mean_e1b_kHz'):.1f} kHz  "
            f"kmeans={payload.get('mean_kmeans_kHz'):.1f} kHz  "
            f"save {payload.get('mean_save_vs_kmeans_kHz'):+.1f} kHz{ci_s}  "
            f"practical {payload.get('n_practical')}/{payload.get('n_runs')}  "
            f"p_greater={p_s}  Path B cell={survives}"
        )
        lines.append(
            f"  mechanism: selection (best cand vs one-shot k-means) "
            f"{decomp.get('mean_selection_save_kHz', 0):+.1f} kHz, "
            f"Hertz polish {decomp.get('mean_polish_save_kHz', 0):+.1f} kHz  "
            f"pre={payload.get('winner_pre_polish')}"
        )
    lines += [
        "",
        "Wording to keep out of the paper:",
        "- Do not write a single 10-18x overprovision. Quote 8.8 MHz / method mean per cell.",
        "- Anchor vs SCA at 500 m J=3: quote ratio-of-means and mean-of-ratios separately.",
        "- 100 m kHz gaps of a few kHz are not a signed Hertz winner (tol 1 kHz, practical bar 50 kHz).",
        "- J=2 k-means vs zenith-subset Hertz has a paired CI in the E1 block above.",
        "",
        f"Pre-registered falsifier (mean save > 50 kHz at both 500 m cells): Path B={path_b}.",
        "Path A (leftover-dump Mbps under the 25% cap) does not depend on E1b.",
        "Do not write Path B as a zenith-subset spectrum claim. Hertz-best vs "
        "inertia-best among the same k-means inits is a covering footnote. "
        "J=3 E1b: point estimate passes 50 kHz, CI lower bound does not.",
        "",
    ]
    reports["path_b_survives"] = path_b
    reports["paper_claim"] = (
        "path_a"
        if not path_b
        else "path_a_plus_e1b_covering_not_zenith_spectrum"
    )
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(text, end="")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
