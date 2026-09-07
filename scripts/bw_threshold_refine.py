"""Refine SCA-vs-random significance threshold in per-link bandwidth cap.

Part 1: 8.8 MHz at max_bw_share 0.21-0.24 (J=3, seeds 1-20).
Part 2: 8.8 MHz cap=50% reseed (seeds 21-40).
Part 3: J-robustness from cap=20% full campaign (re-slice uavs axis).

Outputs:
  results/bw_threshold_refine.json
  results/bw_cap50_reseed.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_bandwidth_screen import ConfigSpec, run_config  # noqa: E402
from paired_winrate import wilcoxon_signed_rank  # noqa: E402

B_SYS = 8_800_000.0
SCREEN_PATH = ROOT / "results" / "bw_screen_default_j3.json"
CAP20_CAMPAIGN = ROOT / "results" / "campaign_8.8mhz_cap20_n20.json"


@dataclass
class CapPoint:
    max_bw_share: float
    per_link_cap_hz: float
    per_link_cap_mhz: float
    j: int
    seed_start: int
    n_runs: int
    means_mbps: dict[str, float]
    feasible_frac: dict[str, float]
    sca_vs_random_delta_mbps: float
    sca_vs_random_p: float
    sca_wins_vs_random: int
    source: str
    fdr_q: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR q-values (two-sided tests)."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(prev, sorted_p[i] * m / rank)
        q[i] = val
        prev = val
    out = np.empty(m, dtype=float)
    out[order] = q
    return out.tolist()


def screen_result_to_cap_point(r, *, source: str, j: int = 3) -> CapPoint:
    share = r.spec.max_bw_share
    assert share is not None
    cap_hz = B_SYS * share
    return CapPoint(
        max_bw_share=share,
        per_link_cap_hz=cap_hz,
        per_link_cap_mhz=cap_hz / 1e6,
        j=j,
        seed_start=1,
        n_runs=r.n_runs,
        means_mbps=r.means,
        feasible_frac=r.feasible_frac,
        sca_vs_random_delta_mbps=r.sca_vs_random_delta,
        sca_vs_random_p=r.sca_vs_random_p,
        sca_wins_vs_random=r.sca_wins_vs_random,
        source=source,
    )


def campaign_j_point(path: Path, j: int, share: float) -> CapPoint:
    d = json.loads(path.read_text(encoding="utf-8"))
    pt = next(x for x in d["points"] if x["axis"] == "uavs" and x["x"] == float(j))
    sca = pt["by_method"]["sca"]["per_seed_Mbps"]
    rnd = pt["by_method"]["random"]["per_seed_Mbps"]
    deltas = np.array(sca) - np.array(rnd)
    w = wilcoxon_signed_rank(deltas.tolist())
    cap_hz = B_SYS * share
    return CapPoint(
        max_bw_share=share,
        per_link_cap_hz=cap_hz,
        per_link_cap_mhz=cap_hz / 1e6,
        j=j,
        seed_start=int(d.get("seed_start", 1)),
        n_runs=int(d["n_runs"]),
        means_mbps={m: pt["by_method"][m]["mean_sum_rate_Mbps"] for m in d["methods"]},
        feasible_frac={m: pt["by_method"][m]["feasible_fraction"] for m in d["methods"]},
        sca_vs_random_delta_mbps=float(np.mean(deltas)),
        sca_vs_random_p=float(w["p_two_sided"]),
        sca_wins_vs_random=int(np.sum(deltas > 0)),
        source=f"campaign:{path.name}",
    )


def load_screen_88_cap_points() -> list[CapPoint]:
    if not SCREEN_PATH.exists():
        return []
    data = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
    out: list[CapPoint] = []
    for row in data["configs"]:
        if row["b_sys_hz"] != B_SYS or row["max_bw_share"] is None:
            continue
        spec = ConfigSpec(B_SYS, row["max_bw_share"])
        # Reconstruct minimal object for conversion
        class _R:
            pass

        r = _R()
        r.spec = spec
        r.means = row["means"]
        r.feasible_frac = row["feasible_frac"]
        r.sca_vs_random_delta = row["sca_vs_random_delta_mbps"]
        r.sca_vs_random_p = row["sca_vs_random_p"]
        r.sca_wins_vs_random = row["sca_wins_vs_random"]
        r.n_runs = data["n_runs"]
        out.append(screen_result_to_cap_point(r, source="bw_screen_default_j3.json"))
    return out


def apply_fdr_j3_cap_sweep(points: list[CapPoint]) -> None:
    """BH-FDR across J=3 8.8 MHz cap-sweep rows only (excludes J-robustness rows)."""
    j3 = [p for p in points if p.j == 3 and p.source != "part3_j_robustness"]
    pvals = [p.sca_vs_random_p for p in j3]
    qs = bh_fdr(pvals)
    for p, q in zip(j3, qs, strict=True):
        p.fdr_q = q


def find_threshold(points: list[CapPoint], *, alpha: float = 0.05) -> dict[str, Any]:
    """Find crossover when tightening cap (decreasing max_bw_share, high MHz -> low)."""
    j3 = sorted(
        [p for p in points if p.j == 3 and p.source != "part3_j_robustness"],
        key=lambda p: p.max_bw_share,
        reverse=True,
    )
    last_insignificant: CapPoint | None = None
    first_significant: CapPoint | None = None
    last_insignificant_fdr: CapPoint | None = None
    first_significant_fdr: CapPoint | None = None
    for p in j3:
        if p.sca_vs_random_p >= alpha:
            last_insignificant = p
        elif first_significant is None:
            first_significant = p
        if p.fdr_q is None or p.fdr_q >= alpha:
            last_insignificant_fdr = p
        elif first_significant_fdr is None:
            first_significant_fdr = p
    return {
        "alpha": alpha,
        "scan_direction": "decreasing max_bw_share (looser -> tighter per-link cap)",
        "first_raw_significant_when_tightening": (
            first_significant.to_dict() if first_significant else None
        ),
        "last_raw_insignificant_before_threshold": (
            last_insignificant.to_dict() if last_insignificant else None
        ),
        "first_fdr_significant_when_tightening": (
            first_significant_fdr.to_dict() if first_significant_fdr else None
        ),
        "last_fdr_insignificant_before_threshold": (
            last_insignificant_fdr.to_dict() if last_insignificant_fdr else None
        ),
        "mid_band_insignificant_shares": [
            p.to_dict()
            for p in sorted(j3, key=lambda x: x.max_bw_share)
            if 0.23 <= p.max_bw_share <= 0.25 and p.sca_vs_random_p >= alpha
        ],
    }


def print_table(points: list[CapPoint]) -> None:
    j3 = sorted(
        [p for p in points if p.j == 3 and p.source != "part3_j_robustness"],
        key=lambda p: p.max_bw_share,
    )
    print("\nPer-link cap vs SCA-vs-random (J=3, 8.8 MHz, seeds 1-20 unless noted)")
    print("-" * 105)
    print(
        f"{'share':>6s} {'cap_MHz':>8s} {'SCA':>8s} {'random':>8s} "
        f"{'delta':>8s} {'p':>10s} {'q_FDR':>10s} {'feas':>6s} {'src':>12s}"
    )
    for p in j3:
        q = f"{p.fdr_q:.4f}" if p.fdr_q is not None else "   —"
        print(
            f"{p.max_bw_share:6.2f} {p.per_link_cap_mhz:8.3f} "
            f"{p.means_mbps['sca']:8.3f} {p.means_mbps['random']:8.3f} "
            f"{p.sca_vs_random_delta_mbps:+8.3f} {p.sca_vs_random_p:10.4f} {q:>10s} "
            f"{p.feasible_frac['sca']:6.0%} {p.source[:12]:>12s}"
        )

    print("\nJ-robustness @ cap=20% (1.760 MHz per link, seeds 1-20)")
    print("-" * 80)
    jrob = sorted([p for p in points if p.source == "part3_j_robustness"], key=lambda p: p.j)
    for p in jrob:
        print(
            f"J={p.j}  SCA={p.means_mbps['sca']:.3f}  random={p.means_mbps['random']:.3f}  "
            f"delta={p.sca_vs_random_delta_mbps:+.3f}  p={p.sca_vs_random_p:.4f}  "
            f"wins={p.sca_wins_vs_random}/{p.n_runs}"
        )


def main() -> int:
    part1_shares = [0.21, 0.22, 0.23, 0.24]
    part1_results = []
    print("Part 1: cap 21-24% @ J=3, seeds 1-20")
    for share in part1_shares:
        spec = ConfigSpec(B_SYS, share)
        print(f"  running cap={share:.0%} ...", flush=True)
        r = run_config(spec, n_runs=20, seed_start=1, solver=None)
        part1_results.append(screen_result_to_cap_point(r, source="part1_refine"))

    print("\nPart 2: cap=50% reseed seeds 21-40")
    cap50_spec = ConfigSpec(B_SYS, 0.50)
    cap50_r = run_config(cap50_spec, n_runs=20, seed_start=21, solver=None)
    cap50_point = screen_result_to_cap_point(
        cap50_r, source="part2_cap50_reseed", j=3
    )
    cap50_point.seed_start = 21
    cap50_out = {
        "b_sys_hz": B_SYS,
        "max_bw_share": 0.50,
        "per_link_cap_hz": B_SYS * 0.50,
        "per_link_cap_mhz": B_SYS * 0.50 / 1e6,
        "seed_start": 21,
        "n_runs": 20,
        "means_mbps": cap50_point.means_mbps,
        "feasible_frac": cap50_point.feasible_frac,
        "sca_vs_random_delta_mbps": cap50_point.sca_vs_random_delta_mbps,
        "sca_vs_random_p": cap50_point.sca_vs_random_p,
        "sca_wins_vs_random": cap50_point.sca_wins_vs_random,
        "prior_screen_seeds_1_20": None,
    }
    if SCREEN_PATH.exists():
        screen = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
        prior = next(
            (
                c
                for c in screen["configs"]
                if c["b_sys_hz"] == B_SYS and c["max_bw_share"] == 0.5
            ),
            None,
        )
        if prior:
            cap50_out["prior_screen_seeds_1_20"] = {
                "sca_vs_random_p": prior["sca_vs_random_p"],
                "sca_vs_random_delta_mbps": prior["sca_vs_random_delta_mbps"],
                "sca_wins_vs_random": prior["sca_wins_vs_random"],
            }
    cap50_path = ROOT / "results" / "bw_cap50_reseed.json"
    cap50_path.write_text(json.dumps(cap50_out, indent=2), encoding="utf-8")
    print(f"  wrote {cap50_path}")
    print(
        f"  reseed p={cap50_point.sca_vs_random_p:.4f}  "
        f"prior={cap50_out['prior_screen_seeds_1_20']}"
    )

    print("\nPart 3: J-robustness from cap=20% campaign")
    j_robust: list[CapPoint] = []
    if CAP20_CAMPAIGN.exists():
        for j in (1, 2, 3, 4, 5):
            p = campaign_j_point(CAP20_CAMPAIGN, j, 0.20)
            p.source = "part3_j_robustness"
            j_robust.append(p)
    else:
        print(f"  WARNING: missing {CAP20_CAMPAIGN}")

    # Combine J=3 cap sweep: old screen (8.8 caps) + part1 new
    old = load_screen_88_cap_points()
    # Deduplicate by share (prefer part1 for 21-24 if overlap)
    by_share: dict[float, CapPoint] = {}
    for p in old:
        by_share[p.max_bw_share] = p
    for p in part1_results:
        by_share[p.max_bw_share] = p
    j3_sweep = sorted(by_share.values(), key=lambda p: p.max_bw_share)

    all_points = j3_sweep + j_robust
    apply_fdr_j3_cap_sweep(all_points)
    threshold = find_threshold(all_points)

    refine_out = {
        "description": (
            "SCA-vs-random significance vs per-link cap at 8.8 MHz. "
            "BH-FDR across J=3 cap-sweep rows (old screen + part1 refine). "
            "Part3 rows are J-robustness at fixed cap=20%."
        ),
        "b_sys_hz": B_SYS,
        "part1_new_shares": part1_shares,
        "part1_points": [p.to_dict() for p in part1_results],
        "j3_cap_sweep_combined": [p.to_dict() for p in j3_sweep],
        "part3_j_robustness_cap20": [p.to_dict() for p in j_robust],
        "threshold_analysis": threshold,
        "cap50_reseed_file": str(cap50_path),
    }
    refine_path = ROOT / "results" / "bw_threshold_refine.json"
    refine_path.write_text(json.dumps(refine_out, indent=2), encoding="utf-8")
    print(f"\nwrote {refine_path}")

    print_table(all_points)

    # Verdict
    jrob_sig = [p.j for p in j_robust if p.sca_vs_random_p < 0.05]
    first = threshold["first_raw_significant_when_tightening"]
    last_ns = threshold["last_raw_insignificant_before_threshold"]
    fdr_first = threshold["first_fdr_significant_when_tightening"]
    fdr_last = threshold["last_fdr_insignificant_before_threshold"]
    print("\nVERDICT:")
    if first and last_ns:
        print(
            f"  Raw J=3 threshold (tightening cap): between "
            f"{last_ns['per_link_cap_mhz']:.3f} MHz ({last_ns['max_bw_share']:.0%}) and "
            f"{first['per_link_cap_mhz']:.3f} MHz ({first['max_bw_share']:.0%}) per link."
        )
    if fdr_first and fdr_last:
        print(
            f"  FDR J=3 threshold: between "
            f"{fdr_last['per_link_cap_mhz']:.3f} MHz ({fdr_last['max_bw_share']:.0%}) and "
            f"{fdr_first['per_link_cap_mhz']:.3f} MHz ({fdr_first['max_bw_share']:.0%})."
        )
    j_sig = [p.j for p in j_robust if p.sca_vs_random_p < 0.05]
    j_nsig = [p.j for p in j_robust if p.sca_vs_random_p >= 0.05]
    if j_sig and j_nsig:
        print(
            "  At fixed 1.76 MHz/link (cap=20%): significance is J-dependent "
            f"(sig J={j_sig}, n.s. J={j_nsig}) — threshold scales with geometry, not fixed Hz."
        )
    elif j_sig and not j_nsig:
        print(
            "  At fixed 1.76 MHz/link (cap=20%): significant at all J tested "
            "— per-link Hz threshold is J-invariant at this cap."
        )
    else:
        print(
            "  At fixed 1.76 MHz/link (cap=20%): not significant at tested J "
            "— placement advantage does not replicate at this cap across J."
        )
    prior_p = cap50_out.get("prior_screen_seeds_1_20", {})
    if prior_p:
        print(
            f"  Cap=50% anomaly: seeds 1-20 p={prior_p['sca_vs_random_p']:.4f}, "
            f"seeds 21-40 p={cap50_point.sca_vs_random_p:.4f} "
            f"({'replicates' if cap50_point.sca_vs_random_p < 0.05 else 'does not replicate'} significance)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
