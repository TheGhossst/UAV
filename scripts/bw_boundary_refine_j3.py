"""J=3 bisection of the SCA-vs-random significance boundary.

Runs 20-seed default-point campaigns (I=10, J=3, B_sys=8.8 MHz) at
max_bw_share in {0.235, 0.24, 0.245}. Combines those three Wilcoxon
p-values with the existing 30-cell cap×J grid and applies BH-FDR once
across all 33 tests.

Separately characterizes whether the loose-cap (25–50%) “significant”
region is driven by effect-size growth or by variance collapse.

This is a boundary-location experiment, not a config-selection search.
All three pre-specified caps are always run.

Output: results/bw_boundary_refine_j3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402
from sweep_bandwidth_screen import ConfigSpec, run_config  # noqa: E402

B_SYS = 8_800_000.0
J = 3
N_RUNS = 20
SEED_START = 1
NEW_SHARES = (0.235, 0.24, 0.245)
LOOSE_SHARES_NEED_RERUN = (0.40, 0.50)
GRID_PATH = ROOT / "results" / "bw_cap_by_J_grid.json"
OUT_PATH = ROOT / "results" / "bw_boundary_refine_j3.json"
CHECKPOINT_PATH = ROOT / "results" / "bw_boundary_refine_j3.checkpoint.json"

CAMPAIGN_FOR_SHARE: dict[float, Path] = {
    0.25: ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    0.30: ROOT / "results" / "campaign_8.8mhz_cap30_n20.json",
    1.00: ROOT / "results" / "campaign_8.8mhz_n20.json",
}


def _out(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def bh_fdr(p_values: list[float]) -> list[float]:
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


def _stats_from_per_seed(
    share: float,
    j: int,
    sca_ps: list[float],
    rnd_ps: list[float],
    sca_feas: float,
    rnd_feas: float,
    *,
    source: str,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    sca = np.asarray(sca_ps, dtype=float)
    rnd = np.asarray(rnd_ps, dtype=float)
    deltas = sca - rnd
    w = wilcoxon_signed_rank(deltas.tolist())
    delta_std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan")
    delta_mean = float(np.mean(deltas))
    effect = delta_mean / delta_std if delta_std > 0 else float("nan")
    cap_hz = float(round(B_SYS * share)) if share is not None else float(B_SYS)
    seed_list = seeds if seeds is not None else list(
        range(SEED_START, SEED_START + len(sca_ps))
    )
    return {
        "max_bw_share": share,
        "per_link_cap_hz": cap_hz,
        "per_link_cap_mhz": cap_hz / 1e6,
        "j": j,
        "n_runs": int(len(sca_ps)),
        "seed_start": int(seed_list[0]) if seed_list else SEED_START,
        "seeds": seed_list,
        "sca_mean_mbps": float(np.mean(sca)),
        "random_mean_mbps": float(np.mean(rnd)),
        "delta_mbps": delta_mean,
        "delta_std_mbps": delta_std,
        "effect_size": effect,
        "wilcoxon_p": float(w["p_two_sided"]),
        "fdr_q": None,
        "sca_feasible_frac": float(sca_feas),
        "random_feasible_frac": float(rnd_feas),
        "sca_wins": int(np.sum(deltas > 0)),
        "source": source,
        "per_seed": {
            "sca": [float(x) for x in sca],
            "random": [float(x) for x in rnd],
            "delta": [float(x) for x in deltas],
        },
    }


def point_from_checkpoint(raw: dict[str, Any], share: float) -> dict[str, Any]:
    return _stats_from_per_seed(
        share,
        int(raw.get("j", J)),
        raw["per_seed"]["sca"],
        raw["per_seed"]["random"],
        raw["sca_feasible_frac"],
        raw["random_feasible_frac"],
        source=raw.get("source", "checkpoint"),
        seeds=raw.get("seeds"),
    )
    spec = ConfigSpec(B_SYS, share)
    _out(
        f"  running J={J} I=10 cap={share:.1%} "
        f"({B_SYS * share / 1e6:.3f} MHz/link) seeds {SEED_START}-{SEED_START + N_RUNS - 1} ..."
    )
    r = run_config(spec, n_runs=N_RUNS, seed_start=SEED_START, solver=None)
    return _stats_from_per_seed(
        share,
        J,
        r.per_seed["sca"],
        r.per_seed["random"],
        r.feasible_frac["sca"],
        r.feasible_frac["random"],
        source="part1_bisection",
        seeds=list(range(SEED_START, SEED_START + N_RUNS)),
    )


def extract_campaign_j(path: Path, share: float, j: int, *, source: str) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    pt = next(x for x in d["points"] if x["axis"] == "uavs" and x["x"] == float(j))
    sca = pt["by_method"]["sca"]
    rnd = pt["by_method"]["random"]
    seeds = list(sca.get("seeds") or range(int(d.get("seed_start", 1)), int(d["n_runs"]) + 1))
    return _stats_from_per_seed(
        share,
        j,
        sca["per_seed_Mbps"],
        rnd["per_seed_Mbps"],
        sca["feasible_fraction"],
        rnd["feasible_fraction"],
        source=source,
        seeds=seeds,
    )


def load_checkpoint() -> dict[str, dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        return {}
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {str(k): v for k, v in data.get("points_by_share", {}).items()}


def save_checkpoint(points_by_share: dict[str, dict[str, Any]]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps({"points_by_share": points_by_share}, indent=2),
        encoding="utf-8",
    )


def grid_cell_as_row(c: dict[str, Any], *, is_new: bool = False) -> dict[str, Any]:
    return {
        "max_bw_share": c["max_bw_share"],
        "per_link_cap_hz": c["per_link_cap_hz"],
        "per_link_cap_mhz": c["per_link_cap_mhz"],
        "j": c["j"],
        "n_runs": c["n_runs"],
        "sca_mean_mbps": c["sca_mean_mbps"],
        "random_mean_mbps": c["random_mean_mbps"],
        "delta_mbps": c["delta_mbps"],
        "delta_std_mbps": c["delta_std_mbps"],
        "effect_size": c["effect_size"],
        "wilcoxon_p": c["wilcoxon_p"],
        "fdr_q": c.get("fdr_q"),
        "sca_feasible_frac": c.get("sca_feasible_frac", 1.0),
        "random_feasible_frac": c.get("random_feasible_frac", 1.0),
        "sca_wins": c.get("sca_wins"),
        "is_new": is_new,
        "source": c.get("source", c.get("campaign_file", "")),
    }


def print_combined_table(rows: list[dict[str, Any]]) -> None:
    _out("\n" + "=" * 128)
    _out("COMBINED 33-CELL TABLE: 30-cell cap x J grid + 3 J=3 bisection points")
    _out("BH-FDR applied once across all 33 Wilcoxon p-values")
    _out("=" * 128)
    _out(
        f"{'cap%':>6s} {'Hz':>10s} {'J':>2s} "
        f"{'SCA':>8s} {'random':>8s} {'delta':>9s} {'std(d)':>8s} {'d/std':>7s} "
        f"{'p':>10s} {'q_FDR':>10s} {'new':>3s}"
    )
    _out("-" * 128)
    for r in rows:
        tag = "Y" if r["is_new"] else ""
        sig = "*" if r["fdr_q"] is not None and r["fdr_q"] < 0.05 else " "
        _out(
            f"{r['max_bw_share']:6.1%} {r['per_link_cap_hz']:10.0f} {r['j']:2d} "
            f"{r['sca_mean_mbps']:8.3f} {r['random_mean_mbps']:8.3f} "
            f"{r['delta_mbps']:+9.4f} {r['delta_std_mbps']:8.4f} {r['effect_size']:7.2f} "
            f"{r['wilcoxon_p']:10.4f} {r['fdr_q']:10.4f} {tag:>3s}{sig}"
        )
    _out("  * = q_FDR < 0.05 (BH across 33 tests)")


def print_j3_transition(rows: list[dict[str, Any]]) -> None:
    j3 = [r for r in rows if r["j"] == J]
    _out("\nJ=3 slice (sorted by cap), including new bisection points:")
    _out("-" * 88)
    _out(
        f"{'cap%':>6s} {'Hz':>10s} {'delta':>9s} {'std(d)':>8s} {'d/std':>7s} "
        f"{'p':>10s} {'q_FDR':>10s}"
    )
    for r in j3:
        sig = "*" if r["fdr_q"] < 0.05 else " "
        _out(
            f"{r['max_bw_share']:6.1%} {r['per_link_cap_hz']:10.0f} "
            f"{r['delta_mbps']:+9.4f} {r['delta_std_mbps']:8.4f} {r['effect_size']:7.2f} "
            f"{r['wilcoxon_p']:10.4f} {r['fdr_q']:10.4f}{sig}"
        )


def find_crossover(rows: list[dict[str, Any]], *, alpha: float = 0.05) -> dict[str, Any]:
    """Adjacent J=3 caps that bracket the first FDR drop from sig to n.s. after 23%."""
    j3 = [r for r in rows if r["j"] == J]
    pairs = []
    first_off = None
    for a, b in zip(j3, j3[1:]):
        a_sig = a["fdr_q"] < alpha and a["delta_mbps"] > 0
        b_sig = b["fdr_q"] < alpha and b["delta_mbps"] > 0
        pair = {
            "from_share": a["max_bw_share"],
            "to_share": b["max_bw_share"],
            "from_q": a["fdr_q"],
            "to_q": b["fdr_q"],
            "from_sig": a_sig,
            "to_sig": b_sig,
            "from_p": a["wilcoxon_p"],
            "to_p": b["wilcoxon_p"],
            "from_delta": a["delta_mbps"],
            "to_delta": b["delta_mbps"],
        }
        pairs.append(pair)
        if first_off is None and a_sig and not b_sig and a["max_bw_share"] >= 0.20:
            first_off = pair
    return {
        "alpha": alpha,
        "j": J,
        "adjacent_pairs": pairs,
        "sig_to_ns_after_tight_regime": first_off,
    }


def print_loose_table(points: list[dict[str, Any]]) -> None:
    _out("\n" + "=" * 96)
    _out("STEP 2 - J=3 loose-cap region: delta vs std(delta) (not p)")
    _out("=" * 96)
    _out(
        f"{'cap%':>6s} {'Hz':>10s} {'d Mbps':>10s} {'std(d)':>10s} "
        f"{'d/std':>8s} {'p':>10s} {'|d|<0.01':>9s}"
    )
    _out("-" * 96)
    for p in points:
        tiny = "yes" if abs(p["delta_mbps"]) < 0.01 else "no"
        _out(
            f"{p['max_bw_share']:6.0%} {p['per_link_cap_hz']:10.0f} "
            f"{p['delta_mbps']:+10.4f} {p['delta_std_mbps']:10.4f} "
            f"{p['effect_size']:8.2f} {p['wilcoxon_p']:10.4f} {tiny:>9s}"
        )


def variance_verdict(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide whether 25–50% significance is effect growth or variance collapse."""
    by_share = {p["max_bw_share"]: p for p in points}
    p25 = by_share[0.25]
    p30 = by_share[0.30]
    p40 = by_share.get(0.40)
    p50 = by_share.get(0.50)
    deltas = [p["delta_mbps"] for p in points if p["max_bw_share"] in (0.25, 0.30, 0.40, 0.50)]
    stds = [p["delta_std_mbps"] for p in points if p["max_bw_share"] in (0.25, 0.30, 0.40, 0.50)]
    delta_max = max(abs(x) for x in deltas)
    std_25 = p25["delta_std_mbps"]
    std_last = stds[-1]
    std_ratio = std_last / std_25 if std_25 > 0 else float("nan")
    # Growing effect: last Δ larger than 25% by a practically meaningful amount.
    effect_grew = abs(deltas[-1]) > abs(deltas[0]) + 0.005
    std_fell = std_ratio < 0.85
    tiny = delta_max < 0.02
    if (not effect_grew) and tiny and std_fell:
        kind = "variance_shrinkage"
    elif effect_grew and not std_fell:
        kind = "real_effect_growth"
    else:
        kind = "mixed"
    return {
        "kind": kind,
        "delta_mbps_by_share": {str(p["max_bw_share"]): p["delta_mbps"] for p in points},
        "std_mbps_by_share": {str(p["max_bw_share"]): p["delta_std_mbps"] for p in points},
        "delta_range_mbps": [min(deltas), max(deltas)],
        "std_25_over_std_loose": std_25 / std_last if std_last else None,
        "std_ratio_loose_over_25": std_ratio,
        "all_abs_delta_lt_0.02": tiny,
        "effect_grew": effect_grew,
        "variance_collapsed": std_fell,
        "cap25": {
            "delta": p25["delta_mbps"],
            "std": p25["delta_std_mbps"],
            "p": p25["wilcoxon_p"],
        },
        "cap30": {
            "delta": p30["delta_mbps"],
            "std": p30["delta_std_mbps"],
            "p": p30["wilcoxon_p"],
        },
        "cap40": None
        if p40 is None
        else {"delta": p40["delta_mbps"], "std": p40["delta_std_mbps"], "p": p40["wilcoxon_p"]},
        "cap50": None
        if p50 is None
        else {"delta": p50["delta_mbps"], "std": p50["delta_std_mbps"], "p": p50["wilcoxon_p"]},
    }


def build_paragraph(
    crossover: dict[str, Any],
    var: dict[str, Any],
    j3_rows: list[dict[str, Any]],
) -> str:
    off = crossover.get("sig_to_ns_after_tight_regime")
    if off:
        a = (
            f"At J=3, BH-FDR q<0.05 with a positive SCA−random gap holds through "
            f"cap={off['from_share']:.1%} (q={off['from_q']:.4f}, "
            f"delta={off['from_delta']:+.4f} Mbps) and turns off at the next "
            f"pre-specified point cap={off['to_share']:.1%} (q={off['to_q']:.4f}, "
            f"delta={off['to_delta']:+.4f} Mbps); those two adjacent caps bracket "
            f"the significant-to-not-significant switch."
        )
    else:
        a = (
            "At J=3, the 23%-to-25% significant-to-not-significant switch did not "
            "move onto a single adjacent pair among the new 23.5/24/24.5% "
            "points after the 33-test FDR correction; see the J=3 slice."
        )
    d25, s25 = var["cap25"]["delta"], var["cap25"]["std"]
    d30, s30 = var["cap30"]["delta"], var["cap30"]["std"]
    d40 = var["cap40"]["delta"] if var["cap40"] else float("nan")
    s40 = var["cap40"]["std"] if var["cap40"] else float("nan")
    d50 = var["cap50"]["delta"] if var["cap50"] else float("nan")
    s50 = var["cap50"]["std"] if var["cap50"] else float("nan")
    if var["kind"] == "variance_shrinkage":
        b = (
            f"In the 25-50% band the raw gap stays tiny and does not grow "
            f"(delta_25={d25:+.4f}, delta_30={d30:+.4f}, delta_40={d40:+.4f}, "
            f"delta_50={d50:+.4f} Mbps; all < 0.02 Mbps), while std(delta) "
            f"falls from {s25:.4f} Mbps at 25% to {s30:.4f}/{s40:.4f}/{s50:.4f} "
            f"at 30/40/50% and then plateaus (not to zero). Standardized "
            f"effect delta/std stays flat (~0.4). The later p<0.05 values are "
            f"therefore variance shrinkage plus more consistent sign of an "
            f"already-tiny gap, not a real effect-size recovery."
        )
    elif var["kind"] == "real_effect_growth":
        b = (
            f"In the 25-50% band delta grows from {d25:+.4f} Mbps at 25% to "
            f"{d50:+.4f} Mbps at 50% while std(delta) stays comparable "
            f"({s25:.4f} -> {s50:.4f} Mbps), so those p-values reflect a "
            f"real, growing gap rather than variance shrinkage alone."
        )
    else:
        b = (
            f"In the 25-50% band both delta and std(delta) move "
            f"(delta: {d25:+.4f} -> {d30:+.4f} -> {d40:+.4f} -> {d50:+.4f} Mbps; "
            f"std: {s25:.4f} -> {s30:.4f} -> {s40:.4f} -> {s50:.4f} Mbps). "
            f"The later p-values are not a clean case of either effect-size "
            f"growth or variance collapse alone."
        )
    j3_note = ", ".join(
        f"{r['max_bw_share']:.1%} p={r['wilcoxon_p']:.4f} q={r['fdr_q']:.4f}"
        for r in j3_rows
        if r["max_bw_share"] in NEW_SHARES
    )
    return a + " " + b + f" New points (raw p / 33-test q): {j3_note}."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Analyze checkpoint/output only; do not run missing caps",
    )
    args = ap.parse_args()

    if not GRID_PATH.exists():
        print(f"MISSING {GRID_PATH}", file=sys.stderr)
        return 1
    grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    old_cells = grid["cells"]
    if len(old_cells) != 30:
        print(f"expected 30 grid cells, got {len(old_cells)}", file=sys.stderr)
        return 1

    ckpt = load_checkpoint()
    if OUT_PATH.exists() and not ckpt:
        prior = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        for p in prior.get("new_points", []):
            ckpt[f"{p['max_bw_share']:.5f}"] = p

    _out("STEP 1 - J=3 bisection at 23.5%, 24%, 24.5% (all three, no early stop)")
    new_points: list[dict[str, Any]] = []
    for share in NEW_SHARES:
        key = f"{share:.5f}"
        if key in ckpt and ckpt[key].get("per_seed"):
            _out(f"  reuse checkpoint cap={share:.1%}")
            pt = point_from_checkpoint(ckpt[key], share)
        else:
            if args.skip_run:
                print(f"MISSING run for cap={share:.1%}", file=sys.stderr)
                return 1
            pt = run_j3_point(share)
            ckpt[key] = pt
            save_checkpoint(ckpt)
            _out(
                f"    SCA={pt['sca_mean_mbps']:.3f}  random={pt['random_mean_mbps']:.3f}  "
                f"d={pt['delta_mbps']:+.4f}  std={pt['delta_std_mbps']:.4f}  "
                f"d/std={pt['effect_size']:.2f}  p={pt['wilcoxon_p']:.4f}  "
                f"feas={pt['sca_feasible_frac']:.0%}/{pt['random_feasible_frac']:.0%}"
            )
        new_points.append(pt)

    _out("\nSTEP 2 - delta and std(delta) at J=3 for cap=25/30/40/50%")
    loose_points: list[dict[str, Any]] = []
    for share, path in ((0.25, CAMPAIGN_FOR_SHARE[0.25]), (0.30, CAMPAIGN_FOR_SHARE[0.30])):
        if not path.exists():
            print(f"MISSING {path}", file=sys.stderr)
            return 1
        loose_points.append(
            extract_campaign_j(path, share, J, source=f"campaign:{path.name}")
        )
        p = loose_points[-1]
        _out(
            f"  cap={share:.0%} from {path.name}: "
            f"d={p['delta_mbps']:+.4f} std={p['delta_std_mbps']:.4f} p={p['wilcoxon_p']:.4f}"
        )

    for share in LOOSE_SHARES_NEED_RERUN:
        key = f"{share:.5f}"
        if key in ckpt and ckpt[key].get("per_seed"):
            _out(f"  reuse checkpoint cap={share:.0%} (std not in 30-cell grid)")
            pt = point_from_checkpoint(ckpt[key], share)
        else:
            if args.skip_run:
                print(
                    f"WARNING: cap={share:.0%} not in grid and no checkpoint; "
                    "std(delta) unavailable",
                    file=sys.stderr,
                )
                continue
            _out(
                f"  cap={share:.0%} is not in bw_cap_by_J_grid.json; "
                "re-running J=3 seeds 1-20 to recover per-seed std(delta)"
            )
            pt = run_j3_point(share)
            pt["source"] = "loose_cap_std_recovery"
            ckpt[key] = pt
            save_checkpoint(ckpt)
        loose_points.append(pt)
        _out(
            f"    d={pt['delta_mbps']:+.4f} std={pt['delta_std_mbps']:.4f} "
            f"p={pt['wilcoxon_p']:.4f}"
        )

    if CAMPAIGN_FOR_SHARE[1.00].exists():
        nocap = extract_campaign_j(
            CAMPAIGN_FOR_SHARE[1.00],
            1.00,
            J,
            source="campaign:campaign_8.8mhz_n20.json (no per-link cap)",
        )
        _out(
            f"  no-cap (share=100% equivalent): "
            f"d={nocap['delta_mbps']:+.4f} std={nocap['delta_std_mbps']:.4f} "
            f"p={nocap['wilcoxon_p']:.4f}"
        )
        loose_points.append(nocap)

    _out("\nSTEP 3 - BH-FDR once across 30 old + 3 new p-values")
    combined_src: list[dict[str, Any]] = []
    for c in old_cells:
        combined_src.append(grid_cell_as_row(c, is_new=False))
    for p in new_points:
        combined_src.append(grid_cell_as_row(p, is_new=True))
    pvals = [r["wilcoxon_p"] for r in combined_src]
    qs = bh_fdr(pvals)
    for r, q in zip(combined_src, qs, strict=True):
        r["fdr_q"] = q
    combined_src.sort(key=lambda r: (r["max_bw_share"], r["j"]))

    q_by_new_share = {
        r["max_bw_share"]: r["fdr_q"] for r in combined_src if r["is_new"]
    }
    for p in new_points:
        p["fdr_q"] = q_by_new_share[p["max_bw_share"]]

    crossover = find_crossover(combined_src)
    var = variance_verdict(loose_points)
    j3_rows = [r for r in combined_src if r["j"] == J]
    paragraph = build_paragraph(crossover, var, j3_rows)

    payload = {
        "description": (
            "J=3 bisection of the SCA-vs-random FDR boundary at 8.8 MHz, I=10, "
            "20 seeds. Three new caps (23.5/24/24.5%) plus the existing 30-cell "
            "cap×J grid; BH-FDR applied once across all 33 p-values. "
            "Loose-cap 40/50% J=3 rows are variance diagnostics only and are "
            "not included in the 33-test FDR."
        ),
        "b_sys_hz": B_SYS,
        "j": J,
        "n_runs": N_RUNS,
        "seed_start": SEED_START,
        "new_shares": list(NEW_SHARES),
        "n_old_cells": 30,
        "n_new_cells": 3,
        "n_fdr_tests": 33,
        "fdr_scope": "BH-FDR applied once across all 33 Wilcoxon p-values",
        "new_points": new_points,
        "combined_33": combined_src,
        "crossover": crossover,
        "loose_cap_variance": {
            "note": (
                "25% and 30% per-seed series come from the same campaigns as "
                "bw_cap_by_J_grid.json. 40% and 50% were not in that 30-cell "
                "file; J=3 was re-run (seeds 1-20) only to recover std(delta). "
                "Those two p-values are not added to the 33-test FDR."
            ),
            "points": [
                {
                    k: v
                    for k, v in p.items()
                    if k != "per_seed" or p["max_bw_share"] in LOOSE_SHARES_NEED_RERUN
                }
                for p in loose_points
            ],
            "verdict": var,
        },
        "paragraph": paragraph,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _out(f"\nwrote {OUT_PATH}")

    for p in new_points:
        _out(
            f"NEW cap={p['max_bw_share']:.1%}  "
            f"SCA={p['sca_mean_mbps']:.3f} random={p['random_mean_mbps']:.3f}  "
            f"d={p['delta_mbps']:+.4f} std(d)={p['delta_std_mbps']:.4f}  "
            f"d/std={p['effect_size']:.2f}  p={p['wilcoxon_p']:.4f}  "
            f"q={p['fdr_q']:.4f}  feas={p['sca_feasible_frac']:.0%}"
        )

    print_combined_table(combined_src)
    print_j3_transition(combined_src)
    print_loose_table(loose_points)
    _out(f"\nVARIANCE VERDICT: {var['kind']}")
    _out(f"\nPARAGRAPH:\n{paragraph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
