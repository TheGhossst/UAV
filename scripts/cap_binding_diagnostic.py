"""Mechanism test for the per-link cap: same geometry, vary only the cap.

Does not score SCA vs random. Does not search for a p-value. Holds
placement (random / k-means) and association fixed, then re-solves the
frozen-q bandwidth LP at no-cap, 15%, and 25%.

Also audits the existing J-sweep and the 30-cell cap×J search against a
practical bar of Δ > 0.05 Mbps (SCA − random). That bar is applied after
seeing the data; it is a diagnostic, not a pre-registered claim.

Output: results/cap_binding_diagnostic.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uavdt.config import SimConfig  # noqa: E402
from uavdt.evaluator import evaluate  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.models import Allocation, Scenario  # noqa: E402
from uavdt.placement.kmeans import place_kmeans  # noqa: E402
from uavdt.placement.random import place_random  # noqa: E402
from uavdt.resources import cpu_stable_processing, nearest_association  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402
from uavdt.sca.cvx_problem import bandwidth_floors_hz, solve_bandwidth_at_fixed_q  # noqa: E402
from uavdt.sca.linearize import spectral_efficiency  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402

B_SYS = 8_800_000.0
J_VALUES = (1, 2, 3, 4, 5)
N_RUNS = 20
SEED_START = 1
SHARES: tuple[float | None, ...] = (None, 0.25, 0.15)
PRACTICAL_MBPS = 0.05
AT_CAP_FRAC = 0.999
OUT = ROOT / "results" / "cap_binding_diagnostic.json"
PAIRED_15 = ROOT / "results" / "campaign_8.8mhz_cap15_n20_paired.json"
GRID = ROOT / "results" / "bw_cap_by_J_grid.json"


def _out(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _share_key(share: float | None) -> str:
    return "none" if share is None else f"{share:.2f}"


def _metrics(
    scenario: Scenario,
    uav: np.ndarray,
    a: np.ndarray,
    proc: np.ndarray,
    bw: np.ndarray,
    ev,
) -> dict[str, Any]:
    cfg = scenario.cfg
    b_sys = float(cfg.b_sys_hz)
    cap_hz = float(cfg.link_bandwidth_cap_hz)
    se = spectral_efficiency(scenario.iot_xyz_m, uav, cfg)
    floors = bandwidth_floors_hz(scenario, a, proc, se)
    assoc = a > 0.5
    b_assoc = bw[assoc]
    n_assoc = int(assoc.sum())
    max_b = float(b_assoc.max()) if n_assoc else 0.0
    leftover = b_sys - float(np.nansum(np.where(assoc, floors, 0.0)))
    se_masked = np.where(assoc, se, -np.inf)
    best = np.unravel_index(int(np.argmax(se_masked)), se.shape)
    extra_best = float(bw[best] - floors[best])
    dump_frac = extra_best / leftover if leftover > 1.0 else float("nan")
    n_at_cap = int(np.sum(b_assoc >= AT_CAP_FRAC * cap_hz)) if cap_hz < b_sys * 0.999 else int(
        np.sum(b_assoc >= AT_CAP_FRAC * b_sys)
    )
    hhi = float(np.sum((b_assoc / b_sys) ** 2)) if n_assoc else 0.0
    return {
        "feasible": bool(ev.feasible),
        "sum_rate_Mbps": float(ev.sum_rate_mbps),
        "max_share": max_b / b_sys,
        "n_assoc": n_assoc,
        "n_at_cap": n_at_cap,
        "frac_assoc_at_cap": (n_at_cap / n_assoc) if n_assoc else 0.0,
        "leftover_hz": leftover,
        "dump_frac_on_best": dump_frac,
        "hhi": hhi,
        "infeasible_lp": False,
    }


def _solve_one(
    sc_base: Scenario,
    uav: np.ndarray,
    a: np.ndarray,
    proc: np.ndarray,
    share: float | None,
) -> dict[str, Any]:
    cfg = replace(sc_base.cfg, max_bw_share=share)
    sc = replace(sc_base, cfg=cfg)
    res = solve_bandwidth_at_fixed_q(sc, uav, a, proc, SCASettings(solver=None))
    if res.infeasible:
        return {
            "feasible": False,
            "sum_rate_Mbps": float("nan"),
            "max_share": float("nan"),
            "n_assoc": int((a > 0.5).sum()),
            "n_at_cap": 0,
            "frac_assoc_at_cap": 0.0,
            "leftover_hz": float("nan"),
            "dump_frac_on_best": float("nan"),
            "hhi": float("nan"),
            "infeasible_lp": True,
        }
    alloc = Allocation(a, proc, res.bandwidth_hz)
    ev = evaluate(sc, uav, alloc)
    return _metrics(sc, uav, a, proc, res.bandwidth_hz, ev)


def run_binding() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for j in J_VALUES:
        for seed in range(SEED_START, SEED_START + N_RUNS):
            cfg0 = SimConfig(b_sys_hz=B_SYS, max_bw_share=None)
            sc0 = generate_scenario(seed, config_for_counts(10, j, cfg0))
            placements = {
                "random": place_random(j, seed, sc0.cfg),
                "kmeans": place_kmeans(sc0, seed),
            }
            for method, uav in placements.items():
                a = nearest_association(sc0.iot_xyz_m, uav)
                proc = cpu_stable_processing(sc0, a)
                by_share: dict[str, dict[str, Any]] = {}
                for share in SHARES:
                    by_share[_share_key(share)] = _solve_one(sc0, uav, a, proc, share)
                none = by_share["none"]
                row = {
                    "j": j,
                    "seed": seed,
                    "method": method,
                    "by_share": by_share,
                    "uncapped_max_share": none["max_share"],
                    "uncapped_exceeds_15": bool(none["max_share"] > 0.15 + 1e-9),
                    "uncapped_exceeds_25": bool(none["max_share"] > 0.25 + 1e-9),
                    "rate_gap_15_vs_none_Mbps": (
                        none["sum_rate_Mbps"] - by_share["0.15"]["sum_rate_Mbps"]
                    ),
                    "rate_gap_25_vs_none_Mbps": (
                        none["sum_rate_Mbps"] - by_share["0.25"]["sum_rate_Mbps"]
                    ),
                }
                rows.append(row)
                _out(
                    f"  J={j} seed={seed} {method:7s}  "
                    f"max_share none={none['max_share']:.3f} "
                    f"25%={by_share['0.25']['max_share']:.3f} "
                    f"15%={by_share['0.15']['max_share']:.3f}  "
                    f"gap15={row['rate_gap_15_vs_none_Mbps']:+.4f} "
                    f"gap25={row['rate_gap_25_vs_none_Mbps']:+.4f}"
                )
    return {"rows": rows}


def _mean(xs: list[float]) -> float:
    arr = np.array(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def summarize_binding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for j in J_VALUES:
        block: dict[str, Any] = {}
        for method in ("random", "kmeans"):
            sub = [r for r in rows if r["j"] == j and r["method"] == method]
            block[method] = {
                "n": len(sub),
                "uncapped_mean_max_share": _mean([r["uncapped_max_share"] for r in sub]),
                "frac_uncapped_exceeds_15": _mean(
                    [float(r["uncapped_exceeds_15"]) for r in sub]
                ),
                "frac_uncapped_exceeds_25": _mean(
                    [float(r["uncapped_exceeds_25"]) for r in sub]
                ),
                "mean_max_share_15": _mean(
                    [r["by_share"]["0.15"]["max_share"] for r in sub]
                ),
                "mean_max_share_25": _mean(
                    [r["by_share"]["0.25"]["max_share"] for r in sub]
                ),
                "mean_n_at_cap_15": _mean(
                    [r["by_share"]["0.15"]["n_at_cap"] for r in sub]
                ),
                "mean_n_at_cap_25": _mean(
                    [r["by_share"]["0.25"]["n_at_cap"] for r in sub]
                ),
                "mean_dump_frac_none": _mean(
                    [r["by_share"]["none"]["dump_frac_on_best"] for r in sub]
                ),
                "mean_dump_frac_15": _mean(
                    [r["by_share"]["0.15"]["dump_frac_on_best"] for r in sub]
                ),
                "mean_dump_frac_25": _mean(
                    [r["by_share"]["0.25"]["dump_frac_on_best"] for r in sub]
                ),
                "mean_rate_gap_15_Mbps": _mean(
                    [r["rate_gap_15_vs_none_Mbps"] for r in sub]
                ),
                "mean_rate_gap_25_Mbps": _mean(
                    [r["rate_gap_25_vs_none_Mbps"] for r in sub]
                ),
            }
        out[str(j)] = block
    return out


def practical_j_sweep() -> dict[str, Any]:
    """Apply Δ > 0.05 Mbps to the existing 15% UAV sweep (seeds 1–20)."""
    if not PAIRED_15.exists():
        return {"error": f"missing {PAIRED_15}"}
    payload = json.loads(PAIRED_15.read_text(encoding="utf-8"))
    rows = []
    for r in payload["rows"]:
        if r["axis"] != "uavs" or r["baseline"] != "random":
            continue
        mean_d = float(r["mean_delta_Mbps"])
        lo = float(r["ci95_lo_Mbps"])
        hi = float(r["ci95_hi_Mbps"])
        rows.append(
            {
                "j": int(r["x"]),
                "mean_delta_Mbps": mean_d,
                "ci95_lo_Mbps": lo,
                "ci95_hi_Mbps": hi,
                "wilcoxon_p": float(r["wilcoxon_p_two_sided"]),
                "wins": f"{r['wins']}/{r['n_seeds']}",
                "mean_clears_0.05": mean_d > PRACTICAL_MBPS,
                "ci_entirely_above_0.05": lo > PRACTICAL_MBPS,
                "ci_entirely_below_0.05": hi < PRACTICAL_MBPS,
            }
        )
    n_mean = sum(1 for r in rows if r["mean_clears_0.05"])
    n_ci = sum(1 for r in rows if r["ci_entirely_above_0.05"])
    return {
        "bar_Mbps": PRACTICAL_MBPS,
        "note": (
            "Bar applied after seeing the data. Mean > 0.05 is not the same "
            "as the 95% CI lying entirely above 0.05."
        ),
        "n_j": len(rows),
        "n_mean_above_bar": n_mean,
        "n_ci_entirely_above_bar": n_ci,
        "all_j_mean_above_bar": n_mean == len(rows) and len(rows) == 5,
        "all_j_ci_above_bar": n_ci == len(rows) and len(rows) == 5,
        "rows": rows,
    }


def search_grid_report() -> dict[str, Any]:
    if not GRID.exists():
        return {"error": f"missing {GRID}"}
    d = json.loads(GRID.read_text(encoding="utf-8"))
    cells = d["cells"]
    n = len(cells)
    n_fdr = sum(1 for c in cells if c["fdr_q"] < 0.05)
    n_bar = sum(1 for c in cells if c["delta_mbps"] > PRACTICAL_MBPS)
    n_both = sum(
        1 for c in cells if c["fdr_q"] < 0.05 and c["delta_mbps"] > PRACTICAL_MBPS
    )
    by_cap: dict[str, Any] = {}
    for share in d["max_bw_shares"]:
        sub = [c for c in cells if abs(c["max_bw_share"] - share) < 1e-12]
        by_cap[f"{share:.2f}"] = {
            "j_mean_delta": {str(c["j"]): c["delta_mbps"] for c in sub},
            "j_fdr_q": {str(c["j"]): c["fdr_q"] for c in sub},
            "n_j_delta_above_0.05": sum(1 for c in sub if c["delta_mbps"] > PRACTICAL_MBPS),
            "n_j_fdr": sum(1 for c in sub if c["fdr_q"] < 0.05),
            "all_j_delta_above_0.05": all(c["delta_mbps"] > PRACTICAL_MBPS for c in sub),
        }
    return {
        "file": GRID.name,
        "n_cells": n,
        "n_fdr_q_lt_0.05": n_fdr,
        "n_delta_gt_0.05": n_bar,
        "n_both": n_both,
        "by_cap": by_cap,
        "note": (
            "Exploratory 6×5 search. FDR is over the 30 cells already "
            "computed. Selecting 15% because it is the tightest cap with "
            "FDR-significant J=3 is still a search-selected result."
        ),
    }


def print_report(
    binding: dict[str, Any],
    practical: dict[str, Any],
    grid: dict[str, Any],
) -> None:
    _out("=" * 88)
    _out("1. 15% is a tighter-cap sensitivity, not an independently chosen primary")
    _out("=" * 88)
    _out(
        "J-sweep practical bar and the 30-cell search are already on disk. "
        "Re-running SCA vs random cannot un-search the grid."
    )
    _out("")
    _out(f"--- Practical bar Δ > {PRACTICAL_MBPS} Mbps (SCA−random, 15%, seeds 1–20) ---")
    if "rows" in practical:
        _out(
            f"{'J':>3} {'Δ mean':>8} {'CI95 lo':>8} {'CI95 hi':>8} "
            f"{'p':>10} {'mean>0.05':>10} {'CI>0.05':>8}"
        )
        for r in practical["rows"]:
            _out(
                f"{r['j']:3d} {r['mean_delta_Mbps']:8.4f} {r['ci95_lo_Mbps']:8.4f} "
                f"{r['ci95_hi_Mbps']:8.4f} {r['wilcoxon_p']:10.4g} "
                f"{str(r['mean_clears_0.05']):>10} {str(r['ci_entirely_above_0.05']):>8}"
            )
        _out(
            f"mean > 0.05 at {practical['n_mean_above_bar']}/5 J; "
            f"CI entirely above 0.05 at {practical['n_ci_entirely_above_bar']}/5 J."
        )
    _out("")
    _out("--- 30-cell cap×J search (exploratory; report as a search) ---")
    if "by_cap" in grid:
        _out(f"{'cap':>6} {'#J Δ>0.05':>10} {'#J FDR':>8} {'all J Δ>0.05':>14}")
        for cap, block in grid["by_cap"].items():
            _out(
                f"{cap:>6} {block['n_j_delta_above_0.05']:10d} "
                f"{block['n_j_fdr']:8d} {str(block['all_j_delta_above_0.05']):>14}"
            )
        _out(
            f"cells with FDR q<0.05 and Δ>0.05: {grid['n_both']}/{grid['n_cells']}"
        )
    _out("")
    _out("--- Binding (frozen q, random + k-means, leftover dump) ---")
    _out(
        "ceil(1/share) links are needed to exhaust B_sys at the cap: "
        "25% → 4 of 10, 15% → 7 of 10. Equal-share at I=10 is 10%."
    )
    _out(
        f"{'J':>3} {'meth':>7} {'uncap max':>10} {'>15%':>6} {'>25%':>6} "
        f"{'max15':>7} {'max25':>7} {'ncap15':>7} {'ncap25':>7} "
        f"{'gap15':>8} {'gap25':>8}"
    )
    for j in J_VALUES:
        for method in ("random", "kmeans"):
            b = binding[str(j)][method]
            _out(
                f"{j:3d} {method:>7} {b['uncapped_mean_max_share']:10.3f} "
                f"{b['frac_uncapped_exceeds_15']:6.0%} {b['frac_uncapped_exceeds_25']:6.0%} "
                f"{b['mean_max_share_15']:7.3f} {b['mean_max_share_25']:7.3f} "
                f"{b['mean_n_at_cap_15']:7.2f} {b['mean_n_at_cap_25']:7.2f} "
                f"{b['mean_rate_gap_15_Mbps']:8.4f} {b['mean_rate_gap_25_Mbps']:8.4f}"
            )


def main() -> int:
    _out("Running frozen-geometry cap binding (J=1..5, 20 seeds, random+kmeans)...")
    raw = run_binding()
    binding = summarize_binding(raw["rows"])
    practical = practical_j_sweep()
    grid = search_grid_report()
    payload = {
        "b_sys_hz": B_SYS,
        "n_runs": N_RUNS,
        "seed_start": SEED_START,
        "practical_bar_Mbps": PRACTICAL_MBPS,
        "claim": (
            "25% is the primary leftover-dump stress test. 15% is a "
            "tighter-cap sensitivity, not an independently chosen primary: "
            "it is a search-selected cap at which leftover dump is forced "
            "across most associated links. The J-sweep does not support a "
            "uniform practical SCA−random gap of 0.05 Mbps."
        ),
        "binding_summary": binding,
        "practical_j_sweep": practical,
        "search_grid": grid,
        "rows": raw["rows"],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_report(binding, practical, grid)
    _out("")
    _out(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
