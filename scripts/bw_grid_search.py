"""Exhaustive B_sys x max_bw_share grid search (J=1-5, SCA vs random).

Screens every (bandwidth, cap) combo at I=10 across J=1..5 with 20 seeds.
Scores vs reference 8.8 MHz / cap=25%. Checkpoints incrementally.

Usage:
  python scripts/bw_grid_search.py --estimate          # grid size + ETA only
  python scripts/bw_grid_search.py --run                # full search (hours)
  python scripts/bw_grid_search.py --run --resume     # continue checkpoint
  python scripts/bw_grid_search.py --run --j3-only      # faster coarse pass

Output: results/bw_grid_search.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uavdt.config import SimConfig  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.experiments.methods import METHODS, run_method  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402
from uavdt.sca import SCASettings  # noqa: E402
from paired_winrate import wilcoxon_signed_rank  # noqa: E402

OUT_PATH = ROOT / "results" / "bw_grid_search.json"
REF_B_HZ = 8_800_000.0
REF_SHARE = 0.25
J_VALUES = (1, 2, 3, 4, 5)

# Seconds per (B_sys, cap, J) cell — calibrated 2026-09-05 on this machine.
SEC_PER_J_CELL = 38.5

# --- search grid (edit to widen/narrow) ---
B_MHZ_VALUES = (
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.4,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.8,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    18.0,
    20.0,
)

CAP_SHARE_VALUES = (
    0.08,
    0.10,
    0.12,
    0.14,
    0.15,
    0.16,
    0.18,
    0.20,
    0.21,
    0.22,
    0.23,
    0.24,
    0.25,
    0.26,
    0.28,
    0.30,
    0.32,
    0.35,
    0.40,
    0.45,
    0.50,
)


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


def grid_pairs() -> list[tuple[float, float]]:
    return [(mhz * 1e6, share) for mhz in B_MHZ_VALUES for share in CAP_SHARE_VALUES]


def config_key(b_hz: float, share: float) -> str:
    return f"{b_hz:g}:{share:g}"


@dataclass
class JCell:
    j: int
    sca_mean_mbps: float
    random_mean_mbps: float
    delta_mbps: float
    delta_std_mbps: float
    effect_size: float
    wilcoxon_p: float
    fdr_q: float | None = None
    sca_feasible_frac: float = 1.0
    random_feasible_frac: float = 1.0
    sca_wins: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfigResult:
    b_sys_hz: float
    b_sys_mhz: float
    max_bw_share: float
    per_link_cap_mhz: float
    per_link_cap_hz: float
    j_cells: list[JCell]
    min_sca_feas: float
    min_random_feas: float
    all_j_positive_delta: bool
    all_j_raw_sig: bool
    all_j_fdr_sig: bool
    mean_delta_j3: float
    spread_j3: float
    beats_ref_j3_delta: bool
    beats_ref_j3_spread: bool
    rank_score: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["j_cells"] = [c.to_dict() for c in self.j_cells]
        return d


def run_j_cell(
    b_hz: float,
    share: float,
    j: int,
    *,
    n_runs: int,
    seed_start: int,
    sca_settings: SCASettings,
) -> JCell:
    cfg = SimConfig(b_sys_hz=b_hz, max_bw_share=share)
    cfg = config_for_counts(10, j, cfg)
    sca_ps: list[float] = []
    rnd_ps: list[float] = []
    sca_feas: list[bool] = []
    rnd_feas: list[bool] = []
    for seed in range(seed_start, seed_start + n_runs):
        scenario = generate_scenario(seed, cfg)
        sca_run = run_method(scenario, "sca", seed, sca_settings=sca_settings)
        rnd_run = run_method(scenario, "random", seed, sca_settings=sca_settings)
        sca_ps.append(sca_run.sum_rate_mbps)
        rnd_ps.append(rnd_run.sum_rate_mbps)
        sca_feas.append(sca_run.feasible)
        rnd_feas.append(rnd_run.feasible)
    sca_arr = np.array(sca_ps)
    rnd_arr = np.array(rnd_ps)
    deltas = sca_arr - rnd_arr
    w = wilcoxon_signed_rank(deltas.tolist())
    dstd = float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan")
    dmean = float(np.mean(deltas))
    return JCell(
        j=j,
        sca_mean_mbps=float(np.mean(sca_arr)),
        random_mean_mbps=float(np.mean(rnd_arr)),
        delta_mbps=dmean,
        delta_std_mbps=dstd,
        effect_size=dmean / dstd if dstd > 0 else float("nan"),
        wilcoxon_p=float(w["p_two_sided"]),
        sca_feasible_frac=float(np.mean(sca_feas)),
        random_feasible_frac=float(np.mean(rnd_feas)),
        sca_wins=int(np.sum(deltas > 0)),
    )


def run_config(
    b_hz: float,
    share: float,
    *,
    j_values: tuple[int, ...],
    n_runs: int,
    seed_start: int,
) -> ConfigResult:
    sca_settings = SCASettings(solver=None, max_iterations=30, step_size_m=20.0)
    cells = [
        run_j_cell(b_hz, share, j, n_runs=n_runs, seed_start=seed_start, sca_settings=sca_settings)
        for j in j_values
    ]
    cap_hz = b_hz * share
    min_sca = min(c.sca_feasible_frac for c in cells)
    min_rnd = min(c.random_feasible_frac for c in cells)
    all_pos = all(c.delta_mbps > 0 for c in cells)
    all_raw = all(c.wilcoxon_p < 0.05 and c.delta_mbps > 0 for c in cells)
    j3 = next(c for c in cells if c.j == 3)
    means_j3 = {m: 0.0 for m in METHODS}  # only sca/random computed here
    means_j3["sca"] = j3.sca_mean_mbps
    means_j3["random"] = j3.random_mean_mbps
    spread_j3 = abs(j3.sca_mean_mbps - j3.random_mean_mbps)
    return ConfigResult(
        b_sys_hz=b_hz,
        b_sys_mhz=b_hz / 1e6,
        max_bw_share=share,
        per_link_cap_mhz=cap_hz / 1e6,
        per_link_cap_hz=cap_hz,
        j_cells=cells,
        min_sca_feas=min_sca,
        min_random_feas=min_rnd,
        all_j_positive_delta=all_pos,
        all_j_raw_sig=all_raw,
        all_j_fdr_sig=False,  # filled later globally
        mean_delta_j3=j3.delta_mbps,
        spread_j3=spread_j3,
        beats_ref_j3_delta=False,
        beats_ref_j3_spread=False,
        rank_score=0.0,
    )


def apply_global_fdr_dicts(results: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for r in results:
        for c in r["j_cells"]:
            flat.append(c)
    ps = [c["wilcoxon_p"] for c in flat]
    qs = bh_fdr(ps)
    for c, q in zip(flat, qs, strict=True):
        c["fdr_q"] = q
    for r in results:
        r["all_j_fdr_sig"] = all(
            c["fdr_q"] < 0.05 and c["delta_mbps"] > 0 for c in r["j_cells"]
        )


def score_vs_reference_dicts(results: list[dict[str, Any]]) -> None:
    ref = next(
        (r for r in results if r["b_sys_hz"] == REF_B_HZ and r["max_bw_share"] == REF_SHARE),
        None,
    )
    ref_j3_delta = ref["mean_delta_j3"] if ref else 0.019
    ref_j3_spread = ref["spread_j3"] if ref else 0.046
    for r in results:
        r["beats_ref_j3_delta"] = r["mean_delta_j3"] > ref_j3_delta
        r["beats_ref_j3_spread"] = r["spread_j3"] > ref_j3_spread
        r["rank_score"] = (
            (10.0 if r["all_j_fdr_sig"] else 0.0)
            + (5.0 if r["all_j_raw_sig"] else 0.0)
            + r["mean_delta_j3"]
            + r["spread_j3"]
            + (1.0 if r["min_sca_feas"] >= 1.0 else -5.0)
        )


def estimate_seconds(*, j_values: tuple[int, ...], n_configs: int) -> float:
    return SEC_PER_J_CELL * len(j_values) * n_configs


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def load_checkpoint() -> dict[str, Any]:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    return {"completed": {}, "results": []}


def save_checkpoint(payload: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_top(payload: dict[str, Any], n: int = 15) -> None:
    results = payload.get("results", [])
    if not results:
        print("No results yet.")
        return
    ranked = sorted(results, key=lambda r: r["rank_score"], reverse=True)
    print(f"\nTOP {n} by rank_score (ref: 8.8 MHz cap=25%):")
    print(
        f"{'MHz':>6s} {'cap%':>5s} {'lnkMHz':>7s} "
        f"{'dJ3':>7s} {'allJ_fdr':>9s} {'feas':>5s} {'score':>7s}"
    )
    for r in ranked[:n]:
        print(
            f"{r['b_sys_mhz']:6.1f} {r['max_bw_share']:5.0%} {r['per_link_cap_mhz']:7.3f} "
            f"{r['mean_delta_j3']:+7.3f} {str(r['all_j_fdr_sig']):>9s} "
            f"{r['min_sca_feas']:5.0%} {r['rank_score']:7.2f}"
        )
    perfect = [r for r in results if r["all_j_fdr_sig"] and r["min_sca_feas"] >= 1.0]
    print(f"\nPerfect (all J FDR-sig, 100% SCA feas): {len(perfect)} configs")
    for r in sorted(perfect, key=lambda x: x["rank_score"], reverse=True)[:10]:
        print(
            f"  {r['b_sys_mhz']:.1f} MHz cap={r['max_bw_share']:.0%} "
            f"({r['per_link_cap_mhz']:.3f} MHz/link)  dJ3={r['mean_delta_j3']:+.3f}  "
            f"score={r['rank_score']:.2f}"
        )


def cmd_estimate(j_values: tuple[int, ...]) -> int:
    pairs = grid_pairs()
    sec = estimate_seconds(j_values=j_values, n_configs=len(pairs))
    print("BANDWIDTH GRID SEARCH — estimate")
    print(f"  B_sys values:   {len(B_MHZ_VALUES)}  ({B_MHZ_VALUES[0]} .. {B_MHZ_VALUES[-1]} MHz)")
    print(f"  cap shares:     {len(CAP_SHARE_VALUES)}  ({CAP_SHARE_VALUES[0]:.0%} .. {CAP_SHARE_VALUES[-1]:.0%})")
    print(f"  configs:        {len(pairs)}")
    print(f"  J per config:   {list(j_values)}")
    print(f"  cells total:    {len(pairs) * len(j_values)}")
    print(f"  sec/cell:       {SEC_PER_J_CELL:.1f} (calibrated)")
    print(f"  ETA:            {fmt_eta(sec)}  ({sec:.0f} s)")
    print(f"  output:         {OUT_PATH}")
    if sec > 30 * 60:
        print("\n  ETA > 30 min — run locally when ready (see command below).")
    return 0


def cmd_run(
    *,
    j_values: tuple[int, ...],
    n_runs: int,
    seed_start: int,
    resume: bool,
) -> int:
    pairs = grid_pairs()
    sec = estimate_seconds(j_values=j_values, n_configs=len(pairs))
    print(f"Grid: {len(pairs)} configs, ETA ~{fmt_eta(sec)}")

    payload = load_checkpoint() if resume else {"completed": {}, "results": []}
    completed: dict[str, bool] = payload.setdefault("completed", {})
    results_by_key: dict[str, dict] = {config_key(r["b_sys_hz"], r["max_bw_share"]): r for r in payload.get("results", [])}

    t0 = time.time()
    for i, (b_hz, share) in enumerate(pairs, 1):
        key = config_key(b_hz, share)
        if completed.get(key):
            continue
        print(
            f"[{i}/{len(pairs)}] {b_hz/1e6:.2g} MHz cap={share:.0%} "
            f"({b_hz*share/1e6:.3f} MHz/link) ...",
            flush=True,
        )
        res = run_config(b_hz, share, j_values=j_values, n_runs=n_runs, seed_start=seed_start)
        results_by_key[key] = res.to_dict()
        completed[key] = True
        payload["results"] = list(results_by_key.values())
        payload["meta"] = {
            "n_runs": n_runs,
            "seed_start": seed_start,
            "j_values": list(j_values),
            "reference": {"b_sys_hz": REF_B_HZ, "max_bw_share": REF_SHARE},
            "elapsed_s": time.time() - t0,
            "done_configs": len(completed),
            "total_configs": len(pairs),
        }
        save_checkpoint(payload)

    # Post-process: FDR + ranking
    apply_global_fdr_dicts(payload["results"])
    score_vs_reference_dicts(payload["results"])
    ref = next(
        (
            r
            for r in payload["results"]
            if r["b_sys_hz"] == REF_B_HZ and r["max_bw_share"] == REF_SHARE
        ),
        None,
    )
    payload["verdict"] = {
        "reference": "8.8 MHz cap=25%",
        "ref_in_grid": ref,
        "n_all_j_fdr_perfect": sum(
            1 for r in payload["results"] if r["all_j_fdr_sig"] and r["min_sca_feas"] >= 1.0
        ),
        "n_beats_ref_j3_delta": sum(1 for r in payload["results"] if r["beats_ref_j3_delta"]),
        "best_all_j_fdr": max(
            (r for r in payload["results"] if r["all_j_fdr_sig"]),
            key=lambda x: x["rank_score"],
            default=None,
        ),
    }
    save_checkpoint(payload)
    print(f"\nwrote {OUT_PATH}")
    print_top(payload)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimate", action="store_true", help="Print grid size and ETA only")
    ap.add_argument("--run", action="store_true", help="Execute search")
    ap.add_argument("--resume", action="store_true", help="Skip completed configs in checkpoint")
    ap.add_argument("--j3-only", action="store_true", help="Screen J=3 only (faster coarse pass)")
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=1)
    args = ap.parse_args()

    j_values: tuple[int, ...] = (3,) if args.j3_only else J_VALUES

    if args.estimate:
        return cmd_estimate(j_values)
    if args.run:
        return cmd_run(
            j_values=j_values,
            n_runs=args.n_runs,
            seed_start=args.seed_start,
            resume=args.resume,
        )
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
