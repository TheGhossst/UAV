"""Cap x J grid: SCA vs random across J=1-5 at 8.8 MHz.

Runs full --axis all campaigns for missing caps, then builds 30-cell table
with global BH-FDR across all p-values.

Output: results/bw_cap_by_J_grid.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402

B_SYS = 8_800_000.0
CAPS = (0.15, 0.18, 0.20, 0.23, 0.25, 0.30)
J_VALUES = (1, 2, 3, 4, 5)
OUT_PATH = ROOT / "results" / "bw_cap_by_J_grid.json"

# Prefer existing campaign files (share -> path)
CAMPAIGN_PATHS: dict[float, Path] = {
    0.15: ROOT / "results" / "campaign_8.8mhz_cap15_n20.json",
    0.18: ROOT / "results" / "campaign_8.8mhz_cap18_n20.json",
    0.20: ROOT / "results" / "campaign_8.8mhz_cap20_n20.json",
    0.23: ROOT / "results" / "campaign_8.8mhz_cap23_n20.json",
    0.25: ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    0.30: ROOT / "results" / "campaign_8.8mhz_cap30_n20.json",
}


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


@dataclass
class GridCell:
    max_bw_share: float
    per_link_cap_mhz: float
    per_link_cap_hz: float
    j: int
    n_runs: int
    seed_start: int
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
    campaign_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_campaign(share: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "uavdt",
        "campaign",
        "--axis",
        "all",
        "--bandwidth",
        str(int(B_SYS)),
        "--max-bw-share",
        str(share),
        "--methods",
        "random,kmeans,pso,sca",
        "--n-runs",
        "20",
        "--seed-start",
        "1",
        "--solver",
        "cvxpy",
        "--out",
        str(out_path),
    ]
    env = {**dict(**{"PYTHONPATH": str(ROOT / "src")})}
    print(f"Running campaign cap={share:.0%} -> {out_path.name}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, check=True)


def extract_cell(campaign_path: Path, share: float, j: int) -> GridCell:
    d = json.loads(campaign_path.read_text(encoding="utf-8"))
    pt = next(x for x in d["points"] if x["axis"] == "uavs" and x["x"] == float(j))
    sca_ps = np.array(pt["by_method"]["sca"]["per_seed_Mbps"], dtype=float)
    rnd_ps = np.array(pt["by_method"]["random"]["per_seed_Mbps"], dtype=float)
    deltas = sca_ps - rnd_ps
    w = wilcoxon_signed_rank(deltas.tolist())
    delta_std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan")
    delta_mean = float(np.mean(deltas))
    effect = delta_mean / delta_std if delta_std > 0 else float("nan")
    cap_hz = B_SYS * share
    return GridCell(
        max_bw_share=share,
        per_link_cap_mhz=cap_hz / 1e6,
        per_link_cap_hz=cap_hz,
        j=j,
        n_runs=int(d["n_runs"]),
        seed_start=int(d.get("seed_start", 1)),
        sca_mean_mbps=float(np.mean(sca_ps)),
        random_mean_mbps=float(np.mean(rnd_ps)),
        delta_mbps=delta_mean,
        delta_std_mbps=delta_std,
        effect_size=effect,
        wilcoxon_p=float(w["p_two_sided"]),
        sca_feasible_frac=float(pt["by_method"]["sca"]["feasible_fraction"]),
        random_feasible_frac=float(pt["by_method"]["random"]["feasible_fraction"]),
        sca_wins=int(np.sum(deltas > 0)),
        campaign_file=str(campaign_path.name),
    )


def print_table(cells: list[GridCell]) -> None:
    print("\n" + "=" * 118)
    print("FULL 30-CELL GRID: SCA vs random @ 8.8 MHz (20 seeds, I=10)")
    print("=" * 118)
    print(
        f"{'cap%':>5s} {'MHz':>6s} {'J':>2s} "
        f"{'SCA':>8s} {'random':>8s} {'delta':>8s} {'std':>7s} {'d/std':>7s} "
        f"{'p':>10s} {'q_FDR':>10s} {'feasS':>6s} {'feasR':>6s} {'W':>5s}"
    )
    print("-" * 118)
    for c in sorted(cells, key=lambda x: (x.max_bw_share, x.j)):
        q = f"{c.fdr_q:.4f}" if c.fdr_q is not None else "     —"
        sig = "*" if c.fdr_q is not None and c.fdr_q < 0.05 else " "
        print(
            f"{c.max_bw_share:5.0%} {c.per_link_cap_mhz:6.3f} {c.j:2d} "
            f"{c.sca_mean_mbps:8.3f} {c.random_mean_mbps:8.3f} {c.delta_mbps:+8.3f} "
            f"{c.delta_std_mbps:7.3f} {c.effect_size:7.2f} "
            f"{c.wilcoxon_p:10.4f} {q:>10s} "
            f"{c.sca_feasible_frac:6.0%} {c.random_feasible_frac:6.0%} "
            f"{c.sca_wins:2d}/20{sig}"
        )
    print("  * = q_FDR < 0.05 (global BH across 30 cells)")


def verdict(cells: list[GridCell]) -> dict[str, Any]:
    by_cap: dict[float, list[GridCell]] = {}
    for c in cells:
        by_cap.setdefault(c.max_bw_share, []).append(c)

    cap_summary = []
    caps_all_j_sig_raw = []
    caps_all_j_sig_fdr = []

    for share in CAPS:
        row_cells = sorted(by_cap[share], key=lambda x: x.j)
        raw_sig_js = [c.j for c in row_cells if c.wilcoxon_p < 0.05 and c.delta_mbps > 0]
        fdr_sig_js = [
            c.j for c in row_cells if c.fdr_q is not None and c.fdr_q < 0.05 and c.delta_mbps > 0
        ]
        all_j_raw = len(raw_sig_js) == 5 and all(c.delta_mbps > 0 for c in row_cells)
        all_j_fdr = len(fdr_sig_js) == 5 and all(c.delta_mbps > 0 for c in row_cells)
        if all_j_raw:
            caps_all_j_sig_raw.append(share)
        if all_j_fdr:
            caps_all_j_sig_fdr.append(share)
        cap_summary.append(
            {
                "max_bw_share": share,
                "per_link_cap_mhz": row_cells[0].per_link_cap_mhz,
                "raw_significant_J": raw_sig_js,
                "fdr_significant_J": fdr_sig_js,
                "all_J_raw_p_lt_0.05_positive_delta": all_j_raw,
                "all_J_fdr_q_lt_0.05_positive_delta": all_j_fdr,
                "J5_delta_mbps": next(c.delta_mbps for c in row_cells if c.j == 5),
                "J5_p": next(c.wilcoxon_p for c in row_cells if c.j == 5),
                "J5_q_fdr": next(c.fdr_q for c in row_cells if c.j == 5),
            }
        )

    if caps_all_j_sig_fdr:
        one_line = (
            f"Cap(s) {', '.join(f'{s:.0%}' for s in caps_all_j_sig_fdr)} remain FDR-significant "
            f"with positive delta at all J=1-5 simultaneously."
        )
    elif caps_all_j_sig_raw:
        one_line = (
            f"Raw p<0.05 at all J only for cap(s) {', '.join(f'{s:.0%}' for s in caps_all_j_sig_raw)}, "
            f"but none survive global BH-FDR across 30 cells."
        )
    else:
        one_line = (
            "No single cap generalizes across UAV count: no max_bw_share achieves "
            "positive SCA advantage over random at all J=1-5 (raw or FDR); the effect is inherently J-dependent."
        )

    return {
        "fdr_scope": "BH-FDR applied once across all 30 Wilcoxon p-values",
        "alpha": 0.05,
        "cap_summaries": cap_summary,
        "caps_significant_all_J_raw": caps_all_j_sig_raw,
        "caps_significant_all_J_fdr": caps_all_j_sig_fdr,
        "one_line_verdict": one_line,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-run", action="store_true", help="Only analyze existing campaigns")
    args = ap.parse_args()

    for share in CAPS:
        path = CAMPAIGN_PATHS[share]
        if not path.exists():
            if args.skip_run:
                print(f"MISSING {path}", file=sys.stderr)
                return 1
            run_campaign(share, path)

    cells: list[GridCell] = []
    for share in CAPS:
        path = CAMPAIGN_PATHS[share]
        for j in J_VALUES:
            cells.append(extract_cell(path, share, j))

    pvals = [c.wilcoxon_p for c in cells]
    qs = bh_fdr(pvals)
    for c, q in zip(cells, qs, strict=True):
        c.fdr_q = q

    v = verdict(cells)
    payload = {
        "description": (
            "Exploratory grid: SCA vs random at 8.8 MHz, I=10, 20 seeds, "
            "full axis campaigns; uavs axis J=1-5 cells."
        ),
        "b_sys_hz": B_SYS,
        "max_bw_shares": list(CAPS),
        "j_values": list(J_VALUES),
        "n_cells": len(cells),
        "cells": [c.to_dict() for c in cells],
        "verdict": v,
        "campaign_files": {str(k): str(v) for k, v in CAMPAIGN_PATHS.items()},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")

    print_table(cells)
    print("\nCAP SUMMARY (significant J lists, raw vs FDR):")
    for s in v["cap_summaries"]:
        print(
            f"  cap={s['max_bw_share']:.0%} ({s['per_link_cap_mhz']:.3f} MHz/link)  "
            f"raw_sig_J={s['raw_significant_J']}  fdr_sig_J={s['fdr_significant_J']}  "
            f"all_J_raw={s['all_J_raw_p_lt_0.05_positive_delta']}  "
            f"all_J_fdr={s['all_J_fdr_q_lt_0.05_positive_delta']}  "
            f"J5: d={s['J5_delta_mbps']:+.3f} p={s['J5_p']:.4f} q={s['J5_q_fdr']:.4f}"
        )
    print(f"\nVERDICT: {v['one_line_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
