"""Screen B_sys and per-link cap configs at the default J=3 point.

Runs all methods on seeds 1..n_runs for each (b_sys_hz, max_bw_share) pair,
scores method separation vs a reference config, and writes a ranked table.

Usage:
    python scripts/sweep_bandwidth_screen.py
    python scripts/sweep_bandwidth_screen.py --reference-b-hz 8800000 --reference-share 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class ConfigSpec:
    b_sys_hz: float
    max_bw_share: float | None

    @property
    def label(self) -> str:
        mhz = self.b_sys_hz / 1e6
        cap = "none" if self.max_bw_share is None else f"{self.max_bw_share:.0%}"
        return f"{mhz:g}MHz cap={cap}"


@dataclass
class ScreenResult:
    spec: ConfigSpec
    means: dict[str, float]
    per_seed: dict[str, list[float]]
    feasible_frac: dict[str, float]
    spread: float
    sca_lead: float
    sca_vs_random_delta: float
    sca_vs_random_p: float
    sca_wins_vs_random: int
    n_runs: int
    score: float

    def to_dict(self) -> dict:
        return {
            "b_sys_hz": self.spec.b_sys_hz,
            "max_bw_share": self.spec.max_bw_share,
            "label": self.spec.label,
            "means": self.means,
            "feasible_frac": self.feasible_frac,
            "spread_mbps": self.spread,
            "sca_lead_mbps": self.sca_lead,
            "sca_vs_random_delta_mbps": self.sca_vs_random_delta,
            "sca_vs_random_p": self.sca_vs_random_p,
            "sca_wins_vs_random": self.sca_wins_vs_random,
            "score": self.score,
        }


def run_config(
    spec: ConfigSpec,
    *,
    n_runs: int,
    seed_start: int,
    solver: str,
) -> ScreenResult:
    cfg = SimConfig(b_sys_hz=spec.b_sys_hz, max_bw_share=spec.max_bw_share)
    cfg = config_for_counts(10, 3, cfg)
    sca_settings = SCASettings(solver=solver, max_iterations=30, step_size_m=20.0)
    seeds = list(range(seed_start, seed_start + n_runs))
    per_seed: dict[str, list[float]] = {m: [] for m in METHODS}
    feas: dict[str, list[bool]] = {m: [] for m in METHODS}
    for seed in seeds:
        scenario = generate_scenario(seed, cfg)
        for method in METHODS:
            run = run_method(
                scenario,
                method,
                seed,
                sca_settings=sca_settings,
            )
            per_seed[method].append(run.sum_rate_mbps)
            feas[method].append(run.feasible)
    means = {m: float(np.mean(per_seed[m])) for m in METHODS}
    feasible_frac = {m: float(np.mean(feas[m])) for m in METHODS}
    rates = list(means.values())
    spread = max(rates) - min(rates)
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    sca_lead = means["sca"] - ranked[1][1] if ranked[0][0] == "sca" else means["sca"] - ranked[0][1]
    deltas = np.array(per_seed["sca"]) - np.array(per_seed["random"])
    sca_vs_random_delta = float(np.mean(deltas))
    sca_vs_random_p = float(wilcoxon_signed_rank(deltas)["p_two_sided"])
    sca_wins = int(np.sum(deltas > 0))
    min_feas = min(feasible_frac.values())
    # Higher spread + SCA lead + significant random beat; penalize infeasibility hard.
    sig_bonus = 0.5 if sca_vs_random_p < 0.05 else 0.0
    score = spread + sca_lead + sig_bonus
    if min_feas < 1.0:
        score -= 10.0 * (1.0 - min_feas)
    return ScreenResult(
        spec=spec,
        means=means,
        per_seed=per_seed,
        feasible_frac=feasible_frac,
        spread=spread,
        sca_lead=sca_lead,
        sca_vs_random_delta=sca_vs_random_delta,
        sca_vs_random_p=sca_vs_random_p,
        sca_wins_vs_random=sca_wins,
        n_runs=n_runs,
        score=score,
    )


def default_grid() -> list[ConfigSpec]:
    b_mhz = [2.4, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 10.0, 12.0, 14.0]
    specs: list[ConfigSpec] = []
    for mhz in b_mhz:
        specs.append(ConfigSpec(mhz * 1e6, 0.15))
    cap_shares = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
    for share in cap_shares:
        specs.append(ConfigSpec(8_800_000.0, share))
    for mhz in [5.0, 6.0, 7.0]:
        specs.append(ConfigSpec(mhz * 1e6, None))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--solver", default=None, help="SCA LP backend; None = auto CVXPY")
    ap.add_argument("--reference-b-hz", type=float, default=8_800_000.0)
    ap.add_argument("--reference-share", type=float, default=0.25)
    ap.add_argument("--out", default="results/bw_screen_default_j3.json")
    args = ap.parse_args()

    ref_spec = ConfigSpec(args.reference_b_hz, args.reference_share)
    grid = default_grid()
    # Ensure reference is in grid
    if ref_spec not in grid:
        grid.insert(0, ref_spec)

    results: list[ScreenResult] = []
    for i, spec in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {spec.label} ...", flush=True)
        results.append(
            run_config(
                spec,
                n_runs=args.n_runs,
                seed_start=args.seed_start,
                solver=args.solver,
            )
        )
        r = results[-1]
        print(
            f"  SCA={r.means['sca']:.3f} spread={r.spread:.3f} "
            f"d_random={r.sca_vs_random_delta:+.3f} p={r.sca_vs_random_p:.4f} "
            f"score={r.score:.3f}",
            flush=True,
        )

    ref = next(r for r in results if r.spec == ref_spec)
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": ref.to_dict(),
        "configs": [r.to_dict() for r in ranked],
        "n_runs": args.n_runs,
        "seed_start": args.seed_start,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"REFERENCE: {ref.spec.label}")
    print(
        f"  spread={ref.spread:.3f}  sca_lead={ref.sca_lead:.3f}  "
        f"vs_random={ref.sca_vs_random_delta:+.3f} p={ref.sca_vs_random_p:.4f}  "
        f"score={ref.score:.3f}"
    )
    print("\nTOP 5 BY SCORE (spread + sca_lead + sig bonus):")
    for r in ranked[:5]:
        beat_ref = r.score > ref.score
        tag = " *** BEATS REF ***" if beat_ref else ""
        print(
            f"  {r.spec.label:22s}  spread={r.spread:.3f}  "
            f"sca={r.means['sca']:.3f}  d_random={r.sca_vs_random_delta:+.3f}  "
            f"p={r.sca_vs_random_p:.4f}  score={r.score:.3f}{tag}"
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
