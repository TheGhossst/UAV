"""T_k=0.8 s follow-ups: cohesive rematch candidate + multi-seed construction.

Does not overwrite the recorded best-SE SCA-joint files or the headline
frozen-SCA campaign.

Outputs:
  results/tk08_scajoint_cohesive.json
  results/tk08_cohesive_construction.json
  results/tk08_sync_tradeoff_gap.json

Usage:
  python scripts/run_tk08_followup.py
  python scripts/run_tk08_followup.py --only cohesive
  python scripts/run_tk08_followup.py --only construction
  python scripts/run_tk08_followup.py --only tradeoff
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.evaluator import evaluate  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.models import Allocation  # noqa: E402
from uavdt.placement.kmeans import place_kmeans  # noqa: E402
from uavdt.resources import (  # noqa: E402
    cpu_stable_processing,
    nearest_association,
    process_consistent_processing,
)
from uavdt.sca.cvx_problem import aodt_upload_slacks_s, solve_bandwidth_at_fixed_q  # noqa: E402
from uavdt.sca.linearize import spectral_efficiency  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402
from uavdt.sca_joint.rematch import (  # noqa: E402
    association_process_cohesive,
    best_se_association,
    centroid_cohesive_association,
    n_forwarding,
)

RESULTS = ROOT / "results"
OUT_COHESIVE = RESULTS / "tk08_scajoint_cohesive.json"
OUT_CONSTRUCTION = RESULTS / "tk08_cohesive_construction.json"
OUT_TRADEOFF = RESULTS / "tk08_sync_tradeoff_gap.json"
PROTECTED = (
    RESULTS / "tk08_scajoint_feasibility.json",
    RESULTS / "campaign_8.8mhz_cap25_si12k_scajoint.json",
    RESULTS / "campaign_8.8mhz_cap25_si12k.json",
    RESULTS / "sca_joint_vs_frozen.json",
    RESULTS / "sca_joint_default_runtime.json",
)
N_RUNS = 20
SEED_START = 1
HELD_OUT_START = 21
HELD_OUT_N = 20


def _log(msg: str) -> None:
    print(msg, flush=True)


def _cfg() -> SimConfig:
    return replace(
        config_for_counts(10, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)),
        aodt_threshold_s=0.8,
    )


def _assert_protected() -> None:
    for path in (OUT_COHESIVE, OUT_CONSTRUCTION):
        for frozen in PROTECTED:
            if path.resolve() == frozen.resolve():
                raise SystemExit(f"refusing to overwrite protected file {frozen}")


def _py(value):
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, np.ndarray):
        return _py(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if np.isnan(x):
            return None
        return x
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _process_map(scenario, association: np.ndarray) -> list[dict]:
    ja = np.argmax(association, axis=1)
    rows = []
    for proc in scenario.processes:
        members = proc.iot_indices
        ids = [int(ja[int(i)]) for i in members]
        unique = sorted(set(ids))
        rows.append(
            {
                "process_id": int(proc.process_id),
                "n_members": int(members.size),
                "uav_ids": unique,
                "split": len(unique) > 1,
            }
        )
    return rows


def run_cohesive_solver() -> dict:
    cfg = _cfg()
    settings = SCASettings(
        solver=None,
        max_iterations=30,
        process_cohesive_candidate=True,
    )
    seeds = tuple(range(SEED_START, SEED_START + N_RUNS))
    rows = []
    for seed in seeds:
        sc = generate_scenario(seed, cfg)
        _log(f"  cohesive-solver  seed={seed}")
        t0 = perf_counter()
        run = run_method(sc, "sca_joint", seed, sca_settings=settings)
        elapsed = perf_counter() - t0
        a = run.allocation.hard_association()
        b = run.allocation.hard_processing()
        rows.append(
            {
                "seed": seed,
                "feasible": run.feasible,
                "sum_rate_Mbps": run.sum_rate_mbps,
                "wall_clock_s": float(run.diagnostics.get("wall_clock_s", elapsed)),
                "n_iterations": run.diagnostics.get("n_iterations"),
                "stop_reason": run.diagnostics.get("stop_reason"),
                "rematch_accepted": run.diagnostics.get("rematch_accepted"),
                "rematch_rejected": run.diagnostics.get("rematch_rejected"),
                "rematch_kinds": run.diagnostics.get("rematch_kinds") or [],
                "process_cohesive_candidate": run.diagnostics.get(
                    "process_cohesive_candidate"
                ),
                "process_cohesive_a": bool(
                    run.diagnostics.get("process_cohesive_a")
                    if run.diagnostics.get("process_cohesive_a") is not None
                    else association_process_cohesive(sc, a)
                ),
                "n_forwarding": int(
                    run.diagnostics.get("n_forwarding")
                    if run.diagnostics.get("n_forwarding") is not None
                    else n_forwarding(a, b)
                ),
                "association_init_equals_final": run.diagnostics.get(
                    "association_init_equals_final"
                ),
                "process_uav_map": _process_map(sc, a),
            }
        )

    n = len(rows)
    n_feas = sum(1 for r in rows if r["feasible"])
    n_cohesive = sum(1 for r in rows if r["process_cohesive_a"])
    n_kind = sum(
        1
        for r in rows
        if any(str(k).endswith("process_cohesive") for k in r["rematch_kinds"])
    )
    payload = {
        "question": (
            "If SCA-joint also tries process-cohesive a_ij as a rematch "
            "candidate, does T_k=0.8s become feasible?"
        ),
        "pre_registered": False,
        "hypothesis": (
            "Best-SE greedy never proposes grouping N_k. Adding that one "
            "candidate should recover feasibility if nothing else blocks "
            "the LP / evaluate gate."
        ),
        "config": {
            "num_iot": 10,
            "num_uav": 3,
            "aodt_threshold_s": 0.8,
            "b_sys_hz": 8.8e6,
            "max_bw_share": PRIMARY_MAX_BW_SHARE,
            "n_runs": N_RUNS,
            "seed_start": SEED_START,
            "process_cohesive_candidate": True,
            "method": "sca_joint",
        },
        "note": (
            "Does not overwrite tk08_scajoint_feasibility.json (best-SE-only). "
            "Frozen SCA is unchanged. Accept rule is unchanged: newly "
            "feasible or true rate improves; newly-feasible ranks first."
        ),
        "summary": {
            "n": n,
            "n_feasible": n_feas,
            "feasible_fraction": n_feas / n if n else 0.0,
            "mean_sum_rate_Mbps": sum(r["sum_rate_Mbps"] for r in rows) / n,
            "mean_wall_clock_s": sum(r["wall_clock_s"] for r in rows) / n,
            "mean_n_iterations": sum(int(r["n_iterations"] or 0) for r in rows) / n,
            "n_process_cohesive_a": n_cohesive,
            "n_accepted_process_cohesive_kind": n_kind,
            "n_zero_forwarding": sum(1 for r in rows if r["n_forwarding"] == 0),
        },
        "seeds": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_COHESIVE.write_text(json.dumps(_py(payload), indent=2), encoding="utf-8")
    _log(f"wrote {OUT_COHESIVE}")
    s = payload["summary"]
    _log(
        f"  sca_joint+cohesive: feasible={s['n_feasible']}/{s['n']}  "
        f"cohesive_a={s['n_process_cohesive_a']}/{s['n']}  "
        f"mean_Mbps={s['mean_sum_rate_Mbps']:.3f}"
    )
    return payload


def _evaluate_association(scenario, uav, association: np.ndarray, label: str) -> dict:
    b_majority = process_consistent_processing(scenario, association)
    b_stable = cpu_stable_processing(scenario, association)
    b_differs = bool(not np.array_equal(b_majority > 0.5, b_stable > 0.5))
    slacks = aodt_upload_slacks_s(scenario, association, b_majority)
    finite = slacks[np.isfinite(slacks)]
    res = solve_bandwidth_at_fixed_q(
        scenario, uav, association, b_majority, SCASettings(solver=None)
    )
    ev = None
    feasible = False
    if not res.infeasible:
        ev = evaluate(
            scenario, uav, Allocation(association, b_majority, res.bandwidth_hz)
        )
        feasible = bool(ev.feasible)
    return {
        "label": label,
        "lp_infeasible": bool(res.infeasible),
        "feasible": feasible,
        "sum_rate_Mbps": None if ev is None else float(ev.sum_rate_mbps),
        "max_aodt_s": None if ev is None else float(np.max(ev.aodt_s)),
        "n_forwarding": n_forwarding(association, b_majority),
        "process_cohesive_a": association_process_cohesive(scenario, association),
        "processing_matches_cpu_stable": not b_differs,
        "min_upload_slack_s": None if finite.size == 0 else float(np.min(finite)),
        "n_negative_slack": int(np.sum(slacks < 0.0)),
        "upload_slacks_s": [None if not np.isfinite(x) else float(x) for x in slacks],
        "process_uav_map": _process_map(scenario, association),
        "lp_status": res.status,
    }


def run_construction() -> dict:
    cfg = _cfg()
    campaign_seeds = tuple(range(SEED_START, SEED_START + N_RUNS))
    held_out_seeds = tuple(range(HELD_OUT_START, HELD_OUT_START + HELD_OUT_N))
    rows = []
    for panel, seeds in (("campaign", campaign_seeds), ("held_out", held_out_seeds)):
        for seed in seeds:
            sc = generate_scenario(seed, cfg)
            _log(f"  construction  panel={panel}  seed={seed}")
            uav = place_kmeans(sc, seed)
            a_cohesive = centroid_cohesive_association(sc, uav)
            a_near = nearest_association(sc.iot_xyz_m, uav)
            a_se = best_se_association(sc, uav)
            cohesive = _evaluate_association(sc, uav, a_cohesive, "centroid_cohesive")
            nearest = _evaluate_association(sc, uav, a_near, "nearest")
            rows.append(
                {
                    "seed": seed,
                    "panel": panel,
                    "nearest_equals_best_se": bool(
                        np.array_equal(a_near > 0.5, a_se > 0.5)
                    ),
                    "cohesive_differs_from_nearest": bool(
                        not np.array_equal(a_cohesive > 0.5, a_near > 0.5)
                    ),
                    "uav_xy_m": uav[:, :2].tolist(),
                    "cohesive": cohesive,
                    "nearest": nearest,
                }
            )

    def _panel(name: str) -> dict:
        subset = [r for r in rows if r["panel"] == name]
        n = len(subset)
        n_feas = sum(1 for r in subset if r["cohesive"]["feasible"])
        n_near = sum(1 for r in subset if r["nearest"]["feasible"])
        n_diff = sum(1 for r in subset if r["cohesive_differs_from_nearest"])
        failed = [int(r["seed"]) for r in subset if not r["cohesive"]["feasible"]]
        return {
            "n": n,
            "n_cohesive_feasible": n_feas,
            "cohesive_feasible_fraction": n_feas / n if n else 0.0,
            "n_nearest_feasible": n_near,
            "n_cohesive_differs_from_nearest": n_diff,
            "failed_seeds": failed,
        }

    campaign = _panel("campaign")
    held = _panel("held_out")
    all_n = len(rows)
    all_feas = sum(1 for r in rows if r["cohesive"]["feasible"])
    payload = {
        "question": (
            "Is process-cohesive feasibility at T_k=0.8s a general property "
            "of this scenario, or a single-geometry accident?"
        ),
        "config": {
            "num_iot": 10,
            "num_uav": 3,
            "aodt_threshold_s": 0.8,
            "b_sys_hz": 8.8e6,
            "max_bw_share": PRIMARY_MAX_BW_SHARE,
            "placement": "kmeans",
            "association": "centroid_cohesive",
            "processing": "process_consistent_processing",
            "campaign_seeds": list(campaign_seeds),
            "held_out_seeds": list(held_out_seeds),
        },
        "note": (
            "Construction only: k-means q, N_k on the UAV nearest that "
            "process centroid, majority-of-association b_ij, frozen-q "
            "bandwidth LP, true evaluate(). Not a solver."
        ),
        "summary": {
            "n": all_n,
            "n_cohesive_feasible": all_feas,
            "cohesive_feasible_fraction": all_feas / all_n if all_n else 0.0,
            "failed_seeds": [
                int(r["seed"]) for r in rows if not r["cohesive"]["feasible"]
            ],
            "campaign": campaign,
            "held_out": held,
        },
        "seeds": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_CONSTRUCTION.write_text(json.dumps(_py(payload), indent=2), encoding="utf-8")
    _log(f"wrote {OUT_CONSTRUCTION}")
    _log(
        f"  construction campaign {campaign['n_cohesive_feasible']}/{campaign['n']}  "
        f"held-out {held['n_cohesive_feasible']}/{held['n']}  "
        f"failed={payload['summary']['failed_seeds']}"
    )
    return payload


def run_tradeoff_gap() -> dict:
    """Best-SE @ T_k=1.2 minus cohesive @ T_k=0.8 at fixed k-means q (40 seeds)."""
    base = config_for_counts(
        10, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)
    )
    cfg08 = replace(base, aodt_threshold_s=0.8)
    cfg12 = replace(base, aodt_threshold_s=1.2)
    campaign_seeds = tuple(range(SEED_START, SEED_START + N_RUNS))
    held_out_seeds = tuple(range(HELD_OUT_START, HELD_OUT_START + HELD_OUT_N))
    rows = []
    for panel, seeds in (("campaign", campaign_seeds), ("held_out", held_out_seeds)):
        for seed in seeds:
            sc08 = generate_scenario(seed, cfg08)
            sc12 = generate_scenario(seed, cfg12)
            _log(f"  tradeoff  panel={panel}  seed={seed}")
            uav = place_kmeans(sc08, seed)
            a_near = nearest_association(sc08.iot_xyz_m, uav)
            a_coh = centroid_cohesive_association(sc08, uav)
            se = spectral_efficiency(sc08.iot_xyz_m, uav, sc08.cfg)
            se_near = float(se[np.arange(10), np.argmax(a_near, 1)].sum())
            se_coh = float(se[np.arange(10), np.argmax(a_coh, 1)].sum())
            n_off = int(np.sum(np.argmax(a_coh, 1) != np.argmax(se, 1)))
            b_near = process_consistent_processing(sc12, a_near)
            b_coh = process_consistent_processing(sc08, a_coh)
            res_near = solve_bandwidth_at_fixed_q(
                sc12, uav, a_near, b_near, SCASettings(solver=None)
            )
            res_coh = solve_bandwidth_at_fixed_q(
                sc08, uav, a_coh, b_coh, SCASettings(solver=None)
            )
            ev_near = evaluate(
                sc12, uav, Allocation(a_near, b_near, res_near.bandwidth_hz)
            )
            ev_coh = evaluate(
                sc08, uav, Allocation(a_coh, b_coh, res_coh.bandwidth_hz)
            )
            rows.append(
                {
                    "seed": seed,
                    "panel": panel,
                    "near_mbps": float(ev_near.sum_rate_mbps),
                    "coh_mbps": float(ev_coh.sum_rate_mbps),
                    "gap_mbps": float(ev_near.sum_rate_mbps - ev_coh.sum_rate_mbps),
                    "se_delta": se_coh - se_near,
                    "n_off_best_se": n_off,
                    "coh_feasible": bool(ev_coh.feasible),
                    "near_feasible": bool(ev_near.feasible),
                }
            )

    gaps = np.array([r["gap_mbps"] for r in rows], dtype=float)
    n_off = np.array([r["n_off_best_se"] for r in rows], dtype=float)
    se_delta = np.array([r["se_delta"] for r in rows], dtype=float)

    def _panel_stats(name: str) -> dict:
        subset = [r["gap_mbps"] for r in rows if r["panel"] == name]
        arr = np.array(subset, dtype=float)
        return {"n": len(arr), "mean": float(arr.mean()) if arr.size else None}

    payload = {
        "question": (
            "At fixed k-means q, what Mbps do we trade for T_k=0.8 s feasibility "
            "via process-cohesive grouping vs per-IoT best-SE?"
        ),
        "definition": (
            "gap = best-SE feasible @ T_k=1.2 s minus process-cohesive feasible "
            "@ T_k=0.8 s, same IoT placement and k-means q"
        ),
        "config": {
            "num_iot": 10,
            "num_uav": 3,
            "b_sys_hz": 8.8e6,
            "max_bw_share": PRIMARY_MAX_BW_SHARE,
            "campaign_seeds": list(campaign_seeds),
            "held_out_seeds": list(held_out_seeds),
        },
        "summary": {
            "n": len(rows),
            "gap_mbps": {
                "mean": float(gaps.mean()),
                "std": float(gaps.std(ddof=1)),
                "min": float(gaps.min()),
                "max": float(gaps.max()),
                "p10": float(np.percentile(gaps, 10)),
                "p50": float(np.percentile(gaps, 50)),
                "p90": float(np.percentile(gaps, 90)),
                "campaign_mean": _panel_stats("campaign")["mean"],
                "held_out_mean": _panel_stats("held_out")["mean"],
                "in_0p5_0p7_fraction": float(
                    sum(0.5 <= r["gap_mbps"] <= 0.7 for r in rows) / len(rows)
                ),
            },
            "spread_mechanism": {
                "pearson_r_gap_vs_n_off_best_se": float(np.corrcoef(gaps, n_off)[0, 1]),
                "pearson_r_gap_vs_neg_se_delta": float(
                    np.corrcoef(gaps, -se_delta)[0, 1]
                ),
                "note": (
                    "Gap spread tracks total SE sacrificed (r≈0.92), not count of "
                    "IoTs off best-SE link (r≈0.14). Example: seed 23 has 8/10 off "
                    "best but gap=0.47 (small per-IoT penalties); seed 16 has 6/10 "
                    "off best but gap=0.98 (large penalties to worse links)."
                ),
            },
            "all_cohesive_feasible": all(r["coh_feasible"] for r in rows),
            "all_near_feasible": all(r["near_feasible"] for r in rows),
        },
        "seeds": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_TRADEOFF.write_text(json.dumps(_py(payload), indent=2), encoding="utf-8")
    _log(f"wrote {OUT_TRADEOFF}")
    s = payload["summary"]["gap_mbps"]
    _log(
        f"  tradeoff gap mean={s['mean']:.3f} median={s['p50']:.3f} "
        f"range=[{s['min']:.3f},{s['max']:.3f}]"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "cohesive", "construction", "tradeoff"),
        default="all",
    )
    args = parser.parse_args(argv)
    _assert_protected()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.only in {"all", "cohesive"}:
        run_cohesive_solver()
    if args.only in {"all", "construction"}:
        run_construction()
    if args.only in {"all", "tradeoff"}:
        run_tradeoff_gap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
