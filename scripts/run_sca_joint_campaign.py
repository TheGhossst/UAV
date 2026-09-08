"""Run the SCA-joint methodology probe in the pre-registered order.

Does not write or overwrite any campaign_*.json whose method set is the
headline frozen-SCA campaign. Outputs:

  results/tk08_scajoint_feasibility.json
  results/campaign_8.8mhz_cap25_si12k_scajoint.json
  results/sca_joint_default_runtime.json
  results/sca_joint_vs_frozen.json  (comparison table payload)

Usage:
  python scripts/run_sca_joint_campaign.py
  python scripts/run_sca_joint_campaign.py --only tk08
  python scripts/run_sca_joint_campaign.py --only campaign
  python scripts/run_sca_joint_campaign.py --only runtime
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.campaign import CampaignSettings, run_point, write_campaign  # noqa: E402
from uavdt.experiments.grids import config_for_counts, iter_axis  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.placement.kmeans import place_kmeans  # noqa: E402
from uavdt.resources import nearest_association, process_consistent_processing  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402
from uavdt.sca_joint.rematch import (  # noqa: E402
    association_process_cohesive,
    centroid_cohesive_association,
    n_forwarding,
)

RESULTS = ROOT / "results"
FROZEN_CAMPAIGN = RESULTS / "campaign_8.8mhz_cap25_si12k.json"
OUT_CAMPAIGN = RESULTS / "campaign_8.8mhz_cap25_si12k_scajoint.json"
OUT_TK08 = RESULTS / "tk08_scajoint_feasibility.json"
OUT_RUNTIME = RESULTS / "sca_joint_default_runtime.json"
OUT_COMPARE = RESULTS / "sca_joint_vs_frozen.json"
PRACTICAL_MBPS = 0.05
N_RUNS = 20
SEED_START = 1


def _cfg() -> SimConfig:
    return SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)


def _settings() -> CampaignSettings:
    return CampaignSettings(
        n_runs=N_RUNS,
        seed_start=SEED_START,
        methods=("sca_joint",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
    )


def _log(msg: str) -> None:
    print(msg, flush=True)


def _seed_list() -> tuple[int, ...]:
    return tuple(range(SEED_START, SEED_START + N_RUNS))


def _tk08_cfg() -> SimConfig:
    return replace(config_for_counts(10, 3, _cfg()), aodt_threshold_s=0.8)


def _default_cfg() -> SimConfig:
    return config_for_counts(10, 3, _cfg())


def _cohesion_snapshot(scenario, uav, alloc) -> dict:
    a = alloc.hard_association()
    b = alloc.hard_processing()
    a_hand = centroid_cohesive_association(scenario, uav)
    a_near = nearest_association(scenario.iot_xyz_m, uav)
    return {
        "process_cohesive_a": association_process_cohesive(scenario, a),
        "n_forwarding": n_forwarding(a, b),
        "matches_centroid_construction": bool(np_equal_one_hot(a, a_hand)),
        "matches_nearest": bool(np_equal_one_hot(a, a_near)),
        "centroid_construction_feasible_at_kmeans_q": None,
    }


def np_equal_one_hot(left, right) -> bool:
    import numpy as np

    return bool(np.array_equal(left > 0.5, right > 0.5))


def _hand_feasible_at_kmeans(scenario, seed: int) -> bool:
    from uavdt.evaluator import evaluate
    from uavdt.models import Allocation
    from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q

    uav = place_kmeans(scenario, seed)
    a = centroid_cohesive_association(scenario, uav)
    b = process_consistent_processing(scenario, a)
    res = solve_bandwidth_at_fixed_q(scenario, uav, a, b, SCASettings(solver=None))
    if res.infeasible:
        return False
    ev = evaluate(scenario, uav, Allocation(a, b, res.bandwidth_hz))
    return bool(ev.feasible)


def run_tk08() -> dict:
    cfg = _tk08_cfg()
    settings = SCASettings(solver=None, max_iterations=30)
    seeds = _seed_list()
    by_method: dict[str, list[dict]] = {"sca": [], "sca_joint": []}
    for method in ("sca", "sca_joint"):
        for seed in seeds:
            sc = generate_scenario(seed, cfg)
            _log(f"  tk08  method={method}  seed={seed}")
            t0 = perf_counter()
            run = run_method(sc, method, seed, sca_settings=settings)
            elapsed = perf_counter() - t0
            snap = _cohesion_snapshot(sc, run.uav_xyz_m, run.allocation)
            snap["centroid_construction_feasible_at_kmeans_q"] = _hand_feasible_at_kmeans(
                sc, seed
            )
            by_method[method].append(
                {
                    "seed": seed,
                    "feasible": run.feasible,
                    "sum_rate_Mbps": run.sum_rate_mbps,
                    "wall_clock_s": float(
                        run.diagnostics.get("wall_clock_s", elapsed)
                    ),
                    "n_iterations": run.diagnostics.get("n_iterations"),
                    "stop_reason": run.diagnostics.get("stop_reason"),
                    "rematch_accepted": run.diagnostics.get("rematch_accepted"),
                    "association_init_equals_final": run.diagnostics.get(
                        "association_init_equals_final"
                    ),
                    **snap,
                }
            )

    def _summ(rows: list[dict]) -> dict:
        n = len(rows)
        n_feas = sum(1 for r in rows if r["feasible"])
        return {
            "n": n,
            "feasible_fraction": n_feas / n if n else 0.0,
            "n_feasible": n_feas,
            "mean_sum_rate_Mbps": sum(r["sum_rate_Mbps"] for r in rows) / n,
            "mean_wall_clock_s": sum(r["wall_clock_s"] for r in rows) / n,
            "n_process_cohesive_a": sum(1 for r in rows if r["process_cohesive_a"]),
            "n_matches_centroid_construction": sum(
                1 for r in rows if r["matches_centroid_construction"]
            ),
            "n_hand_construction_feasible_at_kmeans_q": sum(
                1 for r in rows if r["centroid_construction_feasible_at_kmeans_q"]
            ),
            "seeds": rows,
        }

    payload = {
        "question": (
            "Does SCA-joint reach feasibility at T_k=0.8s where frozen SCA is 0%?"
        ),
        "pre_registered": True,
        "config": {
            "num_iot": 10,
            "num_uav": 3,
            "aodt_threshold_s": 0.8,
            "b_sys_hz": 8.8e6,
            "max_bw_share": PRIMARY_MAX_BW_SHARE,
            "n_runs": N_RUNS,
            "seed_start": SEED_START,
        },
        "note": (
            "SCA-joint is a methodology probe. Frozen SCA remains the headline "
            "method. Hand-construction feasibility is recorded per seed at the "
            "same k-means q; the solver is not iterated to match it."
        ),
        "by_method": {m: _summ(rows) for m, rows in by_method.items()},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_TK08.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"wrote {OUT_TK08}")
    for m, s in payload["by_method"].items():
        _log(
            f"  {m}: feasible={s['feasible_fraction']:.0%}  "
            f"cohesive_a={s['n_process_cohesive_a']}/20  "
            f"matches_hand={s['n_matches_centroid_construction']}/20"
        )
    return payload


def run_campaign_joint() -> dict:
    if OUT_CAMPAIGN.exists() and FROZEN_CAMPAIGN.exists():
        # Never replace the headline frozen-SCA file.
        assert OUT_CAMPAIGN.resolve() != FROZEN_CAMPAIGN.resolve()
    cfg = _cfg()
    settings = _settings()
    points = []
    checkpoint = RESULTS / "campaign_8.8mhz_cap25_si12k_scajoint.partial.json"
    if checkpoint.exists():
        partial = json.loads(checkpoint.read_text(encoding="utf-8"))
        points = list(partial.get("points", []))
        done = {(p["axis"], float(p["x"])) for p in points}
        _log(f"resuming campaign; {len(done)} points already on disk")
    else:
        done = set()
    for axis in ("uavs", "iots", "lambda", "aodt", "cpu"):
        for point in iter_axis(axis, cfg):
            key = (point.axis, float(point.x_value))
            if key in done:
                _log(f"  skip {point.label} (checkpoint)")
                continue
            _log(f"=== {point.label}  sca_joint ===")
            row = run_point(point, settings)
            points.append(row)
            payload = _campaign_payload(points, cfg)
            checkpoint.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload = _campaign_payload(points, cfg)
    write_campaign(payload, OUT_CAMPAIGN)
    if checkpoint.exists():
        checkpoint.unlink()
    _log(f"wrote {OUT_CAMPAIGN}")
    return payload


def _campaign_payload(points: list[dict], cfg: SimConfig) -> dict:
    return {
        "paper": "Khalaf et al. IEEE TNSM 2026 §VII Figs. 6–10 axes",
        "sca_frozen": True,
        "sca_joint_probe": True,
        "headline_method": "sca",
        "n_runs": N_RUNS,
        "seed_start": SEED_START,
        "methods": ["sca_joint"],
        "note": (
            "Methodology probe: SCA-joint only. Frozen SCA / random / k-means / "
            "PSO numbers stay in campaign_8.8mhz_cap25_si12k.json and are not "
            "overwritten. Do not treat sca_joint vs baselines as a replacement "
            "headline comparison."
        ),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "area_m": [cfg.area_x_m, cfg.area_y_m],
        "points": points,
    }


def run_default_runtime() -> dict:
    cfg = _default_cfg()
    settings = SCASettings(solver=None, max_iterations=30)
    seeds = _seed_list()
    by_method: dict[str, list[dict]] = {"sca": [], "sca_joint": []}
    for method in ("sca", "sca_joint"):
        for seed in seeds:
            sc = generate_scenario(seed, cfg)
            _log(f"  default  method={method}  seed={seed}")
            run = run_method(sc, method, seed, sca_settings=settings)
            by_method[method].append(
                {
                    "seed": seed,
                    "feasible": run.feasible,
                    "sum_rate_Mbps": run.sum_rate_mbps,
                    "wall_clock_s": float(run.diagnostics.get("wall_clock_s", 0.0)),
                    "n_iterations": int(run.diagnostics.get("n_iterations") or 0),
                    "accepted_steps": run.diagnostics.get("accepted_steps"),
                    "rematch_accepted": run.diagnostics.get("rematch_accepted"),
                    "stop_reason": run.diagnostics.get("stop_reason"),
                }
            )

    def _summ(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "mean_wall_clock_s": sum(r["wall_clock_s"] for r in rows) / n,
            "mean_n_iterations": sum(r["n_iterations"] for r in rows) / n,
            "mean_sum_rate_Mbps": sum(r["sum_rate_Mbps"] for r in rows) / n,
            "feasible_fraction": sum(1 for r in rows if r["feasible"]) / n,
            "seeds": rows,
        }

    payload = {
        "point": "default J=3 I=10 lambda=2 Tk=2.8s CPU=2e8 8.8MHz/25% 100x100m",
        "by_method": {m: _summ(rows) for m, rows in by_method.items()},
    }
    sca_t = payload["by_method"]["sca"]["mean_wall_clock_s"]
    joint_t = payload["by_method"]["sca_joint"]["mean_wall_clock_s"]
    payload["wall_clock_ratio_joint_over_frozen"] = (
        joint_t / sca_t if sca_t > 0 else None
    )
    payload["iteration_ratio_joint_over_frozen"] = (
        payload["by_method"]["sca_joint"]["mean_n_iterations"]
        / payload["by_method"]["sca"]["mean_n_iterations"]
        if payload["by_method"]["sca"]["mean_n_iterations"]
        else None
    )
    OUT_RUNTIME.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"wrote {OUT_RUNTIME}")
    _log(
        f"  frozen SCA {sca_t:.2f}s/seed  sca_joint {joint_t:.2f}s/seed  "
        f"ratio={payload['wall_clock_ratio_joint_over_frozen']}"
    )
    return payload


def _write_comparison(campaign: dict | None = None) -> dict:
    if campaign is None:
        campaign = json.loads(OUT_CAMPAIGN.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_CAMPAIGN.read_text(encoding="utf-8"))
    frozen_by = {(p["axis"], float(p["x"])): p for p in frozen["points"]}
    rows = []
    n_practical = 0
    n_feas_diff = 0
    n_joint_worse = 0
    for pt in campaign["points"]:
        key = (pt["axis"], float(pt["x"]))
        fr = frozen_by[key]["by_method"]["sca"]
        jn = pt["by_method"]["sca_joint"]
        delta = jn["mean_sum_rate_Mbps"] - fr["mean_sum_rate_Mbps"]
        feas_delta = jn["feasible_fraction"] - fr["feasible_fraction"]
        practical = abs(delta) > PRACTICAL_MBPS or abs(feas_delta) > 1e-12
        if abs(delta) > PRACTICAL_MBPS:
            n_practical += 1
        if abs(feas_delta) > 1e-12:
            n_feas_diff += 1
        if delta < -PRACTICAL_MBPS:
            n_joint_worse += 1
        if practical:
            if abs(feas_delta) > 1e-12:
                verdict = "feasibility differs"
            elif delta > PRACTICAL_MBPS:
                verdict = "practical (joint higher)"
            else:
                verdict = "practical (joint lower)"
        else:
            verdict = "negligible"
        rows.append(
            {
                "axis": pt["axis"],
                "x_name": pt["x_name"],
                "x": pt["x"],
                "label": pt["label"],
                "frozen_Mbps": fr["mean_sum_rate_Mbps"],
                "frozen_feasible": fr["feasible_fraction"],
                "joint_Mbps": jn["mean_sum_rate_Mbps"],
                "joint_feasible": jn["feasible_fraction"],
                "delta_Mbps": delta,
                "delta_feasible": feas_delta,
                "verdict": verdict,
            }
        )
    payload = {
        "practical_effect_bar_Mbps": PRACTICAL_MBPS,
        "frozen_source": str(FROZEN_CAMPAIGN.as_posix()),
        "joint_source": str(OUT_CAMPAIGN.as_posix()),
        "n_points": len(rows),
        "n_practical_rate": n_practical,
        "n_feasibility_diff": n_feas_diff,
        "n_joint_worse_than_bar": n_joint_worse,
        "pre_registered_joint_geq_frozen": n_joint_worse == 0,
        "rows": rows,
    }
    OUT_COMPARE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"wrote {OUT_COMPARE}")
    _log(
        f"{'point':40s} {'frozen':>8s} {'feas':>5s} {'joint':>8s} {'feas':>5s} "
        f"{'dMbps':>8s}  verdict"
    )
    for r in rows:
        _log(
            f"{r['label'][:40]:40s} {r['frozen_Mbps']:8.3f} {r['frozen_feasible']:5.0%} "
            f"{r['joint_Mbps']:8.3f} {r['joint_feasible']:5.0%} "
            f"{r['delta_Mbps']:+8.3f}  {r['verdict']}"
        )
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        choices=("tk08", "campaign", "runtime", "compare", "all"),
        default="all",
    )
    args = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if FROZEN_CAMPAIGN.exists():
        # Guardrail: never write the headline frozen-SCA campaign.
        assert str(OUT_CAMPAIGN.resolve()) != str(FROZEN_CAMPAIGN.resolve())
    if args.only in {"tk08", "all"}:
        run_tk08()
    if args.only in {"campaign", "all"}:
        camp = run_campaign_joint()
        if FROZEN_CAMPAIGN.exists():
            _write_comparison(camp)
    if args.only in {"runtime", "all"}:
        run_default_runtime()
    if args.only == "compare":
        _write_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
