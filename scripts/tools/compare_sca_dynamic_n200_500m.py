"""Paired frozen SCA vs dynamic_assignment on the n200 500 m scenario bank."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.grids import config_for_counts
from uavdt.experiments.scenario_bank import eval_cfg_from_bank, load_bank, scenario_from_record
from uavdt.sca import SCASettings, solve_sca
from uavdt.sca.algorithm import true_gate_ok


def _log(msg: str) -> None:
    print(msg, flush=True)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _summarize(rows: list[dict]) -> dict:
    frozen = np.array([r["frozen_Mbps"] for r in rows], dtype=float)
    dynamic = np.array([r["dynamic_Mbps"] for r in rows], dtype=float)
    delta = dynamic - frozen
    feas_f = np.array([r["frozen_feasible"] for r in rows], dtype=bool)
    feas_d = np.array([r["dynamic_feasible"] for r in rows], dtype=bool)
    assoc_changed = np.array([r["assoc_changed"] for r in rows], dtype=bool)
    proc_changed = np.array([r["proc_changed"] for r in rows], dtype=bool)
    practical = np.abs(delta) > 0.05
    return {
        "n": len(rows),
        "frozen_mean_Mbps": float(np.mean(frozen)),
        "frozen_std_Mbps": float(np.std(frozen, ddof=1)) if len(rows) > 1 else 0.0,
        "dynamic_mean_Mbps": float(np.mean(dynamic)),
        "dynamic_std_Mbps": float(np.std(dynamic, ddof=1)) if len(rows) > 1 else 0.0,
        "delta_mean_Mbps": float(np.mean(delta)),
        "delta_std_Mbps": float(np.std(delta, ddof=1)) if len(rows) > 1 else 0.0,
        "delta_median_Mbps": float(np.median(delta)),
        "frozen_feasible_fraction": float(np.mean(feas_f)),
        "dynamic_feasible_fraction": float(np.mean(feas_d)),
        "feasibility_gain_count": int(np.sum(feas_d & ~feas_f)),
        "feasibility_loss_count": int(np.sum(feas_f & ~feas_d)),
        "dynamic_higher_count": int(np.sum(delta > 0)),
        "dynamic_lower_count": int(np.sum(delta < 0)),
        "practical_gain_count": int(np.sum(practical & (delta > 0))),
        "practical_loss_count": int(np.sum(practical & (delta < 0))),
        "assoc_changed_fraction": float(np.mean(assoc_changed)),
        "proc_changed_fraction": float(np.mean(proc_changed)),
        "assignment_accepted_mean": float(
            np.mean([r.get("assignment_accepted", 0) for r in rows])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT / "data" / "scenario_bank" / "n200_i10_j3_500m.json",
    )
    parser.add_argument("--n-scenarios", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "n200_500m_cap25" / "sca_dynamic_vs_frozen.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "n200_500m_cap25" / "sca_dynamic_vs_frozen.checkpoint.json",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    bank = load_bank(args.bank)
    overlay = config_for_counts(
        10,
        3,
        SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE),
    ).with_square_area_m(500.0)
    cfg = eval_cfg_from_bank(bank, overlay)
    records = list(bank["scenarios"])[: int(args.n_scenarios)]
    frozen_settings = SCASettings(solver=None, max_iterations=int(args.max_iterations))
    dynamic_settings = SCASettings(
        solver=None,
        max_iterations=int(args.max_iterations),
        dynamic_assignment=True,
    )

    done: dict[str, dict] = {}
    if args.checkpoint.exists() and not args.no_resume:
        payload = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            done[str(row["seed"])] = row
        _log(f"resume {len(done)}/{len(records)} seeds from {args.checkpoint.name}")

    rows: list[dict] = []
    t_all = time.perf_counter()
    for i, rec in enumerate(records, start=1):
        seed = int(rec["seed"])
        key = str(seed)
        if key in done:
            rows.append(done[key])
            continue
        sc = scenario_from_record(rec, cfg)
        _log(f"[{i}/{len(records)}] seed={seed}")
        t0 = time.perf_counter()
        fr = solve_sca(sc, seed, settings=frozen_settings)
        dr = solve_sca(sc, seed, settings=dynamic_settings)
        wall = time.perf_counter() - t0
        row = {
            "scenario_id": int(rec["id"]),
            "seed": seed,
            "frozen_Mbps": float(fr.true_eval.sum_rate_mbps),
            "dynamic_Mbps": float(dr.true_eval.sum_rate_mbps),
            "delta_Mbps": float(dr.true_eval.sum_rate_mbps - fr.true_eval.sum_rate_mbps),
            "frozen_feasible": bool(fr.true_eval.feasible),
            "dynamic_feasible": bool(dr.true_eval.feasible),
            "frozen_true_gate": bool(true_gate_ok(fr.true_eval)),
            "dynamic_true_gate": bool(true_gate_ok(dr.true_eval)),
            "assoc_changed": not np.array_equal(
                fr.allocation.hard_association(),
                dr.allocation.hard_association(),
            ),
            "proc_changed": not np.array_equal(
                fr.allocation.hard_processing(),
                dr.allocation.hard_processing(),
            ),
            "assignment_accepted": int(dr.diagnostics.get("assignment_accepted", 0)),
            "assignment_rejected": int(dr.diagnostics.get("assignment_rejected", 0)),
            "frozen_stop": fr.diagnostics.get("stop_reason"),
            "dynamic_stop": dr.diagnostics.get("stop_reason"),
            "wall_clock_s": wall,
        }
        rows.append(row)
        done[key] = row
        _atomic_write(
            args.checkpoint,
            {"rows": [done[str(r["seed"])] for r in records if str(r["seed"]) in done]},
        )
        _log(
            f"    frozen={row['frozen_Mbps']:.4f}  dynamic={row['dynamic_Mbps']:.4f}  "
            f"dM={row['delta_Mbps']:+.4f}  feas {int(row['frozen_feasible'])}/{int(row['dynamic_feasible'])}  "
            f"{wall:.1f}s"
        )

    rows = [done[str(int(r["seed"]))] for r in records if str(int(r["seed"])) in done]
    summary = _summarize(rows)
    out = {
        "bank_path": str(args.bank),
        "area_m": 500.0,
        "n_scenarios": len(rows),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "max_iterations": int(args.max_iterations),
        "summary": summary,
        "rows": rows,
        "wall_clock_s": time.perf_counter() - t_all,
    }
    _atomic_write(args.out, out)
    _log("")
    _log("=== n200 @ 500 m: frozen SCA vs dynamic_assignment ===")
    for k, v in summary.items():
        _log(f"  {k}: {v}")
    _log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
