"""Monte Carlo: train/eval every method on a frozen scenario bank."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from uavdt.config import SimConfig
from uavdt.experiments.methods import MethodRun, run_method
from uavdt.experiments.scenario_bank import (
    cfg_to_dict,
    eval_cfg_from_bank,
    scenario_from_record,
    uav_from_record,
)
from uavdt.placement.pso import PSOSettings
from uavdt.sca.settings import SCASettings
from uavdt.sca_anchor import AnchorSettings
from uavdt.sca_multistart import MultiStartSettings
from uavdt.td3.settings import TD3Settings


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _run_key(seed: int, method: str) -> str:
    return f"{int(seed)}:{method}"


def _atomic_write_json(path: Path, payload: dict, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    tmp.replace(path)


def summarize_runs(runs: list[MethodRun]) -> dict:
    rates = np.array([r.sum_rate_bit_per_s for r in runs], dtype=float)
    feas = np.array([r.feasible for r in runs], dtype=bool)
    aodt = np.vstack([r.true_eval.aodt_s for r in runs])
    std = float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0
    return {
        "n": len(runs),
        "mean_sum_rate_bit_per_s": float(np.mean(rates)),
        "std_sum_rate_bit_per_s": std,
        "mean_sum_rate_Mbps": float(np.mean(rates) / 1e6),
        "std_sum_rate_Mbps": std / 1e6,
        "feasible_fraction": float(np.mean(feas)),
        "mean_max_AoDT_s": float(np.mean(np.max(aodt, axis=1))),
        "std_max_AoDT_s": (
            float(np.std(np.max(aodt, axis=1), ddof=1)) if rates.size > 1 else 0.0
        ),
        "seeds": [r.seed for r in runs],
        "per_seed_Mbps": [r.sum_rate_mbps for r in runs],
        "per_seed_feasible": [r.feasible for r in runs],
        "diagnostics": [r.diagnostics for r in runs],
    }


def run_to_record(run: MethodRun, *, scenario_id: int) -> dict:
    ev = run.true_eval
    c = ev.constraints
    return {
        "scenario_id": int(scenario_id),
        "seed": int(run.seed),
        "method": run.method,
        "sum_rate_bit_per_s": float(run.sum_rate_bit_per_s),
        "sum_rate_Mbps": float(run.sum_rate_mbps),
        "feasible": bool(run.feasible),
        "aodt_s": [float(x) for x in ev.aodt_s],
        "max_AoDT_s": float(np.max(ev.aodt_s)),
        "rho": [float(x) for x in ev.rho],
        "uav_xyz_m": np.asarray(run.uav_xyz_m, dtype=float).tolist(),
        "violations": {
            "qos": int(c.qos_violations),
            "aodt": int(c.aodt_violations),
            "sep": int(c.sep_violations),
            "cpu": int(c.cpu_unstable_count),
            "bw_excess_hz": float(c.bw_excess_hz),
        },
        "diagnostics": _jsonable(run.diagnostics),
    }


def _paired_stats(by_method: dict[str, dict], champion: str = "sca") -> dict:
    if champion not in by_method:
        return {}
    champ = np.asarray(by_method[champion]["per_seed_Mbps"], dtype=float)
    out = {}
    for name, stats in by_method.items():
        if name == champion:
            continue
        other = np.asarray(stats["per_seed_Mbps"], dtype=float)
        if other.shape != champ.shape:
            continue
        delta = champ - other
        out[name] = {
            "mean_delta_Mbps": float(np.mean(delta)),
            "std_delta_Mbps": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0,
            "win_fraction": float(np.mean(delta > 0.0)),
            "n": int(delta.size),
        }
    return out


def _load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", payload)
    if isinstance(runs, list):
        return {_run_key(r["seed"], r["method"]): r for r in runs}
    return {str(k): v for k, v in runs.items()}


def evaluate_bank(
    bank: dict,
    overlay: SimConfig,
    methods: tuple[str, ...],
    *,
    sca_settings: SCASettings | None = None,
    pso_settings: PSOSettings | None = None,
    td3_settings: TD3Settings | None = None,
    multistart_settings: MultiStartSettings | None = None,
    anchor_settings: AnchorSettings | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    bank_path: str | Path | None = None,
) -> dict:
    cfg = eval_cfg_from_bank(bank, overlay)
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    done: dict[str, dict] = _load_checkpoint(ckpt) if (ckpt and resume) else {}
    records = list(bank["scenarios"])
    n = len(records)
    total = n * len(methods)
    if done:
        _log(f"  resume checkpoint    {len(done)}/{total} runs already stored")
    finished = 0
    for rec in records:
        sc = scenario_from_record(rec, cfg)
        init_uav = uav_from_record(rec)
        for method in methods:
            key = _run_key(sc.seed, method)
            finished += 1
            if key in done:
                continue
            _log(f"  [{finished}/{total}] {rec['id']}  method={method}  seed={sc.seed}")
            t0 = perf_counter()
            uav_arg = init_uav if method == "random" else None
            run = run_method(
                sc,
                method,
                sc.seed,
                sca_settings=sca_settings,
                pso_settings=pso_settings,
                td3_settings=td3_settings,
                multistart_settings=multistart_settings,
                anchor_settings=anchor_settings,
                uav_xyz_m=uav_arg,
            )
            run.diagnostics.setdefault("wall_clock_s", perf_counter() - t0)
            done[key] = run_to_record(run, scenario_id=int(rec["id"]))
            if ckpt is not None:
                _atomic_write_json(ckpt, {"runs": done})
            if method in {"td3", "sca_multistart", "sca_anchor"}:
                _log(
                    f"           done {perf_counter() - t0:.1f}s  "
                    f"{run.sum_rate_mbps:.4f} Mbps  "
                    f"feas={int(run.feasible)}  "
                    f"[{finished}/{total}]"
                )

    ordered: list[dict] = []
    for rec in records:
        for method in methods:
            ordered.append(done[_run_key(int(rec["seed"]), method)])

    by_method: dict[str, dict] = {}
    for method in methods:
        rows = [r for r in ordered if r["method"] == method]
        rates = np.array([r["sum_rate_bit_per_s"] for r in rows], dtype=float)
        feas = np.array([r["feasible"] for r in rows], dtype=bool)
        max_aodt = np.array([r["max_AoDT_s"] for r in rows], dtype=float)
        std = float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0
        by_method[method] = {
            "n": len(rows),
            "mean_sum_rate_bit_per_s": float(np.mean(rates)),
            "std_sum_rate_bit_per_s": std,
            "mean_sum_rate_Mbps": float(np.mean(rates) / 1e6),
            "std_sum_rate_Mbps": std / 1e6,
            "feasible_fraction": float(np.mean(feas)),
            "mean_max_AoDT_s": float(np.mean(max_aodt)),
            "std_max_AoDT_s": (
                float(np.std(max_aodt, ddof=1)) if max_aodt.size > 1 else 0.0
            ),
            "seeds": [r["seed"] for r in rows],
            "per_seed_Mbps": [r["sum_rate_Mbps"] for r in rows],
            "per_seed_feasible": [r["feasible"] for r in rows],
            "per_seed_max_AoDT_s": [r["max_AoDT_s"] for r in rows],
        }

    return {
        "paper": "Khalaf et al. IEEE TNSM 2026 — 100-scenario Monte Carlo",
        "note": (
            "Each saved scenario is one random area + IoT + UAV layout. "
            "SCA is the per-instance optimizer. TD3 is opt-in via --methods. "
            "Plotted points are means over the bank. "
            f"Area is {cfg.area_x_m:g}×{cfg.area_y_m:g} m, "
            f"I={cfg.num_iot}, J={cfg.num_uav}."
        ),
        "n_scenarios": n,
        "seed_start": bank.get("seed_start"),
        "methods": list(methods),
        "bank_path": None if bank_path is None else str(bank_path),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "area_m": [cfg.area_x_m, cfg.area_y_m],
        "num_iot": cfg.num_iot,
        "num_uav": cfg.num_uav,
        "cfg": cfg_to_dict(cfg),
        "geometry": bank["geometry"],
        "by_method": by_method,
        "sca_minus_baseline_Mbps": _paired_stats(by_method),
        "runs": ordered,
    }


def write_eval(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    rows = []
    for run in payload["runs"]:
        rows.append(
            {
                "scenario_id": run["scenario_id"],
                "seed": run["seed"],
                "method": run["method"],
                "sum_rate_Mbps": run["sum_rate_Mbps"],
                "feasible": run["feasible"],
                "max_AoDT_s": run["max_AoDT_s"],
            }
        )
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary_path = path.with_name(path.stem + "_summary.csv")
    summary_rows = []
    for method, stats in payload["by_method"].items():
        summary_rows.append(
            {
                "method": method,
                "n": stats["n"],
                "mean_Mbps": stats["mean_sum_rate_Mbps"],
                "std_Mbps": stats["std_sum_rate_Mbps"],
                "feasible_fraction": stats["feasible_fraction"],
                "mean_max_AoDT_s": stats["mean_max_AoDT_s"],
                "std_max_AoDT_s": stats["std_max_AoDT_s"],
            }
        )
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    return path
