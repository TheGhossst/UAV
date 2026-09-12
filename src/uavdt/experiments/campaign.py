"""Paper-style experimental campaign. Does not modify frozen SCA code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uavdt.config import SimConfig
from uavdt.experiments.grids import AXES, SweepPoint, iter_axis
from uavdt.experiments.methods import METHODS, MethodRun, run_method
from uavdt.placement.pso import PSOSettings
from uavdt.sca.settings import SCASettings
from uavdt.sca_multistart import MultiStartSettings
from uavdt.scenario import generate_scenario
from uavdt.td3.settings import TD3Settings


@dataclass
class CampaignSettings:
    n_runs: int = 5
    seed_start: int = 1
    methods: tuple[str, ...] = METHODS
    sca_settings: SCASettings | None = None
    pso_settings: PSOSettings | None = None
    td3_settings: TD3Settings | None = None
    multistart_settings: MultiStartSettings | None = None


def _seed_list(settings: CampaignSettings) -> tuple[int, ...]:
    return tuple(range(settings.seed_start, settings.seed_start + settings.n_runs))


def _summarize(runs: list[MethodRun]) -> dict:
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
        "seeds": [r.seed for r in runs],
        "per_seed_Mbps": [r.sum_rate_mbps for r in runs],
        "per_seed_feasible": [r.feasible for r in runs],
        "diagnostics": [r.diagnostics for r in runs],
    }


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def run_point(
    point: SweepPoint,
    settings: CampaignSettings,
) -> dict:
    seeds = _seed_list(settings)
    by_method: dict[str, dict] = {}
    for method in settings.methods:
        runs: list[MethodRun] = []
        for seed in seeds:
            sc = generate_scenario(seed, point.cfg)
            _log(f"  {point.label}  method={method}  seed={seed}")
            runs.append(
                run_method(
                    sc,
                    method,
                    seed,
                    sca_settings=settings.sca_settings,
                    pso_settings=settings.pso_settings,
                    td3_settings=settings.td3_settings,
                    multistart_settings=settings.multistart_settings,
                )
            )
        by_method[method] = _summarize(runs)
    return {
        "axis": point.axis,
        "x_name": point.x_name,
        "x": point.x_value,
        "label": point.label,
        "b_sys_hz": point.cfg.b_sys_hz,
        "max_bw_share": point.cfg.max_bw_share,
        "num_iot": point.cfg.num_iot,
        "num_uav": point.cfg.num_uav,
        "by_method": by_method,
    }


def _campaign_header(cfg: SimConfig, settings: CampaignSettings) -> dict:
    return {
        "paper": "Khalaf et al. IEEE TNSM 2026 §VII Figs. 6–10 axes",
        "sca_frozen": True,
        "sca_joint_probe": "sca_joint" in settings.methods,
        "sca_multistart": "sca_multistart" in settings.methods,
        "td3_opt_in": "td3" in settings.methods,
        "n_runs": settings.n_runs,
        "seed_start": settings.seed_start,
        "methods": list(settings.methods),
        "note": (
            "Paper averages 20 runs per point. This file's n_runs is the "
            f"campaign setting. Area is {cfg.area_x_m:g}×{cfg.area_y_m:g} m "
            "(paper §VII is 500×500 m; this reproduction's default is "
            "100×100 m). Headline B_sys is this reproduction's experiment "
            "parameter, not Table II 20 kHz. PSO is EXTERNAL (not a paper "
            "baseline). Placement methods are re-scored with the frozen-q "
            "bandwidth LP."
        ),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "area_m": [cfg.area_x_m, cfg.area_y_m],
    }


def _checkpoint_matches(old: dict, cfg: SimConfig, settings: CampaignSettings) -> bool:
    return (
        float(old.get("b_sys_hz", -1)) == float(cfg.b_sys_hz)
        and old.get("max_bw_share") == cfg.max_bw_share
        and int(old.get("n_runs", -1)) == int(settings.n_runs)
        and int(old.get("seed_start", -1)) == int(settings.seed_start)
        and list(old.get("methods", [])) == list(settings.methods)
        and list(old.get("area_m", [])) == [cfg.area_x_m, cfg.area_y_m]
    )


def _load_checkpoint(
    path: Path, cfg: SimConfig, settings: CampaignSettings
) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not _checkpoint_matches(old, cfg, settings):
        _log(f"  checkpoint mismatch, ignoring {path}")
        return {}
    done: dict[tuple, dict] = {}
    for pt in old.get("points", []):
        done[(pt["axis"], float(pt["x"]))] = pt
    if done:
        _log(f"  resume {len(done)} point(s) from {path}")
    return done


def _write_checkpoint(path: Path, header: dict, points: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(header)
    payload["points"] = points
    payload["checkpoint"] = True
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_campaign(
    axes: tuple[str, ...] | list[str],
    cfg: SimConfig,
    settings: CampaignSettings | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    settings = settings or CampaignSettings()
    wanted = tuple(a.lower().strip() for a in axes)
    for a in wanted:
        if a not in AXES:
            raise ValueError(f"unknown axis {a!r}; expected {AXES}")
    header = _campaign_header(cfg, settings)
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
    completed = _load_checkpoint(ckpt, cfg, settings) if ckpt is not None else {}
    points_out: list[dict] = []
    for axis in wanted:
        for point in iter_axis(axis, cfg):
            key = (point.axis, float(point.x_value))
            if key in completed:
                _log(f"  resume skip {point.label}")
                points_out.append(completed[key])
            else:
                points_out.append(run_point(point, settings))
            if ckpt is not None:
                _write_checkpoint(ckpt, header, points_out)
    payload = dict(header)
    payload["points"] = points_out
    return payload


def replace_points(payload: dict, new_points: list[dict]) -> dict:
    """Swap in re-run rows matched by (axis, x). Other points stay untouched."""
    by_key = {(p["axis"], float(p["x"])): p for p in new_points}
    out = dict(payload)
    pts = []
    for old in payload["points"]:
        key = (old["axis"], float(old["x"]))
        pts.append(by_key.pop(key, old))
    if by_key:
        missing = ", ".join(f"{a} x={x:g}" for a, x in by_key)
        raise ValueError(f"no matching campaign rows for {missing}")
    out["points"] = pts
    return out


def write_campaign(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    rows = []
    for pt in payload["points"]:
        for method, stats in pt["by_method"].items():
            rows.append(
                {
                    "axis": pt["axis"],
                    "x_name": pt["x_name"],
                    "x": pt["x"],
                    "method": method,
                    "mean_Mbps": stats["mean_sum_rate_Mbps"],
                    "std_Mbps": stats["std_sum_rate_Mbps"],
                    "feasible_fraction": stats["feasible_fraction"],
                    "mean_max_AoDT_s": stats["mean_max_AoDT_s"],
                    "n": stats["n"],
                }
            )
    if rows:
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path
