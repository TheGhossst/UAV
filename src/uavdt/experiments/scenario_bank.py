"""Persist random area + IoT + UAV layouts for Monte Carlo experiments.

A scenario bank freezes geometry so later SCA/baseline runs replay the same
deployments. Radio/task knobs (B_sys, T_k, …) stay on the eval config.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from uavdt.config import SimConfig
from uavdt.experiments.grids import config_for_counts
from uavdt.models import Scenario
from uavdt.placement.random import place_random
from uavdt.scenario import generate_scenario

SCHEMA = "uavdt.scenario_bank.v1"


def cfg_to_dict(cfg: SimConfig) -> dict:
    return asdict(cfg)


def cfg_from_dict(payload: dict) -> SimConfig:
    allowed = {f.name for f in fields(SimConfig)}
    return SimConfig(**{k: v for k, v in payload.items() if k in allowed})


def _as_list(arr: np.ndarray) -> list:
    return np.asarray(arr).tolist()


def record_from_scenario(scenario: Scenario, uav_xyz_m: np.ndarray, *, id_: int) -> dict:
    return {
        "id": int(id_),
        "seed": int(scenario.seed),
        "iot_xyz_m": _as_list(scenario.iot_xyz_m),
        "process_id_of_iot": np.asarray(scenario.process_id_of_iot, dtype=int).tolist(),
        "lambdas_per_s": _as_list(scenario.lambdas_per_s),
        "uav_xyz_m": _as_list(uav_xyz_m),
    }


def scenario_from_record(record: dict, cfg: SimConfig) -> Scenario:
    iot = np.asarray(record["iot_xyz_m"], dtype=float)
    lam = np.asarray(record["lambdas_per_s"], dtype=float)
    return generate_scenario(
        int(record["seed"]),
        cfg,
        lambdas_per_s=lam,
        iot_xyz_m=iot,
    )


def uav_from_record(record: dict) -> np.ndarray:
    return np.asarray(record["uav_xyz_m"], dtype=float)


def generate_bank(
    n_scenarios: int,
    cfg: SimConfig,
    *,
    seed_start: int = 1,
    num_iot: int | None = None,
    num_uav: int | None = None,
) -> dict:
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be >= 1")
    i = int(num_iot if num_iot is not None else cfg.num_iot)
    j = int(num_uav if num_uav is not None else cfg.num_uav)
    geo = config_for_counts(i, j, cfg)
    records = []
    for k in range(n_scenarios):
        seed = int(seed_start) + k
        sc = generate_scenario(seed, geo)
        uav = place_random(geo.num_uav, seed, geo)
        records.append(record_from_scenario(sc, uav, id_=k))
    return {
        "schema": SCHEMA,
        "n_scenarios": n_scenarios,
        "seed_start": int(seed_start),
        "note": (
            "Frozen area + IoT (z=0) + random UAV layouts. "
            "SCA / k-means / PSO ignore saved UAV xy and place or optimize "
            "their own. The random method replays uav_xyz_m. "
            "B_sys and other radio knobs are eval-time, not geometry."
        ),
        "geometry": {
            "area_x_m": geo.area_x_m,
            "area_y_m": geo.area_y_m,
            "num_iot": geo.num_iot,
            "num_uav": geo.num_uav,
            "num_processes": geo.num_processes,
            "iots_per_process": geo.iots_per_process,
            "uav_height_m": geo.uav_height_m,
            "uav_min_separation_m": geo.uav_min_separation_m,
        },
        "cfg": cfg_to_dict(geo),
        "scenarios": records,
    }


def write_bank(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_bank(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported scenario bank schema {schema!r}")
    n = int(payload["n_scenarios"])
    if len(payload["scenarios"]) != n:
        raise ValueError(
            f"bank n_scenarios={n} but found {len(payload['scenarios'])} records"
        )
    return payload


def eval_cfg_from_bank(bank: dict, overlay: SimConfig) -> SimConfig:
    """Keep frozen I/J/area; take radio/task knobs from the eval overlay."""
    geo = bank["geometry"]
    base = replace(
        overlay,
        area_x_m=float(geo["area_x_m"]),
        area_y_m=float(geo["area_y_m"]),
        uav_height_m=float(geo.get("uav_height_m", overlay.uav_height_m)),
        uav_min_separation_m=float(
            geo.get("uav_min_separation_m", overlay.uav_min_separation_m)
        ),
    )
    return config_for_counts(int(geo["num_iot"]), int(geo["num_uav"]), base)
