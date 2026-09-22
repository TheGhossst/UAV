"""E1: min-spectrum B_sys at frozen leftover-dump geometry.

For each seed and method: place at 8.8 MHz / 25% cap (same radio as the
headline leftover-dump campaigns), freeze (q, a, b), then binary-search
the smallest B_sys that still meets R_min and AoDT. Score is Hertz, not
Mbps. Does not overwrite headline campaign JSON.

Prediction. 500 m / J=3: zenith-anchor needs less Hertz than k-means SCA
if leftover-dump geometry also covers AoDT floors. 100 m: Hertz almost
identical (SE is uniformly high). Surprising if 500 m Hertz is a tie —
then leftover-dump Mbps was hiding a covering/max-min-SE problem that
k-means already solves.

Usage:
  python scripts/campaigns/run_min_spectrum.py
  python scripts/campaigns/run_min_spectrum.py --only n20_500m
  python scripts/campaigns/run_min_spectrum.py --n-runs 2 --only n20_100m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.min_spectrum import min_bsys_frozen_q  # noqa: E402
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
    ROOT / "results" / "n100_500m_cap15" / "eval.json",
}

PLACE_B_SYS_HZ = 8.8e6
METHODS = ("random", "kmeans", "pso", "sca", "sca_anchor", "sca_medoid")
CASES = {
    "n20_100m": {
        "area_m": 100.0,
        "num_uav": 3,
        "out": ROOT / "results" / "min_spectrum_n20_100m.json",
    },
    "n20_500m": {
        "area_m": 500.0,
        "num_uav": 3,
        "out": ROOT / "results" / "min_spectrum_n20_500m.json",
    },
    "n20_500m_j2": {
        "area_m": 500.0,
        "num_uav": 2,
        "out": ROOT / "results" / "min_spectrum_n20_500m_j2.json",
    },
}


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _refuse(path: Path) -> None:
    resolved = path.resolve()
    for p in PROTECTED:
        if resolved == p.resolve():
            raise SystemExit(f"refusing to overwrite protected {p}")


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
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _cell_complete(path: Path, n_runs: int, seed_start: int, methods: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    seeds = payload.get("seeds") or []
    want = list(range(int(seed_start), int(seed_start) + int(n_runs)))
    if list(seeds) != want:
        return False
    means = payload.get("mean_kHz") or {}
    return all(m in means for m in methods)


def _summarize(rows: list[dict], methods: tuple[str, ...]) -> dict:
    out: dict[str, dict] = {}
    for method in methods:
        hz = []
        for row in rows:
            cell = row.get(method) or {}
            val = cell.get("min_b_sys_hz")
            hz.append(float(val) if val is not None else float("nan"))
        arr = np.asarray(hz, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[method] = {
            "n_finite": int(finite.size),
            "mean_hz": float(np.mean(finite)) if finite.size else None,
            "std_hz": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "median_hz": float(np.median(finite)) if finite.size else None,
            "min_hz": float(np.min(finite)) if finite.size else None,
            "max_hz": float(np.max(finite)) if finite.size else None,
            "per_seed_hz": [None if not np.isfinite(x) else float(x) for x in arr],
        }
    return out


def run_case(
    name: str,
    *,
    n_runs: int,
    seed_start: int,
    methods: tuple[str, ...],
    max_bw_share: float,
    skip_if_complete: bool,
) -> dict:
    spec = CASES[name]
    out: Path = spec["out"]
    _refuse(out)
    if skip_if_complete and _cell_complete(out, n_runs, seed_start, methods):
        _log(f"skip complete {name} -> {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    area_m = float(spec["area_m"])
    num_uav = int(spec["num_uav"])
    base = SimConfig(
        b_sys_hz=PLACE_B_SYS_HZ, max_bw_share=float(max_bw_share)
    ).with_square_area_m(area_m)
    cfg = config_for_counts(10, num_uav, base)
    sca_settings = SCASettings(solver=None, max_iterations=30)
    pso_settings = PSOSettings()
    anchor_settings = AnchorSettings()
    seeds = tuple(range(int(seed_start), int(seed_start) + int(n_runs)))
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    done: dict[str, dict] = {}
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            for row in prev.get("per_seed") or []:
                seed = row.get("seed")
                if seed is not None:
                    done[str(seed)] = dict(row)
            if done:
                _log(f"resume {len(done)} seeds from {out.name}")
        except (OSError, json.JSONDecodeError):
            done = {}
    if ckpt.exists():
        ckpt_rows = json.loads(ckpt.read_text(encoding="utf-8"))
        done.update(ckpt_rows)
        _log(f"resume {len(ckpt_rows)}/{len(seeds)} seeds from {ckpt.name}")

    _log(
        f"{name}  seeds={seeds[0]}..{seeds[-1]}  "
        f"area={area_m:g} m  J={num_uav}  place={PLACE_B_SYS_HZ/1e6:g} MHz  "
        f"cap={max_bw_share:.0%}  methods={list(methods)}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        key = str(seed)
        rec = done.get(key) or {"seed": int(seed)}
        missing = [m for m in methods if m not in rec]
        if not missing:
            done[key] = rec
            continue
        sc = generate_scenario(seed, cfg)
        _log(f"  seed {seed}")
        for method in missing:
            t0 = perf_counter()
            run = run_method(
                sc,
                method,
                seed,
                sca_settings=sca_settings,
                pso_settings=pso_settings,
                anchor_settings=anchor_settings,
            )
            t_place = perf_counter() - t0
            ms = min_bsys_frozen_q(
                sc,
                run.uav_xyz_m,
                run.allocation,
                max_bw_share=float(max_bw_share),
                hi_hz=PLACE_B_SYS_HZ,
                place_eval=run.true_eval,
                settings=sca_settings,
            )
            rec[method] = {
                "min_b_sys_hz": ms.min_b_sys_hz,
                "bound_hz": ms.bound_hz,
                "binding": ms.binding,
                "feasible_at_place_radio": ms.feasible_at_place_radio,
                "feasible_at_min": ms.feasible_at_min,
                "n_lp": ms.n_lp,
                "sum_floors_hz": ms.sum_floors_hz,
                "max_floor_hz": ms.max_floor_hz,
                "min_assoc_se": ms.min_assoc_se,
                "n_forwarding": ms.n_forwarding,
                "sum_rate_Mbps_at_min": ms.sum_rate_Mbps_at_min,
                "place_sum_rate_Mbps": ms.place_sum_rate_Mbps,
                "place_feasible": bool(run.feasible),
                "place_wall_s": t_place,
                "search_wall_s": ms.wall_clock_s,
                "message": ms.message,
            }
            hz = ms.min_b_sys_hz
            hz_s = "inf" if hz is None else f"{hz/1e3:.1f} kHz"
            _log(
                f"    {method:<12} {hz_s:>12}  bind={ms.binding}  "
                f"place={ms.place_sum_rate_Mbps:.3f} Mbps  "
                f"{t_place + ms.wall_clock_s:.1f}s"
            )
        done[key] = rec
        ckpt.write_text(json.dumps(_jsonable(done), indent=2), encoding="utf-8")

    rows = [done[str(s)] for s in seeds]
    stats = _summarize(rows, methods)
    payload = {
        "experiment": "min_spectrum_frozen_q",
        "case": name,
        "area_m": area_m,
        "num_iot": 10,
        "num_uav": num_uav,
        "place_b_sys_hz": PLACE_B_SYS_HZ,
        "max_bw_share": float(max_bw_share),
        "n_runs": int(n_runs),
        "seed_start": int(seed_start),
        "seeds": list(seeds),
        "methods": list(methods),
        "score": "min_b_sys_hz",
        "mean_kHz": {
            m: (None if s["mean_hz"] is None else s["mean_hz"] / 1.0e3)
            for m, s in stats.items()
        },
        "by_method": stats,
        "per_seed": rows,
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    _log(f"wrote {out}")
    for method in methods:
        mean = payload["mean_kHz"][method]
        n_fin = stats[method]["n_finite"]
        mean_s = "n/a" if mean is None else f"{mean:.1f} kHz"
        _log(f"  mean {method:<12} {mean_s}  n={n_fin}/{n_runs}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default="", help="comma-separated case names")
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--max-bw-share", type=float, default=PRIMARY_MAX_BW_SHARE)
    p.add_argument("--no-skip-complete", action="store_true")
    args = p.parse_args()
    if args.only.strip():
        names = tuple(x.strip() for x in args.only.split(",") if x.strip())
    else:
        names = ("n20_100m", "n20_500m", "n20_500m_j2")
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases {unknown}; known={sorted(CASES)}")
    methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    t0 = perf_counter()
    for name in names:
        run_case(
            name,
            n_runs=int(args.n_runs),
            seed_start=int(args.seed_start),
            methods=methods,
            max_bw_share=float(args.max_bw_share),
            skip_if_complete=not args.no_skip_complete,
        )
    _log(f"done in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
