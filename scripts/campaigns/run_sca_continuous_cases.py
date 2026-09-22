"""Continuous-candidate control: ~120 uniform UAV layouts, LP + top-3 polish.

Same leftover-dump LP scoring, top-K Algorithm 1 polish, and keep-best
as zenith-subset, but candidates are not parked on IoTs. If this matches
zenith-anchor, the mechanism is many candidates plus LP screening, not
IoT parking.

Does not overwrite headline 25% campaigns or n100/eval.json.

Usage:
  python scripts/campaigns/run_sca_continuous_cases.py
  python scripts/campaigns/run_sca_continuous_cases.py --only n20_500m
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

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
}

BASELINE_JSON = {
    "n20_500m": ROOT / "results" / "sca_anchor_n20_500m.json",
}

CASES = {
    "n20_500m": {
        "area_m": 500.0,
        "num_uav": 3,
        "n_continuous": 120,
        "out": ROOT / "results" / "sca_continuous_n20_500m.json",
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
    return value


def _baselines_from_anchor(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for row in payload.get("per_seed") or []:
        seed = int(row.get("seed", -1))
        cell = {}
        for method in (
            "random",
            "kmeans",
            "pso",
            "sca",
            "sca_anchor",
            "sca_multistart",
            "sca_medoid",
        ):
            if method in row:
                cell[method] = row[method]
        out[seed] = cell
    return out


def _complete(path: Path, n_runs: int, seed_start: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    seeds = payload.get("seeds") or []
    want = list(range(int(seed_start), int(seed_start) + int(n_runs)))
    return list(seeds) == want and "sca_continuous" in (payload.get("mean_Mbps") or {})


def run_case(name: str, *, n_runs: int, seed_start: int, skip_if_complete: bool) -> dict:
    spec = CASES[name]
    out: Path = spec["out"]
    _refuse(out)
    if skip_if_complete and _complete(out, n_runs, seed_start):
        _log(f"skip complete {name} -> {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    area_m = float(spec["area_m"])
    num_uav = int(spec["num_uav"])
    n_cont = int(spec["n_continuous"])
    cfg = config_for_counts(
        10,
        num_uav,
        SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE).with_square_area_m(
            area_m
        ),
    )
    sca_settings = SCASettings(solver=None, max_iterations=30)
    anc = AnchorSettings(selection="continuous", n_continuous=n_cont, top_k=3)
    seeds = tuple(range(int(seed_start), int(seed_start) + int(n_runs)))
    reuse = _baselines_from_anchor(BASELINE_JSON.get(name, Path()))
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    done: dict[str, dict] = {}
    if ckpt.exists():
        done = json.loads(ckpt.read_text(encoding="utf-8"))
        _log(f"resume {len(done)}/{len(seeds)} from {ckpt.name}")

    _log(
        f"{name}  sca_continuous  n={n_cont}  seeds={seeds[0]}..{seeds[-1]}  "
        f"area={area_m:g} m  J={num_uav}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        key = str(seed)
        rec = done.get(key) or {"seed": int(seed)}
        if "sca_continuous" not in rec:
            sc = generate_scenario(seed, cfg)
            t0 = perf_counter()
            run = run_method(
                sc,
                "sca_continuous",
                seed,
                sca_settings=sca_settings,
                anchor_settings=anc,
            )
            rec["sca_continuous"] = {
                "sum_rate_Mbps": float(run.sum_rate_mbps),
                "feasible": bool(run.feasible),
                "wall_clock_s": float(perf_counter() - t0),
                "source": "run",
                "diagnostics": {
                    k: run.diagnostics.get(k)
                    for k in (
                        "selection",
                        "enum_mode",
                        "winner_kind",
                        "winner_combo",
                        "n_lp",
                        "n_continuous",
                        "lp_best_Mbps",
                        "frozen_Mbps",
                        "delta_vs_frozen_Mbps",
                    )
                    if k in run.diagnostics
                },
            }
            _log(
                f"  seed {seed}  continuous {run.sum_rate_mbps:.4f} Mbps  "
                f"kind={run.diagnostics.get('winner_kind')}  "
                f"n_lp={run.diagnostics.get('n_lp')}  "
                f"{perf_counter() - t0:.1f}s"
            )
        for method, cell in (reuse.get(int(seed)) or {}).items():
            rec.setdefault(method, cell)
        done[key] = rec
        ckpt.write_text(json.dumps(_jsonable(done), indent=2), encoding="utf-8")

    rows = [done[str(s)] for s in seeds]
    present = sorted({m for r in rows for m in r if m != "seed"})
    payload = {
        "label": f"sca_continuous uniform-xy LP+polish  {area_m:g} m J={num_uav}",
        "experiment": "sca_continuous",
        "case": name,
        "b_sys_hz": 8.8e6,
        "max_bw_share": PRIMARY_MAX_BW_SHARE,
        "area_m": [area_m, area_m],
        "num_uav": num_uav,
        "n_continuous": n_cont,
        "n_runs": int(n_runs),
        "seed_start": int(seed_start),
        "seeds": list(seeds),
        "per_seed": rows,
        "mean_Mbps": {
            m: float(np.mean([r[m]["sum_rate_Mbps"] for r in rows if m in r]))
            for m in present
        },
        "notes": [
            "120 uniform continuous UAV layouts, leftover-dump LP, top-3 SCA polish.",
            "Same keep-best-with-frozen construction as sca_anchor.",
            "Candidates are not restricted to IoT xy.",
            "Does not overwrite headline 8.8 MHz campaign JSON.",
        ],
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}")
    _log("means  " + "  ".join(f"{m}={payload['mean_Mbps'][m]:.4f}" for m in present))
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default="")
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--no-skip-complete", action="store_true")
    args = p.parse_args()
    names = (
        tuple(x.strip() for x in args.only.split(",") if x.strip())
        if args.only.strip()
        else ("n20_500m",)
    )
    t0 = perf_counter()
    for name in names:
        if name not in CASES:
            raise SystemExit(f"unknown case {name}; known={sorted(CASES)}")
        run_case(
            name,
            n_runs=int(args.n_runs),
            seed_start=int(args.seed_start),
            skip_if_complete=not args.no_skip_complete,
        )
    _log(f"done in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
