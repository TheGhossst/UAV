"""E1b: min-Hertz keep-best placement (time-boxed).

Falsifier (stated first): if keep-best Hertz at 500 m J=2 and J=3 is
within 50 kHz of one-shot k-means, drop Path B (spectrum-efficient
placement). K-means is in the candidate pool, so E1b cannot lose to
k-means on the closed-form bound; the question is whether extras+polish
move Hertz by a practical amount.

Usage:
  python scripts/campaigns/run_e1b.py
  python scripts/campaigns/run_e1b.py --only n20_500m_j2
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

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.min_spectrum_place import HertzPlaceSettings, solve_min_hertz_place  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PRACTICAL_HZ = 50.0e3
PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
}
CASES = {
    "n20_500m": {
        "area_m": 500.0,
        "num_uav": 3,
        "out": ROOT / "results" / "e1b_n20_500m.json",
    },
    "n20_500m_j2": {
        "area_m": 500.0,
        "num_uav": 2,
        "out": ROOT / "results" / "e1b_n20_500m_j2.json",
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _complete(path: Path, n_runs: int, seed_start: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    seeds = payload.get("seeds") or []
    want = list(range(int(seed_start), int(seed_start) + int(n_runs)))
    return list(seeds) == want and payload.get("n_finite") == int(n_runs)


def run_case(name: str, *, n_runs: int, seed_start: int, skip_if_complete: bool) -> dict:
    spec = CASES[name]
    out: Path = spec["out"]
    _refuse(out)
    if skip_if_complete and _complete(out, n_runs, seed_start):
        _log(f"skip complete {name} -> {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    area_m = float(spec["area_m"])
    num_uav = int(spec["num_uav"])
    cfg = config_for_counts(
        10,
        num_uav,
        SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE).with_square_area_m(
            area_m
        ),
    )
    settings = HertzPlaceSettings()
    seeds = tuple(range(int(seed_start), int(seed_start) + int(n_runs)))
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    done: dict[str, dict] = {}
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            for row in prev.get("per_seed") or []:
                seed = row.get("seed")
                if seed is not None and row.get("min_b_sys_hz") is not None:
                    done[str(seed)] = dict(row)
            if done:
                _log(f"resume {len(done)} seeds from {out.name}")
        except (OSError, json.JSONDecodeError):
            done = {}
    if ckpt.exists():
        ckpt_rows = json.loads(ckpt.read_text(encoding="utf-8"))
        done.update(ckpt_rows)
        _log(f"resume {len(ckpt_rows)}/{len(seeds)} from {ckpt.name}")

    _log(
        f"E1b {name}  seeds={seeds[0]}..{seeds[-1]}  "
        f"area={area_m:g} m  J={num_uav}  "
        f"falsifier: |Hertz-kmeans| < {PRACTICAL_HZ/1e3:.0f} kHz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        key = str(seed)
        if key in done and done[key].get("min_b_sys_hz") is not None:
            continue
        sc = generate_scenario(seed, cfg)
        t0 = perf_counter()
        rec = solve_min_hertz_place(sc, seed, settings)
        rec["wall_clock_s"] = perf_counter() - t0
        done[key] = rec
        km = rec.get("kmeans_bound_hz")
        hz = rec.get("min_b_sys_hz")
        km_s = "n/a" if km is None else f"{float(km)/1e3:.1f}"
        hz_s = "n/a" if hz is None else f"{float(hz)/1e3:.1f}"
        _log(
            f"  seed {seed}  e1b={hz_s} kHz  kmeans={km_s} kHz  "
            f"pre={rec['pre_polish_kind']}  "
            f"moves={rec['polish']['n_moves']}  "
            f"{rec['wall_clock_s']:.1f}s"
        )
        ckpt.write_text(json.dumps(_jsonable(done), indent=2), encoding="utf-8")

    rows = [done[str(s)] for s in seeds]
    e1b = np.array([float(r["min_b_sys_hz"]) for r in rows], dtype=float)
    km = np.array(
        [
            float("nan") if r.get("kmeans_bound_hz") is None else float(r["kmeans_bound_hz"])
            for r in rows
        ],
        dtype=float,
    )
    d = km - e1b
    w = wilcoxon_signed_rank(d, alternative="greater")
    tstat = paired_t(d)
    mean_save = float(np.mean(d))
    payload = {
        "experiment": "e1b_min_hertz_place",
        "case": name,
        "area_m": area_m,
        "num_uav": num_uav,
        "n_runs": int(n_runs),
        "seed_start": int(seed_start),
        "seeds": list(seeds),
        "practical_hz": PRACTICAL_HZ,
        "falsifier": (
            f"drop Path B if mean Hertz save vs k-means < {PRACTICAL_HZ/1e3:.0f} kHz "
            "at this cell"
        ),
        "mean_e1b_kHz": float(np.mean(e1b) / 1e3),
        "mean_kmeans_kHz": float(np.nanmean(km) / 1e3),
        "mean_save_vs_kmeans_kHz": mean_save / 1e3,
        "std_save_kHz": float(np.std(d, ddof=1) / 1e3) if d.size > 1 else 0.0,
        "n_finite": int(np.isfinite(e1b).sum()),
        "n_e1b_better": int(np.sum(d > 1.0e3)),
        "n_practical": int(np.sum(d > PRACTICAL_HZ)),
        "path_b_survives": bool(mean_save > PRACTICAL_HZ),
        "wilcoxon": w,
        "paired_t": tstat,
        "winner_pre_polish": {},
        "per_seed": rows,
        "notes": [
            "Oracle is AoDT/QoS floor Hertz, not leftover-dump Mbps.",
            "K-means is in the pool: vs k-means delta >= 0 on the bound.",
            "Polish is pattern search on Hertz, not Algorithm 1.",
        ],
    }
    kinds: dict[str, int] = {}
    for r in rows:
        k = str(r.get("pre_polish_kind") or "unknown")
        kinds[k] = kinds.get(k, 0) + 1
    payload["winner_pre_polish"] = kinds
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}")
    _log(
        f"  mean e1b {payload['mean_e1b_kHz']:.1f} kHz  "
        f"kmeans {payload['mean_kmeans_kHz']:.1f} kHz  "
        f"save {payload['mean_save_vs_kmeans_kHz']:+.1f} kHz  "
        f"practical {payload['n_practical']}/{payload['n_runs']}  "
        f"Path B survives={payload['path_b_survives']}"
    )
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
        else ("n20_500m", "n20_500m_j2")
    )
    t0 = perf_counter()
    for name in names:
        if name not in CASES:
            raise SystemExit(f"unknown case {name}")
        run_case(
            name,
            n_runs=int(args.n_runs),
            seed_start=int(args.seed_start),
            skip_if_complete=not args.no_skip_complete,
        )
    _log(f"done in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
