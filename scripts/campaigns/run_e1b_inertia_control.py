"""Path B footnote: Hertz-best vs inertia-best among the same k-means inits.

If both selections land within 50 kHz, Path B is 'multi-start covering helps',
not a Hertz-specific geometry family.

Usage:
  python scripts/campaigns/run_e1b_inertia_control.py
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
from uavdt.min_spectrum import hertz_bound_at_layout  # noqa: E402
from uavdt.min_spectrum_place import HertzPlaceSettings, _kmeans_seeds  # noqa: E402
from uavdt.placement.kmeans import place_kmeans  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PRACTICAL_HZ = 50.0e3
OUT = ROOT / "results" / "e1b_inertia_control.json"
CASES = {
    "n50_500m_j3": {"area_m": 500.0, "num_uav": 3},
    "n50_500m_j2": {"area_m": 500.0, "num_uav": 2},
}


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def inertia_sse(iot_xy: np.ndarray, uav_xy: np.ndarray) -> float:
    pts = np.asarray(iot_xy, dtype=float)[:, :2]
    ctr = np.asarray(uav_xy, dtype=float)[:, :2]
    d2 = np.sum((pts[:, None, :] - ctr[None, :, :]) ** 2, axis=-1)
    return float(np.min(d2, axis=1).sum())


def run_case(name: str, *, n_runs: int, seed_start: int) -> dict:
    spec = CASES[name]
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
    seeds = list(range(int(seed_start), int(seed_start) + int(n_runs)))
    rows = []
    _log(f"{name}  seeds={seeds[0]}..{seeds[-1]}  J={num_uav}  5 k-means inits")
    for seed in seeds:
        sc = generate_scenario(seed, cfg)
        iot_xy = sc.iot_xyz_m[:, :2]
        cands = []
        for init in _kmeans_seeds(seed, settings.n_kmeans):
            uav = place_kmeans(sc, init)
            hz = float(hertz_bound_at_layout(sc, uav)["bound_hz"])
            sse = inertia_sse(iot_xy, uav)
            cands.append({"init_seed": int(init), "bound_hz": hz, "inertia_sse": sse})
        hertz_best = min(cands, key=lambda r: r["bound_hz"])
        inertia_best = min(cands, key=lambda r: r["inertia_sse"])
        oneshot = next(r for r in cands if r["init_seed"] == int(seed))
        rec = {
            "seed": int(seed),
            "oneshot_hz": oneshot["bound_hz"],
            "hertz_best_hz": hertz_best["bound_hz"],
            "inertia_best_hz": inertia_best["bound_hz"],
            "same_init": hertz_best["init_seed"] == inertia_best["init_seed"],
            "n_inits": len(cands),
        }
        rec["hertz_save_vs_oneshot_hz"] = rec["oneshot_hz"] - rec["hertz_best_hz"]
        rec["inertia_save_vs_oneshot_hz"] = rec["oneshot_hz"] - rec["inertia_best_hz"]
        rec["hertz_minus_inertia_hz"] = rec["inertia_best_hz"] - rec["hertz_best_hz"]
        rows.append(rec)
        _log(
            f"  seed {seed}  oneshot={oneshot['bound_hz']/1e3:.1f}  "
            f"hertz-best={hertz_best['bound_hz']/1e3:.1f}  "
            f"inertia-best={inertia_best['bound_hz']/1e3:.1f}  "
            f"same={rec['same_init']}"
        )
    h = np.array([r["hertz_best_hz"] for r in rows])
    inn = np.array([r["inertia_best_hz"] for r in rows])
    one = np.array([r["oneshot_hz"] for r in rows])
    d = inn - h
    return {
        "case": name,
        "area_m": area_m,
        "num_uav": num_uav,
        "n_runs": int(n_runs),
        "n_kmeans": int(settings.n_kmeans),
        "mean_oneshot_kHz": float(np.mean(one) / 1e3),
        "mean_hertz_best_kHz": float(np.mean(h) / 1e3),
        "mean_inertia_best_kHz": float(np.mean(inn) / 1e3),
        "mean_hertz_save_vs_oneshot_kHz": float(np.mean(one - h) / 1e3),
        "mean_inertia_save_vs_oneshot_kHz": float(np.mean(one - inn) / 1e3),
        "mean_hertz_minus_inertia_kHz": float(np.mean(d) / 1e3),
        "n_same_init": int(sum(r["same_init"] for r in rows)),
        "n_hertz_better_50kHz": int(np.sum(d > PRACTICAL_HZ)),
        "wilcoxon": wilcoxon_signed_rank(d, alternative="greater"),
        "paired_t": paired_t(d),
        "per_seed": rows,
        "footnote": (
            "If Hertz-best and inertia-best save the same Hertz vs one-shot, "
            "Path B is multi-start k-means on a covering objective."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=1)
    args = p.parse_args()
    t0 = perf_counter()
    payload = {
        "experiment": "e1b_hertz_vs_inertia_kmeans",
        "practical_hz": PRACTICAL_HZ,
        "cases": {},
    }
    for name in CASES:
        payload["cases"][name] = run_case(
            name, n_runs=int(args.n_runs), seed_start=int(args.seed_start)
        )
        c = payload["cases"][name]
        _log(
            f"  {name}  Hertz-best {c['mean_hertz_best_kHz']:.1f} kHz  "
            f"inertia-best {c['mean_inertia_best_kHz']:.1f} kHz  "
            f"Hertz extra save {c['mean_hertz_minus_inertia_kHz']:+.1f} kHz  "
            f"same init {c['n_same_init']}/{c['n_runs']}"
        )
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT}  ({perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
