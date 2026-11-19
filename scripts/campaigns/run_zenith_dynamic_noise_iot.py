"""100-seed zenith-anchor IoT sweep at 10 MHz, dynamic bound, proportional noise.

Nominal task rate is 2 tasks/s. Spot checks at I=10 and I=30 (seed 1) give
the same sum rate at 1 and 3.5 tasks/s, so this sweep is the curve drawn
for every arrival rate. Writes one JSON per (I, seed) and a summary CSV.
"""

from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "run_10mhz_dynamic_noise" / "100m" / "zenith_iot_tasks"
IOTS = (10, 15, 20, 25, 30)
SEEDS = range(1, 101)
WORKERS = 12


def _point_path(num_iot: int, seed: int) -> Path:
    return OUT / "points" / f"i{num_iot}_s{seed}.json"


def _eval(job: tuple[int, int]) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning)
    num_iot, seed = job
    from uavdt.config import headline_sim_config
    from uavdt.experiments.grids import config_for_iot_sweep
    from uavdt.experiments.methods import run_method
    from uavdt.scenario import generate_scenario
    from uavdt.sca_anchor import AnchorSettings

    anchor = AnchorSettings(max_enumerate=10000, top_k=3, beam_width=10)
    base = replace(
        headline_sim_config(area_m=100.0, max_bw_share=None),
        dynamic_cap_alpha=1.0,
        bandwidth_proportional_noise=True,
        lambda_i_per_s=2.0,
    )
    cfg = config_for_iot_sweep(num_iot, base)
    scenario = generate_scenario(seed, cfg)
    run = run_method(scenario, "sca_anchor", seed, anchor_settings=anchor)
    return {
        "num_iot": num_iot,
        "seed": seed,
        "mbps": float(run.sum_rate_mbps),
        "feasible": bool(run.feasible),
    }


def _write(row: dict) -> None:
    path = _point_path(int(row["num_iot"]), int(row["seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row), encoding="utf-8")
    tmp.replace(path)


def _summarize() -> None:
    import numpy as np

    lines = ["num_iot,n,mean_Mbps,std_Mbps,feasible_fraction"]
    for num_iot in IOTS:
        rows = []
        for seed in SEEDS:
            payload = json.loads(_point_path(num_iot, seed).read_text(encoding="utf-8"))
            rows.append(payload)
        rates = np.array([row["mbps"] for row in rows], dtype=float)
        feas = np.array([row["feasible"] for row in rows], dtype=float)
        std = float(np.std(rates, ddof=1))
        lines.append(
            f"{num_iot},{len(rows)},{float(np.mean(rates)):.12g},{std:.12g},{float(np.mean(feas)):.12g}"
        )
        print(
            f"I={num_iot} mean={float(np.mean(rates)):.6f} std={std:.6f} feas={float(np.mean(feas)):.3f}",
            flush=True,
        )
    summary = OUT / "summary.csv"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {summary}", flush=True)


def main() -> None:
    pending = [
        (num_iot, seed)
        for num_iot in IOTS
        for seed in SEEDS
        if not _point_path(num_iot, seed).is_file()
    ]
    print(f"pending {len(pending)} of {len(IOTS) * len(list(SEEDS))}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_eval, job): job for job in pending}
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                _write(row)
                done += 1
                if done % 10 == 0 or done == len(pending):
                    print(
                        f"done {done}/{len(pending)}  I={row['num_iot']} seed={row['seed']} {row['mbps']:.4f}",
                        flush=True,
                    )
    _summarize()


if __name__ == "__main__":
    main()
