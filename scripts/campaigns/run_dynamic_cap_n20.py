"""20-seed dynamic per-link cap at 100 m, 10 MHz.

Cap on an associated link is max(rate-floor bandwidth, alpha * B_sys / I).
alpha = 1 is an equal share: every device can receive leftover. Methods are
headline SCA and zenith-subset. IoT counts follow the existing IoT axis
(equal groups, K = 5) at I = 10, 20, 30.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from uavdt.config import headline_sim_config
from uavdt.experiments.grids import config_for_iot_sweep
from uavdt.experiments.methods import run_method
from uavdt.scenario import generate_scenario

ROOT = Path(__file__).resolve().parents[2]
IOTS = (10, 20, 30)
METHODS = ("sca", "sca_anchor")
SEEDS = range(1, 21)


def _cfg(num_iot: int, alpha: float):
    base = headline_sim_config(area_m=100.0, max_bw_share=None)
    base = replace(base, dynamic_cap_alpha=alpha)
    return config_for_iot_sweep(num_iot, base)


def _done(path: Path) -> set[tuple[int, str, int]]:
    if not path.exists():
        return set()
    out = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.add((int(row["num_iot"]), row["method"], int(row["seed"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    alpha = float(args.alpha)
    out = ROOT / "results" / f"run_dynamic_cap_alpha{alpha:g}" / "100m"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "n20.csv"
    seen = _done(path)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "num_iot",
                "cap_share",
                "method",
                "seed",
                "sum_rate_Mbps",
                "feasible",
                "max_aodt_s",
                "max_link_share",
                "wall_clock_s",
            ],
        )
        if new:
            writer.writeheader()
        for num_iot in IOTS:
            cfg = _cfg(num_iot, alpha)
            share = alpha / num_iot
            for method in METHODS:
                for seed in SEEDS:
                    key = (num_iot, method, seed)
                    if key in seen:
                        continue
                    scenario = generate_scenario(seed, cfg)
                    run = run_method(scenario, method, seed)
                    bw = run.allocation.bandwidth_hz
                    total = float(bw.sum())
                    max_share = float(bw.max() / cfg.b_sys_hz) if cfg.b_sys_hz else 0.0
                    aodt = run.true_eval.aodt_s
                    finite = aodt[np.isfinite(aodt)] if hasattr(aodt, "__len__") else aodt
                    row = {
                        "num_iot": num_iot,
                        "cap_share": share,
                        "method": method,
                        "seed": seed,
                        "sum_rate_Mbps": run.true_eval.sum_rate_mbps,
                        "feasible": int(bool(run.true_eval.feasible)),
                        "max_aodt_s": float(np.max(finite)) if np.size(finite) else "",
                        "max_link_share": max_share,
                        "wall_clock_s": run.diagnostics.get("wall_clock_s", ""),
                    }
                    writer.writerow(row)
                    handle.flush()
                    print(
                        f"I={num_iot} {method} seed={seed} "
                        f"{row['sum_rate_Mbps']:.3f} Mbps "
                        f"feas={row['feasible']} max_share={max_share:.3f} "
                        f"t={row['wall_clock_s']}",
                        flush=True,
                    )
                    _ = total
    summary = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for num_iot in IOTS:
        for method in METHODS:
            got = [r for r in rows if int(r["num_iot"]) == num_iot and r["method"] == method]
            rates = [float(r["sum_rate_Mbps"]) for r in got]
            feas = [int(r["feasible"]) for r in got]
            shares = [float(r["max_link_share"]) for r in got]
            summary.append(
                {
                    "num_iot": num_iot,
                    "method": method,
                    "n": len(rates),
                    "mean_Mbps": sum(rates) / len(rates) if rates else None,
                    "feasible_fraction": sum(feas) / len(feas) if feas else None,
                    "mean_max_link_share": sum(shares) / len(shares) if shares else None,
                }
            )
    (out / "n20_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
