"""Compare old placeholder S_i/L vs settled experimental defaults."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from uavdt.config import SimConfig
from uavdt.experiments.fig11 import run_fig11, sensibility_checks
from uavdt.experiments.methods import METHODS, run_method
from uavdt.scenario import generate_scenario
from uavdt.sca.settings import SCASettings

OLD_S_BITS = 10_000.0
OLD_L = 1.0e6
NEW_S_BITS = 96_000.0
NEW_L = 3.75e6

SEEDS = tuple(range(1, 21))
BASE = SimConfig(b_sys_hz=8_800_000.0, max_bw_share=0.25)


def _cfg(label: str) -> SimConfig:
    if label == "old":
        return replace(BASE, task_size_bits=OLD_S_BITS, task_cycles=OLD_L)
    if label == "new":
        return replace(BASE, task_size_bits=NEW_S_BITS, task_cycles=NEW_L)
    raise ValueError(label)


def run_methods_j3(cfg: SimConfig, seeds: tuple[int, ...]) -> dict:
    sca_settings = SCASettings(solver=None, max_iterations=30)
    rows = {m: [] for m in METHODS}
    for seed in seeds:
        sc = generate_scenario(seed, cfg)
        for method in METHODS:
            run = run_method(sc, method, seed, sca_settings=sca_settings)
            rows[method].append(
                {
                    "seed": seed,
                    "mbps": run.sum_rate_mbps,
                    "feasible": run.feasible,
                    "max_aodt_s": float(np.max(run.true_eval.aodt_s)),
                    "aodt_ok": bool(np.all(run.true_eval.aodt_satisfied)),
                    "qos_viol": run.true_eval.constraints.qos_violations,
                    "aodt_viol": run.true_eval.constraints.aodt_violations,
                    "cpu_bad": run.true_eval.constraints.cpu_unstable_count,
                    "mu_per_s": run.true_eval.mu_per_s,
                }
            )
    summary = {}
    for method, items in rows.items():
        rates = np.array([x["mbps"] for x in items], dtype=float)
        feas = np.array([x["feasible"] for x in items], dtype=bool)
        max_aodt = np.array([x["max_aodt_s"] for x in items], dtype=float)
        summary[method] = {
            "mean_Mbps": float(np.mean(rates)),
            "std_Mbps": float(np.std(rates, ddof=1)),
            "feasible_fraction": float(np.mean(feas)),
            "mean_max_AoDT_s": float(np.mean(max_aodt)),
            "per_seed": items,
        }
    return {
        "task_size_bits": cfg.task_size_bits,
        "task_size_bytes": cfg.task_size_bits / 8.0,
        "task_cycles": cfg.task_cycles,
        "mu_per_s": cfg.service_rate_per_s,
        "summary": summary,
    }


def main() -> None:
    out: dict = {"seeds": list(SEEDS), "configs": {}}
    for label in ("old", "new"):
        cfg = _cfg(label)
        print(f"=== {label}: S={cfg.task_size_bits/8:.0f} bytes, L={cfg.task_cycles:g}, mu={cfg.service_rate_per_s:.4g}/s ===")
        methods = run_methods_j3(cfg, SEEDS)
        fig11 = run_fig11(
            cfg,
            n_runs=len(SEEDS),
            seed_start=SEEDS[0],
            uav_counts=(3,),
            horizon_s=80.0,
            warmup_s=16.0,
        )
        sens = sensibility_checks(fig11)
        out["configs"][label] = {
            "methods_j3": methods,
            "fig11_j3": fig11["points"][0]["by_pattern"],
            "fig11_sensibility": {
                "n_pass": sens["n_pass"],
                "n_checks": sens["n_checks"],
                "all_ok": sens["all_ok"],
            },
        }
        for m in METHODS:
            s = methods["summary"][m]
            print(
                f"  {m:7s}  {s['mean_Mbps']:.6f} Mbps  "
                f"feas={s['feasible_fraction']:.0%}  "
                f"maxAoDT={s['mean_max_AoDT_s']:.3f}s"
            )
        print(f"  fig11 sensibility {sens['n_pass']}/{sens['n_checks']}")

    # Paired deltas new - old for SCA vs baselines
    deltas = {}
    for m in METHODS:
        old_rates = [x["mbps"] for x in out["configs"]["old"]["methods_j3"]["summary"][m]["per_seed"]]
        new_rates = [x["mbps"] for x in out["configs"]["new"]["methods_j3"]["summary"][m]["per_seed"]]
        deltas[m] = {
            "mean_delta_Mbps": float(np.mean(np.array(new_rates) - np.array(old_rates))),
            "old_mean": out["configs"]["old"]["methods_j3"]["summary"][m]["mean_Mbps"],
            "new_mean": out["configs"]["new"]["methods_j3"]["summary"][m]["mean_Mbps"],
        }
    out["paired_delta_new_minus_old_Mbps"] = deltas

    path = Path("results/compare_si_l_defaults.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
