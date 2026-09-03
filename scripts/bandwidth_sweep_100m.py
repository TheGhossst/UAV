"""Bandwidth sweep on a 100 x 100 m field.

Does not change src/config.py. Area 100 m is an advisor experimental choice
(paper Table II / config default is 500 m). B_sys is swept; everything else is
the current repo compute-on defaults.

    python -m scripts.bandwidth_sweep_100m
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.config import DEFAULT, PAPER_SCENARIO_SEEDS
from src.logutil import Counter, configure_logging, log
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.pso import solve_pso_placement
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca

OUT = Path("results") / "run_20260902_bw_sweep_100m"
SEEDS = PAPER_SCENARIO_SEEDS
METHODS = ("random", "kmeans", "pso", "sca")
# Hz. 8.8e6 is the current calibrated default, included as a reference point
# not as a paper-matching target.
B_SYS_HZ = (
    1.0e6,
    2.0e6,
    3.0e6,
    4.0e6,
    5.0e6,
    6.0e6,
    7.0e6,
    8.0e6,
    8.8e6,
)
CVX_SEEDS = (100, 109, 119)


def _base_cfg():
    return replace(
        DEFAULT.with_compute(),
        area_x=100.0,
        area_y=100.0,
        num_iot=10,
        num_uav=3,
        iots_per_process=5,
        max_bw_share=None,
        bandwidth_scope="system",
    )


def _cfg_snapshot(cfg) -> dict:
    d = asdict(cfg)
    return {
        "area_x": d["area_x"],
        "area_y": d["area_y"],
        "num_iot": d["num_iot"],
        "num_uav": d["num_uav"],
        "num_processes": d["num_processes"],
        "iots_per_process": d["iots_per_process"],
        "uav_height": d["uav_height"],
        "uav_min_distance": d["uav_min_distance"],
        "r_min": d["r_min"],
        "b_sys": d["b_sys"],
        "noise_power": d["noise_power"],
        "bandwidth_scope": d["bandwidth_scope"],
        "max_bw_share": d["max_bw_share"],
        "p_i": d["p_i"],
        "sigma": d["sigma"],
        "f_c": d["f_c"],
        "lambda_i": d["lambda_i"],
        "uav_cpu": d["uav_cpu"],
        "aodt_threshold": d["aodt_threshold"],
        "t_u2u": d["t_u2u"],
        "task_size_bytes": d["task_size_bytes"],
        "task_cycles": d["task_cycles"],
        "use_compute_model": d["use_compute_model"],
        "los_angle_unit": d["los_angle_unit"],
    }


def _row(seed, b_hz, method, result, rt) -> dict:
    return {
        "seed": seed,
        "b_sys_hz": b_hz,
        "method": method,
        "sum_rate": result.sum_rate,
        "qos": result.qos_violations,
        "aodt_viol": result.aodt_violations,
        "feasible": bool(result.feasible),
        "runtime": rt,
        "aodt_mean": float(np.nanmean(result.aodt)) if result.compute_available else None,
        "cpu_unstable": result.cpu_unstable,
    }


def _summarize(rows: list[dict]) -> list[dict]:
    out = []
    for b_hz in B_SYS_HZ:
        for method in METHODS:
            subset = [r for r in rows if r["b_sys_hz"] == b_hz and r["method"] == method]
            if not subset:
                continue
            rates = np.array([r["sum_rate"] for r in subset], dtype=float)
            feas = np.array([r["feasible"] for r in subset], dtype=float)
            qos = np.array([r["qos"] for r in subset], dtype=float)
            aodt_v = np.array([r["aodt_viol"] for r in subset], dtype=float)
            rts = np.array([r["runtime"] for r in subset], dtype=float)
            aodt = [r["aodt_mean"] for r in subset if r["aodt_mean"] is not None]
            qos_seed_frac = float(np.mean(qos > 0))
            aodt_seed_frac = float(np.mean(aodt_v > 0))
            out.append(
                {
                    "b_sys_mhz": b_hz / 1e6,
                    "b_sys_hz": b_hz,
                    "method": method,
                    "mean_mbps": float(rates.mean()) / 1e6,
                    "std_mbps": float(rates.std(ddof=1) / 1e6) if rates.size > 1 else 0.0,
                    "feasible_pct": 100.0 * float(feas.mean()),
                    "aodt_mean_s": float(np.mean(aodt)) if aodt else None,
                    "qos_mean_count": float(qos.mean()),
                    "qos_seed_pct": 100.0 * qos_seed_frac,
                    "aodt_viol_mean_count": float(aodt_v.mean()),
                    "aodt_viol_seed_pct": 100.0 * aodt_seed_frac,
                    "runtime_s": float(rts.mean()),
                    "n": len(subset),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_md(summary: list[dict], path: Path) -> None:
    lines = [
        "# Bandwidth sweep, 100 x 100 m, I=10, J=3, 20 seeds, compute on, no per-link cap",
        "",
        "Mean associated uplink sum rate in Mbps. Area 100 m is an advisor experimental choice;",
        "config.py / paper Table II still default to 500 m. This run does not change config.py.",
        "",
        "| B (MHz) | Method | Mbps | Std | Feasible % | AoDT (s) | QoS count | QoS seed % | AoDT viol count | AoDT viol seed % | Runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        aodt = "n/a" if row["aodt_mean_s"] is None else f"{row['aodt_mean_s']:.3f}"
        lines.append(
            f"| {row['b_sys_mhz']:.1f} | {row['method']} | {row['mean_mbps']:.3f} | "
            f"{row['std_mbps']:.3f} | {row['feasible_pct']:.0f} | {aodt} | "
            f"{row['qos_mean_count']:.2f} | {row['qos_seed_pct']:.0f} | "
            f"{row['aodt_viol_mean_count']:.2f} | {row['aodt_viol_seed_pct']:.0f} | "
            f"{row['runtime_s']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_highs() -> list[dict]:
    rows: list[dict] = []
    n_jobs = len(SEEDS) * len(B_SYS_HZ) * len(METHODS)
    jobs = Counter("bw-sweep-100m", n_jobs)
    for b_hz in B_SYS_HZ:
        cfg = replace(_base_cfg(), b_sys=b_hz)
        log.info("B_sys=%.1f MHz  area=%.0fx%.0f  cap=%s  compute=%s  Si=%s  L=%s",
                 b_hz / 1e6, cfg.area_x, cfg.area_y, cfg.max_bw_share,
                 cfg.use_compute_model, cfg.task_size_bytes, cfg.task_cycles)
        for seed in SEEDS:
            scenario = generate_scenario(seed, cfg)
            for name in METHODS:
                if name == "random":
                    _xy, result, rt = solve_random(scenario, seed=seed)
                elif name == "kmeans":
                    _xy, result, rt = solve_kmeans(scenario, seed=seed)
                elif name == "pso":
                    _xy, result, rt, _ = solve_pso_placement(scenario, seed=seed)
                else:
                    _xy, result, rt = solve_sca(scenario, seed=seed)
                rec = _row(seed, b_hz, name, result, rt)
                rows.append(rec)
                jobs.tick(f"B={b_hz/1e6:.1f} seed={seed} {name:8}", result, rt)
        _write_csv(OUT / "raw_highs.csv", rows)
        _write_csv(OUT / "summary_highs.csv", _summarize(rows))
    return rows


def _run_cvx(highs_summary: list[dict]) -> list[dict]:
    from src.solvers.sca_cvx import MatlabCvxSession, _convexified_lp_cvx

    sca_rows = [r for r in highs_summary if r["method"] == "sca"]
    full_feas = [r["b_sys_hz"] for r in sca_rows if r["feasible_pct"] >= 95.0]
    selected_b = []
    if sca_rows:
        selected_b.append(sca_rows[0]["b_sys_hz"])
    if full_feas:
        selected_b.append(full_feas[0])
    selected_b.append(8.8e6)
    selected_b = list(dict.fromkeys(selected_b))
    log.info("CVX validation B_sys=%s seeds=%s", selected_b, CVX_SEEDS)
    rows: list[dict] = []
    with MatlabCvxSession() as session:
        def lp(sc, q, b, a, proc, tr):
            return _convexified_lp_cvx(sc, q, b, a, proc, tr, session)

        for b_hz in selected_b:
            cfg = replace(_base_cfg(), b_sys=b_hz)
            for seed in CVX_SEEDS:
                scenario = generate_scenario(seed, cfg)
                _xy_h, res_h, rt_h = solve_sca(scenario, seed=seed)
                _xy_c, res_c, rt_c = solve_sca(
                    scenario, seed=seed, lp_solver=lp, backend="cvx-mosek/shared"
                )
                rows.append(
                    {
                        "seed": seed,
                        "b_sys_hz": b_hz,
                        "highs_mbps": res_h.sum_rate / 1e6,
                        "cvx_mbps": res_c.sum_rate / 1e6,
                        "abs_diff_mbps": abs(res_h.sum_rate - res_c.sum_rate) / 1e6,
                        "highs_feasible": bool(res_h.feasible),
                        "cvx_feasible": bool(res_c.feasible),
                        "highs_s": rt_h,
                        "cvx_s": rt_c,
                    }
                )
                log.info(
                    "CVX check B=%.1f seed=%d  HiGHS=%.3f  CVX=%.3f  d=%.4f",
                    b_hz / 1e6, seed, res_h.sum_rate / 1e6, res_c.sum_rate / 1e6,
                    abs(res_h.sum_rate - res_c.sum_rate) / 1e6,
                )
    return rows


def main() -> int:
    configure_logging("INFO")
    OUT.mkdir(parents=True, exist_ok=True)
    probe = replace(_base_cfg(), b_sys=B_SYS_HZ[0])
    snap = _cfg_snapshot(probe)
    snap["b_sys_note"] = "swept; value shown is the first grid point only"
    snap["area_source"] = "advisor experimental choice (paper/config default 500 m)"
    snap["noise_source"] = "calibrated profile: Table II label sigma=0.01 W used as Eq.(6) sigma^2"
    snap["b_sys_grid_hz"] = list(B_SYS_HZ)
    snap["S_i_source"] = "repo experimental default EXPERIMENTAL_TASK_SIZE_BYTES (not Table II)"
    snap["L_source"] = "repo experimental default EXPERIMENTAL_TASK_CYCLES (not Table II)"
    snap["max_bw_share_source"] = "config.py current truth: None (no per-link cap)"
    (OUT / "config_used.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    log.info("config snapshot written to %s", OUT / "config_used.json")
    log.info(
        "area=%.0fx%.0f I=%d J=%d compute=%s Si=%s L=%s noise=%s cap=%s",
        probe.area_x, probe.area_y, probe.num_iot, probe.num_uav,
        probe.use_compute_model, probe.task_size_bytes, probe.task_cycles,
        probe.noise_power, probe.max_bw_share,
    )

    highs_rows = _run_highs()
    summary = _summarize(highs_rows)
    _write_csv(OUT / "raw_highs.csv", highs_rows)
    _write_csv(OUT / "summary_highs.csv", summary)
    _write_md(summary, OUT / "table_highs.md")

    try:
        cvx_rows = _run_cvx(summary)
        _write_csv(OUT / "cvx_validation.csv", cvx_rows)
    except Exception as exc:
        log.warning("CVX validation skipped: %s", exc)

    log.info("wrote %s", OUT / "table_highs.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
