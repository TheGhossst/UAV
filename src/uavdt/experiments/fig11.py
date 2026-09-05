"""Fig. 11: uniform-fast / uniform-slow / heterogeneous λ within a group.

PAPER: I=10, J varies, two groups of five. Uniform fast (λ1=2, λ2=3),
uniform slow (λ1=0.8, λ2=1), heterogeneous (0.8 ≤ λ ≤ 3 within each
group). Used to inspect the AoDT expression, not as a Mbps target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uavdt.aodt_sim import DISCIPLINES, compare_closed_and_sim
from uavdt.config import SimConfig
from uavdt.evaluator import evaluate
from uavdt.experiments.grids import UAV_COUNTS, config_for_counts
from uavdt.placement.kmeans import place_kmeans
from uavdt.scenario import generate_scenario

LAMBDA_PATTERNS = ("uniform_fast", "uniform_slow", "heterogeneous")

# PAPER Fig. 11 captions.
UNIFORM_FAST = (2.0, 3.0)
UNIFORM_SLOW = (0.8, 1.0)
HETERO_LO = 0.8
HETERO_HI = 3.0


def lambdas_for_pattern(cfg: SimConfig, pattern: str) -> np.ndarray:
    """Per-IoT λ vector for a Fig. 11 arrival-rate pattern."""
    name = pattern.strip().lower()
    n = cfg.num_iot
    n_per = cfg.iots_per_process
    out = np.zeros(n, dtype=float)
    if name == "uniform_fast":
        group_rates = UNIFORM_FAST
    elif name == "uniform_slow":
        group_rates = UNIFORM_SLOW
    elif name == "heterogeneous":
        group_rates = None
    else:
        raise ValueError(f"unknown lambda pattern {pattern!r}")

    for k in range(cfg.num_processes):
        sl = slice(k * n_per, (k + 1) * n_per)
        if name == "heterogeneous":
            out[sl] = np.linspace(HETERO_LO, HETERO_HI, n_per)
        else:
            out[sl] = float(group_rates[k] if k < len(group_rates) else group_rates[-1])
    return out


def run_fig11(
    cfg: SimConfig | None = None,
    *,
    n_runs: int = 20,
    seed_start: int = 1,
    uav_counts: tuple[int, ...] = UAV_COUNTS,
    horizon_s: float = 80.0,
    warmup_s: float = 16.0,
    disciplines: tuple[str, ...] = DISCIPLINES,
) -> dict:
    """K-means placement, same IoT geometry across λ patterns for each seed."""
    base = cfg or SimConfig()
    points = []
    for j in uav_counts:
        point_cfg = config_for_counts(10, int(j), base)
        by_pattern: dict[str, dict] = {}
        for pattern in LAMBDA_PATTERNS:
            eq17 = []
            eq15 = []
            fcfs_c = []
            sim_max = {d: [] for d in disciplines}
            sim_mean_src = {d: [] for d in disciplines}
            feas = []
            rates = []
            for offset in range(n_runs):
                seed = seed_start + offset
                print(
                    f"  Fig11 J={j}  {pattern}  seed={seed}",
                    flush=True,
                )
                lam = lambdas_for_pattern(point_cfg, pattern)
                sc = generate_scenario(seed, point_cfg, lambdas_per_s=lam)
                uav = place_kmeans(sc, seed)
                ev = evaluate(sc, uav)
                a = ev.extras["association"]
                b = ev.extras["processing"]
                mu_vec = np.full(uav.shape[0], ev.mu_per_s)
                cmp = compare_closed_and_sim(
                    sc,
                    a,
                    b,
                    ev.rates_bit_per_s,
                    mu_vec,
                    disciplines=disciplines,
                    horizon_s=horizon_s,
                    warmup_s=warmup_s,
                    seed=seed,
                )
                eq17.append(cmp["eq17_max_s"])
                eq15.append(cmp["eq15_max_s"])
                fcfs_c.append(cmp["fcfs_closed_max_s"])
                feas.append(bool(ev.feasible))
                rates.append(float(ev.sum_rate_mbps))
                for d in disciplines:
                    sim_max[d].append(cmp["sim"][d]["mean_max_process_age_s"])
                    src = cmp["sim"][d]["mean_source_age_s"]
                    flat = [x for row in src for x in row]
                    sim_mean_src[d].append(float(np.mean(flat)) if flat else np.inf)
            by_pattern[pattern] = {
                "mean_eq17_max_s": float(np.mean(eq17)),
                "std_eq17_max_s": float(np.std(eq17, ddof=1)) if n_runs > 1 else 0.0,
                "mean_eq15_max_s": float(np.mean(eq15)),
                "mean_fcfs_closed_max_s": float(np.mean(fcfs_c)),
                "mean_sum_rate_Mbps": float(np.mean(rates)),
                "feasible_fraction": float(np.mean(feas)),
                "sim_mean_max_process_age_s": {
                    d: float(np.mean(sim_max[d])) for d in disciplines
                },
                "sim_mean_source_age_s": {
                    d: float(np.mean(sim_mean_src[d])) for d in disciplines
                },
                "per_seed_eq17_max_s": [float(x) for x in eq17],
            }
        points.append(
            {
                "num_uav": int(j),
                "label": f"PAPER Fig. 11  I=10  J={j}",
                "by_pattern": by_pattern,
            }
        )
    return {
        "paper": "Khalaf et al. IEEE TNSM 2026 Fig. 11",
        "n_runs": n_runs,
        "seed_start": seed_start,
        "placement": "kmeans",
        "horizon_s": horizon_s,
        "warmup_s": warmup_s,
        "disciplines": list(disciplines),
        "b_sys_hz": base.b_sys_hz,
        "max_bw_share": base.max_bw_share,
        "area_m": [base.area_x_m, base.area_y_m],
        "patterns": {
            "uniform_fast": {"N1": UNIFORM_FAST[0], "N2": UNIFORM_FAST[1]},
            "uniform_slow": {"N1": UNIFORM_SLOW[0], "N2": UNIFORM_SLOW[1]},
            "heterogeneous": {
                "within_group": f"linspace({HETERO_LO}, {HETERO_HI}, |N_k|)"
            },
        },
        "note": (
            "Fig. 11 is an AoDT-expression check, not a Mbps target. "
            "Same IoT coordinates across λ patterns for a given seed. "
            "Eq. (17) remains the Problem (P) score. Simulator is extra."
        ),
        "points": points,
    }


def write_fig11(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    rows = []
    for pt in payload["points"]:
        for pattern, stats in pt["by_pattern"].items():
            row = {
                "num_uav": pt["num_uav"],
                "pattern": pattern,
                "mean_eq17_max_s": stats["mean_eq17_max_s"],
                "mean_eq15_max_s": stats["mean_eq15_max_s"],
                "mean_fcfs_closed_max_s": stats["mean_fcfs_closed_max_s"],
                "mean_sum_rate_Mbps": stats["mean_sum_rate_Mbps"],
                "feasible_fraction": stats["feasible_fraction"],
            }
            for d, val in stats["sim_mean_max_process_age_s"].items():
                row[f"sim_{d}_max_s"] = val
            for d, val in stats["sim_mean_source_age_s"].items():
                row[f"sim_{d}_mean_src_s"] = val
            rows.append(row)
    if rows:
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path


def sensibility_checks(payload: dict) -> dict:
    """Qualitative gates for Fig. 11 / FCFS vs LCFS-S. Not paper Mbps."""
    checks = []
    for pt in payload["points"]:
        bp = pt["by_pattern"]
        j = pt["num_uav"]
        fast = bp["uniform_fast"]["mean_eq17_max_s"]
        slow = bp["uniform_slow"]["mean_eq17_max_s"]
        hetero = bp["heterogeneous"]["mean_eq17_max_s"]
        checks.append(
            {
                "name": f"J={j} eq17 fast < slow",
                "ok": fast < slow - 1e-6,
                "fast": fast,
                "slow": slow,
                "hetero": hetero,
            }
        )
        checks.append(
            {
                "name": f"J={j} eq17 fast < hetero",
                "ok": fast < hetero - 1e-6,
                "fast": fast,
                "hetero": hetero,
            }
        )
        # Slowest-limited: hetero min λ = 0.8, same as uniform slow N1,
        # so Eq. (17) hetero sits near slow, not near fast.
        checks.append(
            {
                "name": f"J={j} eq17 hetero closer to slow than to fast",
                "ok": abs(hetero - slow) < abs(hetero - fast),
                "fast": fast,
                "slow": slow,
                "hetero": hetero,
            }
        )
        src_fast = bp["uniform_fast"]["sim_mean_source_age_s"]["fcfs"]
        src_slow = bp["uniform_slow"]["sim_mean_source_age_s"]["fcfs"]
        src_het = bp["heterogeneous"]["sim_mean_source_age_s"]["fcfs"]
        checks.append(
            {
                "name": f"J={j} sim mean-source fast < hetero < slow",
                "ok": src_fast < src_het < src_slow,
                "fast": src_fast,
                "hetero": src_het,
                "slow": src_slow,
            }
        )
        mx_fast = bp["uniform_fast"]["sim_mean_max_process_age_s"]["fcfs"]
        mx_slow = bp["uniform_slow"]["sim_mean_max_process_age_s"]["fcfs"]
        mx_het = bp["heterogeneous"]["sim_mean_max_process_age_s"]["fcfs"]
        checks.append(
            {
                "name": f"J={j} sim process-max fast < hetero < slow",
                "ok": mx_fast < mx_het < mx_slow,
                "fast": mx_fast,
                "hetero": mx_het,
                "slow": mx_slow,
            }
        )
        for d_lo, d_hi in (("lcfs_s", "fcfs_p"), ("fcfs_p", "fcfs")):
            lo = bp["uniform_fast"]["sim_mean_source_age_s"][d_lo]
            hi = bp["uniform_fast"]["sim_mean_source_age_s"][d_hi]
            checks.append(
                {
                    "name": f"J={j} fast mean-source {d_lo} finite and ~ {d_hi}",
                    "ok": np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi > 0.0,
                    "lo": lo,
                    "hi": hi,
                }
            )
        # Process-max age is E[max_i ζ_i], larger than a single-source closed form.
        sim_max = bp["uniform_fast"]["sim_mean_max_process_age_s"]["fcfs"]
        checks.append(
            {
                "name": f"J={j} sim process-max > eq17 (max of group ages)",
                "ok": sim_max > fast - 0.05,
                "sim_max": sim_max,
                "eq17": fast,
            }
        )
        eq15 = bp["uniform_fast"]["mean_eq15_max_s"]
        eq17 = fast
        fcfs_c = bp["uniform_fast"]["mean_fcfs_closed_max_s"]
        checks.append(
            {
                "name": f"J={j} eq17 >= eq15 (extra sum-lambda load)",
                "ok": eq17 + 1e-9 >= eq15,
                "eq15": eq15,
                "eq17": eq17,
            }
        )
        checks.append(
            {
                "name": f"J={j} FCFS closed > LCFS-S eq15",
                "ok": fcfs_c > eq15 + 1e-9,
                "eq15": eq15,
                "fcfs_closed": fcfs_c,
            }
        )
    return {
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["ok"]),
        "all_ok": all(c["ok"] for c in checks),
        "checks": checks,
    }
