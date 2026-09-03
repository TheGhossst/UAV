"""I=10 J=3 radio-stack comparison: 20 kHz, 2.4 MHz, 8.8 MHz, 8.8 MHz+25% cap.

20 paper seeds. Methods: random, k-means, placement PSO, SCA/HiGHS, SCA/CVX+MOSEK.
Compute/AoDT is on (same protocol as --mode compare --paper-runs --compute).

    python -m scripts.radio_stack_compare
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import DEFAULT, PAPER_SCENARIO_SEEDS, SIGMA
from src.logutil import Counter, configure_logging, log
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.pso import solve_pso_placement
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca
from src.solvers.sca_cvx import MatlabCvxSession, _convexified_lp_cvx

OUT = Path("results") / "run_20260902_radio_compare"
SEEDS = PAPER_SCENARIO_SEEDS
METHODS = ("random", "kmeans", "pso", "sca", "sca_cvx")

# Paper Fig. 6 at J=3 (figure reads; approximate).
PAPER_FIG6_J3_MBPS = {
    "random": 3.0,
    "kmeans": 4.0,
    "pso": None,
    "sca": 7.1,
    "sca_cvx": 7.1,
}

RADIOS = (
    {
        "id": "20khz_table2",
        "label": "20 kHz (paper Table II)",
        "b_sys": 20_000.0,
        "noise_power": SIGMA**2,
        "max_bw_share": None,
    },
    {
        "id": "2p4mhz_reverse",
        "label": "2.4 MHz (reverse-engineered B)",
        "b_sys": 2.4e6,
        "noise_power": SIGMA,
        "max_bw_share": None,
    },
    {
        "id": "8p8mhz_impl",
        "label": "8.8 MHz (implementation)",
        "b_sys": 8.8e6,
        "noise_power": SIGMA,
        "max_bw_share": None,
    },
    {
        "id": "8p8mhz_cap25",
        "label": "8.8 MHz (25% per-link cap)",
        "b_sys": 8.8e6,
        "noise_power": SIGMA,
        "max_bw_share": 0.25,
    },
)


def _cfg(radio: dict):
    return replace(
        DEFAULT.with_compute(),
        num_iot=10,
        num_uav=3,
        iots_per_process=5,
        b_sys=radio["b_sys"],
        noise_power=radio["noise_power"],
        max_bw_share=radio["max_bw_share"],
        bandwidth_scope="system",
    )


def _row(seed, radio, method, result, rt) -> dict:
    return {
        "seed": seed,
        "radio": radio["id"],
        "radio_label": radio["label"],
        "b_sys_hz": radio["b_sys"],
        "noise_power": radio["noise_power"],
        "max_bw_share": radio["max_bw_share"],
        "method": method,
        "sum_rate": result.sum_rate,
        "min_rate": result.min_assoc_rate,
        "qos": result.qos_violations,
        "feasible": bool(result.feasible),
        "runtime": rt,
        "aodt_mean": float(np.nanmean(result.aodt)) if result.compute_available else None,
        "aodt_viol": result.aodt_violations,
    }


def _summarize(rows: list[dict]) -> list[dict]:
    out = []
    keys = {(r["radio"], r["method"]) for r in rows}
    for radio_id, method in sorted(keys):
        subset = [r for r in rows if r["radio"] == radio_id and r["method"] == method]
        rates = np.array([r["sum_rate"] for r in subset], dtype=float)
        feas = np.array([r["feasible"] for r in subset], dtype=float)
        qos = np.array([r["qos"] for r in subset], dtype=float)
        rts = np.array([r["runtime"] for r in subset], dtype=float)
        aodt = [r["aodt_mean"] for r in subset if r["aodt_mean"] is not None]
        paper = PAPER_FIG6_J3_MBPS.get(method)
        out.append(
            {
                "radio": radio_id,
                "radio_label": subset[0]["radio_label"],
                "b_sys_hz": subset[0]["b_sys_hz"],
                "max_bw_share": subset[0]["max_bw_share"],
                "method": method,
                "repo_mbps": float(rates.mean()) / 1e6,
                "std_mbps": float(rates.std(ddof=1) / 1e6) if rates.size > 1 else 0.0,
                "paper_mbps": paper,
                "feasible_frac": float(feas.mean()),
                "qos_mean": float(qos.mean()),
                "aodt_mean": float(np.mean(aodt)) if aodt else None,
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


def _write_tables(summary: list[dict], path: Path) -> None:
    method_order = list(METHODS)
    method_header = {
        "random": "Random",
        "kmeans": "K-means",
        "pso": "PSO",
        "sca": "SCA (HiGHS)",
        "sca_cvx": "SCA (CVX+MOSEK)",
    }
    lines = [
        "# Radio-stack comparison, I=10, J=3, 20 seeds, `--compute`",
        "",
        "Mean associated uplink sum rate in **Mbps**. Paper column is IEEE Fig. 6 at J=3 (approximate figure read).",
        "",
        "Noise: `20 kHz` uses Table II literal `σ² = 10⁻⁴ W`. The other three use Table II’s labelled noise power `σ = 0.01 W` as Eq. (6)’s `σ²`.",
        "",
        "## Sum rate (Mbps)",
        "",
        "| Radio | Random | K-means | PSO | SCA (HiGHS) | SCA (CVX+MOSEK) | Paper SCA | Paper KM | Paper Random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_radio = {}
    for row in summary:
        by_radio.setdefault(row["radio"], {})[row["method"]] = row
    for radio in RADIOS:
        cells = by_radio.get(radio["id"], {})
        def mb(m):
            r = cells.get(m)
            return f"{r['repo_mbps']:.3f}" if r else "—"
        lines.append(
            f"| {radio['label']} | {mb('random')} | {mb('kmeans')} | {mb('pso')} | "
            f"{mb('sca')} | {mb('sca_cvx')} | ~7.1 | ~4.0 | ~3.0 |"
        )
    lines += [
        "",
        "## Feasible fraction / AoDT mean (s) / runtime (s)",
        "",
        "| Radio | Method | Mbps | Std | Feasible | AoDT (s) | QoS | Runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for radio in RADIOS:
        for method in method_order:
            r = by_radio.get(radio["id"], {}).get(method)
            if not r:
                continue
            aodt = "—" if r["aodt_mean"] is None else f"{r['aodt_mean']:.3f}"
            lines.append(
                f"| {radio['label']} | {method_header[method]} | {r['repo_mbps']:.3f} | "
                f"{r['std_mbps']:.3f} | {r['feasible_frac']:.2f} | {aodt} | "
                f"{r['qos_mean']:.2f} | {r['runtime_s']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_logging("INFO")
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_jobs = len(SEEDS) * len(RADIOS) * len(METHODS)
    jobs = Counter("radio-compare", n_jobs)

    log.info("I=10 J=3  seeds=%d  radios=%d  methods=%s", len(SEEDS), len(RADIOS), ",".join(METHODS))
    log.info("output %s", OUT.resolve())

    with MatlabCvxSession() as session:
        def sca_cvx_lp(sc, q, b, a, proc, tr):
            return _convexified_lp_cvx(sc, q, b, a, proc, tr, session)

        for radio in RADIOS:
            cfg = _cfg(radio)
            log.info(
                "radio %s  B=%g Hz  N=%g  cap=%s",
                radio["label"],
                cfg.b_sys,
                cfg.noise_power,
                cfg.max_bw_share,
            )
            for seed in SEEDS:
                scenario = generate_scenario(seed, cfg)
                for name in METHODS:
                    if name == "random":
                        _xy, result, rt = solve_random(scenario, seed=seed)
                    elif name == "kmeans":
                        _xy, result, rt = solve_kmeans(scenario, seed=seed)
                    elif name == "pso":
                        _xy, result, rt, _ = solve_pso_placement(scenario, seed=seed)
                    elif name == "sca":
                        _xy, result, rt = solve_sca(scenario, seed=seed)
                    else:
                        _xy, result, rt = solve_sca(
                            scenario,
                            seed=seed,
                            lp_solver=sca_cvx_lp,
                            backend="cvx-mosek/shared",
                        )
                    rec = _row(seed, radio, name, result, rt)
                    rows.append(rec)
                    jobs.tick(f"{radio['id']} seed={seed} {name:8}", result, rt)
            _write_csv(OUT / "raw.csv", rows)
            _write_csv(OUT / "summary.csv", _summarize(rows))

    summary = _summarize(rows)
    _write_csv(OUT / "raw.csv", rows)
    _write_csv(OUT / "summary.csv", summary)
    _write_tables(summary, OUT / "comparison_table.md")
    meta = {
        "I": 10,
        "J": 3,
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "compute": True,
        "methods": list(METHODS),
        "radios": RADIOS,
        "paper_fig6_j3_mbps": PAPER_FIG6_J3_MBPS,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s", OUT / "comparison_table.md")
    print((OUT / "comparison_table.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
