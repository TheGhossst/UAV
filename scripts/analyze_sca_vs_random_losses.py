"""Characterize SCA-vs-random losses and map the advantage across unique axes.

Uses campaign JSON for paired rates; re-runs the default J=3 I=10 scenario
for losing (and a few winning) seeds to inspect geometry / LP vs equal-share.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paired_winrate import (  # noqa: E402
    analyze_campaign,
    compare_pair,
    fmt_p,
    student_t_sf_two_sided,
    wilcoxon_signed_rank,
    bh_qvalues,
)
from uavdt.config import SimConfig  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.resources import equal_share_bandwidth_hz  # noqa: E402
from uavdt.evaluator import evaluate  # noqa: E402
from uavdt.models import Allocation  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402
from uavdt.sca.linearize import spectral_efficiency  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402


def bonferroni(p: float, m: int) -> float:
    return min(1.0, float(p) * m)


def unique_rows(rows: list[dict], baseline: str) -> list[dict]:
    return [
        r
        for r in rows
        if r["baseline"] == baseline and not r.get("repeated_default")
    ]


def point_from_campaign(payload: dict, axis: str, x: float) -> dict:
    for pt in payload["points"]:
        if pt["axis"] == axis and abs(float(pt["x"]) - x) < 1e-9:
            return pt
    raise KeyError(f"{axis}={x}")


def seed_table(pt: dict, champ: str, baseline: str) -> list[dict]:
    c = pt["by_method"][champ]
    b = pt["by_method"][baseline]
    cmap = {int(s): float(r) for s, r in zip(c["seeds"], c["per_seed_Mbps"])}
    bmap = {int(s): float(r) for s, r in zip(b["seeds"], b["per_seed_Mbps"])}
    cdiag = {int(s): d for s, d in zip(c["seeds"], c.get("diagnostics", [{}] * len(c["seeds"])))}
    rows = []
    for s in sorted(set(cmap) & set(bmap)):
        d = cmap[s] - bmap[s]
        rows.append(
            {
                "seed": s,
                "sca_Mbps": cmap[s],
                "random_Mbps": bmap[s],
                "delta_Mbps": d,
                "sca_diag": cdiag.get(s, {}),
            }
        )
    return sorted(rows, key=lambda r: r["delta_Mbps"])


def geom_features(run) -> dict:
    sc = run  # MethodRun
    uav = sc.uav_xyz_m
    alloc = sc.allocation
    ev = sc.true_eval
    return {
        "uav_xy": uav[:, :2].tolist(),
        "sum_rate_Mbps": sc.sum_rate_mbps,
        "feasible": sc.feasible,
        "max_B_Hz": float(alloc.bandwidth_hz.max()),
        "n_assoc": int((alloc.association > 0.5).sum()),
        "min_assoc_dist_m": float(
            ev.distances_m[alloc.association > 0.5].min()
            if np.any(alloc.association > 0.5)
            else float("nan")
        ),
        "mean_assoc_dist_m": float(
            ev.distances_m[alloc.association > 0.5].mean()
            if np.any(alloc.association > 0.5)
            else float("nan")
        ),
        "max_assoc_snr": float(
            ev.snr[alloc.association > 0.5].max()
            if np.any(alloc.association > 0.5)
            else float("nan")
        ),
        "aodt_max_s": float(np.max(ev.aodt_s)),
        "diagnostics": sc.diagnostics,
    }


def lp_vs_equal(scenario, uav, alloc) -> dict:
    ev_lp = evaluate(scenario, uav, alloc)
    bw_eq = equal_share_bandwidth_hz(alloc.association, scenario.cfg)
    alloc_eq = Allocation(alloc.association, alloc.processing, bw_eq)
    ev_eq = evaluate(scenario, uav, alloc_eq)
    cap = scenario.cfg.link_bandwidth_cap_hz
    near = float(np.sum((alloc.bandwidth_hz > 0.99 * cap) & (alloc.association > 0.5)))
    se = spectral_efficiency(scenario.iot_xyz_m, uav, scenario.cfg)
    assoc = alloc.association > 0.5
    return {
        "lp_Mbps": float(ev_lp.sum_rate_mbps),
        "equal_share_Mbps": float(ev_eq.sum_rate_mbps),
        "lp_minus_equal_Mbps": float(ev_lp.sum_rate_mbps - ev_eq.sum_rate_mbps),
        "n_links_near_cap": int(near),
        "max_assoc_SE": float(se[assoc].max()) if np.any(assoc) else float("nan"),
        "mean_assoc_SE": float(se[assoc].mean()) if np.any(assoc) else float("nan"),
    }


def iot_layout(scenario) -> dict:
    xy = scenario.iot_xyz_m[:, :2]
    return {
        "std_x_m": float(xy[:, 0].std()),
        "std_y_m": float(xy[:, 1].std()),
        "centroid_xy": [float(xy[:, 0].mean()), float(xy[:, 1].mean())],
        "min_xy": [float(xy[:, 0].min()), float(xy[:, 1].min())],
        "max_xy": [float(xy[:, 0].max()), float(xy[:, 1].max())],
        "xy": xy.tolist(),
    }


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/campaign_20260904_cap25.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    analyzed = analyze_campaign(payload, champion="sca")
    rows = analyzed["rows"]

    j3 = point_from_campaign(payload, "uavs", 3.0)
    seeds = seed_table(j3, "sca", "random")
    losses = [s for s in seeds if s["delta_Mbps"] < -1e-9]
    wins = [s for s in seeds if s["delta_Mbps"] > 1e-9]
    deltas = np.array([s["delta_Mbps"] for s in seeds], dtype=float)
    wx = wilcoxon_signed_rank(deltas)
    mean = float(deltas.mean())
    std = float(deltas.std(ddof=1))
    sem = std / math.sqrt(deltas.size)
    t = mean / sem
    p_t = student_t_sf_two_sided(t, deltas.size - 1)

    # Bonferroni over 3 baselines at J=3
    j3_vs = [r for r in rows if r["axis"] == "uavs" and r["x"] == 3.0]
    m_base = len(j3_vs)
    j3_corr = []
    for r in j3_vs:
        j3_corr.append(
            {
                "baseline": r["baseline"],
                "mean_delta_Mbps": r["mean_delta_Mbps"],
                "std_delta_Mbps": r["std_delta_Mbps"],
                "wins": r["wins"],
                "n": r["n_seeds"],
                "wilcoxon_p": r["wilcoxon_p_two_sided"],
                "t_p": r["t_p_two_sided"],
                "bonferroni_p": bonferroni(r["wilcoxon_p_two_sided"], m_base),
                "quote": r["quote"],
            }
        )

    # Regime map: unique SCA-vs-random, BH-FDR
    rand_unique = unique_rows(rows, "random")
    q_rand = bh_qvalues([r["wilcoxon_p_two_sided"] for r in rand_unique])
    regime = []
    for r, q in zip(rand_unique, q_rand):
        regime.append(
            {
                "axis": r["axis"],
                "x": r["x"],
                "label": r["label"],
                "mean_delta_Mbps": r["mean_delta_Mbps"],
                "std_delta_Mbps": r["std_delta_Mbps"],
                "wins": r["wins"],
                "n": r["n_seeds"],
                "wilcoxon_p": r["wilcoxon_p_two_sided"],
                "t_p": r["t_p_two_sided"],
                "bh_q": q,
                "sig_raw": r["wilcoxon_p_two_sided"] < 0.05,
                "sig_fdr": q < 0.05,
            }
        )

    # Also FDR on unique SCA-vs-kmeans and vs-pso for the structural claim
    km_unique = unique_rows(rows, "kmeans")
    pso_unique = unique_rows(rows, "pso")
    q_km = bh_qvalues([r["wilcoxon_p_two_sided"] for r in km_unique])
    q_pso = bh_qvalues([r["wilcoxon_p_two_sided"] for r in pso_unique])

    cfg = config_for_counts(
        10,
        3,
        SimConfig(b_sys_hz=8_800_000.0, max_bw_share=0.25),
    )
    inspect_seeds = [s["seed"] for s in losses] + [
        s["seed"] for s in sorted(wins, key=lambda x: -x["delta_Mbps"])[:3]
    ]
    seen: set[int] = set()
    inspect_seeds = [s for s in inspect_seeds if not (s in seen or seen.add(s))]

    sca_settings = SCASettings(solver=None, max_iterations=30)
    inspected = []
    for seed in inspect_seeds:
        sc = generate_scenario(seed, cfg)
        layout = iot_layout(sc)
        methods = {}
        for name in ("random", "kmeans", "pso", "sca"):
            run = run_method(sc, name, seed, sca_settings=sca_settings)
            g = geom_features(run)
            extra = lp_vs_equal(sc, run.uav_xyz_m, run.allocation)
            methods[name] = {**g, **extra}
        inspected.append(
            {
                "seed": seed,
                "kind": "loss" if seed in {s["seed"] for s in losses} else "win",
                "delta_sca_minus_random": methods["sca"]["sum_rate_Mbps"]
                - methods["random"]["sum_rate_Mbps"],
                "iot": {k: layout[k] for k in layout if k != "xy"},
                "iot_xy": layout["xy"],
                "methods": methods,
            }
        )

    # All 20 seeds: equal-share vs LP gap (no SCA — campaign JSON already has it)
    proxy: dict[str, list] = {m: [] for m in ("random", "kmeans", "pso")}
    for seed in range(1, 21):
        sc = generate_scenario(seed, cfg)
        for name in ("random", "kmeans", "pso"):
            run = run_method(sc, name, seed)
            extra = lp_vs_equal(sc, run.uav_xyz_m, run.allocation)
            proxy[name].append({"seed": seed, **extra})

    sca_vs_kmeans_same_seed = []
    sca_m = j3["by_method"]["sca"]
    km_m = j3["by_method"]["kmeans"]
    sca_map = {int(s): float(r) for s, r in zip(sca_m["seeds"], sca_m["per_seed_Mbps"])}
    km_map = {int(s): float(r) for s, r in zip(km_m["seeds"], km_m["per_seed_Mbps"])}
    sca_diag = {
        int(s): d
        for s, d in zip(sca_m["seeds"], sca_m.get("diagnostics", [{}] * 20))
    }
    for seed in sorted(set(sca_map) & set(km_map)):
        sca_vs_kmeans_same_seed.append(
            {
                "seed": seed,
                "sca_minus_kmeans_Mbps": sca_map[seed] - km_map[seed],
                "sca_accepted": sca_diag.get(seed, {}).get("accepted_steps"),
                "sca_stop": sca_diag.get(seed, {}).get("stop_reason"),
            }
        )

    proxy_summary = {}
    for name, lst in proxy.items():
        lp = np.array([x["lp_Mbps"] for x in lst])
        eq = np.array([x["equal_share_Mbps"] for x in lst])
        proxy_summary[name] = {
            "mean_lp_Mbps": float(lp.mean()),
            "mean_equal_Mbps": float(eq.mean()),
            "mean_lp_minus_equal_Mbps": float((lp - eq).mean()),
        }

    out = {
        "file": str(path),
        "j3_sca_vs_random": {
            "mean_delta_Mbps": mean,
            "std_delta_Mbps": std,
            "wilcoxon_p": wx["p_two_sided"],
            "t_p": p_t,
            "t_stat": t,
            "wins": int((deltas > 1e-9).sum()),
            "losses": int((deltas < -1e-9).sum()),
            "n": int(deltas.size),
            "loss_seeds": [s["seed"] for s in losses],
            "loss_deltas_Mbps": [s["delta_Mbps"] for s in losses],
            "min_delta_Mbps": float(deltas.min()),
            "max_delta_Mbps": float(deltas.max()),
            "skew": float(
                ((deltas - mean) ** 3).mean() / (std**3) if std else 0.0
            ),
            "all_seeds": seeds,
        },
        "j3_three_baselines_bonferroni_m3": j3_corr,
        "regime_sca_vs_random_unique": regime,
        "regime_kmeans_fdr_sig_count": int(sum(q < 0.05 for q in q_km)),
        "regime_pso_fdr_sig_count": int(sum(q < 0.05 for q in q_pso)),
        "regime_kmeans": [
            {
                "axis": r["axis"],
                "x": r["x"],
                "mean_delta_Mbps": r["mean_delta_Mbps"],
                "wins": r["wins"],
                "n": r["n_seeds"],
                "wilcoxon_p": r["wilcoxon_p_two_sided"],
                "bh_q": q,
            }
            for r, q in zip(km_unique, q_km)
        ],
        "regime_pso": [
            {
                "axis": r["axis"],
                "x": r["x"],
                "mean_delta_Mbps": r["mean_delta_Mbps"],
                "wins": r["wins"],
                "n": r["n_seeds"],
                "wilcoxon_p": r["wilcoxon_p_two_sided"],
                "bh_q": q,
            }
            for r, q in zip(pso_unique, q_pso)
        ],
        "inspected_seeds": inspected,
        "proxy_mismatch_j3": proxy_summary,
        "sca_vs_kmeans_all20": sca_vs_kmeans_same_seed,
        "note": (
            "SCA initializes from k-means (initialize_sca), not from random. "
            "PSO inner fitness is equal-share B; campaign score is frozen-q LP."
        ),
    }
    dest = path.with_name(path.stem + "_losses.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")

    print()
    print("=== J=3 SCA vs random deltas ===")
    print(
        f"mean={mean:+.4f} std={std:.4f} wilcox {fmt_p(wx['p_two_sided'])} "
        f"t {fmt_p(p_t)}  {int((deltas>1e-9).sum())}/20 wins"
    )
    print("losses:")
    for s in losses:
        print(
            f"  seed {s['seed']:2d}  d={s['delta_Mbps']:+.4f}  "
            f"SCA={s['sca_Mbps']:.4f}  rnd={s['random_Mbps']:.4f}  "
            f"stop={s['sca_diag'].get('stop_reason')}  "
            f"acc={s['sca_diag'].get('accepted_steps')}"
        )
    print("Bonferroni m=3 at J=3:")
    for r in j3_corr:
        print(
            f"  vs {r['baseline']:8}  wilcox={r['wilcoxon_p']:.4g}  "
            f"bonf={r['bonferroni_p']:.4g}  {r['wins']}/{r['n']}"
        )
    print()
    print("=== unique SCA-vs-random (BH-FDR) ===")
    for r in regime:
        mark = "FDR*" if r["sig_fdr"] else ("raw*" if r["sig_raw"] else "    ")
        print(
            f"  {mark} {r['axis']:6} x={r['x']:<8g}  "
            f"d={r['mean_delta_Mbps']:+.4f}  W={r['wins']}/{r['n']}  "
            f"p={r['wilcoxon_p']:.4g}  q={r['bh_q']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
