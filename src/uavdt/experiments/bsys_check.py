"""Table II 20 kHz check: model-free ceiling and paper-area control.

Eq. (6) + constraint (27) give R_sum ≤ B_sys · log2(1 + SNR_max) for any
path-loss, power, or noise values. This module also evaluates the written
model at 100×100 m and at the paper's 500×500 m field so a critic cannot
attribute infeasibility to the area modification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uavdt.channel import (
    link_metrics,
    per_link_hz_for_target_rate,
    sum_rate_ceiling_bit_per_s,
)
from uavdt.config import PAPER_AREA_X_M, PAPER_B_SYS_HZ, SimConfig
from uavdt.experiments.methods import run_method
from uavdt.scenario import generate_scenario
from uavdt.sca.settings import SCASettings

GENEROUS_SNR_MAX = 1.0e15
DEFAULT_AREAS_M = (100.0, PAPER_AREA_X_M)
DEFAULT_METHODS = ("random", "kmeans")

# Published §VII numbers (paper text, not our simulator).
PAPER_RATE_ANCHORS = (
    {
        "fig": "6",
        "quote": "SCA ~8.8 Mbps at five UAVs",
        "n_links": 10,
        "mbps": 8.8,
    },
    {
        "fig": "7",
        "quote": "SCA >~14 Mbps at 32 IoTs",
        "n_links": 32,
        "mbps": 14.0,
    },
    {
        "fig": "6-10 band",
        "quote": "7 Mbps low end of the I=10 plots",
        "n_links": 10,
        "mbps": 7.0,
    },
)


def model_free_ceilings_bps(b_sys_hz: float = PAPER_B_SYS_HZ) -> dict:
    """Shannon ceilings that do not use the channel model or the simulator."""
    snr_grid = (1.0, 1.0e3, 1.0e6, GENEROUS_SNR_MAX)
    rows = []
    for snr in snr_grid:
        bps = sum_rate_ceiling_bit_per_s(b_sys_hz, snr)
        rows.append(
            {
                "snr_max": snr,
                "spectral_efficiency_bit_per_s_per_hz": float(np.log2(1.0 + snr)),
                "ceiling_bit_per_s": bps,
                "ceiling_Mbps": bps / 1.0e6,
            }
        )
    generous = rows[-1]
    return {
        "b_sys_hz": b_sys_hz,
        "formula": "R_sum <= B_sys * log2(1 + SNR_max)",
        "from": "Eq. (6) and constraint (27)",
        "area_independent": True,
        "channel_constants_independent": True,
        "generous_snr_max": GENEROUS_SNR_MAX,
        "generous_ceiling_Mbps": generous["ceiling_Mbps"],
        "grid": rows,
    }


def reading_b_magnitude(
    snr: float,
    b_floor_hz: float = PAPER_B_SYS_HZ,
    anchors: tuple[dict, ...] = PAPER_RATE_ANCHORS,
) -> dict:
    """Per-link B_i that would make published Mbps reachable under Reading B.

    Reading B: Table II 20 kHz is a per-link floor, not the (27) cap.
    Equal split across ``n_links`` associated devices, SNR held fixed
    (best-case = observed/max SNR; generous = 1e15).
    """
    se = float(np.log2(1.0 + snr))
    rows = []
    for a in anchors:
        b_i = per_link_hz_for_target_rate(a["mbps"] * 1.0e6, a["n_links"], snr)
        rows.append(
            {
                "fig": a["fig"],
                "quote": a["quote"],
                "n_links": a["n_links"],
                "target_Mbps": a["mbps"],
                "snr": snr,
                "se_bit_per_s_per_hz": se,
                "per_link_hz": b_i,
                "per_link_kHz": b_i / 1.0e3,
                "vs_stated_20kHz": b_i / b_floor_hz,
                "implied_sum_hz": b_i * a["n_links"],
                "implied_sum_vs_20kHz": (b_i * a["n_links"]) / b_floor_hz,
            }
        )
    return {
        "stated_floor_hz": b_floor_hz,
        "snr": snr,
        "anchors": rows,
    }


def _associated_se(scenario, uav: np.ndarray, association: np.ndarray) -> dict:
    dummy = np.ones((scenario.cfg.num_iot, uav.shape[0]))
    m = link_metrics(scenario.iot_xyz_m, uav, dummy, scenario.cfg)
    snr = np.asarray(m["snr"], dtype=float)
    a = np.asarray(association) > 0.5
    snr_assoc = snr[a]
    se_all = np.log2(1.0 + np.maximum(snr, 0.0))
    se_assoc = np.log2(1.0 + np.maximum(snr_assoc, 0.0))
    n = int(a.sum())
    mean_se = float(np.mean(se_assoc)) if n else 0.0
    max_se = float(np.max(se_all)) if se_all.size else 0.0
    b_sys = float(scenario.cfg.b_sys_hz)
    return {
        "n_associated": n,
        "max_snr": float(np.max(snr)) if snr.size else 0.0,
        "mean_associated_snr": float(np.mean(snr_assoc)) if n else 0.0,
        "max_se": max_se,
        "mean_associated_se": mean_se,
        "equal_share_pred_Mbps": (b_sys * mean_se) / 1.0e6,
        "best_link_dump_Mbps": (b_sys * max_se) / 1.0e6,
        "equal_share_B_hz": (b_sys / n) if n else 0.0,
    }


def run_area_control(
    *,
    area_m: float,
    n_runs: int = 20,
    seed_start: int = 1,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    b_sys_hz: float = PAPER_B_SYS_HZ,
) -> dict:
    cfg = SimConfig(b_sys_hz=b_sys_hz).with_square_area_m(area_m)
    seeds = tuple(range(seed_start, seed_start + n_runs))
    sca_settings = SCASettings(solver=None, max_iterations=30)
    by_method: dict[str, dict] = {}
    geo_max_snr: list[float] = []
    mean_se_all: list[float] = []
    max_se_all: list[float] = []
    pred_eq: list[float] = []
    pred_dump: list[float] = []
    for method in methods:
        rates = []
        feas = []
        qos = []
        aodt = []
        mean_se = []
        for seed in seeds:
            sc = generate_scenario(seed, cfg)
            run = run_method(sc, method, seed, sca_settings=sca_settings)
            rates.append(run.sum_rate_bit_per_s)
            feas.append(run.feasible)
            qos.append(run.true_eval.constraints.qos_violations)
            aodt.append(run.true_eval.constraints.aodt_violations)
            se = _associated_se(sc, run.uav_xyz_m, run.allocation.association)
            geo_max_snr.append(se["max_snr"])
            mean_se.append(se["mean_associated_se"])
            mean_se_all.append(se["mean_associated_se"])
            max_se_all.append(se["max_se"])
            pred_eq.append(se["equal_share_pred_Mbps"])
            pred_dump.append(se["best_link_dump_Mbps"])
        rates_a = np.asarray(rates, dtype=float)
        std = float(np.std(rates_a, ddof=1)) if rates_a.size > 1 else 0.0
        by_method[method] = {
            "mean_sum_rate_bit_per_s": float(np.mean(rates_a)),
            "std_sum_rate_bit_per_s": std,
            "mean_sum_rate_Mbps": float(np.mean(rates_a) / 1.0e6),
            "std_sum_rate_Mbps": std / 1.0e6,
            "max_sum_rate_Mbps": float(np.max(rates_a) / 1.0e6),
            "feasible_fraction": float(np.mean(np.asarray(feas, dtype=float))),
            "mean_qos_violations": float(np.mean(qos)),
            "mean_aodt_violations": float(np.mean(aodt)),
            "mean_associated_se": float(np.mean(mean_se)),
            "per_seed_Mbps": [float(x) / 1.0e6 for x in rates],
            "per_seed_feasible": [bool(x) for x in feas],
        }
    max_snr = float(max(geo_max_snr)) if geo_max_snr else 0.0
    model_ceiling = sum_rate_ceiling_bit_per_s(b_sys_hz, max_snr)
    return {
        "area_m": [cfg.area_x_m, cfg.area_y_m],
        "b_sys_hz": b_sys_hz,
        "n_runs": n_runs,
        "seed_start": seed_start,
        "methods": list(methods),
        "seeds": list(seeds),
        "observed_max_snr": max_snr,
        "mean_associated_se": float(np.mean(mean_se_all)) if mean_se_all else 0.0,
        "mean_max_se": float(np.mean(max_se_all)) if max_se_all else 0.0,
        "model_snr_ceiling_Mbps": model_ceiling / 1.0e6,
        "equal_share_pred_mean_Mbps": float(np.mean(pred_eq)) if pred_eq else 0.0,
        "best_link_dump_mean_Mbps": float(np.mean(pred_dump)) if pred_dump else 0.0,
        "realized_rate_mechanism": (
            "QoS bandwidth LP is infeasible at 20 kHz, so evaluate falls "
            "back to equal-share B_sys/I. Then R_sum = B_sys * mean_associated_SE. "
            "The max-SNR ceiling is B_sys * max_SE (dump the pool on the best "
            "link). Compact fields have mean_SE ~ max_SE; large fields do not."
        ),
        "by_method": by_method,
    }


def run_bsys_20khz_check(
    *,
    n_runs: int = 20,
    seed_start: int = 1,
    areas_m: tuple[float, ...] = DEFAULT_AREAS_M,
    methods: tuple[str, ...] = DEFAULT_METHODS,
) -> dict:
    ceilings = model_free_ceilings_bps()
    areas = [run_area_control(
        area_m=side,
        n_runs=n_runs,
        seed_start=seed_start,
        methods=methods,
    ) for side in areas_m]
    any_feasible = any(
        stats["feasible_fraction"] > 0.0
        for area in areas
        for stats in area["by_method"].values()
    )
    max_realized = max(
        stats["max_sum_rate_Mbps"]
        for area in areas
        for stats in area["by_method"].values()
    )
    written_snr = max(a["observed_max_snr"] for a in areas) if areas else 1.0
    return {
        "paper": "Khalaf et al. IEEE TNSM 2026 Table II B_sys / constraint (27)",
        "reading": (
            "B_sys in Table II is treated as the sum cap in constraint (27). "
            "Table II's row text is 'Minimum bandwidth allocation'."
        ),
        "model_free": ceilings,
        "fig_6_10_rules_out": (
            "Reading A: Figs. 6-10 sit at 7-14 Mbps and Fig. 7 grows with I, "
            "both impossible under a 20 kHz sum cap. Reading B's *shape* "
            "(rate scales with I) matches Fig. 7, but the stated 20 kHz floor "
            "is tens of times too small on the written channel."
        ),
        "reading_b_written_snr": reading_b_magnitude(written_snr),
        "reading_b_generous_snr": reading_b_magnitude(GENEROUS_SNR_MAX),
        "areas": areas,
        "all_areas_zero_feasible": not any_feasible,
        "max_realized_Mbps": max_realized,
        "generous_ceiling_Mbps": ceilings["generous_ceiling_Mbps"],
        "realized_below_generous_ceiling": max_realized
        < ceilings["generous_ceiling_Mbps"],
    }


def write_bsys_check(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
