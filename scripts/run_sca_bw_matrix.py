"""Seed-1 SCA matrix: 20 kHz / 2.4 MHz / 8.8 MHz, with and without 25% B_ij cap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uavdt.config import BANDWIDTH_PRESETS, SimConfig
from uavdt.sca import SCASettings, gap_vs_uncapped, solve_sca
from uavdt.sca.algorithm import write_history
from uavdt.scenario import generate_scenario

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STEM = {
    "20khz": "20khz",
    "2.4mhz": "2.4mhz",
    "8.8mhz": "8.8mhz",
}


def _row(preset: str, share: float | None, result) -> dict:
    ev = result.true_eval
    c = ev.constraints
    bw = result.allocation.bandwidth_hz
    init = result.history[0].true_objective if result.history else float("nan")
    return {
        "preset": preset,
        "max_bw_share": share,
        "b_ij_cap_hz": None,
        "stop_reason": result.diagnostics.get("stop_reason"),
        "solver_status": result.solver_status,
        "solver_name": result.solver_name,
        "accepted_steps": result.diagnostics.get("accepted_steps"),
        "rejected_steps": result.diagnostics.get("rejected_steps"),
        "n_iterations": result.n_iterations,
        "feasible": bool(ev.feasible),
        "sum_rate_Mbps": float(ev.sum_rate_mbps),
        "sum_rate_bit_per_s": float(ev.sum_rate_bit_per_s),
        "initial_true_obj": float(init),
        "final_true_obj": float(result.true_objective),
        "qos_violations": int(c.qos_violations),
        "aodt_violations": int(c.aodt_violations),
        "sep_violations": int(c.sep_violations),
        "cpu_unstable_count": int(c.cpu_unstable_count),
        "bw_excess_hz": float(c.bw_excess_hz),
        "bandwidth_used_hz": float(bw.sum()),
        "max_link_B_hz": float(bw.max()),
        "n_links_near_cap": None,
        "aodt_s": ev.aodt_s.tolist(),
        "rho": ev.rho.tolist(),
        "uav_xy_m": result.uav_xyz_m[:, :2].tolist(),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for preset, stem in STEM.items():
        hz = BANDWIDTH_PRESETS[preset]
        for share in (None, 0.25):
            suffix = "" if share is None else "_cap25"
            cfg = SimConfig(b_sys_hz=hz, max_bw_share=share)
            sc = generate_scenario(1, cfg)
            print(f"=== {preset}  share={share}  B_sys={hz:g} Hz ===", flush=True)
            result = solve_sca(
                sc,
                seed=1,
                settings=SCASettings(solver="matlab", max_iterations=30, step_size_m=20.0),
            )
            write_history(result, str(RESULTS / f"sca_history_{stem}{suffix}.json"))
            write_history(result, str(RESULTS / f"sca_history_{stem}{suffix}.csv"))
            row = _row(preset, share, result)
            row["b_sys_hz"] = hz
            row["b_ij_cap_hz"] = cfg.link_bandwidth_cap_hz
            cap = cfg.link_bandwidth_cap_hz
            bw = result.allocation.bandwidth_hz
            row["n_links_near_cap"] = int(np.sum(bw > 0.9 * cap))
            rows.append(row)
            print(
                f"stop={row['stop_reason']} feasible={row['feasible']} "
                f"Mbps={row['sum_rate_Mbps']:.6g} maxB={row['max_link_B_hz']:.6g} "
                f"qos={row['qos_violations']} aodt={row['aodt_violations']}",
                flush=True,
            )
    gaps = []
    by_preset: dict[str, dict] = {}
    for row in rows:
        by_preset.setdefault(row["preset"], {})[row["max_bw_share"]] = row
    for preset, pair in by_preset.items():
        uncapped = pair.get(None)
        capped = pair.get(0.25)
        if uncapped is None or capped is None:
            continue
        gap_bps, pct = gap_vs_uncapped(
            float(uncapped["sum_rate_bit_per_s"]),
            float(capped["sum_rate_bit_per_s"]),
        )
        gap_mbps, _ = gap_vs_uncapped(
            float(uncapped["sum_rate_Mbps"]),
            float(capped["sum_rate_Mbps"]),
        )
        gaps.append(
            {
                "preset": preset,
                "uncapped_rate_bit_per_s": float(uncapped["sum_rate_bit_per_s"]),
                "capped_rate_bit_per_s": float(capped["sum_rate_bit_per_s"]),
                "uncapped_rate_Mbps": float(uncapped["sum_rate_Mbps"]),
                "capped_rate_Mbps": float(capped["sum_rate_Mbps"]),
                "gap_vs_uncapped_bit_per_s": gap_bps,
                "gap_vs_uncapped_Mbps": gap_mbps,
                "gap_vs_uncapped_pct": pct,
            }
        )
    out = RESULTS / "sca_bw_share_sweep_seed1.json"
    payload = {
        "seed": 1,
        "gap_definition": (
            "gap_vs_uncapped = uncapped_rate - capped_rate; "
            "gap_vs_uncapped_pct = 100 * gap / uncapped_rate. "
            "Same unit on both rates; no extra factor of 10."
        ),
        "rows": rows,
        "gaps_vs_uncapped": gaps,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
