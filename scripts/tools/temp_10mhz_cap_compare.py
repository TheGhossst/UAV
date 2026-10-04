"""Temporary: 10 MHz B_sys at 500×500 m, cap=25% vs no cap (n20, J=3 I=10)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402

from uavdt.config import SimConfig  # noqa: E402
from uavdt.experiments.campaign import CampaignSettings, run_point  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.experiments.grids import SweepPoint  # noqa: E402
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402

OUT_DIR = ROOT / "results" / "_tmp_10mhz_500m_cap_ablation"
B_HZ = 10_000_000.0
METHODS = ("random", "kmeans", "pso", "sca")
N_RUNS = 20


def _point(max_bw_share: float | None) -> SweepPoint:
    base = SimConfig(b_sys_hz=B_HZ, max_bw_share=max_bw_share).with_square_area_m(500.0)
    cfg = config_for_counts(10, 3, base)
    cap = "none" if max_bw_share is None else f"{max_bw_share:.0%}"
    return SweepPoint(
        "tmp",
        "cap",
        float(max_bw_share or -1.0),
        cfg,
        f"10 MHz 500m  cap={cap}",
    )


def _run_label(share: float | None) -> str:
    return "nocap" if share is None else f"cap{int(share * 100)}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = CampaignSettings(
        n_runs=N_RUNS,
        seed_start=1,
        methods=METHODS,
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
    )
    payloads: dict[str, dict] = {}
    for share in (0.25, None):
        label = _run_label(share)
        out = OUT_DIR / f"n20_{label}.json"
        if out.exists():
            payloads[label] = json.loads(out.read_text(encoding="utf-8"))
            print(f"loaded {out}", flush=True)
            continue
        t0 = perf_counter()
        print(f"=== run {label} ===", flush=True)
        pt = _point(share)
        result = run_point(pt, settings)
        payload = {
            "b_sys_hz": B_HZ,
            "max_bw_share": share,
            "area_m": 500.0,
            "n_runs": N_RUNS,
            "methods": list(METHODS),
            "point": result,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payloads[label] = payload
        print(f"wrote {out} ({perf_counter() - t0:.1f}s)", flush=True)

    compare: dict[str, dict] = {}
    for method in METHODS:
        cap_rates = np.array(
            payloads["cap25"]["point"]["by_method"][method]["per_seed_Mbps"], dtype=float
        )
        no_rates = np.array(
            payloads["nocap"]["point"]["by_method"][method]["per_seed_Mbps"], dtype=float
        )
        d = cap_rates - no_rates
        w = wilcoxon_signed_rank(d)
        compare[method] = {
            "mean_cap_Mbps": float(np.mean(cap_rates)),
            "mean_nocap_Mbps": float(np.mean(no_rates)),
            "mean_delta_cap_minus_nocap_Mbps": float(np.mean(d)),
            "wilcoxon_p_two_sided": float(w["p_two_sided"]),
            "significant_p05": bool(w["p_two_sided"] < 0.05),
        }

    summary = {
        "configs": ["cap25", "nocap"],
        "b_sys_hz": B_HZ,
        "area_m": 500.0,
        "n_runs": N_RUNS,
        "cap_vs_nocap": compare,
        "any_method_significant": any(c["significant_p05"] for c in compare.values()),
    }
    summary_path = OUT_DIR / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- cap25 vs no cap (paired per seed) ---", flush=True)
    for method, c in compare.items():
        sig = "*" if c["significant_p05"] else ""
        print(
            f"  {method:8s}  cap={c['mean_cap_Mbps']:.4f}  "
            f"nocap={c['mean_nocap_Mbps']:.4f}  "
            f"delta={c['mean_delta_cap_minus_nocap_Mbps']:+.4f}  "
            f"p={c['wilcoxon_p_two_sided']:.4g}{sig}",
            flush=True,
        )
    print(f"\nwrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
