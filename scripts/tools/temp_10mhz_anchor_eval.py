"""10 MHz / 500 m / 25% cap: zenith-anchor (n20) + comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "campaigns"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from run_sca_anchor_eval import _pair  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

OUT_DIR = ROOT / "results" / "_tmp_10mhz_500m_cap_ablation"
BASE_10 = OUT_DIR / "n20_cap25.json"
REF_88_ANCHOR = ROOT / "results" / "sca_anchor_n20_500m.json"
REF_88_CAMPAIGN = ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json"
B_HZ = 10_000_000.0
N_RUNS = 20
SEED_START = 1


def _cfg() -> SimConfig:
    base = SimConfig(b_sys_hz=B_HZ, max_bw_share=PRIMARY_MAX_BW_SHARE).with_square_area_m(
        500.0
    )
    return config_for_counts(10, 3, base)


def _baseline_10(method: str) -> dict[str, float]:
    payload = json.loads(BASE_10.read_text(encoding="utf-8"))
    rates = payload["point"]["by_method"][method]["per_seed_Mbps"]
    return {int(s): float(r) for s, r in zip(range(SEED_START, SEED_START + N_RUNS), rates)}


def _run_anchor_eval() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sca_anchor_n20_cap25.json"
    ckpt = OUT_DIR / "sca_anchor_n20_cap25.checkpoint.json"
    cfg = _cfg()
    sca_settings = SCASettings(solver=None, max_iterations=30)
    anc = AnchorSettings()
    seeds = list(range(SEED_START, SEED_START + N_RUNS))
    done: dict[str, dict] = {}
    if ckpt.exists():
        done = json.loads(ckpt.read_text(encoding="utf-8"))
        print(f"resume anchor {len(done)}/{len(seeds)} from {ckpt.name}", flush=True)

    baselines = {m: _baseline_10(m) for m in ("random", "kmeans", "pso", "sca")}

    for seed in seeds:
        key = str(seed)
        if key in done and "sca_anchor" in done[key]:
            continue
        print(f"  seed {seed}", flush=True)
        sc = generate_scenario(seed, cfg)
        t0 = perf_counter()
        run = run_method(
            sc,
            "sca_anchor",
            seed,
            sca_settings=sca_settings,
            anchor_settings=anc,
        )
        rec: dict = {"seed": seed}
        rec["sca_anchor"] = {
            "sum_rate_Mbps": float(run.sum_rate_mbps),
            "feasible": bool(run.feasible),
            "wall_clock_s": float(perf_counter() - t0),
            "diagnostics": {
                k: run.diagnostics.get(k)
                for k in (
                    "winner_kind",
                    "delta_vs_frozen_Mbps",
                    "frozen_Mbps",
                    "best_Mbps",
                    "n_lp",
                )
                if k in run.diagnostics
            },
        }
        for method, by_seed in baselines.items():
            rec[method] = {
                "sum_rate_Mbps": by_seed[seed],
                "feasible": True,
                "source": "n20_cap25.json",
            }
        done[key] = rec
        ckpt.write_text(json.dumps(done, indent=2), encoding="utf-8")
        print(
            f"    anchor {run.sum_rate_mbps:.4f} Mbps  "
            f"vs sca {baselines['sca'][seed]:.4f}  "
            f"{perf_counter() - t0:.1f}s",
            flush=True,
        )

    rows = [done[str(s)] for s in seeds]
    pairs = {
        "vs_sca": _pair(rows, "sca_anchor", "sca"),
        "vs_pso": _pair(rows, "sca_anchor", "pso"),
        "vs_random": _pair(rows, "sca_anchor", "random"),
    }
    mean_mbps = {
        m: float(np.mean([r[m]["sum_rate_Mbps"] for r in rows]))
        for m in ("sca_anchor", "sca", "pso", "kmeans", "random")
    }
    payload = {
        "label": "10 MHz 500m 25% cap — sca_anchor vs baselines (n20)",
        "b_sys_hz": B_HZ,
        "max_bw_share": PRIMARY_MAX_BW_SHARE,
        "area_m": [500.0, 500.0],
        "n_runs": N_RUNS,
        "per_seed": rows,
        "mean_Mbps": mean_mbps,
        **pairs,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if ckpt.exists():
        ckpt.unlink()
    print(f"wrote {out}", flush=True)
    return payload


def _compare_88(payload_10: dict) -> dict:
    ref = json.loads(REF_88_ANCHOR.read_text(encoding="utf-8"))
    camp = json.loads(REF_88_CAMPAIGN.read_text(encoding="utf-8"))
    pt88 = [x for x in camp["points"] if x.get("num_uav") == 3][0]

    def rates_from_anchor_file() -> list[float]:
        by_seed = {int(r["seed"]): r["sca_anchor"]["sum_rate_Mbps"] for r in ref["per_seed"]}
        return [float(by_seed[s]) for s in range(SEED_START, SEED_START + N_RUNS)]

    def rates_from_10() -> list[float]:
        return [float(r["sca_anchor"]["sum_rate_Mbps"]) for r in payload_10["per_seed"]]

    a88 = np.array(rates_from_anchor_file())
    a10 = np.array(rates_from_10())
    sca88 = np.array(pt88["by_method"]["sca"]["per_seed_Mbps"], dtype=float)
    sca10 = np.array(
        [_baseline_10("sca")[s] for s in range(SEED_START, SEED_START + N_RUNS)],
        dtype=float,
    )

    from paired_winrate import wilcoxon_signed_rank

    d_anchor = a10 - a88
    d_sca = sca10 - sca88
    w_a = wilcoxon_signed_rank(d_anchor)
    w_s = wilcoxon_signed_rank(d_sca)
    rows_10 = payload_10["per_seed"]
    vs_sca_88 = {
        "mean_anchor_88_Mbps": float(ref["mean_Mbps"]["sca_anchor"]),
        "mean_anchor_10_Mbps": float(payload_10["mean_Mbps"]["sca_anchor"]),
        "mean_sca_88_Mbps": float(np.mean(sca88)),
        "mean_sca_10_Mbps": float(np.mean(sca10)),
        "anchor_10_minus_88_Mbps": float(np.mean(d_anchor)),
        "sca_10_minus_88_Mbps": float(np.mean(d_sca)),
        "anchor_ratio_10_over_88": float(np.mean(a10) / np.mean(a88)),
        "sca_ratio_10_over_88": float(np.mean(sca10) / np.mean(sca88)),
        "wilcoxon_anchor_10_vs_88_p": float(w_a["p_two_sided"]),
        "wilcoxon_sca_10_vs_88_p": float(w_s["p_two_sided"]),
        "vs_sca_at_10MHz": payload_10["vs_sca"],
        "vs_sca_at_88MHz_from_file": ref["vs_sca"],
    }
    summary_path = OUT_DIR / "anchor_compare_summary.json"
    summary_path.write_text(json.dumps(vs_sca_88, indent=2), encoding="utf-8")
    return vs_sca_88


def main() -> int:
    if not BASE_10.exists():
        raise SystemExit(f"missing {BASE_10}; run temp_10mhz_cap_compare.py first")
    t0 = perf_counter()
    out = OUT_DIR / "sca_anchor_n20_cap25.json"
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        print(f"loaded {out}", flush=True)
    else:
        payload = _run_anchor_eval()
    summary = _compare_88(payload)
    print(f"\n--- 10 MHz anchor means ---", flush=True)
    for m, v in payload["mean_Mbps"].items():
        print(f"  {m:12s}  {v:.4f} Mbps", flush=True)
    vs = payload["vs_sca"]
    print(
        f"\nanchor vs sca @10 MHz:  delta={vs['mean_delta_Mbps']:+.4f}  "
        f"wins/losses={vs['wins']}/{vs['losses']}  "
        f"n_practical={vs['n_practical']}  p={vs['wilcoxon']['p_two_sided']:.4g}",
        flush=True,
    )
    print(
        f"\n8.8→10 scaling: anchor ratio {summary['anchor_ratio_10_over_88']:.4f}  "
        f"(sca {summary['sca_ratio_10_over_88']:.4f})",
        flush=True,
    )
    print(
        f"88 MHz anchor lift vs sca: {summary['vs_sca_at_88MHz_from_file']['mean_delta_Mbps']:+.4f}  "
        f"10 MHz: {summary['vs_sca_at_10MHz']['mean_delta_Mbps']:+.4f}",
        flush=True,
    )
    print(f"\nwrote {OUT_DIR / 'anchor_compare_summary.json'}  ({perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
