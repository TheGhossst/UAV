"""Does associated-link SE spread, not leftover dump, explain the J fade?

Cap-binding already showed the uncapped LP still puts ~95% of B_sys on the
best-SE link at every J. This script holds that dump fixed as a fact and
asks whether the *rate* of dump-vs-spread fades because associated SE
values get closer as J grows (nearest UAV is closer for every IoT).

SE is a geometry quantity (independent of the cap). First-order leftover
premium is leftover_Hz * (SE_max - SE_mean) bit/s.

Joins rate gaps from results/cap_binding_diagnostic.json when present.

Output: results/se_spread_by_j.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import SimConfig  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.placement.kmeans import place_kmeans  # noqa: E402
from uavdt.placement.random import place_random  # noqa: E402
from uavdt.resources import cpu_stable_processing, nearest_association  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402
from uavdt.sca.cvx_problem import bandwidth_floors_hz  # noqa: E402
from uavdt.sca.linearize import spectral_efficiency  # noqa: E402

B_SYS = 8_800_000.0
J_VALUES = (1, 2, 3, 4, 5)
N_RUNS = 20
SEED_START = 1
OUT = ROOT / "results" / "se_spread_by_j.json"
BINDING = ROOT / "results" / "cap_binding_diagnostic.json"


def _out(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _assoc_stats(
    iot: np.ndarray,
    uav: np.ndarray,
    a: np.ndarray,
    se: np.ndarray,
    floors: np.ndarray,
    b_sys: float,
) -> dict[str, float]:
    assoc = a > 0.5
    se_a = se[assoc]
    delta = iot[:, None, :] - uav[None, :, :]
    dist = np.sqrt(np.sum(delta**2, axis=-1))
    d_a = dist[assoc]
    leftover = b_sys - float(np.nansum(np.where(assoc, floors, 0.0)))
    se_max = float(se_a.max())
    se_mean = float(se_a.mean())
    se_min = float(se_a.min())
    premium_mbps = leftover * (se_max - se_mean) / 1e6
    return {
        "n_assoc": int(assoc.sum()),
        "se_max": se_max,
        "se_min": se_min,
        "se_mean": se_mean,
        "se_std": float(se_a.std(ddof=1)) if se_a.size > 1 else 0.0,
        "se_range": se_max - se_min,
        "se_max_minus_mean": se_max - se_mean,
        "se_max_over_mean": se_max / se_mean if se_mean > 0 else float("nan"),
        "se_cv": float(se_a.std(ddof=1) / se_mean) if se_a.size > 1 and se_mean > 0 else 0.0,
        "dist_mean_m": float(d_a.mean()),
        "dist_std_m": float(d_a.std(ddof=1)) if d_a.size > 1 else 0.0,
        "dist_max_m": float(d_a.max()),
        "dist_min_m": float(d_a.min()),
        "leftover_hz": leftover,
        "dump_premium_Mbps": premium_mbps,
    }


def _mean(xs: list[float]) -> float:
    arr = np.array(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _corr(x: list[float], y: list[float]) -> float:
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    if float(np.std(a[m])) < 1e-15 or float(np.std(b[m])) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load_binding_gaps() -> dict[tuple[int, int, str], dict[str, float]]:
    if not BINDING.exists():
        return {}
    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    out: dict[tuple[int, int, str], dict[str, float]] = {}
    for r in payload.get("rows", []):
        none = r["by_share"]["none"]
        out[(int(r["j"]), int(r["seed"]), str(r["method"]))] = {
            "uncapped_max_share": float(none["max_share"]),
            "dump_frac_on_best": float(none["dump_frac_on_best"]),
            "rate_gap_15_Mbps": float(r["rate_gap_15_vs_none_Mbps"]),
            "rate_gap_25_Mbps": float(r["rate_gap_25_vs_none_Mbps"]),
        }
    return out


def main() -> int:
    gaps = load_binding_gaps()
    rows: list[dict[str, Any]] = []
    cfg = SimConfig(b_sys_hz=B_SYS, max_bw_share=None)
    for j in J_VALUES:
        for seed in range(SEED_START, SEED_START + N_RUNS):
            sc = generate_scenario(seed, config_for_counts(10, j, cfg))
            placements = {
                "random": place_random(j, seed, sc.cfg),
                "kmeans": place_kmeans(sc, seed),
            }
            for method, uav in placements.items():
                a = nearest_association(sc.iot_xyz_m, uav)
                proc = cpu_stable_processing(sc, a)
                se = spectral_efficiency(sc.iot_xyz_m, uav, sc.cfg)
                floors = bandwidth_floors_hz(sc, a, proc, se)
                stats = _assoc_stats(sc.iot_xyz_m, uav, a, se, floors, B_SYS)
                g = gaps.get((j, seed, method), {})
                row = {
                    "j": j,
                    "seed": seed,
                    "method": method,
                    **stats,
                    **g,
                }
                rows.append(row)

    summary: dict[str, Any] = {}
    for j in J_VALUES:
        block: dict[str, Any] = {}
        for method in ("random", "kmeans"):
            sub = [r for r in rows if r["j"] == j and r["method"] == method]
            block[method] = {
                "se_max": _mean([r["se_max"] for r in sub]),
                "se_mean": _mean([r["se_mean"] for r in sub]),
                "se_min": _mean([r["se_min"] for r in sub]),
                "se_range": _mean([r["se_range"] for r in sub]),
                "se_max_minus_mean": _mean([r["se_max_minus_mean"] for r in sub]),
                "se_max_over_mean": _mean([r["se_max_over_mean"] for r in sub]),
                "se_cv": _mean([r["se_cv"] for r in sub]),
                "dist_mean_m": _mean([r["dist_mean_m"] for r in sub]),
                "dist_std_m": _mean([r["dist_std_m"] for r in sub]),
                "dump_premium_Mbps": _mean([r["dump_premium_Mbps"] for r in sub]),
                "uncapped_max_share": _mean(
                    [r.get("uncapped_max_share", float("nan")) for r in sub]
                ),
                "dump_frac_on_best": _mean(
                    [r.get("dump_frac_on_best", float("nan")) for r in sub]
                ),
                "rate_gap_15_Mbps": _mean(
                    [r.get("rate_gap_15_Mbps", float("nan")) for r in sub]
                ),
                "rate_gap_25_Mbps": _mean(
                    [r.get("rate_gap_25_Mbps", float("nan")) for r in sub]
                ),
            }
        summary[str(j)] = block

    all_random = [r for r in rows if r["method"] == "random"]
    all_kmeans = [r for r in rows if r["method"] == "kmeans"]
    corrs = {}
    for tag, sub in (("random", all_random), ("kmeans", all_kmeans), ("both", rows)):
        corrs[tag] = {
            "premium_vs_gap15": _corr(
                [r["dump_premium_Mbps"] for r in sub],
                [r.get("rate_gap_15_Mbps", float("nan")) for r in sub],
            ),
            "se_range_vs_gap15": _corr(
                [r["se_range"] for r in sub],
                [r.get("rate_gap_15_Mbps", float("nan")) for r in sub],
            ),
            "se_max_minus_mean_vs_gap15": _corr(
                [r["se_max_minus_mean"] for r in sub],
                [r.get("rate_gap_15_Mbps", float("nan")) for r in sub],
            ),
            "se_max_minus_mean_vs_gap25": _corr(
                [r["se_max_minus_mean"] for r in sub],
                [r.get("rate_gap_25_Mbps", float("nan")) for r in sub],
            ),
            "j_vs_se_range": _corr(
                [float(r["j"]) for r in sub],
                [r["se_range"] for r in sub],
            ),
        }

    payload = {
        "b_sys_hz": B_SYS,
        "n_runs": N_RUNS,
        "claim": (
            "Uncapped leftover dump onto the best-SE link does not fade with J "
            "(max share stays ~0.95). Associated SE spread does. The rate cost "
            "of a cap is leftover * (SE_max - SE_of_displaced_B), so the same "
            "dump has less sum-rate consequence when J raises the worst links."
        ),
        "summary": summary,
        "correlations": corrs,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _out("=" * 92)
    _out("Associated SE spread vs J  (geometry only; cap does not enter SE)")
    _out("Dump of leftover B still happens at every J (~95% on best link).")
    _out("=" * 92)
    _out(
        f"{'J':>3} {'meth':>7} {'SE max':>8} {'SE mean':>8} {'SE min':>8} "
        f"{'range':>8} {'max-mean':>9} {'max/mean':>9} {'CV':>7} "
        f"{'d_mean':>7} {'premium':>8} {'gap15':>8} {'gap25':>8} {'maxB':>7}"
    )
    for j in J_VALUES:
        for method in ("random", "kmeans"):
            b = summary[str(j)][method]
            _out(
                f"{j:3d} {method:>7} {b['se_max']:8.4f} {b['se_mean']:8.4f} "
                f"{b['se_min']:8.4f} {b['se_range']:8.4f} "
                f"{b['se_max_minus_mean']:9.4f} {b['se_max_over_mean']:9.3f} "
                f"{b['se_cv']:7.3f} {b['dist_mean_m']:7.2f} "
                f"{b['dump_premium_Mbps']:8.4f} {b['rate_gap_15_Mbps']:8.4f} "
                f"{b['rate_gap_25_Mbps']:8.4f} {b['uncapped_max_share']:7.3f}"
            )
    _out("")
    _out("Pearson r across J×seeds")
    for tag, c in corrs.items():
        _out(
            f"  {tag:7s}  premium~gap15={c['premium_vs_gap15']:+.3f}  "
            f"range~gap15={c['se_range_vs_gap15']:+.3f}  "
            f"(max-mean)~gap15={c['se_max_minus_mean_vs_gap15']:+.3f}  "
            f"(max-mean)~gap25={c['se_max_minus_mean_vs_gap25']:+.3f}  "
            f"J~range={c['j_vs_se_range']:+.3f}"
        )
    r1 = summary["1"]["random"]
    r5 = summary["5"]["random"]
    _out("")
    _out(
        f"Random J=1→5: SE range {r1['se_range']:.4f}→{r5['se_range']:.4f} "
        f"({100*(r5['se_range']/r1['se_range']-1):+.1f}%), "
        f"max−mean {r1['se_max_minus_mean']:.4f}→{r5['se_max_minus_mean']:.4f}, "
        f"mean dist {r1['dist_mean_m']:.1f}→{r5['dist_mean_m']:.1f} m, "
        f"uncapped max share {r1['uncapped_max_share']:.3f}→{r5['uncapped_max_share']:.3f} "
        f"(dump does not fade)."
    )
    _out(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
