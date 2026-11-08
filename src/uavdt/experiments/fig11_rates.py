"""Sum rates for the Fig. 11 arrival patterns, under the age floors.

Khalaf et al. Fig. 11 plots sum rate (Mbps) against the number of UAVs
for three arrival patterns. Eq. (17) is an age. An equal split of the
pool does not depend on the arrival rates, so that score cannot order
the curves.

This module holds the IoT geometry and the k-means layout fixed, then
solves the bandwidth LP (36) separately for each pattern. The floors
are (35), from the Eq. (17) slack (32). A pattern with a larger queue
term has a smaller slack, a larger floor, and, when the leftover hertz
are poured onto stronger links, a lower sum rate.

Optional: one frozen-assignment SCA trajectory from that same k-means
start, again separately for each pattern. Association and processing
stay at the start. Positions and bandwidth move.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uavdt.aodt import queueing_term_s
from uavdt.computation import offered_load, queue_unstable, service_rate_per_s
from uavdt.config import SimConfig
from uavdt.evaluator import evaluate
from uavdt.experiments.fig11 import LAMBDA_PATTERNS, lambdas_for_pattern
from uavdt.experiments.grids import config_for_counts
from uavdt.models import Allocation
from uavdt.placement.kmeans import place_kmeans
from uavdt.resources import cpu_stable_processing, nearest_association
from uavdt.sca import SCASettings, solve_sca
from uavdt.sca.cvx_problem import bandwidth_floors_hz, solve_bandwidth_at_fixed_q
from uavdt.sca.initialize import qos_floor_bandwidth
from uavdt.sca.linearize import spectral_efficiency
from uavdt.scenario import generate_scenario


@dataclass(frozen=True)
class RateCell:
    pattern: str
    equal_mbps: float
    lp_mbps: float | None
    lp_feasible: bool
    lp_max_aodt_s: float | None
    floor_hz: float
    n_forwarded: int
    queue_s: tuple[float, ...]
    sca_mbps: float | None = None
    sca_feasible: bool | None = None
    sca_max_aodt_s: float | None = None
    sca_move_m: float | None = None


def queue_terms_s(cfg: SimConfig, pattern: str) -> tuple[float, ...]:
    """Eq. (17) queue term per process. Independent of geometry."""
    lam = lambdas_for_pattern(cfg, pattern)
    mu = service_rate_per_s(cfg)
    n_per = cfg.iots_per_process
    out = []
    for k in range(cfg.num_processes):
        sl = lam[k * n_per : (k + 1) * n_per]
        out.append(float(queueing_term_s(sl, mu)))
    return tuple(out)


def score_pattern_at_layout(
    seed: int,
    cfg: SimConfig,
    pattern: str,
    uav_xyz_m: np.ndarray,
    *,
    run_sca: bool = False,
    sca_settings: SCASettings | None = None,
) -> RateCell:
    """Equal split and the bandwidth LP at one frozen layout.

    ``uav_xyz_m`` is shared across patterns. Nearest association and
    CPU-stable processing are rebuilt from this pattern's arrivals.
    On these loads the assignment matches across patterns; the caller
    checks that.
    """
    lam = lambdas_for_pattern(cfg, pattern)
    scenario = generate_scenario(seed, cfg, lambdas_per_s=lam)
    uav = np.asarray(uav_xyz_m, dtype=float)
    a = nearest_association(scenario.iot_xyz_m, uav)
    b = cpu_stable_processing(scenario, a)
    equal = evaluate(scenario, uav)
    se = spectral_efficiency(scenario.iot_xyz_m, uav, cfg)
    floors = bandwidth_floors_hz(scenario, a, b, se)
    finite = np.isfinite(floors)
    floor_hz = float(floors[finite].sum()) if np.any(finite) else float("inf")
    j_assoc = np.argmax(a, axis=1)
    forwarded = int(np.sum(b[np.arange(a.shape[0]), j_assoc] < 0.5))
    lp = solve_bandwidth_at_fixed_q(
        scenario, uav, a, b, SCASettings(solver=None)
    )
    lp_mbps = None
    lp_age = None
    lp_ok = not lp.infeasible
    if lp_ok:
        ev = evaluate(scenario, uav, Allocation(a, b, lp.bandwidth_hz))
        lp_mbps = float(ev.sum_rate_mbps)
        lp_age = float(np.max(ev.aodt_s))
        lp_ok = bool(ev.feasible)
        if not lp_ok:
            lp_mbps = None
    sca_mbps = None
    sca_ok = None
    sca_age = None
    sca_move = None
    if run_sca:
        settings = sca_settings or SCASettings(solver=None, dynamic_assignment=False)
        # Same start initialize_sca would build at this layout. Passing both
        # is required: solve_sca rebuilds the start if either argument is omitted.
        start = Allocation(a, b, qos_floor_bandwidth(a, se, cfg))
        mu = service_rate_per_s(cfg)
        if np.any(queue_unstable(b, scenario.lambdas_per_s, mu)):
            rho = offered_load(b, scenario.lambdas_per_s, mu)
            raise RuntimeError(f"CPU-unstable start: rho={rho}")
        result = solve_sca(
            scenario,
            seed,
            settings=settings,
            uav_xyz_m=uav,
            allocation=start,
        )
        sca_ok = bool(result.true_eval.feasible)
        sca_age = float(np.max(result.true_eval.aodt_s))
        if sca_ok:
            sca_mbps = float(result.true_eval.sum_rate_mbps)
        sca_move = float(
            np.linalg.norm(result.uav_xyz_m[:, :2] - uav[:, :2], axis=1).mean()
        )
    return RateCell(
        pattern=pattern,
        equal_mbps=float(equal.sum_rate_mbps),
        lp_mbps=lp_mbps,
        lp_feasible=lp_ok,
        lp_max_aodt_s=lp_age,
        floor_hz=floor_hz,
        n_forwarded=forwarded,
        queue_s=queue_terms_s(cfg, pattern),
        sca_mbps=sca_mbps,
        sca_feasible=sca_ok,
        sca_max_aodt_s=sca_age,
        sca_move_m=sca_move,
    )


def run_fig11_rates(
    cfg: SimConfig,
    *,
    n_runs: int = 20,
    seed_start: int = 1,
    uav_counts: tuple[int, ...] = (1, 2, 3, 4, 5),
    sca_counts: tuple[int, ...] = (),
    sca_settings: SCASettings | None = None,
) -> dict:
    """Shared k-means layout, three patterns, equal split and bandwidth LP.

    SCA, when requested, starts from that layout with frozen association
    and processing. ``sca_counts`` is the subset of ``uav_counts`` to solve.
    """
    points = []
    for j in uav_counts:
        point_cfg = config_for_counts(10, int(j), cfg)
        rows: list[dict] = []
        for offset in range(n_runs):
            seed = seed_start + offset
            geom = generate_scenario(seed, point_cfg)
            uav = place_kmeans(geom, seed)
            cells = []
            assignments = []
            for pattern in LAMBDA_PATTERNS:
                cell = score_pattern_at_layout(
                    seed,
                    point_cfg,
                    pattern,
                    uav,
                    run_sca=int(j) in sca_counts,
                    sca_settings=sca_settings,
                )
                lam = lambdas_for_pattern(point_cfg, pattern)
                scenario = generate_scenario(seed, point_cfg, lambdas_per_s=lam)
                a = nearest_association(scenario.iot_xyz_m, uav)
                b = cpu_stable_processing(scenario, a)
                assignments.append((a, b))
                cells.append(cell)
            same_a = all(np.array_equal(assignments[0][0], ab[0]) for ab in assignments[1:])
            same_b = all(np.array_equal(assignments[0][1], ab[1]) for ab in assignments[1:])
            rows.append(
                {
                    "seed": seed,
                    "same_association": bool(same_a),
                    "same_processing": bool(same_b),
                    "by_pattern": {c.pattern: _cell_dict(c) for c in cells},
                }
            )
            print(
                f"  J={j} seed={seed}  "
                + "  ".join(
                    f"{c.pattern[:4]} lp={_fmt(c.lp_mbps)} eq={c.equal_mbps:.4f}"
                    for c in cells
                ),
                flush=True,
            )
        points.append({"num_uav": int(j), "seeds": rows, "summary": _summarize(rows)})
    return {
        "paper": "Khalaf et al. Fig. 11 sum-rate order, scored with Eq. (17) floors",
        "placement": "k-means, shared across the three patterns",
        "association": "nearest UAV",
        "processing": "CPU-stable, process-consistent",
        "bandwidth": "LP (36) at the frozen layout; equal split recorded beside it",
        "sca": (
            "frozen-assignment SCA from the shared k-means start"
            if sca_counts
            else "not run"
        ),
        "sca_counts": list(sca_counts),
        "n_runs": n_runs,
        "seed_start": seed_start,
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "area_m": [cfg.area_x_m, cfg.area_y_m],
        "aodt_threshold_s": cfg.aodt_threshold_s,
        "queue_terms_s": {p: list(queue_terms_s(cfg, p)) for p in LAMBDA_PATTERNS},
        "note": (
            "Equal-split sum rate does not depend on lambda. "
            "The LP sum rate does, through the Eq. (17) bandwidth floors."
        ),
        "points": points,
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "infeas"
    return f"{value:.4f}"


def _cell_dict(cell: RateCell) -> dict:
    return {
        "equal_mbps": cell.equal_mbps,
        "lp_mbps": cell.lp_mbps,
        "lp_feasible": cell.lp_feasible,
        "lp_max_aodt_s": cell.lp_max_aodt_s,
        "floor_hz": cell.floor_hz,
        "n_forwarded": cell.n_forwarded,
        "queue_s": list(cell.queue_s),
        "sca_mbps": cell.sca_mbps,
        "sca_feasible": cell.sca_feasible,
        "sca_max_aodt_s": cell.sca_max_aodt_s,
        "sca_move_m": cell.sca_move_m,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _summarize(rows: list[dict]) -> dict:
    summary = {"n_seeds": len(rows)}
    by = {}
    for pattern in LAMBDA_PATTERNS:
        equal = [r["by_pattern"][pattern]["equal_mbps"] for r in rows]
        lp = [
            r["by_pattern"][pattern]["lp_mbps"]
            for r in rows
            if r["by_pattern"][pattern]["lp_feasible"]
            and r["by_pattern"][pattern]["lp_mbps"] is not None
        ]
        floors = [r["by_pattern"][pattern]["floor_hz"] for r in rows]
        ages = [
            r["by_pattern"][pattern]["lp_max_aodt_s"]
            for r in rows
            if r["by_pattern"][pattern]["lp_max_aodt_s"] is not None
        ]
        n_at_deadline = sum(
            1
            for age in ages
            if age is not None and abs(age - 2.8) <= 1e-6
        )
        sca = [
            r["by_pattern"][pattern]["sca_mbps"]
            for r in rows
            if r["by_pattern"][pattern]["sca_feasible"]
            and r["by_pattern"][pattern]["sca_mbps"] is not None
        ]
        moves = [
            r["by_pattern"][pattern]["sca_move_m"]
            for r in rows
            if r["by_pattern"][pattern]["sca_move_m"] is not None
        ]
        by[pattern] = {
            "mean_equal_mbps": _mean(equal),
            "mean_lp_mbps": _mean(lp),
            "lp_feasible": len(lp),
            "mean_floor_hz": _mean(floors),
            "mean_lp_max_aodt_s": _mean(ages),
            "n_age_at_2_8s": n_at_deadline,
            "mean_sca_mbps": _mean(sca),
            "sca_feasible": len(sca),
            "mean_sca_move_m": _mean(moves),
        }
    summary["by_pattern"] = by
    summary["same_association"] = sum(1 for r in rows if r["same_association"])
    summary["same_processing"] = sum(1 for r in rows if r["same_processing"])
    summary["order"] = _order_counts(rows, "lp_mbps", "lp_feasible")
    summary["sca_order"] = _order_counts(rows, "sca_mbps", "sca_feasible")
    return summary


def _order_counts(rows: list[dict], key: str, flag: str) -> dict:
    """How the three sum rates line up. Fast, slow, heterogeneous."""
    n_fast_gt_slow = 0
    n_het_lt_slow = 0
    n_het_gt_slow = 0
    n_fast_gt_het = 0
    n_all_feasible = 0
    gaps = []
    fast_slow = []
    for row in rows:
        bp = row["by_pattern"]
        if not all(bp[p][flag] and bp[p][key] is not None for p in LAMBDA_PATTERNS):
            continue
        n_all_feasible += 1
        fast = bp["uniform_fast"][key]
        slow = bp["uniform_slow"][key]
        het = bp["heterogeneous"][key]
        n_fast_gt_slow += int(fast > slow + 1e-9)
        n_het_lt_slow += int(het < slow - 1e-9)
        n_het_gt_slow += int(het > slow + 1e-9)
        n_fast_gt_het += int(fast > het + 1e-9)
        gaps.append(slow - het)
        fast_slow.append(fast - slow)
    return {
        "n_all_feasible": n_all_feasible,
        "n_fast_above_slow": n_fast_gt_slow,
        "n_fast_above_heterogeneous": n_fast_gt_het,
        "n_heterogeneous_under_slow": n_het_lt_slow,
        "n_heterogeneous_above_slow": n_het_gt_slow,
        "mean_slow_minus_heterogeneous_mbps": _mean(gaps),
        "mean_fast_minus_slow_mbps": _mean(fast_slow),
    }


def write_fig11_rates(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _print_summary(label: str, payload: dict) -> None:
    print(f"\n== {label} ==")
    q = payload["queue_terms_s"]
    print("queue terms (s)", {k: [round(v, 4) for v in vs] for k, vs in q.items()})
    for pt in payload["points"]:
        s = pt["summary"]
        bp = s["by_pattern"]
        print(
            f"J={pt['num_uav']}  assoc {s['same_association']}/{s['n_seeds']}"
            f"  proc {s['same_processing']}/{s['n_seeds']}"
        )
        for pattern in LAMBDA_PATTERNS:
            row = bp[pattern]
            print(
                f"  {pattern:16}  equal {row['mean_equal_mbps']:.6f}"
                f"  lp {row['mean_lp_mbps']}"
                f"  feas {row['lp_feasible']}"
                f"  floor_kHz {None if row['mean_floor_hz'] is None else row['mean_floor_hz']/1e3:.1f}"
                f"  age {row['mean_lp_max_aodt_s']}"
                f"  sca {row['mean_sca_mbps']}"
                f"  move {row['mean_sca_move_m']}"
            )
        print("  lp order", s["order"])
        print("  sca order", s["sca_order"])
