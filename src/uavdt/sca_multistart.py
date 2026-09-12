"""Keep-best multi-start SCA. Does not edit frozen Algorithm 1.

Experiment A (docs/RESULTS.md §2.9): extra random / k-means inits find
16–72 m basins that one-shot k-means SCA misses. This module is that
keep-best loop as a campaign method (`sca_multistart`). Frozen SCA stays
the headline solver. Extra starts reuse `solve_sca` with a different q0;
a_ij is still nearest at that init.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from uavdt.models import Allocation, Scenario
from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.random import place_random
from uavdt.resources import cpu_stable_processing, nearest_association
from uavdt.sca.algorithm import SCAResult, solve_sca
from uavdt.sca.initialize import qos_floor_bandwidth
from uavdt.sca.linearize import spectral_efficiency
from uavdt.sca.settings import SCASettings


@dataclass(frozen=True)
class MultiStartSettings:
    """How many extra SCA inits. Not Table II / Problem (P).

    Defaults match Experiment A: 2 random + 2 k-means with placement
    seeds ``scenario_seed * 1000 + {1,2}``, plus the frozen k-means start.
    """

    n_random: int = 2
    n_kmeans: int = 2
    include_frozen: bool = True


def extra_starts(
    seed: int,
    settings: MultiStartSettings | None = None,
) -> tuple[tuple[str, int], ...]:
    """Placement-seed pairs for extra SCA inits. Same formula as Experiment A."""
    ms = settings or MultiStartSettings()
    out: list[tuple[str, int]] = []
    for i in range(1, int(ms.n_random) + 1):
        out.append(("random", int(seed) * 1000 + i))
    for i in range(1, int(ms.n_kmeans) + 1):
        out.append(("kmeans", int(seed) * 1000 + i))
    return tuple(out)


def _rank_key(feasible: bool, rate_mbps: float) -> tuple[int, float]:
    return (int(bool(feasible)), float(rate_mbps))


def _init_alloc(scenario: Scenario, uav: np.ndarray) -> Allocation:
    """Same start convention as ``initialize_sca``: nearest a, CPU-stable b, QoS-floor B."""
    a = nearest_association(scenario.iot_xyz_m, uav)
    b = cpu_stable_processing(scenario, a)
    se = spectral_efficiency(scenario.iot_xyz_m, uav, scenario.cfg)
    bw = qos_floor_bandwidth(a, se, scenario.cfg)
    return Allocation(a, b, bw)


def _place(scenario: Scenario, kind: str, init_seed: int) -> np.ndarray:
    if kind == "random":
        return place_random(scenario.cfg.num_uav, init_seed, scenario.cfg)
    if kind == "kmeans":
        return place_kmeans(scenario, init_seed)
    raise ValueError(f"unknown start kind {kind!r}")


def _run_one(
    scenario: Scenario,
    seed: int,
    sca_settings: SCASettings,
    *,
    uav: np.ndarray | None = None,
    allocation: Allocation | None = None,
) -> tuple[SCAResult | None, float, str | None]:
    t0 = perf_counter()
    try:
        result = solve_sca(
            scenario,
            seed,
            settings=sca_settings,
            uav_xyz_m=uav,
            allocation=allocation,
        )
    except RuntimeError as exc:
        if "CPU stability" not in str(exc):
            raise
        return None, perf_counter() - t0, "init_cpu_unstable"
    return result, perf_counter() - t0, None


def _start_row(
    kind: str,
    init_seed: int,
    result: SCAResult | None,
    wall_s: float,
    fail: str | None,
) -> dict:
    if result is None:
        return {
            "kind": kind,
            "init_seed": int(init_seed),
            "sum_rate_Mbps": 0.0,
            "feasible": False,
            "n_iterations": 0,
            "stop_reason": fail or "fail",
            "wall_clock_s": float(wall_s),
            "failed": True,
        }
    ev = result.true_eval
    return {
        "kind": kind,
        "init_seed": int(init_seed),
        "sum_rate_Mbps": float(ev.sum_rate_mbps),
        "feasible": bool(ev.feasible),
        "n_iterations": int(result.n_iterations),
        "stop_reason": result.diagnostics.get("stop_reason"),
        "wall_clock_s": float(wall_s),
        "failed": False,
    }


def solve_sca_multistart(
    scenario: Scenario,
    seed: int,
    settings: SCASettings | None = None,
    multistart: MultiStartSettings | None = None,
) -> SCAResult:
    """Run frozen SCA plus extra inits; return the keep-best ``evaluate()`` point.

    With default ``include_frozen=True``, one-shot k-means SCA is always a
    candidate. Keep-best is lexicographic (feasible, rate), so the return
    is never worse than that frozen start **by construction** — not an
    empirical claim about a seed set. ``--no-frozen-start`` drops the
    guarantee. Does not rematch a_ij inside a start.
    """
    sca_settings = settings or SCASettings()
    ms = multistart or MultiStartSettings()
    rows: list[dict] = []
    results: list[SCAResult | None] = []

    if ms.include_frozen:
        frozen, wall_s, fail = _run_one(scenario, seed, sca_settings)
        rows.append(_start_row("frozen_kmeans", seed, frozen, wall_s, fail))
        results.append(frozen)

    for kind, init_seed in extra_starts(seed, ms):
        try:
            uav = _place(scenario, kind, init_seed)
            alloc = _init_alloc(scenario, uav)
        except RuntimeError as exc:
            rows.append(_start_row(kind, init_seed, None, 0.0, str(exc)))
            results.append(None)
            continue
        extra, wall_s, fail = _run_one(
            scenario, seed, sca_settings, uav=uav, allocation=alloc
        )
        rows.append(_start_row(kind, init_seed, extra, wall_s, fail))
        results.append(extra)

    best_i: int | None = None
    best_key: tuple[int, float] | None = None
    for i, row in enumerate(rows):
        if row.get("failed") and results[i] is None:
            continue
        key = _rank_key(bool(row["feasible"]), float(row["sum_rate_Mbps"]))
        if best_key is None or key > best_key:
            best_key = key
            best_i = i

    if best_i is None or results[best_i] is None:
        raise RuntimeError(
            f"sca_multistart: every start failed for seed={seed} "
            f"(n_starts={len(rows)})"
        )

    winner = results[best_i]
    win_row = rows[best_i]
    frozen_row = next((r for r in rows if r["kind"] == "frozen_kmeans"), None)
    frozen_mbps = (
        float(frozen_row["sum_rate_Mbps"]) if frozen_row is not None else None
    )
    best_mbps = float(win_row["sum_rate_Mbps"])
    diag = dict(winner.diagnostics)
    diag.update(
        {
            "method": "sca_multistart",
            "n_starts": len(rows),
            "n_ok": sum(1 for r in rows if not r.get("failed")),
            "winner_kind": win_row["kind"],
            "winner_init_seed": win_row["init_seed"],
            "frozen_Mbps": frozen_mbps,
            "best_Mbps": best_mbps,
            "delta_vs_frozen_Mbps": (
                None
                if frozen_mbps is None
                else best_mbps - frozen_mbps
            ),
            "starts": rows,
            "wall_clock_s": float(sum(float(r["wall_clock_s"]) for r in rows)),
        }
    )
    winner.diagnostics = diag
    return winner
