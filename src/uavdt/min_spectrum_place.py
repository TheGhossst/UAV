"""E1b: min-Hertz placement keep-best. Does not edit frozen SCA.

Oracle
------
Score a layout by the closed-form AoDT/QoS floor pool
``max(sum floors, max(floors)/c)`` (see ``hertz_bound_at_layout``).
That is the E1 bound, not leftover-dump Mbps.

Candidates are *not* restricted to IoT xy: k-means centroids (several
inits), discrete p-median / k-medoids zenith, random, and (when cheap)
every zenith J-subset scored by Hertz. Keep-best includes one-shot
k-means, so Hertz is never worse than k-means by construction while
``include_kmeans`` is True.

Polish is a coordinate pattern search on the Hertz bound, not Algorithm 1
(which maximises leftover-dump rate and would walk off this objective).

Falsifier (stated in advance)
-----------------------------
If the keep-best Hertz at 500 m / J=2 and J=3 lands within 50 kHz of
one-shot k-means, drop the spectrum-efficient placement claim (Path B).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from uavdt.min_spectrum import hertz_bound_at_layout, min_bsys_frozen_q
from uavdt.models import Scenario
from uavdt.placement.kmeans import _enforce_min_separation, place_kmeans
from uavdt.placement.kmedoids import covering_combos, place_kmedoids
from uavdt.placement.random import place_random
from uavdt.sca_anchor import uav_for_combo
from uavdt.scenario import make_uav_xyz_m


@dataclass(frozen=True)
class HertzPlaceSettings:
    n_kmeans: int = 5
    n_random: int = 2
    n_medoid_inits: int = 5
    enum_zenith_max: int = 200
    include_kmeans: bool = True
    polish: bool = True
    polish_steps_m: tuple[float, ...] = (20.0, 10.0, 5.0, 2.0)
    polish_max_rounds: int = 8
    verify_search: bool = True


def _finite(hz: float) -> bool:
    return bool(np.isfinite(hz) and hz > 0.0)


def _score(scenario: Scenario, uav: np.ndarray, kind: str, extra: dict | None = None) -> dict:
    info = hertz_bound_at_layout(scenario, uav)
    rec = {
        "kind": kind,
        "bound_hz": float(info["bound_hz"]),
        "binding": str(info["binding"]),
        "uav_xyz_m": np.asarray(uav, dtype=float).copy(),
        "allocation": info["allocation"],
        "min_assoc_se": float(info["min_assoc_se"]),
        "n_forwarding": int(info["n_forwarding"]),
    }
    if extra:
        rec.update(extra)
    return rec


def _kmeans_seeds(seed: int, n: int) -> list[int]:
    out = [int(seed)]
    for i in range(1, max(0, int(n))):
        out.append(int(seed) * 1000 + i)
    return out[: max(1, int(n))]


def _collect_candidates(
    scenario: Scenario,
    seed: int,
    settings: HertzPlaceSettings,
) -> list[dict]:
    cfg = scenario.cfg
    j = int(cfg.num_uav)
    i = int(scenario.num_iot)
    rows: list[dict] = []
    if settings.include_kmeans or settings.n_kmeans > 0:
        n_km = max(int(settings.n_kmeans), 1 if settings.include_kmeans else 0)
        for init in _kmeans_seeds(seed, n_km):
            uav = place_kmeans(scenario, init)
            kind = "kmeans" if init == int(seed) else "kmeans_extra"
            rows.append(_score(scenario, uav, kind, {"init_seed": int(init)}))
    for i_rand in range(int(settings.n_random)):
        uav = place_random(j, int(seed) * 1000 + 50 + i_rand, cfg)
        rows.append(_score(scenario, uav, "random", {"init_seed": i_rand}))
    uav_m = place_kmedoids(scenario, seed)
    rows.append(_score(scenario, uav_m, "medoid_place"))
    rng = np.random.default_rng(int(seed) + 7701)
    for combo in covering_combos(
        scenario.iot_xyz_m[:, :2],
        j,
        rng,
        n_pam_inits=int(settings.n_medoid_inits),
    ):
        uav = uav_for_combo(scenario, combo, seed)
        if uav is None:
            continue
        rows.append(
            _score(scenario, uav, "medoid_zenith", {"combo": list(combo)})
        )
    n_all = 0 if j > i or j < 0 else int(math.comb(i, j))
    if 0 < n_all <= int(settings.enum_zenith_max):
        for combo in itertools.combinations(range(i), j):
            uav = uav_for_combo(scenario, combo, seed)
            if uav is None:
                continue
            rows.append(
                _score(scenario, uav, "zenith_hertz", {"combo": list(combo)})
            )
    return rows


def _neighbors(xy: np.ndarray, j: int, step_m: float) -> list[np.ndarray]:
    out = []
    for axis in (0, 1):
        for sign in (-1.0, 1.0):
            cand = xy.copy()
            cand[j, axis] += sign * float(step_m)
            out.append(cand)
    for dx, dy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        cand = xy.copy()
        cand[j, 0] += dx * float(step_m)
        cand[j, 1] += dy * float(step_m)
        out.append(cand)
    return out


def polish_hertz(
    scenario: Scenario,
    uav: np.ndarray,
    settings: HertzPlaceSettings,
) -> tuple[np.ndarray, dict]:
    """Greedy pattern search on the Hertz bound. Not Algorithm 1."""
    cfg = scenario.cfg
    rng = np.random.default_rng(7)
    xy = np.asarray(uav[:, :2], dtype=float).copy()
    best = hertz_bound_at_layout(scenario, uav)
    best_hz = float(best["bound_hz"])
    n_moves = 0
    n_eval = 1
    if not settings.polish or not _finite(best_hz):
        return np.asarray(uav, dtype=float).copy(), {
            "n_moves": 0,
            "n_eval": n_eval,
            "bound_hz": best_hz,
        }
    j_n = xy.shape[0]
    for step in settings.polish_steps_m:
        for _round in range(int(settings.polish_max_rounds)):
            improved = False
            order = rng.permutation(j_n)
            for j in order:
                for cand_xy in _neighbors(xy, int(j), float(step)):
                    cand_xy[:, 0] = np.clip(cand_xy[:, 0], 0.0, cfg.area_x_m)
                    cand_xy[:, 1] = np.clip(cand_xy[:, 1], 0.0, cfg.area_y_m)
                    try:
                        cand_xy = _enforce_min_separation(cand_xy, cfg, rng)
                    except RuntimeError:
                        continue
                    cand_uav = make_uav_xyz_m(cand_xy, cfg.uav_height_m)
                    info = hertz_bound_at_layout(scenario, cand_uav)
                    n_eval += 1
                    hz = float(info["bound_hz"])
                    if _finite(hz) and hz + 1.0 < best_hz:
                        xy = cand_xy
                        best_hz = hz
                        best = info
                        n_moves += 1
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    uav_out = make_uav_xyz_m(xy, cfg.uav_height_m)
    return uav_out, {
        "n_moves": int(n_moves),
        "n_eval": int(n_eval),
        "bound_hz": float(best_hz),
        "binding": str(best.get("binding")),
        "allocation": best.get("allocation"),
    }


def _rank_key(bound_hz: float) -> tuple[int, float]:
    ok = 1 if _finite(bound_hz) else 0
    return (ok, -float(bound_hz) if ok else 0.0)


def solve_min_hertz_place(
    scenario: Scenario,
    seed: int,
    settings: HertzPlaceSettings | None = None,
) -> dict:
    """Keep-best min-Hertz layout. Never worse than k-means if include_kmeans."""
    ms = settings or HertzPlaceSettings()
    t0 = perf_counter()
    cands = _collect_candidates(scenario, seed, ms)
    if not cands:
        raise RuntimeError(f"e1b: no candidates for seed={seed}")
    best = max(cands, key=lambda r: _rank_key(float(r["bound_hz"])))
    pre_hz = float(best["bound_hz"])
    pre_kind = str(best["kind"])
    uav = np.asarray(best["uav_xyz_m"], dtype=float)
    polish_info: dict = {"n_moves": 0, "n_eval": 0, "bound_hz": pre_hz}
    if ms.polish:
        uav, polish_info = polish_hertz(scenario, uav, ms)
    verified = None
    if ms.verify_search:
        from types import SimpleNamespace

        alloc = hertz_bound_at_layout(scenario, uav)["allocation"]
        verified = min_bsys_frozen_q(
            scenario,
            uav,
            alloc,
            hi_hz=float(scenario.cfg.b_sys_hz),
            place_eval=SimpleNamespace(feasible=True, sum_rate_mbps=float("nan")),
        )
    kinds: dict[str, int] = {}
    for r in cands:
        kinds[str(r["kind"])] = kinds.get(str(r["kind"]), 0) + 1
    kmeans_hz = None
    for r in cands:
        if r["kind"] == "kmeans":
            kmeans_hz = float(r["bound_hz"])
            break
    min_hz = (
        None
        if verified is None
        else verified.min_b_sys_hz
    )
    if min_hz is None:
        min_hz = float(polish_info.get("bound_hz", pre_hz))
        if not _finite(min_hz):
            min_hz = None
    return {
        "seed": int(seed),
        "min_b_sys_hz": min_hz,
        "bound_hz": float(polish_info.get("bound_hz", pre_hz)),
        "pre_polish_bound_hz": pre_hz,
        "pre_polish_kind": pre_kind,
        "winner_kind": pre_kind if polish_info.get("n_moves", 0) == 0 else "polished",
        "kmeans_bound_hz": kmeans_hz,
        "n_candidates": len(cands),
        "candidate_kinds": kinds,
        "polish": {
            "n_moves": int(polish_info.get("n_moves", 0)),
            "n_eval": int(polish_info.get("n_eval", 0)),
        },
        "uav_xyz_m": np.asarray(uav, dtype=float).tolist(),
        "verified_message": None if verified is None else verified.message,
        "wall_clock_s": perf_counter() - t0,
    }
