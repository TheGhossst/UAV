"""Cap-aware zenith-anchor enumeration + SCA polish. Does not edit frozen SCA.

Leftover-dump (27) plus a per-link cap puts leftover Hertz on about
``1/cap`` highest-SE links. SE is maximal at zenith, so a natural prior
is ``UAV j`` hovering over a distinct IoT. This module enumerates those
J-subsets (or beam-searches when ``C(I,J)`` is large), scores each with
one frozen-q bandwidth LP, polishes the top-K with unmodified
``solve_sca``, and keep-bests against frozen k-means SCA.

Opt-in method ``sca_anchor``. Frozen SCA stays the headline solver.
Never worse than one-shot SCA while ``include_frozen=True``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from uavdt.evaluator import evaluate
from uavdt.models import Allocation, Scenario
from uavdt.placement.kmeans import place_kmeans
from uavdt.resources import cpu_stable_processing, nearest_association
from uavdt.sca.algorithm import SCAResult, solve_sca
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q
from uavdt.sca.settings import SCASettings
from uavdt.scenario import make_uav_xyz_m


@dataclass(frozen=True)
class AnchorSettings:
    """Zenith-subset search knobs. Not Table II / Problem (P)."""

    top_k: int = 3
    max_enumerate: int = 1000
    beam_width: int = 10
    include_frozen: bool = True
    process_cohesive_candidate: bool = False
    # "enum" = combinatorial (full or beam). "random" = ablation control:
    # sample n_random J-subsets uniformly, LP-score, polish top-K.
    selection: str = "enum"
    n_random: int = 1


def n_anchor_combos(num_iot: int, num_uav: int) -> int:
    i = int(num_iot)
    j = int(num_uav)
    if j < 0 or i < 0 or j > i:
        return 0
    return int(math.comb(i, j))


def anchors_separated(xy: np.ndarray, theta_m: float) -> bool:
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    for p in range(n):
        for q in range(p + 1, n):
            if float(np.linalg.norm(pts[p] - pts[q])) < float(theta_m):
                return False
    return True


def repair_anchor_xy(
    xy: np.ndarray,
    theta_m: float,
    cfg,
) -> tuple[np.ndarray | None, int]:
    """Push later UAVs ``theta`` m along the connecting line.

    Used to be a hard skip when two host IoTs were closer than ``theta``.
    Skipping discards the subset; jitter keeps it and only fails if the
    field box cannot fit the pair.
    """
    out = np.asarray(xy, dtype=float).reshape(-1, 2).copy()
    theta = float(theta_m)
    n_jitter = 0
    n = out.shape[0]
    xmin, xmax = 0.0, float(cfg.area_x_m)
    ymin, ymax = 0.0, float(cfg.area_y_m)
    for q in range(1, n):
        for p in range(q):
            delta = out[q] - out[p]
            dist = float(np.linalg.norm(delta))
            if dist + 1e-12 >= theta:
                continue
            if dist < 1e-12:
                direction = np.array([1.0, 0.0], dtype=float)
            else:
                direction = delta / dist
            placed = False
            for sign in (1.0, -1.0):
                cand = out[p] + sign * direction * theta
                cand[0] = float(np.clip(cand[0], xmin, xmax))
                cand[1] = float(np.clip(cand[1], ymin, ymax))
                if float(np.linalg.norm(cand - out[p])) + 1e-9 < theta:
                    continue
                if all(
                    float(np.linalg.norm(cand - out[r])) + 1e-9 >= theta
                    for r in range(q)
                ):
                    out[q] = cand
                    n_jitter += 1
                    placed = True
                    break
            if not placed:
                return None, n_jitter
    if not anchors_separated(out, theta):
        return None, n_jitter
    return out, n_jitter


def leftover_dump_slots(max_bw_share: float | None) -> int:
    """How many cap-filling leftover links: ``ceil(1/c)``, or 1 if uncapped."""
    if max_bw_share is None or float(max_bw_share) <= 0.0:
        return 1
    return int(math.ceil(1.0 / float(max_bw_share) - 1e-12))


def leftover_dump_upper_bound(scenario: Scenario) -> dict:
    """Closed-form leftover-dump upper bound at frozen nearest-a.

    Every IoT gets its QoS floor. Leftover Hertz is then given to at most
    ``k = ceil(1/c)`` links (one link if there is no cap), each at most
    ``B_cap``. The J zenith hosts are scored at ``SE_zenith``; remaining
    dump links use SE to each IoT's nearest UAV among a host J-subset.
    The reported bound is the max over subsets, clipped at
    ``B_sys · SE_zenith``.
    """
    from uavdt.sca.linearize import spectral_efficiency

    cfg = scenario.cfg
    iot = scenario.iot_xyz_m
    i = int(scenario.num_iot)
    j = int(cfg.num_uav)
    r_min = float(cfg.r_min_bit_per_s)
    b_sys = float(cfg.b_sys_hz)
    b_cap = float(cfg.link_bandwidth_cap_hz)
    k = leftover_dump_slots(cfg.max_bw_share)
    h = float(cfg.uav_height_m)
    zenith_uav = make_uav_xyz_m(iot[0:1, :2], h)
    se_z = float(spectral_efficiency(iot[0:1], zenith_uav, cfg)[0, 0])
    radio_ceil = b_sys * se_z
    floors = float(i) * r_min
    leftover = max(0.0, b_sys - floors / max(se_z, 1e-15))

    def _rate_from_ses(dump_ses: list[float]) -> float:
        extra = 0.0
        pool = leftover
        for se_m in dump_ses:
            take = min(b_cap, pool)
            extra += float(se_m) * take
            pool -= take
            if pool <= 1e-12:
                break
        return min(radio_ceil, floors + extra)

    best_bps = 0.0
    best_combo: tuple[int, ...] | None = None
    if j <= 0 or i < j:
        best_bps = _rate_from_ses([se_z] * max(1, min(k, 1)))
    else:
        for combo in itertools.combinations(range(i), j):
            xy, _ = _zenith_xy(scenario, combo)
            if xy is None:
                continue
            uav = make_uav_xyz_m(xy, h)
            se = spectral_efficiency(iot, uav, cfg)
            se_i = se.max(axis=1)
            host = set(combo)
            remaining = sorted(
                (float(se_i[t]) for t in range(i) if t not in host),
                reverse=True,
            )
            n_host_dump = min(j, k)
            dump_ses = [se_z] * n_host_dump + remaining[: max(0, k - n_host_dump)]
            rate = _rate_from_ses(dump_ses)
            if rate > best_bps:
                best_bps = rate
                best_combo = tuple(int(x) for x in combo)

    return {
        "se_zenith": se_z,
        "k_dump": int(k),
        "b_cap_hz": b_cap,
        "radio_ceiling_bps": radio_ceil,
        "radio_ceiling_Mbps": radio_ceil / 1e6,
        "bound_bps": float(best_bps),
        "bound_Mbps": float(best_bps) / 1e6,
        "combo": best_combo,
    }


def _rank_key(feasible: bool, rate_mbps: float) -> tuple[int, float]:
    return (int(bool(feasible)), float(rate_mbps))


def _combo_seed(seed: int, combo: tuple[int, ...]) -> int:
    acc = (int(seed) * 1009 + 17) % 2_147_483_647
    for i in combo:
        acc = (acc * 131 + int(i) + 1) % 2_147_483_647
    return int(acc)


def lp_point(
    scenario: Scenario,
    uav: np.ndarray,
    settings: SCASettings | None = None,
    association: np.ndarray | None = None,
) -> tuple | None:
    a = (
        nearest_association(scenario.iot_xyz_m, uav)
        if association is None
        else np.asarray(association, dtype=float)
    )
    b = cpu_stable_processing(scenario, a)
    res = solve_bandwidth_at_fixed_q(
        scenario, uav, a, b, settings or SCASettings(solver=None)
    )
    if res.infeasible:
        return None
    alloc = Allocation(a, b, res.bandwidth_hz)
    ev = evaluate(scenario, uav, alloc)
    return ev, alloc


def _zenith_xy(
    scenario: Scenario, combo: tuple[int, ...]
) -> tuple[np.ndarray | None, int]:
    if not combo:
        return None, 0
    xy = scenario.iot_xyz_m[list(combo), :2].astype(float, copy=True)
    return repair_anchor_xy(xy, scenario.cfg.uav_min_separation_m, scenario.cfg)


def _repair_pad_rows(
    xy: np.ndarray,
    n_fixed: int,
    cfg,
    rng: np.random.Generator,
    max_tries: int = 5_000,
) -> np.ndarray | None:
    out = xy.copy()
    theta = float(cfg.uav_min_separation_m)
    for j in range(int(n_fixed), out.shape[0]):
        for _ in range(max_tries):
            if j == 0:
                break
            d = np.linalg.norm(out[:j] - out[j][None, :], axis=1)
            if np.all(d >= theta):
                break
            out[j] = np.array(
                [
                    rng.uniform(0.0, cfg.area_x_m),
                    rng.uniform(0.0, cfg.area_y_m),
                ]
            )
        else:
            return None
    out[:, 0] = np.clip(out[:, 0], 0.0, cfg.area_x_m)
    out[:, 1] = np.clip(out[:, 1], 0.0, cfg.area_y_m)
    return out


def uav_for_combo(
    scenario: Scenario,
    combo: tuple[int, ...],
    seed: int,
) -> np.ndarray | None:
    """Zenith over ``combo``. Pad leftover UAVs with k-means if ``len < J``."""
    cfg = scenario.cfg
    j = int(cfg.num_uav)
    t = len(combo)
    if t == 0 or t > j:
        return None
    xy_anchor, _n_jit = _zenith_xy(scenario, combo)
    if xy_anchor is None:
        return None
    if t == j:
        return make_uav_xyz_m(xy_anchor, cfg.uav_height_m)
    extra = place_kmeans(scenario, _combo_seed(seed, combo), num_uav=j - t)
    xy = np.vstack([xy_anchor, extra[:, :2]])
    rng = np.random.default_rng(_combo_seed(seed, combo) + 7)
    xy = _repair_pad_rows(xy, t, cfg, rng)
    if xy is None:
        return None
    return make_uav_xyz_m(xy, cfg.uav_height_m)


def _row(
    feasible: int,
    rate_bps: float,
    combo: tuple[int, ...],
    uav: np.ndarray,
    alloc: Allocation,
    assoc_kind: str = "nearest",
) -> dict:
    return {
        "feasible": bool(feasible),
        "sum_rate_bit_per_s": float(rate_bps),
        "sum_rate_Mbps": float(rate_bps) / 1e6,
        "combo": tuple(int(i) for i in combo),
        "uav_xyz_m": np.asarray(uav, dtype=float),
        "allocation": alloc,
        "assoc_kind": assoc_kind,
    }


def _lp_row(
    scenario: Scenario,
    uav: np.ndarray,
    combo: tuple[int, ...],
    settings: SCASettings,
    association: np.ndarray | None,
    assoc_kind: str,
) -> tuple[dict | None, int]:
    out = lp_point(scenario, uav, settings, association=association)
    if out is None:
        return None, 1
    ev, alloc = out
    return (
        _row(
            int(ev.feasible),
            float(ev.sum_rate_bit_per_s),
            combo,
            uav,
            alloc,
            assoc_kind,
        ),
        1,
    )


def _score_one(
    scenario: Scenario,
    combo: tuple[int, ...],
    seed: int,
    settings: SCASettings,
    anchor: AnchorSettings | None = None,
) -> tuple[dict | None, int, int, int]:
    """Return ``(row, n_lp, n_sep_skip, n_jitter)``."""
    xy_anchor, n_jit = _zenith_xy(scenario, combo)
    if xy_anchor is None:
        return None, 0, 1, int(n_jit)
    uav = uav_for_combo(scenario, combo, seed)
    if uav is None:
        return None, 0, 1, int(n_jit)
    ms = anchor or AnchorSettings()
    rows: list[dict] = []
    n_lp = 0
    row, used = _lp_row(scenario, uav, combo, settings, None, "nearest")
    n_lp += used
    if row is not None:
        row["n_jittered_sep"] = int(n_jit)
        rows.append(row)
    if ms.process_cohesive_candidate:
        from uavdt.sca_joint.rematch import centroid_cohesive_association

        a = centroid_cohesive_association(scenario, uav)
        row, used = _lp_row(scenario, uav, combo, settings, a, "process_cohesive")
        n_lp += used
        if row is not None:
            row["n_jittered_sep"] = int(n_jit)
            rows.append(row)
    if not rows:
        return None, n_lp, 0, int(n_jit)
    rows.sort(
        key=lambda r: (int(r["feasible"]), float(r["sum_rate_bit_per_s"])),
        reverse=True,
    )
    return rows[0], n_lp, 0, int(n_jit)


def _collect(
    scenario: Scenario,
    combos: list[tuple[int, ...]],
    seed: int,
    top_k: int,
    settings: SCASettings,
    anchor: AnchorSettings | None = None,
) -> tuple[list[dict], int, int, int]:
    rows: list[dict] = []
    n_lp = 0
    n_skip = 0
    n_jit = 0
    for combo in combos:
        row, used, skipped, jittered = _score_one(
            scenario, combo, seed, settings, anchor
        )
        n_lp += used
        n_skip += skipped
        n_jit += jittered
        if row is None:
            continue
        rows.append(row)
    rows.sort(
        key=lambda r: (int(r["feasible"]), float(r["sum_rate_bit_per_s"])),
        reverse=True,
    )
    return rows[: max(0, int(top_k))], n_lp, n_skip, n_jit


def _full_enum(
    scenario: Scenario,
    seed: int,
    top_k: int,
    settings: SCASettings,
    anchor: AnchorSettings | None = None,
) -> tuple[list[dict], int, int, int]:
    i = scenario.num_iot
    j = int(scenario.cfg.num_uav)
    combos = list(itertools.combinations(range(i), j))
    return _collect(scenario, combos, seed, top_k, settings, anchor)


def _random_enum(
    scenario: Scenario,
    seed: int,
    top_k: int,
    settings: SCASettings,
    anchor: AnchorSettings,
) -> tuple[list[dict], int, int, int]:
    i = scenario.num_iot
    j = int(scenario.cfg.num_uav)
    n_all = n_anchor_combos(i, j)
    want = max(1, int(anchor.n_random))
    if n_all == 0:
        return [], 0, 0, 0
    if want >= n_all:
        combos = list(itertools.combinations(range(i), j))
    else:
        rng = np.random.default_rng(int(seed) + 4001)
        seen: set[tuple[int, ...]] = set()
        combos = []
        while len(combos) < want:
            combo = tuple(sorted(int(x) for x in rng.choice(i, size=j, replace=False)))
            if combo in seen:
                continue
            seen.add(combo)
            combos.append(combo)
    return _collect(scenario, combos, seed, top_k, settings, anchor)


def _beam_search(
    scenario: Scenario,
    seed: int,
    settings: AnchorSettings,
    sca_settings: SCASettings,
) -> tuple[list[dict], int, int, int]:
    i = scenario.num_iot
    j = int(scenario.cfg.num_uav)
    width = max(1, int(settings.beam_width))
    beam: list[tuple[int, ...]] = [()]
    n_lp = 0
    n_skip = 0
    n_jit = 0
    last: list[dict] = []
    for depth in range(j):
        scored: list[dict] = []
        seen: set[tuple[int, ...]] = set()
        for partial in beam:
            used = set(partial)
            for idx in range(i):
                if idx in used:
                    continue
                combo = tuple(sorted(partial + (idx,)))
                if combo in seen:
                    continue
                seen.add(combo)
                row, lp_used, skipped, jittered = _score_one(
                    scenario, combo, seed, sca_settings, settings
                )
                n_lp += lp_used
                n_skip += skipped
                n_jit += jittered
                if row is None:
                    continue
                scored.append(row)
        scored.sort(
            key=lambda r: (int(r["feasible"]), float(r["sum_rate_bit_per_s"])),
            reverse=True,
        )
        beam = [r["combo"] for r in scored[:width]]
        last = scored
        if not beam:
            break
    last.sort(
        key=lambda r: (int(r["feasible"]), float(r["sum_rate_bit_per_s"])),
        reverse=True,
    )
    return last[: max(0, int(settings.top_k))], n_lp, n_skip, n_jit


def score_anchor_combos(
    scenario: Scenario,
    seed: int = 1,
    settings: AnchorSettings | None = None,
    sca_settings: SCASettings | None = None,
) -> dict:
    """LP-score zenith J-subsets. Does not run SCA polish."""
    ms = settings or AnchorSettings()
    sca = sca_settings or SCASettings(solver=None)
    i = scenario.num_iot
    j = int(scenario.cfg.num_uav)
    n_combos = n_anchor_combos(i, j)
    if j <= 0 or n_combos == 0 or int(ms.top_k) <= 0:
        return {
            "mode": "none",
            "n_combos": n_combos,
            "n_lp": 0,
            "n_skipped_sep": 0,
            "n_jittered_sep": 0,
            "top": [],
        }
    sel = str(ms.selection or "enum").lower()
    if sel == "random":
        top, n_lp, n_skip, n_jit = _random_enum(
            scenario, seed, ms.top_k, sca, ms
        )
        mode = "random"
    elif n_combos <= int(ms.max_enumerate):
        top, n_lp, n_skip, n_jit = _full_enum(scenario, seed, ms.top_k, sca, ms)
        mode = "full"
    else:
        top, n_lp, n_skip, n_jit = _beam_search(scenario, seed, ms, sca)
        mode = "beam"
    return {
        "mode": mode,
        "n_combos": n_combos,
        "n_lp": int(n_lp),
        "n_skipped_sep": int(n_skip),
        "n_jittered_sep": int(n_jit),
        "top": top,
    }


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


def solve_sca_anchor(
    scenario: Scenario,
    seed: int,
    settings: SCASettings | None = None,
    anchor: AnchorSettings | None = None,
) -> SCAResult:
    """Enumerate zenith anchors, polish top-K, keep-best with frozen SCA."""
    sca_settings = settings or SCASettings()
    ms = anchor or AnchorSettings()
    scored = score_anchor_combos(
        scenario, seed, settings=ms, sca_settings=sca_settings
    )
    candidates: list[tuple[SCAResult | None, dict]] = []

    if ms.include_frozen:
        frozen, wall_s, fail = _run_one(scenario, seed, sca_settings)
        frozen_row = {
            "kind": "frozen",
            "combo": None,
            "lp_Mbps": None,
            "sum_rate_Mbps": (
                0.0 if frozen is None else float(frozen.true_eval.sum_rate_mbps)
            ),
            "feasible": False if frozen is None else bool(frozen.true_eval.feasible),
            "n_iterations": 0 if frozen is None else int(frozen.n_iterations),
            "wall_clock_s": float(wall_s),
            "failed": frozen is None,
            "assoc_kind": "nearest",
            "stop_reason": fail if frozen is None else frozen.diagnostics.get("stop_reason"),
        }
        candidates.append((frozen, frozen_row))

    if ms.include_frozen and ms.process_cohesive_candidate:
        from uavdt.placement.kmeans import place_kmeans as _place_kmeans
        from uavdt.sca_joint.rematch import centroid_cohesive_association

        uav_f = _place_kmeans(scenario, seed)
        a_c = centroid_cohesive_association(scenario, uav_f)
        out_c = lp_point(scenario, uav_f, sca_settings, association=a_c)
        if out_c is not None:
            _, alloc_c = out_c
            extra, wall_s, fail = _run_one(
                scenario,
                seed,
                sca_settings,
                uav=uav_f,
                allocation=alloc_c,
            )
            rec = {
                "kind": "frozen_cohesive",
                "combo": None,
                "assoc_kind": "process_cohesive",
                "lp_Mbps": float(out_c[0].sum_rate_mbps),
                "sum_rate_Mbps": (
                    0.0 if extra is None else float(extra.true_eval.sum_rate_mbps)
                ),
                "feasible": False if extra is None else bool(extra.true_eval.feasible),
                "n_iterations": 0 if extra is None else int(extra.n_iterations),
                "wall_clock_s": float(wall_s),
                "failed": extra is None,
                "stop_reason": fail
                if extra is None
                else extra.diagnostics.get("stop_reason"),
            }
            candidates.append((extra, rec))

    def _assoc_labels(uav: np.ndarray) -> tuple[int, ...]:
        a = nearest_association(scenario.iot_xyz_m, uav)
        return tuple(int(x) for x in np.argmax(a, axis=1))

    polished: list[dict] = []
    for row in scored["top"]:
        uav0 = np.asarray(row["uav_xyz_m"], dtype=float)
        extra, wall_s, fail = _run_one(
            scenario,
            seed,
            sca_settings,
            uav=uav0,
            allocation=row["allocation"],
        )
        combo = [int(i) for i in row["combo"]]
        if extra is None:
            polished.append(
                {
                    "kind": "anchor",
                    "combo": combo,
                    "assoc_kind": row.get("assoc_kind", "nearest"),
                    "lp_Mbps": float(row["sum_rate_Mbps"]),
                    "sum_rate_Mbps": 0.0,
                    "feasible": False,
                    "n_iterations": 0,
                    "wall_clock_s": float(wall_s),
                    "failed": True,
                    "stop_reason": fail or "fail",
                    "polish_mean_disp_m": None,
                    "polish_max_disp_m": None,
                    "polish_assoc_changed": None,
                }
            )
            candidates.append((None, polished[-1]))
            continue
        disp = np.linalg.norm(extra.uav_xyz_m[:, :2] - uav0[:, :2], axis=1)
        rec = {
            "kind": "anchor",
            "combo": combo,
            "assoc_kind": row.get("assoc_kind", "nearest"),
            "lp_Mbps": float(row["sum_rate_Mbps"]),
            "sum_rate_Mbps": float(extra.true_eval.sum_rate_mbps),
            "feasible": bool(extra.true_eval.feasible),
            "n_iterations": int(extra.n_iterations),
            "wall_clock_s": float(wall_s),
            "failed": False,
            "stop_reason": extra.diagnostics.get("stop_reason"),
            "polish_mean_disp_m": float(np.mean(disp)),
            "polish_max_disp_m": float(np.max(disp)),
            "polish_assoc_changed": _assoc_labels(uav0) != _assoc_labels(extra.uav_xyz_m),
        }
        polished.append(rec)
        candidates.append((extra, rec))

    best_i: int | None = None
    best_key: tuple[int, float] | None = None
    for i, (result, rec) in enumerate(candidates):
        if result is None:
            continue
        key = _rank_key(bool(rec["feasible"]), float(rec["sum_rate_Mbps"]))
        if best_key is None or key > best_key:
            best_key = key
            best_i = i

    if best_i is None or candidates[best_i][0] is None:
        raise RuntimeError(
            f"sca_anchor: every candidate failed for seed={seed} "
            f"(n_cand={len(candidates)})"
        )

    winner, win_row = candidates[best_i]
    frozen_row = next((r for _, r in candidates if r["kind"] == "frozen"), None)
    frozen_mbps = (
        None if frozen_row is None else float(frozen_row["sum_rate_Mbps"])
    )
    best_mbps = float(win_row["sum_rate_Mbps"])
    lp_best = None
    if scored["top"]:
        lp_best = float(scored["top"][0]["sum_rate_Mbps"])
    polish_ok = [
        r
        for r in polished
        if not r.get("failed") and r.get("polish_mean_disp_m") is not None
    ]
    bound = leftover_dump_upper_bound(scenario)
    diag = dict(winner.diagnostics)
    diag.update(
        {
            "method": "sca_anchor",
            "enum_mode": scored["mode"],
            "selection": str(ms.selection),
            "n_random": int(ms.n_random),
            "n_combos": scored["n_combos"],
            "n_lp": scored["n_lp"],
            "n_skipped_sep": scored["n_skipped_sep"],
            "n_jittered_sep": scored.get("n_jittered_sep", 0),
            "top_k": int(ms.top_k),
            "beam_width": int(ms.beam_width),
            "max_enumerate": int(ms.max_enumerate),
            "winner_kind": win_row["kind"],
            "winner_combo": win_row.get("combo"),
            "winner_assoc_kind": win_row.get("assoc_kind"),
            "process_cohesive_candidate": bool(ms.process_cohesive_candidate),
            "lp_best_Mbps": lp_best,
            "frozen_Mbps": frozen_mbps,
            "best_Mbps": best_mbps,
            "delta_vs_frozen_Mbps": (
                None if frozen_mbps is None else best_mbps - frozen_mbps
            ),
            "winner_polish_mean_disp_m": win_row.get("polish_mean_disp_m"),
            "winner_polish_max_disp_m": win_row.get("polish_max_disp_m"),
            "winner_polish_assoc_changed": win_row.get("polish_assoc_changed"),
            "polish_mean_disp_m": (
                float(np.mean([r["polish_mean_disp_m"] for r in polish_ok]))
                if polish_ok
                else None
            ),
            "polish_frac_assoc_changed": (
                float(np.mean([int(bool(r["polish_assoc_changed"])) for r in polish_ok]))
                if polish_ok
                else None
            ),
            "bound_Mbps": bound["bound_Mbps"],
            "bound_combo": list(bound["combo"]) if bound["combo"] is not None else None,
            "se_zenith": bound["se_zenith"],
            "radio_ceiling_Mbps": bound["radio_ceiling_Mbps"],
            "gap_vs_bound_Mbps": (
                None if bound["bound_Mbps"] is None else bound["bound_Mbps"] - best_mbps
            ),
            "polished": polished,
            "wall_clock_s": float(
                sum(float(r["wall_clock_s"]) for _, r in candidates)
            ),
        }
    )
    winner.diagnostics = diag
    return winner
