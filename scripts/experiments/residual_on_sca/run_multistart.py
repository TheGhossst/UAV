"""Experiment A: is q even the hole?

Frozen k-means-init SCA vs four extra SCA starts (2 random + 2 k-means
with different placement seeds). Keep-best among extras. Same evaluate()
and primary 8.8 MHz / 25% cap geometry as the headline campaign.

Does not overwrite campaign_8.8mhz_cap25_si12k.json. Does not edit
src/uavdt/sca/.

Usage:
  python scripts/experiments/residual_on_sca/run_multistart.py
  python scripts/experiments/residual_on_sca/run_multistart.py --n-runs 2 --seed-start 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from _paths import OUT_MULTISTART, PROTECTED, bootstrap  # noqa: E402

bootstrap()

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.models import Allocation  # noqa: E402
from uavdt.placement.kmeans import place_kmeans  # noqa: E402
from uavdt.placement.random import place_random  # noqa: E402
from uavdt.resources import cpu_stable_processing, nearest_association  # noqa: E402
from uavdt.sca.algorithm import solve_sca  # noqa: E402
from uavdt.sca.initialize import qos_floor_bandwidth  # noqa: E402
from uavdt.sca.linearize import spectral_efficiency  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

OUT_DEFAULT = OUT_MULTISTART
RANDOM_BEAT_SCA = (7, 11, 13, 18, 19)
PRACTICAL_MBPS = 0.05
FLAT_MBPS = 0.01


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _cfg() -> SimConfig:
    return SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _rank_key(feasible: bool, rate_mbps: float) -> tuple[int, float]:
    return (int(bool(feasible)), float(rate_mbps))


def _mean_xy_dist_m(uav_a: np.ndarray, uav_b: np.ndarray) -> float:
    a = np.asarray(uav_a, dtype=float)[:, :2]
    b = np.asarray(uav_b, dtype=float)[:, :2]
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _alloc_at_positions(scenario, uav: np.ndarray) -> Allocation:
    a = nearest_association(scenario.iot_xyz_m, uav)
    b = cpu_stable_processing(scenario, a)
    se = spectral_efficiency(scenario.iot_xyz_m, uav, scenario.cfg)
    bw = qos_floor_bandwidth(a, se, scenario.cfg)
    return Allocation(a, b, bw)


def _record_from_result(kind: str, init_seed: int, result, wall_s: float, q0: np.ndarray) -> dict:
    ev = result.true_eval
    return {
        "kind": kind,
        "init_seed": int(init_seed),
        "sum_rate_Mbps": float(ev.sum_rate_mbps),
        "feasible": bool(ev.feasible),
        "n_iterations": int(result.n_iterations),
        "stop_reason": result.diagnostics.get("stop_reason"),
        "wall_clock_s": float(wall_s),
        "mean_xy_dist_to_frozen_m": _mean_xy_dist_m(result.uav_xyz_m, q0),
        "uav_xyz_m": np.asarray(result.uav_xyz_m, dtype=float).tolist(),
    }


def _failed_record(kind: str, init_seed: int, reason: str, q0: np.ndarray) -> dict:
    return {
        "kind": kind,
        "init_seed": int(init_seed),
        "sum_rate_Mbps": 0.0,
        "feasible": False,
        "n_iterations": 0,
        "stop_reason": reason,
        "wall_clock_s": 0.0,
        "mean_xy_dist_to_frozen_m": 0.0,
        "uav_xyz_m": np.asarray(q0, dtype=float).tolist(),
        "failed": True,
    }


def _run_sca(scenario, seed: int, settings: SCASettings, uav=None, allocation=None):
    t0 = perf_counter()
    try:
        result = solve_sca(
            scenario,
            seed,
            settings=settings,
            uav_xyz_m=uav,
            allocation=allocation,
        )
    except RuntimeError as exc:
        if "CPU stability" not in str(exc):
            raise
        return None, perf_counter() - t0, "init_cpu_unstable"
    return result, perf_counter() - t0, None


def extra_starts(seed: int) -> tuple[tuple[str, int], ...]:
    return (
        ("random", seed * 1000 + 1),
        ("random", seed * 1000 + 2),
        ("kmeans", seed * 1000 + 1),
        ("kmeans", seed * 1000 + 2),
    )


def run_seed(seed: int, cfg: SimConfig, settings: SCASettings) -> dict:
    sc = generate_scenario(seed, cfg)
    frozen, frozen_s, frozen_fail = _run_sca(sc, seed, settings)
    if frozen is None:
        q_ref = place_kmeans(sc, seed)
        frozen_rec = _failed_record("frozen_kmeans", seed, frozen_fail or "fail", q_ref)
        extras = [_failed_record(k, s, "skipped_frozen_fail", q_ref) for k, s in extra_starts(seed)]
        return {
            "seed": int(seed),
            "frozen": frozen_rec,
            "extras": extras,
            "best_extra": None,
            "best_all": frozen_rec,
            "delta_best_extra_Mbps": None,
            "delta_best_all_Mbps": 0.0,
        }

    q0 = np.asarray(frozen.uav_xyz_m, dtype=float)
    frozen_rec = _record_from_result("frozen_kmeans", seed, frozen, frozen_s, q0)
    extras: list[dict] = []
    for kind, init_seed in extra_starts(seed):
        if kind == "random":
            uav = place_random(sc.cfg.num_uav, init_seed, sc.cfg)
        else:
            uav = place_kmeans(sc, init_seed)
        try:
            alloc = _alloc_at_positions(sc, uav)
        except RuntimeError as exc:
            extras.append(_failed_record(kind, init_seed, str(exc), q0))
            continue
        result, wall_s, fail = _run_sca(sc, seed, settings, uav=uav, allocation=alloc)
        if result is None:
            extras.append(_failed_record(kind, init_seed, fail or "fail", q0))
            continue
        extras.append(_record_from_result(kind, init_seed, result, wall_s, q0))

    def better(a: dict, b: dict) -> bool:
        return _rank_key(a["feasible"], a["sum_rate_Mbps"]) > _rank_key(
            b["feasible"], b["sum_rate_Mbps"]
        )

    best_extra = extras[0]
    for row in extras[1:]:
        if better(row, best_extra):
            best_extra = row
    best_all = frozen_rec
    if better(best_extra, frozen_rec):
        best_all = best_extra
    d_extra = float(best_extra["sum_rate_Mbps"] - frozen_rec["sum_rate_Mbps"])
    d_all = float(best_all["sum_rate_Mbps"] - frozen_rec["sum_rate_Mbps"])
    extra_better = better(best_extra, frozen_rec)
    return {
        "seed": int(seed),
        "frozen": frozen_rec,
        "extras": extras,
        "best_extra": best_extra,
        "best_all": best_all,
        "delta_best_extra_Mbps": d_extra,
        "delta_best_all_Mbps": d_all,
        "extra_wins": bool(extra_better),
        "extra_wins_practical": bool(extra_better and d_extra > PRACTICAL_MBPS),
    }


def summarize(rows: list[dict]) -> dict:
    seeds = [int(r["seed"]) for r in rows]
    frozen = np.array([r["frozen"]["sum_rate_Mbps"] for r in rows], dtype=float)
    extra = np.array([r["best_extra"]["sum_rate_Mbps"] for r in rows], dtype=float)
    deltas = extra - frozen
    win_seeds = [int(r["seed"]) for r in rows if r.get("extra_wins")]
    practical = [int(r["seed"]) for r in rows if r.get("extra_wins_practical")]
    overlap = [s for s in win_seeds if s in RANDOM_BEAT_SCA]
    w = wilcoxon_signed_rank(deltas)
    tstat = paired_t(deltas)
    max_i = int(np.argmax(deltas)) if deltas.size else 0
    max_delta = float(np.max(deltas)) if deltas.size else 0.0
    local_or_far = []
    for r in rows:
        if not r.get("extra_wins"):
            continue
        dist = float(r["best_extra"]["mean_xy_dist_to_frozen_m"])
        local_or_far.append(
            {
                "seed": int(r["seed"]),
                "delta_Mbps": float(r["delta_best_extra_Mbps"]),
                "mean_xy_dist_to_frozen_m": dist,
                "basin": "local" if dist <= 15.0 else "distant",
                "kind": r["best_extra"]["kind"],
            }
        )
    n_distant = sum(1 for x in local_or_far if x["basin"] == "distant")
    extras_flat = max_delta <= FLAT_MBPS and w["p_greater"] >= 0.05
    return {
        "n": len(rows),
        "seeds": seeds,
        "mean_frozen_Mbps": float(np.mean(frozen)),
        "mean_best_extra_Mbps": float(np.mean(extra)),
        "mean_delta_Mbps": float(np.mean(deltas)),
        "std_delta_Mbps": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
        "max_delta_Mbps": max_delta,
        "max_delta_seed": int(seeds[max_i]) if seeds else None,
        "min_delta_Mbps": float(np.min(deltas)) if deltas.size else 0.0,
        "win_seeds": win_seeds,
        "n_extra_wins": len(win_seeds),
        "practical_win_seeds": practical,
        "random_beat_sca_seeds": list(RANDOM_BEAT_SCA),
        "overlap_random_beat_sca": overlap,
        "wilcoxon": w,
        "paired_t": tstat,
        "winning_inits": local_or_far,
        "n_distant_basin_wins": n_distant,
        "stop_rule": {
            "extras_flat": bool(extras_flat),
            "flat_threshold_Mbps": FLAT_MBPS,
            "justifies_residual_on_sca": bool(
                (not extras_flat) and (len(overlap) > 0 or len(win_seeds) > 0)
            ),
            "prefer_larger_box_or_cmaes": bool(n_distant > 0),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment A: multi-start SCA vs frozen k-means SCA")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.resolve() == PROTECTED.resolve():
        raise SystemExit(f"refusing to overwrite protected {PROTECTED}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _cfg()
    settings = SCASettings(solver=None, max_iterations=30)
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.n_runs)))
    _log(
        f"Experiment A  seeds={seeds[0]}..{seeds[-1]}  "
        f"B_sys=8.8 MHz  cap=25%  extras=2 random + 2 k-means"
    )
    rows = []
    for seed in seeds:
        _log(f"  seed {seed}")
        row = run_seed(seed, cfg, settings)
        d = row["delta_best_extra_Mbps"]
        flag = "WIN" if row.get("extra_wins") else "tie/loss"
        _log(
            f"    frozen={row['frozen']['sum_rate_Mbps']:.4f}  "
            f"best_extra={row['best_extra']['sum_rate_Mbps']:.4f}  "
            f"d={d:+.4f} Mbps  {flag}  kind={row['best_extra']['kind']}"
        )
        rows.append(row)
    summary = summarize(rows)
    payload = {
        "experiment": "A",
        "label": "frozen k-means SCA vs 4 extra SCA inits (keep-best extra)",
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "n_runs": int(args.n_runs),
        "seed_start": int(args.seed_start),
        "seeds": list(seeds),
        "random_beat_sca_seeds": list(RANDOM_BEAT_SCA),
        "per_seed": rows,
        "summary": summary,
        "notes": [
            "Does not overwrite campaign_8.8mhz_cap25_si12k.json.",
            "Will not produce a 1 Mbps win at 100 m / 25% cap.",
            "Will not fix T_k=0.8 s (same frozen nearest-a after each start).",
        ],
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    _log(f"wrote {out}")
    s = summary
    _log(
        f"mean d extra-frozen = {s['mean_delta_Mbps']:+.4f} +/- {s['std_delta_Mbps']:.4f} Mbps  "
        f"max={s['max_delta_Mbps']:+.4f} (seed {s['max_delta_seed']})  "
        f"wins={s['n_extra_wins']}/{s['n']}  "
        f"Wilcoxon p_greater={s['wilcoxon']['p_greater']:.4g}"
    )
    _log(f"win seeds: {s['win_seeds']}")
    _log(f"overlap random-beats-SCA {list(RANDOM_BEAT_SCA)}: {s['overlap_random_beat_sca']}")
    _log(f"stop_rule: {s['stop_rule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
