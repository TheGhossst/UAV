"""Default-point eval: sca_multistart vs frozen SCA and random.

Does not overwrite campaign_8.8mhz_cap25_si12k.json, the 500 m campaign,
or Experiment A results/residual_on_sca/multistart_n20.json.

Usage:
  python scripts/run_sca_multistart_eval.py
  python scripts/run_sca_multistart_eval.py --n-runs 2 --seed-start 1
  python scripts/run_sca_multistart_eval.py --area-m 500 --out results/sca_multistart_n20_500m.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_multistart import MultiStartSettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PROTECTED = (
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "residual_on_sca" / "multistart_n20.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
)
OUT_DEFAULT = ROOT / "results" / "sca_multistart_n20.json"
RANDOM_BEAT_SCA = (7, 11, 13, 18, 19)
PRACTICAL_MBPS = 0.05
METHODS = ("random", "sca", "sca_multistart")


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


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


def _pair(rows: list[dict], left: str, right: str) -> dict:
    d = np.array(
        [float(r[left]["sum_rate_Mbps"]) - float(r[right]["sum_rate_Mbps"]) for r in rows],
        dtype=float,
    )
    wins = int(np.sum(d > 1e-9))
    losses = int(np.sum(d < -1e-9))
    w = wilcoxon_signed_rank(d)
    tstat = paired_t(d)
    loss_seeds = [int(r["seed"]) for r, delta in zip(rows, d) if delta < -1e-9]
    return {
        "mean_delta_Mbps": float(np.mean(d)),
        "std_delta_Mbps": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "wins": wins,
        "losses": losses,
        "n": int(d.size),
        "loss_seeds": loss_seeds,
        "wilcoxon": w,
        "paired_t": tstat,
        "per_seed_delta_Mbps": d.tolist(),
    }


def _refuse_protected(out: Path) -> None:
    resolved = out.resolve()
    for path in PROTECTED:
        if resolved == path.resolve():
            raise SystemExit(f"refusing to overwrite protected {path}")


def n20_complete(path: Path, n_runs: int, seed_start: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    seeds = payload.get("seeds") or []
    want = list(range(int(seed_start), int(seed_start) + int(n_runs)))
    return list(seeds) == want and "sca_multistart" in (payload.get("mean_Mbps") or {})


def run_n20_eval(
    *,
    out: Path,
    n_runs: int = 20,
    seed_start: int = 1,
    area_m: float = 100.0,
    max_bw_share: float = PRIMARY_MAX_BW_SHARE,
    n_random: int = 2,
    n_kmeans: int = 2,
    skip_if_complete: bool = False,
) -> dict:
    _refuse_protected(out)
    if skip_if_complete and n20_complete(out, n_runs, seed_start):
        payload = json.loads(out.read_text(encoding="utf-8"))
        _log(f"skip complete n20 {out}")
        return payload
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    cfg = SimConfig(
        b_sys_hz=8.8e6, max_bw_share=float(max_bw_share)
    ).with_square_area_m(float(area_m))
    sca_settings = SCASettings(solver=None, max_iterations=30)
    ms = MultiStartSettings(
        n_random=int(n_random),
        n_kmeans=int(n_kmeans),
        include_frozen=True,
    )
    seeds = tuple(range(int(seed_start), int(seed_start) + int(n_runs)))
    done: dict[str, dict] = {}
    if ckpt.exists():
        done = json.loads(ckpt.read_text(encoding="utf-8"))
        _log(f"resume {len(done)}/{len(seeds)} seeds from {ckpt}")
    cap = "none" if cfg.max_bw_share is None else f"{cfg.max_bw_share:.0%}"
    _log(
        f"sca_multistart eval  seeds={seeds[0]}..{seeds[-1]}  "
        f"area={cfg.area_x_m:g}x{cfg.area_y_m:g} m  "
        f"B_sys=8.8 MHz  cap={cap}  extras={ms.n_random} random + {ms.n_kmeans} k-means"
    )
    for seed in seeds:
        key = str(seed)
        if key in done:
            continue
        _log(f"  seed {seed}")
        sc = generate_scenario(seed, cfg)
        rec: dict = {"seed": int(seed)}
        for method in METHODS:
            t0 = perf_counter()
            run = run_method(
                sc,
                method,
                seed,
                sca_settings=sca_settings,
                multistart_settings=ms if method == "sca_multistart" else None,
            )
            rec[method] = {
                "sum_rate_Mbps": float(run.sum_rate_mbps),
                "feasible": bool(run.feasible),
                "wall_clock_s": float(perf_counter() - t0),
                "diagnostics": {
                    k: run.diagnostics.get(k)
                    for k in (
                        "stop_reason",
                        "accepted_steps",
                        "n_iterations",
                        "winner_kind",
                        "winner_init_seed",
                        "delta_vs_frozen_Mbps",
                        "n_starts",
                    )
                    if k in run.diagnostics
                },
            }
            _log(
                f"    {method:16s}  {run.sum_rate_mbps:.4f} Mbps  "
                f"feas={int(run.feasible)}  {perf_counter() - t0:.1f}s"
            )
        d_s = rec["sca_multistart"]["sum_rate_Mbps"] - rec["sca"]["sum_rate_Mbps"]
        d_r = rec["sca_multistart"]["sum_rate_Mbps"] - rec["random"]["sum_rate_Mbps"]
        rec["delta_vs_sca_Mbps"] = d_s
        rec["delta_vs_random_Mbps"] = d_r
        rec["closes_random_loss"] = bool(seed in RANDOM_BEAT_SCA and d_r > -1e-9)
        _log(
            f"    vs SCA {d_s:+.4f}  vs random {d_r:+.4f}  "
            f"kind={rec['sca_multistart']['diagnostics'].get('winner_kind')}"
        )
        done[key] = rec
        ckpt.write_text(json.dumps(_jsonable(done), indent=2), encoding="utf-8")

    rows = [done[str(s)] for s in seeds]
    vs_sca = _pair(rows, "sca_multistart", "sca")
    vs_random = _pair(rows, "sca_multistart", "random")
    vs_sca_rand = _pair(rows, "sca", "random")
    beat_sca = [
        int(r["seed"])
        for r in rows
        if float(r["sca"]["sum_rate_Mbps"]) + 1e-9 < float(r["random"]["sum_rate_Mbps"])
    ]
    overlap = [
        int(r["seed"])
        for r in rows
        if int(r["seed"]) in beat_sca and float(r["delta_vs_random_Mbps"]) > -1e-9
    ]
    still_lose = [
        int(r["seed"])
        for r in rows
        if int(r["seed"]) in beat_sca and float(r["delta_vs_random_Mbps"]) < -1e-9
    ]
    kinds: dict[str, int] = {}
    for r in rows:
        kind = str(r["sca_multistart"]["diagnostics"].get("winner_kind") or "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
    walls = [float(r["sca_multistart"]["wall_clock_s"]) for r in rows]
    payload = {
        "label": (
            "sca_multistart vs frozen SCA and random "
            f"(default J=3, {cfg.area_x_m:g}x{cfg.area_y_m:g} m)"
        ),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "area_m": [cfg.area_x_m, cfg.area_y_m],
        "n_runs": int(n_runs),
        "seed_start": int(seed_start),
        "seeds": list(seeds),
        "multistart": {
            "n_random": ms.n_random,
            "n_kmeans": ms.n_kmeans,
            "include_frozen": ms.include_frozen,
        },
        "random_beat_sca_seeds": beat_sca,
        "known_100m_random_beat_sca": list(RANDOM_BEAT_SCA),
        "per_seed": rows,
        "mean_Mbps": {
            m: float(np.mean([r[m]["sum_rate_Mbps"] for r in rows]))
            for m in METHODS
        },
        "mean_wall_clock_s": {
            m: float(np.mean([r[m]["wall_clock_s"] for r in rows]))
            for m in METHODS
        },
        "winner_kinds": kinds,
        "vs_sca": vs_sca,
        "vs_random": vs_random,
        "sca_vs_random": vs_sca_rand,
        "overlap_random_beat_sca_now_tied_or_won": overlap,
        "still_lose_to_random": still_lose,
        "n_practical_vs_sca": int(
            sum(1 for r in rows if r["delta_vs_sca_Mbps"] > PRACTICAL_MBPS)
        ),
        "mean_multistart_wall_s": float(np.mean(walls)),
        "notes": [
            "Does not overwrite campaign_8.8mhz_cap25_si12k.json.",
            "Does not overwrite campaign_8.8mhz_cap25_si12k_500m.json.",
            "Does not overwrite results/residual_on_sca/multistart_n20.json.",
            "Keep-best includes the frozen k-means start, so vs SCA Δ >= 0.",
        ],
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}")
    _log(
        f"means  SCA={payload['mean_Mbps']['sca']:.4f}  "
        f"multistart={payload['mean_Mbps']['sca_multistart']:.4f}  "
        f"random={payload['mean_Mbps']['random']:.4f}"
    )
    _log(
        f"vs SCA     {vs_sca['mean_delta_Mbps']:+.4f} +/- {vs_sca['std_delta_Mbps']:.4f}  "
        f"{vs_sca['wins']}/{vs_sca['n']}  p={vs_sca['wilcoxon']['p_greater']:.4g}"
    )
    _log(
        f"vs random  {vs_random['mean_delta_Mbps']:+.4f} +/- {vs_random['std_delta_Mbps']:.4f}  "
        f"{vs_random['wins']}/{vs_random['n']}  p={vs_random['wilcoxon']['p_greater']:.4g}"
    )
    _log(f"random-beats-SCA seeds now tied/won: {overlap}")
    _log(f"still lose to random: {still_lose}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="sca_multistart vs frozen SCA and random at default J=3"
    )
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    parser.add_argument("--area-m", type=float, default=100.0)
    parser.add_argument("--max-bw-share", type=float, default=PRIMARY_MAX_BW_SHARE)
    parser.add_argument("--multistart-random", type=int, default=2)
    parser.add_argument("--multistart-kmeans", type=int, default=2)
    args = parser.parse_args(argv)
    run_n20_eval(
        out=Path(args.out),
        n_runs=int(args.n_runs),
        seed_start=int(args.seed_start),
        area_m=float(args.area_m),
        max_bw_share=float(args.max_bw_share),
        n_random=int(args.multistart_random),
        n_kmeans=int(args.multistart_kmeans),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
