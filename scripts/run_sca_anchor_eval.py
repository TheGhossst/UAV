"""Default-point eval: sca_anchor vs frozen SCA, multi-start, and baselines.

Does not overwrite campaign_8.8mhz_cap25_si12k.json, the 500 m campaign,
or n100/eval.json.

Usage:
  python scripts/run_sca_anchor_eval.py
  python scripts/run_sca_anchor_eval.py --area-m 500 --out results/sca_anchor_n20_500m.json
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
from uavdt.sca_anchor import AnchorSettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PROTECTED = (
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "residual_on_sca" / "multistart_n20.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
)
OUT_DEFAULT = ROOT / "results" / "sca_anchor_n20.json"
PRACTICAL_MBPS = 0.05
COMPARE_METHODS = (
    "random",
    "kmeans",
    "pso",
    "sca",
    "sca_multistart",
    "sca_anchor",
)


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
        "n_practical": int(np.sum(d > PRACTICAL_MBPS)),
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
    return list(seeds) == want and "sca_anchor" in (payload.get("mean_Mbps") or {})


def _campaign_j3(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for pt in payload.get("points") or []:
        if pt.get("axis") == "uavs" and int(float(pt.get("x", -1))) == 3:
            return pt.get("by_method") or {}
    return None


def _baseline_cell(by_method: dict, method: str, index: int) -> dict | None:
    stats = by_method.get(method) or {}
    rates = stats.get("per_seed_Mbps") or []
    if index >= len(rates):
        return None
    feas = stats.get("per_seed_feasible") or []
    return {
        "sum_rate_Mbps": float(rates[index]),
        "feasible": bool(feas[index]) if index < len(feas) else True,
        "wall_clock_s": None,
        "source": "campaign",
        "diagnostics": {},
    }


def _multistart_row(path: Path, seed: int) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("per_seed") or []:
        if int(row.get("seed", -1)) == int(seed):
            cell = dict(row.get("sca_multistart") or {})
            if not cell:
                return None
            cell.setdefault("source", "sca_multistart_n20")
            return cell
    return None


_DIAG_KEYS = (
    "stop_reason",
    "n_iterations",
    "winner_kind",
    "winner_combo",
    "winner_assoc_kind",
    "enum_mode",
    "selection",
    "n_lp",
    "n_skipped_sep",
    "n_jittered_sep",
    "lp_best_Mbps",
    "delta_vs_frozen_Mbps",
    "frozen_Mbps",
    "best_Mbps",
    "winner_polish_mean_disp_m",
    "winner_polish_max_disp_m",
    "winner_polish_assoc_changed",
    "polish_mean_disp_m",
    "polish_frac_assoc_changed",
    "bound_Mbps",
    "bound_combo",
    "se_zenith",
    "radio_ceiling_Mbps",
    "gap_vs_bound_Mbps",
    "polished",
    "top_k",
)


def run_n20_eval(
    *,
    out: Path,
    n_runs: int = 20,
    seed_start: int = 1,
    area_m: float = 100.0,
    max_bw_share: float | None = PRIMARY_MAX_BW_SHARE,
    los_angle_unit: str = "rad",
    skip_if_complete: bool = False,
    campaign_path: Path | None = None,
    multistart_path: Path | None = None,
    anchor_settings: AnchorSettings | None = None,
    run_baselines: bool = True,
) -> dict:
    _refuse_protected(out)
    if skip_if_complete and n20_complete(out, n_runs, seed_start):
        payload = json.loads(out.read_text(encoding="utf-8"))
        _log(f"skip complete n20 {out}")
        return payload
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    cfg = SimConfig(
        b_sys_hz=8.8e6,
        max_bw_share=max_bw_share,
        los_angle_unit=los_angle_unit,  # type: ignore[arg-type]
    ).with_square_area_m(float(area_m))
    sca_settings = SCASettings(solver=None, max_iterations=30)
    anc = anchor_settings or AnchorSettings()
    seeds = tuple(range(int(seed_start), int(seed_start) + int(n_runs)))
    done: dict[str, dict] = {}
    if ckpt.exists():
        done = json.loads(ckpt.read_text(encoding="utf-8"))
        _log(f"resume {len(done)}/{len(seeds)} seeds from {ckpt}")
    camp = _campaign_j3(campaign_path) if campaign_path else None
    cap = "none" if cfg.max_bw_share is None else f"{cfg.max_bw_share:.0%}"
    _log(
        f"sca_anchor eval  seeds={seeds[0]}..{seeds[-1]}  "
        f"area={cfg.area_x_m:g}x{cfg.area_y_m:g} m  "
        f"B_sys=8.8 MHz  cap={cap}  top_k={anc.top_k}"
    )
    for i, seed in enumerate(seeds):
        key = str(seed)
        if key in done and "sca_anchor" in done[key]:
            continue
        _log(f"  seed {seed}")
        sc = generate_scenario(seed, cfg)
        rec: dict = {"seed": int(seed)}
        t0 = perf_counter()
        run = run_method(
            sc,
            "sca_anchor",
            seed,
            sca_settings=sca_settings,
            anchor_settings=anc,
        )
        rec["sca_anchor"] = {
            "sum_rate_Mbps": float(run.sum_rate_mbps),
            "feasible": bool(run.feasible),
            "wall_clock_s": float(perf_counter() - t0),
            "source": "run",
            "diagnostics": {
                k: run.diagnostics.get(k)
                for k in _DIAG_KEYS
                if k in run.diagnostics
            },
        }
        _log(
            f"    sca_anchor       {run.sum_rate_mbps:.4f} Mbps  "
            f"feas={int(run.feasible)}  {perf_counter() - t0:.1f}s  "
            f"kind={run.diagnostics.get('winner_kind')}  "
            f"n_lp={run.diagnostics.get('n_lp')}"
        )
        if run_baselines:
            if camp is not None:
                for method in ("random", "kmeans", "pso", "sca"):
                    cell = _baseline_cell(camp, method, i)
                    if cell is not None:
                        rec[method] = cell
            else:
                for method in ("random", "kmeans", "pso", "sca"):
                    t1 = perf_counter()
                    base = run_method(sc, method, seed, sca_settings=sca_settings)
                    rec[method] = {
                        "sum_rate_Mbps": float(base.sum_rate_mbps),
                        "feasible": bool(base.feasible),
                        "wall_clock_s": float(perf_counter() - t1),
                        "source": "run",
                        "diagnostics": {},
                    }
                    _log(
                        f"    {method:16s}  {base.sum_rate_mbps:.4f} Mbps  "
                        f"{perf_counter() - t1:.1f}s"
                    )
        if multistart_path is not None:
            ms_cell = _multistart_row(multistart_path, seed)
            if ms_cell is not None:
                rec["sca_multistart"] = ms_cell
        done[key] = rec
        ckpt.write_text(json.dumps(_jsonable(done), indent=2), encoding="utf-8")

    rows = [done[str(s)] for s in seeds]
    present = [m for m in COMPARE_METHODS if all(m in r for r in rows)]
    pairs = {}
    if "sca_anchor" in present and "sca" in present:
        pairs["vs_sca"] = _pair(rows, "sca_anchor", "sca")
    if "sca_anchor" in present and "sca_multistart" in present:
        pairs["vs_multistart"] = _pair(rows, "sca_anchor", "sca_multistart")
    if "sca_anchor" in present and "pso" in present:
        pairs["vs_pso"] = _pair(rows, "sca_anchor", "pso")
    if "sca_anchor" in present and "random" in present:
        pairs["vs_random"] = _pair(rows, "sca_anchor", "random")
    if "sca_anchor" in present and "kmeans" in present:
        pairs["vs_kmeans"] = _pair(rows, "sca_anchor", "kmeans")

    kinds: dict[str, int] = {}
    for r in rows:
        kind = str(r["sca_anchor"]["diagnostics"].get("winner_kind") or "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
    payload = {
        "label": (
            "sca_anchor vs frozen SCA / multi-start / baselines "
            f"(default J=3, {cfg.area_x_m:g}x{cfg.area_y_m:g} m)"
        ),
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "area_m": [cfg.area_x_m, cfg.area_y_m],
        "n_runs": int(n_runs),
        "seed_start": int(seed_start),
        "seeds": list(seeds),
        "los_angle_unit": cfg.los_angle_unit,
        "anchor": {
            "top_k": anc.top_k,
            "max_enumerate": anc.max_enumerate,
            "beam_width": anc.beam_width,
            "include_frozen": anc.include_frozen,
            "selection": anc.selection,
            "n_random": anc.n_random,
        },
        "per_seed": rows,
        "mean_Mbps": {
            m: float(np.mean([r[m]["sum_rate_Mbps"] for r in rows]))
            for m in present
        },
        "mean_wall_clock_s": {
            m: float(
                np.mean(
                    [
                        r[m]["wall_clock_s"]
                        for r in rows
                        if r[m].get("wall_clock_s") is not None
                    ]
                    or [0.0]
                )
            )
            for m in present
        },
        "winner_kinds": kinds,
        **pairs,
        "notes": [
            "Does not overwrite campaign_8.8mhz_cap25_si12k.json.",
            "Does not overwrite campaign_8.8mhz_cap25_si12k_500m.json.",
            "Does not overwrite n100/eval.json.",
            "Keep-best includes the frozen k-means start, so vs SCA Δ >= 0.",
            "random/kmeans/pso/sca reused from the matching 25% campaign J=3 row when present.",
            "sca_multistart reused from sca_multistart_n20*.json when present.",
        ],
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}")
    means = payload["mean_Mbps"]
    _log(
        "means  "
        + "  ".join(f"{m}={means[m]:.4f}" for m in present)
    )
    for key, label in (
        ("vs_sca", "SCA"),
        ("vs_multistart", "multi-start"),
        ("vs_pso", "PSO"),
        ("vs_random", "random"),
        ("vs_kmeans", "k-means"),
    ):
        p = payload.get(key)
        if not p:
            continue
        _log(
            f"vs {label:12s} {p['mean_delta_Mbps']:+.4f} +/- {p['std_delta_Mbps']:.4f}  "
            f"{p['wins']}/{p['n']}  practical>{PRACTICAL_MBPS:g}: {p['n_practical']}  "
            f"p={p['wilcoxon']['p_greater']:.4g}"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="sca_anchor vs frozen SCA / multi-start at default J=3"
    )
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    parser.add_argument("--area-m", type=float, default=100.0)
    parser.add_argument("--max-bw-share", type=float, default=PRIMARY_MAX_BW_SHARE)
    parser.add_argument("--campaign", type=str, default="")
    parser.add_argument("--multistart", type=str, default="")
    args = parser.parse_args(argv)
    run_n20_eval(
        out=Path(args.out),
        n_runs=int(args.n_runs),
        seed_start=int(args.seed_start),
        area_m=float(args.area_m),
        max_bw_share=float(args.max_bw_share),
        campaign_path=Path(args.campaign) if args.campaign else None,
        multistart_path=Path(args.multistart) if args.multistart else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
