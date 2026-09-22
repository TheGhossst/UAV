"""Replay the random method on 15% / 12% campaigns and banks (post-fix RNG).

SCA and k-means do not call place_random. PSO particle inits do
(`place_random(seed*10000+p)`); replay those with replay_pso_tight_caps.py.
Does not overwrite protected 25% headline JSON.

Usage:
  python scripts/campaigns/replay_random_tight_caps.py
  python scripts/campaigns/replay_random_tight_caps.py --only camp15,n100_15_500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
sys.path.insert(0, str(ROOT / "scripts" / "campaigns"))

from run_sca_anchor_eval import _pair  # noqa: E402

from uavdt.config import SimConfig  # noqa: E402
from uavdt.experiments.campaign import (  # noqa: E402
    CampaignSettings,
    run_point,
    write_campaign,
)
from uavdt.experiments.grids import iter_axis  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.experiments.n100 import (  # noqa: E402
    _paired_stats,
    evaluate_bank,
    write_eval,
)
from uavdt.experiments.scenario_bank import load_bank  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
}

TAG_CAMP = "random method replayed 2026-09-20 (post-fix salted RNG)."
TAG_BANK = "random method replayed 2026-09-20 (post-fix bank UAV xy)."


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _refuse(path: Path) -> None:
    resolved = path.resolve()
    for p in PROTECTED:
        if resolved == p.resolve():
            raise SystemExit(f"refusing to overwrite protected {p}")


def _already_replayed(payload: dict, tag: str) -> bool:
    return tag in str(payload.get("note") or "")


def replay_campaign_random(
    path: Path,
    *,
    axes: tuple[str, ...] | None = None,
    force: bool = False,
) -> None:
    _refuse(path)
    if not path.exists():
        _log(f"skip missing {path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not force and _already_replayed(payload, TAG_CAMP):
        _log(f"skip already-replayed campaign {path.name}")
        return
    area = float((payload.get("area_m") or [100.0, 100.0])[0])
    share = payload.get("max_bw_share")
    b_sys = float(payload.get("b_sys_hz") or 8.8e6)
    n_runs = int(payload.get("n_runs") or 20)
    seed_start = int(payload.get("seed_start") or 1)
    cfg = SimConfig(b_sys_hz=b_sys, max_bw_share=share).with_square_area_m(area)
    settings = CampaignSettings(
        n_runs=n_runs,
        seed_start=seed_start,
        methods=("random",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
    )
    axis_filter = frozenset(axes) if axes else None
    new_points = []
    t0 = perf_counter()
    for pt in payload["points"]:
        axis = pt["axis"]
        if axis_filter is not None and axis not in axis_filter:
            new_points.append(pt)
            continue
        x = float(pt["x"])
        match = next(
            (
                cand
                for cand in iter_axis(axis, cfg)
                if abs(float(cand.x_value) - x) < 1e-9
            ),
            None,
        )
        if match is None:
            raise SystemExit(f"no sweep point {axis} x={x} in {path}")
        _log(f"  {path.name}  {pt.get('label')}  random")
        one = run_point(match, settings)
        merged = dict(pt)
        bm = dict(pt.get("by_method") or {})
        bm["random"] = one["by_method"]["random"]
        merged["by_method"] = bm
        new_points.append(merged)
    payload["points"] = new_points
    note = str(payload.get("note") or "")
    payload["note"] = (note + " " + TAG_CAMP).strip()
    write_campaign(payload, path)
    _log(f"wrote {path}  ({perf_counter() - t0:.1f}s)")


def replay_bank_random(*, bank_path: Path, eval_path: Path, share: float) -> None:
    _refuse(eval_path)
    if not eval_path.exists():
        _log(f"skip missing {eval_path}")
        return
    old = json.loads(eval_path.read_text(encoding="utf-8"))
    if _already_replayed(old, TAG_BANK):
        _log(f"skip already-replayed bank {eval_path}")
        return
    bank = load_bank(bank_path)
    overlay = SimConfig(b_sys_hz=8.8e6, max_bw_share=share)
    ckpt = eval_path.with_name(eval_path.stem + ".random_replay.checkpoint.json")
    _log(f"=== bank random {eval_path} cap={share} ===")
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        overlay,
        ("random",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        checkpoint_path=ckpt,
        resume=True,
        bank_path=bank_path,
    )
    old["by_method"]["random"] = payload["by_method"]["random"]
    new_rand = {(r["seed"], r["method"]): r for r in payload.get("runs") or []}
    runs = []
    for r in old.get("runs") or []:
        if r["method"] == "random":
            runs.append(new_rand.get((r["seed"], "random"), r))
        else:
            runs.append(r)
    old["runs"] = runs
    old["sca_minus_baseline_Mbps"] = _paired_stats(old["by_method"])
    note = str(old.get("note") or "")
    old["note"] = (note + " " + TAG_BANK).strip()
    write_eval(old, eval_path)
    if ckpt.exists():
        ckpt.unlink()
    stats = old["by_method"]["random"]
    _log(
        f"wrote {eval_path}  random={stats['mean_sum_rate_Mbps']:.4f} +/- "
        f"{stats['std_sum_rate_Mbps']:.4f}  ({perf_counter() - t0:.1f}s)"
    )


def copy_bank_random(src: Path, dst: Path) -> None:
    _refuse(dst)
    if not src.exists() or not dst.exists():
        _log(f"skip copy-random missing {src} or {dst}")
        return
    src_p = json.loads(src.read_text(encoding="utf-8"))
    dst_p = json.loads(dst.read_text(encoding="utf-8"))
    if "random" not in (src_p.get("by_method") or {}):
        _log(f"skip copy-random: no random in {src}")
        return
    dst_p["by_method"]["random"] = src_p["by_method"]["random"]
    new_rand = {
        (r["seed"], r["method"]): r
        for r in src_p.get("runs") or []
        if r.get("method") == "random"
    }
    runs = []
    for r in dst_p.get("runs") or []:
        if r["method"] == "random" and (r["seed"], "random") in new_rand:
            runs.append(new_rand[(r["seed"], "random")])
        else:
            runs.append(r)
    dst_p["runs"] = runs
    dst_p["sca_minus_baseline_Mbps"] = _paired_stats(dst_p["by_method"])
    note = str(dst_p.get("note") or "")
    if TAG_BANK not in note:
        dst_p["note"] = (note + " " + TAG_BANK).strip()
    write_eval(dst_p, dst)
    _log(f"copied random {src.name} -> {dst}")


def replay_n20_eval_random(path: Path) -> None:
    _refuse(path)
    if not path.exists():
        _log(f"skip missing {path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _already_replayed(payload, TAG_CAMP):
        _log(f"skip already-replayed n20 {path.name}")
        return
    area = float((payload.get("area_m") or [500.0, 500.0])[0])
    share = payload.get("max_bw_share")
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=share).with_square_area_m(area)
    t0 = perf_counter()
    for row in payload.get("per_seed") or []:
        seed = int(row["seed"])
        sc = generate_scenario(seed, cfg)
        t1 = perf_counter()
        run = run_method(sc, "random", seed)
        row["random"] = {
            "sum_rate_Mbps": float(run.sum_rate_mbps),
            "feasible": bool(run.feasible),
            "wall_clock_s": float(perf_counter() - t1),
            "source": "random_replay_postfix",
            "diagnostics": {},
        }
        _log(
            f"  {path.name}  seed {seed}  random {run.sum_rate_mbps:.4f} Mbps"
        )
    rows = payload["per_seed"]
    present = [m for m in payload.get("mean_Mbps") or {} if all(m in r for r in rows)]
    if "random" not in present:
        present.append("random")
    payload["mean_Mbps"] = {
        m: float(sum(r[m]["sum_rate_Mbps"] for r in rows) / len(rows))
        for m in present
        if all(m in r for r in rows)
    }
    if all("sca_anchor" in r and "random" in r for r in rows):
        payload["vs_random"] = _pair(rows, "sca_anchor", "random")
    note_key = "notes"
    notes = list(payload.get(note_key) or [])
    notes.append(TAG_CAMP)
    payload[note_key] = notes
    payload["note"] = TAG_CAMP
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(
        f"wrote {path}  random={payload['mean_Mbps']['random']:.4f}  "
        f"({perf_counter() - t0:.1f}s)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="",
        help=(
            "Comma list: camp15,camp12_100,camp12_500,"
            "n100_15_100,n100_15_500,n100_12_100,n100_12_500,n20_15_500"
        ),
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=None,
        help="Replay random on this campaign JSON (optional --axes filter)",
    )
    parser.add_argument(
        "--axes",
        default="",
        help="Comma axes to replay (default: all points), e.g. aodt",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replay even if campaign note already has the replay tag",
    )
    args = parser.parse_args(argv)
    if args.campaign is not None:
        ax = tuple(x.strip() for x in str(args.axes).split(",") if x.strip()) or None
        t0 = perf_counter()
        replay_campaign_random(args.campaign, axes=ax, force=bool(args.force))
        _log(f"random replay done in {perf_counter() - t0:.1f}s")
        return 0
    wanted = {x.strip() for x in str(args.only).split(",") if x.strip()} or {
        "camp15",
        "camp12_100",
        "camp12_500",
        "n100_15_100",
        "n100_15_500",
        "n100_12_100",
        "n100_12_500",
        "n20_15_500",
    }
    t0 = perf_counter()
    if "camp15" in wanted:
        replay_campaign_random(ROOT / "results" / "campaign_8.8mhz_cap15_n20.json")
    if "camp12_100" in wanted:
        replay_campaign_random(ROOT / "results" / "campaign_8.8mhz_cap12_n20.json")
    if "camp12_500" in wanted:
        replay_campaign_random(ROOT / "results" / "campaign_8.8mhz_cap12_n20_500m.json")
    if "n100_15_100" in wanted:
        replay_bank_random(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
            eval_path=ROOT / "results" / "n100_cap15" / "eval.json",
            share=0.15,
        )
    if "n100_15_500" in wanted:
        replay_bank_random(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            eval_path=ROOT / "results" / "n100_500m_cap15" / "eval.json",
            share=0.15,
        )
        copy_bank_random(
            ROOT / "results" / "n100_500m_cap15" / "eval.json",
            ROOT / "results" / "n100_500m_cap15" / "eval_anchor.json",
        )
    if "n100_12_100" in wanted:
        replay_bank_random(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
            eval_path=ROOT / "results" / "n100_cap12" / "eval.json",
            share=0.12,
        )
    if "n100_12_500" in wanted:
        replay_bank_random(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            eval_path=ROOT / "results" / "n100_500m_cap12" / "eval.json",
            share=0.12,
        )
    if "n20_15_500" in wanted:
        replay_n20_eval_random(
            ROOT / "results" / "sca_anchor_ablations" / "cap_15%_500m.json"
        )
    _log(f"random replay done in {perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
