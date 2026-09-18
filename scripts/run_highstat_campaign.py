"""Run §VII sweep axes at n_runs seeds (100 m or 500 m field), merge to one JSON.

Usage:
  python scripts/run_highstat_campaign.py --n-runs 200
  python scripts/run_highstat_campaign.py --n-runs 100 --area-m 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign
from uavdt.experiments.grids import AXES
from uavdt.placement.pso import PSOSettings
from uavdt.sca.settings import SCASettings


def _axis_complete(path: Path, n_runs: int) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("n_runs") or 0) != int(n_runs):
        return None
    if not payload.get("points"):
        return None
    return payload


def _merge(parts: list[dict], n_runs: int, area_m: float) -> dict:
    base = dict(parts[0])
    points: list[dict] = []
    for part in parts:
        points.extend(part["points"])
    base["points"] = points
    base["note"] = (
        f"{n_runs}-seed replay of §VII Figs. 6–10 axes at {area_m:g}×{area_m:g} m. "
        + str(base.get("note", ""))
    )
    return base


def _default_paths(n_runs: int, area_m: float) -> tuple[Path, Path]:
    tag = f"n{n_runs}" if n_runs != 20 else "si12k"
    if area_m >= 400:
        tag = f"{tag}_500m"
    out_dir = ROOT / "results" / f"n{n_runs}_campaign_{int(area_m)}m"
    merged = ROOT / "results" / f"campaign_8.8mhz_cap25_{tag}.json"
    if n_runs == 20:
        merged = ROOT / "results" / (
            "campaign_8.8mhz_cap25_si12k_500m.json"
            if area_m >= 400
            else "campaign_8.8mhz_cap25_si12k.json"
        )
    elif n_runs == 100 and area_m < 400:
        merged = ROOT / "results" / "campaign_8.8mhz_cap25_n100.json"
    elif n_runs == 100:
        merged = ROOT / "results" / "campaign_8.8mhz_cap25_n100_500m.json"
    elif n_runs == 200 and area_m < 400:
        merged = ROOT / "results" / "campaign_8.8mhz_cap25_n200.json"
    else:
        merged = ROOT / "results" / "campaign_8.8mhz_cap25_n200_500m.json"
    return out_dir, merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, required=True)
    parser.add_argument("--area-m", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--merged-out", type=Path, default=None)
    parser.add_argument("--axes", type=str, default="all", help="Comma list or 'all'")
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite merged n20 campaigns and re-run incomplete axis JSON",
    )
    parser.add_argument(
        "--skip-if-merged",
        action="store_true",
        help="Exit 0 when merged campaign JSON already exists (resume mode)",
    )
    args = parser.parse_args(argv)

    n_runs = int(args.n_runs)
    area_m = float(args.area_m)
    out_dir, merged = args.out_dir, args.merged_out
    if out_dir is None or merged is None:
        d, m = _default_paths(n_runs, area_m)
        out_dir = out_dir or d
        merged = merged or m

    if n_runs == 20 and merged.exists() and not args.force_overwrite:
        if args.skip_if_merged:
            print(f"skip (merged exists) {merged}", flush=True)
            return 0
        print(f"refusing to overwrite primary n20 campaign {merged}", flush=True)
        return 1
    if args.force_overwrite:
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in list(out_dir.glob("*.json")):
            path.unlink()
            print(f"removed {path}", flush=True)
        if merged.exists():
            merged.unlink()
            print(f"removed {merged}", flush=True)

    axes = list(AXES) if args.axes.strip().lower() == "all" else [
        x.strip() for x in args.axes.split(",") if x.strip()
    ]
    cfg = SimConfig(
        b_sys_hz=8_800_000.0,
        max_bw_share=PRIMARY_MAX_BW_SHARE,
    ).with_square_area_m(area_m)
    settings = CampaignSettings(
        n_runs=n_runs,
        seed_start=1,
        methods=("random", "kmeans", "pso", "sca"),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[dict] = []
    for axis in axes:
        path = out_dir / f"{axis}.json"
        existing = _axis_complete(path, n_runs)
        if existing is not None:
            print(f"skip {axis}: {path} already has n_runs={n_runs}", flush=True)
            parts.append(existing)
            continue
        print(f"=== axis {axis}  n_runs={n_runs}  area={area_m:g} m ===", flush=True)
        ckpt = path.with_suffix(".checkpoint.json")
        payload = run_campaign((axis,), cfg, settings, checkpoint_path=ckpt)
        write_campaign(payload, path)
        if ckpt.exists():
            ckpt.unlink()
        print(f"wrote {path}", flush=True)
        parts.append(payload)
    merged_payload = _merge(parts, n_runs, area_m)
    write_campaign(merged_payload, merged)
    print(f"wrote {merged}", flush=True)
    print(f"points={len(merged_payload['points'])}  n_runs={merged_payload['n_runs']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
