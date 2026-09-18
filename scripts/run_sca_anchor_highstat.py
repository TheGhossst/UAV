"""Zenith-anchor on §VII sweep axes at n_runs=100 or 200 (all four axes).

Usage:
  python scripts/run_sca_anchor_highstat.py --n-runs 100 --area-m 100 --axis iots
  python scripts/run_sca_anchor_highstat.py --n-runs 200 --area-m 500 --axis all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign
from uavdt.sca.settings import SCASettings
from uavdt.sca_anchor import AnchorSettings

AXES = ("uavs", "iots", "lambda", "aodt")


def _out_path(n_runs: int, area_m: float, axis: str) -> Path:
    field = int(area_m)
    return ROOT / "results" / f"campaign_sca_anchor_{axis}_n{n_runs}_{field}m.json"


def _complete(path: Path, axis: str, n_runs: int) -> bool:
    if not path.exists():
        return False
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("n_runs") or 0) != int(n_runs):
        return False
    pts = [p for p in payload.get("points") or [] if p.get("axis") == axis]
    return bool(pts) and all("sca_anchor" in (p.get("by_method") or {}) for p in pts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, required=True)
    parser.add_argument("--area-m", type=float, default=100.0)
    parser.add_argument("--axis", type=str, default="all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output JSON looks complete",
    )
    args = parser.parse_args(argv)
    n_runs = int(args.n_runs)
    area_m = float(args.area_m)
    axes = list(AXES) if args.axis.strip().lower() == "all" else [
        x.strip() for x in args.axis.split(",") if x.strip()
    ]
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE).with_square_area_m(
        area_m
    )
    for axis in axes:
        out = _out_path(n_runs, area_m, axis)
        if not args.force and _complete(out, axis, n_runs):
            print(f"skip {out}", flush=True)
            continue
        if args.force and out.exists():
            out.unlink()
            ckpt_old = out.with_name(out.stem + ".checkpoint.json")
            if ckpt_old.exists():
                ckpt_old.unlink()
        anc = (
            AnchorSettings(max_enumerate=10_000)
            if axis == "iots"
            else AnchorSettings()
        )
        settings = CampaignSettings(
            n_runs=n_runs,
            seed_start=1,
            methods=("sca_anchor",),
            sca_settings=SCASettings(solver=None, max_iterations=30),
            anchor_settings=anc,
        )
        ckpt = out.with_name(out.stem + ".checkpoint.json")
        print(f"=== {axis}  n_runs={n_runs}  area={area_m:g} m ===", flush=True)
        t0 = perf_counter()
        payload = run_campaign((axis,), cfg, settings, checkpoint_path=ckpt)
        write_campaign(payload, out)
        if ckpt.exists():
            ckpt.unlink()
        print(f"wrote {out}  ({perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
