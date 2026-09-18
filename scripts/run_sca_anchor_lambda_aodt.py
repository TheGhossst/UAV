"""Zenith-anchor sweeps on λ and T_k axes (20 seeds), 100 m and 500 m.

Merges into standalone JSON (plot script overlays on headline campaigns).

Usage:
  python scripts/run_sca_anchor_lambda_aodt.py
  python scripts/run_sca_anchor_lambda_aodt.py --only lambda_100m,aodt_500m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sca_anchor_cases import _merge_axis_campaign  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings  # noqa: E402

CASES = {
    "lambda_100m": (100.0, "lambda", ROOT / "results" / "campaign_sca_anchor_lambda.json"),
    "aodt_100m": (100.0, "aodt", ROOT / "results" / "campaign_sca_anchor_aodt.json"),
    "lambda_500m": (500.0, "lambda", ROOT / "results" / "campaign_sca_anchor_lambda_500m.json"),
    "aodt_500m": (500.0, "aodt", ROOT / "results" / "campaign_sca_anchor_aodt_500m.json"),
}

BASE_CAMPAIGN = {
    100.0: ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    500.0: ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _complete(path: Path, axis: str) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    pts = [p for p in payload.get("points") or [] if p.get("axis") == axis]
    return bool(pts) and all("sca_anchor" in (p.get("by_method") or {}) for p in pts)


def run_case(key: str) -> None:
    area_m, axis, out = CASES[key]
    if _complete(out, axis):
        _log(f"skip complete {key} -> {out}")
        return
    base_path = BASE_CAMPAIGN[area_m]
    if not base_path.exists():
        raise FileNotFoundError(f"missing base campaign {base_path}")
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE).with_square_area_m(
        area_m
    )
    settings = CampaignSettings(
        n_runs=20,
        seed_start=1,
        methods=("sca_anchor",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        anchor_settings=AnchorSettings(),
    )
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    _log(f"{key}  axis={axis}  area={area_m:g} m  n_runs=20")
    t0 = perf_counter()
    extra = run_campaign((axis,), cfg, settings, checkpoint_path=ckpt)
    payload = _merge_axis_campaign(base_path, extra, axis)
    write_campaign(payload, out)
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default="")
    args = parser.parse_args(argv)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()} or set(CASES)
    for key in CASES:
        if key in wanted:
            run_case(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
