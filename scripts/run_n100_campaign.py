"""Run the 100-seed §VII campaign axis-by-axis and merge.

Each axis writes its own JSON so a crash can resume. Headline settings:
8.8 MHz, 25% per-link cap, 100 x 100 m, seeds 1..100.

Usage (repo root):
    $env:PYTHONPATH = "src"
    python scripts/run_n100_campaign.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign
from uavdt.experiments.grids import AXES
from uavdt.placement.pso import PSOSettings
from uavdt.sca.settings import SCASettings

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "n100_campaign"
MERGED = ROOT / "results" / "campaign_8.8mhz_cap25_n100.json"


def _axis_complete(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("n_runs") or 0) != 100:
        return None
    if not payload.get("points"):
        return None
    return payload


def _merge(parts: list[dict]) -> dict:
    base = dict(parts[0])
    points = []
    for part in parts:
        points.extend(part["points"])
    base["points"] = points
    base["note"] = (
        "100-seed replay of §VII Figs. 6–10 axes. "
        + str(base.get("note", ""))
    )
    return base


def main() -> int:
    cfg = SimConfig(b_sys_hz=8_800_000.0, max_bw_share=PRIMARY_MAX_BW_SHARE)
    settings = CampaignSettings(
        n_runs=100,
        seed_start=1,
        methods=("random", "kmeans", "pso", "sca"),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[dict] = []
    for axis in AXES:
        path = OUT_DIR / f"{axis}.json"
        existing = _axis_complete(path)
        if existing is not None:
            print(f"skip {axis}: {path} already has n_runs=100", flush=True)
            parts.append(existing)
            continue
        print(f"=== axis {axis}  n_runs=100 ===", flush=True)
        payload = run_campaign((axis,), cfg, settings)
        write_campaign(payload, path)
        print(f"wrote {path}", flush=True)
        parts.append(payload)
    merged = _merge(parts)
    write_campaign(merged, MERGED)
    print(f"wrote {MERGED}", flush=True)
    print(f"points={len(merged['points'])}  n_runs={merged['n_runs']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
