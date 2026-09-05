"""Re-run the three sweep points whose SCA init collided on constraint (24).

After cpu_stable_processing, majority-of-association is kept when it is
already queue-stable. Only I=28, I=32, and f_j=0.5e8 change. Merges those
rows into the primary campaign JSON; other points are untouched.

Usage:
    $env:PYTHONPATH="src"
    python scripts/rerun_init_repair_points.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from uavdt.config import SimConfig
from uavdt.experiments.campaign import (
    CampaignSettings,
    replace_points,
    run_point,
    write_campaign,
)
from uavdt.experiments.grids import iter_axis
from uavdt.placement.pso import PSOSettings
from uavdt.sca.settings import SCASettings

PRIMARY = Path("results/campaign_8.8mhz_cap25_si12k.json")
# (axis, x) — the only points where majority vote can violate (24).
TARGETS = (
    ("iots", 28.0),
    ("iots", 32.0),
    ("cpu", 5.0e7),
)


def _wanted(axis: str, x: float) -> bool:
    for a, xv in TARGETS:
        if axis == a and abs(float(x) - xv) < 1e-6:
            return True
    return False


def main() -> int:
    if not PRIMARY.exists():
        raise SystemExit(f"missing {PRIMARY}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = PRIMARY.with_name(f"{PRIMARY.stem}_pre_init_repair_{stamp}.json")
    shutil.copy2(PRIMARY, backup)
    print(f"backup {backup}", flush=True)

    payload = json.loads(PRIMARY.read_text(encoding="utf-8"))
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.25)
    settings = CampaignSettings(
        n_runs=20,
        seed_start=1,
        methods=("random", "kmeans", "pso", "sca"),
        sca_settings=SCASettings(
            max_iterations=30,
            epsilon=1e-4,
            step_size_m=20.0,
            solver=None,
        ),
        pso_settings=PSOSettings(),
    )
    new_points = []
    for axis in ("iots", "cpu"):
        for point in iter_axis(axis, cfg):
            if not _wanted(point.axis, point.x_value):
                continue
            print(f"RUN {point.label}", flush=True)
            new_points.append(run_point(point, settings))
    if len(new_points) != len(TARGETS):
        raise SystemExit(f"expected {len(TARGETS)} points, got {len(new_points)}")
    merged = replace_points(payload, new_points)
    out = write_campaign(merged, PRIMARY)
    print(f"wrote {out}", flush=True)
    print(f"wrote {out.with_suffix('.csv')}", flush=True)
    for pt in new_points:
        print(f"--- {pt['axis']} x={pt['x']:g} ---", flush=True)
        for method, stats in pt["by_method"].items():
            print(
                f"  {method}: Mbps={stats['mean_sum_rate_Mbps']:.4f} "
                f"feas={stats['feasible_fraction']:.2%} n={stats['n']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
