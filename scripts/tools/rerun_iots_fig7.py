"""Re-run Fig. 7 IoT sweep only (10 MHz, 25%, 100 seeds), patch campaigns, replot fig07."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "campaigns"))

from run_sca_anchor_cases import _merge_axis_campaign  # noqa: E402

from uavdt.config import PRIMARY_CAMPAIGN_REL, headline_sim_config  # noqa: E402
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign  # noqa: E402
from uavdt.experiments.grids import IOT_COUNTS  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings  # noqa: E402

EVAL = ROOT / "results" / "eval_10mhz_cap25_100m_n100"
FIG_DIR = EVAL / "figures"
BASELINE_OUT = EVAL / "iots_baseline_rerun.json"
ANCHOR_OUT = EVAL / "anchor_iots_n100_100m.json"
HEADLINE = ROOT / PRIMARY_CAMPAIGN_REL
MERGED = EVAL / "campaign_10mhz_cap25_n100_with_anchor.json"
PY = sys.executable
N_RUNS = 100


def _patch_iots_points(campaign_path: Path, iots_payload: dict) -> None:
    base = json.loads(campaign_path.read_text(encoding="utf-8"))
    new_pts = [p for p in iots_payload.get("points") or [] if p.get("axis") == "iots"]
    if len(new_pts) != len(IOT_COUNTS):
        raise SystemExit(
            f"expected {len(IOT_COUNTS)} iots points, got {len(new_pts)} in rerun"
        )
    kept = [p for p in base.get("points") or [] if p.get("axis") != "iots"]
    kept.extend(sorted(new_pts, key=lambda p: float(p["x"])))
    base["points"] = kept
    base["note"] = (
        f"Fig. 7 IoT axis re-run {N_RUNS} seeds at I={list(IOT_COUNTS)}. "
        + str(base.get("note") or "")
    )
    write_campaign(base, campaign_path)
    print(f"patched iots -> {campaign_path}", flush=True)


def _run_baseline() -> dict:
    cfg = headline_sim_config(area_m=100.0)
    settings = CampaignSettings(
        n_runs=N_RUNS,
        seed_start=1,
        methods=("random", "kmeans", "pso", "sca"),
        sca_settings=SCASettings(solver=None, max_iterations=30),
    )
    ckpt = BASELINE_OUT.with_name(BASELINE_OUT.stem + ".checkpoint.json")
    print("=== iots baseline methods ===", flush=True)
    t0 = perf_counter()
    payload = run_campaign(("iots",), cfg, settings, checkpoint_path=ckpt)
    write_campaign(payload, BASELINE_OUT)
    if ckpt.exists():
        ckpt.unlink()
    print(f"wrote {BASELINE_OUT} ({perf_counter() - t0:.1f}s)", flush=True)
    return payload


def _run_anchor() -> dict:
    cfg = headline_sim_config(area_m=100.0)
    settings = CampaignSettings(
        n_runs=N_RUNS,
        seed_start=1,
        methods=("sca_anchor",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        anchor_settings=AnchorSettings(max_enumerate=10_000),
    )
    ckpt = ANCHOR_OUT.with_name(ANCHOR_OUT.stem + ".checkpoint.json")
    print("=== iots sca_anchor ===", flush=True)
    t0 = perf_counter()
    payload = run_campaign(("iots",), cfg, settings, checkpoint_path=ckpt)
    write_campaign(payload, ANCHOR_OUT)
    if ckpt.exists():
        ckpt.unlink()
    print(f"wrote {ANCHOR_OUT} ({perf_counter() - t0:.1f}s)", flush=True)
    return payload


def _rebuild_merged() -> Path:
    merged = _merge_axis_campaign(HEADLINE, json.loads(ANCHOR_OUT.read_text()), "iots")
    write_campaign(merged, MERGED)
    print(f"wrote {MERGED}", flush=True)
    return MERGED


def _replot(merged: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "plot"))
    from plot_paper_figures import _load, plot_sum_rate_figure  # noqa: E402

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    camp = _load(merged)
    plot_sum_rate_figure(
        camp,
        "iots",
        "Number of IoT devices ($I$)",
        "Fig. 7 analogue — sum rate vs IoT count ($J=3$)",
        7,
        FIG_DIR,
        smooth_monotone=True,
        error_bars=False,
    )
    print(f"updated {FIG_DIR / 'fig07_sum_rate.png'}", flush=True)


def main() -> int:
    EVAL.mkdir(parents=True, exist_ok=True)
    baseline = _run_baseline()
    _patch_iots_points(HEADLINE, baseline)
    anchor = _run_anchor()
    merged = _rebuild_merged()
    _replot(merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
