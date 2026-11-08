"""Dynamic-assignment multi-start on 100×100 m, 10 MHz, 25% per-link cap.

Runs:
  1) Default — frozen n100 bank (I=10, J=3 layouts), 100 seeds
  2) §VII axis sweep — J, I, λ, T_k, f_j (NOT the per-link cap fraction study)

Outputs: results/run_10mhz_open/multistart_dyn_100m/

Usage (repo root):
  $env:PYTHONPATH = "src"
  python scripts/experiments/run_multistart_dyn_100m.py
  python scripts/experiments/run_multistart_dyn_100m.py --default-only
  python scripts/experiments/run_multistart_dyn_100m.py --campaign-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import PRIMARY_MAX_BW_SHARE, headline_sim_config  # noqa: E402
from uavdt.experiments.campaign import (  # noqa: E402
    AXES,
    CampaignSettings,
    run_campaign,
    write_campaign,
)
from uavdt.experiments.n100 import evaluate_bank, write_eval  # noqa: E402
from uavdt.experiments.scenario_bank import load_bank  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_multistart import MultiStartSettings  # noqa: E402

OUT_ROOT = ROOT / "results" / "run_10mhz_open" / "multistart_dyn_100m"
BANK = ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json"
METHODS = ("sca_multistart_dyn",)
N_RUNS = 100


def _run_default(*, force: bool) -> Path:
    out_path = OUT_ROOT / "default" / "n100_eval.json"
    ckpt = OUT_ROOT / "default" / "n100_eval.checkpoint.json"
    legacy = OUT_ROOT / "cap25" / "n100_eval.json"
    if not force and out_path.is_file():
        print(f"skip complete {out_path}", flush=True)
        return out_path
    if not force and legacy.is_file() and not out_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"linked default from prior cap25 run -> {out_path}", flush=True)
        return out_path

    bank = load_bank(BANK)
    cfg = headline_sim_config(area_m=100.0, max_bw_share=PRIMARY_MAX_BW_SHARE)
    ms = MultiStartSettings(n_random=2, n_kmeans=2, include_frozen=True)
    sca = SCASettings(solver=None, max_iterations=30)
    print("=== default n100  I=10 J=3 bank  25% cap ===", flush=True)
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        cfg,
        METHODS,
        sca_settings=sca,
        multistart_settings=ms,
        checkpoint_path=ckpt,
        resume=not force,
        bank_path=BANK,
    )
    payload["experiment"] = "multistart_dyn_100m_default"
    write_eval(payload, out_path)
    if ckpt.exists():
        ckpt.unlink()
    print(f"wrote {out_path} ({perf_counter() - t0:.1f}s)", flush=True)
    for method, stats in payload["by_method"].items():
        print(
            f"  {method:20s}  {stats['mean_sum_rate_Mbps']:.4f} ± "
            f"{stats['std_sum_rate_Mbps']:.4f} Mbps",
            flush=True,
        )
    return out_path


def _run_campaign(*, force: bool) -> Path:
    out_path = OUT_ROOT / "campaign_eval.json"
    ckpt = OUT_ROOT / "campaign_eval.checkpoint.json"
    if not force and out_path.is_file():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if int(payload.get("n_runs") or 0) == N_RUNS and not payload.get("checkpoint"):
            print(f"skip complete {out_path}", flush=True)
            return out_path

    cfg = headline_sim_config(area_m=100.0, max_bw_share=PRIMARY_MAX_BW_SHARE)
    settings = CampaignSettings(
        n_runs=N_RUNS,
        seed_start=1,
        methods=METHODS,
        sca_settings=SCASettings(solver=None, max_iterations=30),
        multistart_settings=MultiStartSettings(n_random=2, n_kmeans=2),
    )
    print(f"=== §VII campaign  axes={AXES}  25% cap  100×100 m ===", flush=True)
    t0 = perf_counter()
    payload = run_campaign(tuple(AXES), cfg, settings, checkpoint_path=ckpt)
    payload["experiment"] = "multistart_dyn_100m_campaign"
    write_campaign(payload, out_path)
    if ckpt.exists():
        ckpt.unlink()
    print(f"wrote {out_path} ({perf_counter() - t0:.1f}s)", flush=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--default-only", action="store_true")
    ap.add_argument("--campaign-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_default = not args.campaign_only
    run_camp = not args.default_only
    paths: list[str] = []
    if run_default:
        paths.append(str(_run_default(force=args.force).relative_to(ROOT)))
    if run_camp:
        paths.append(str(_run_campaign(force=args.force).relative_to(ROOT)))
    manifest = {
        "note": "Default = n100 bank J=3 I=10. Campaign = uavs/iots/lambda/aodt/cpu axes.",
        "outputs": paths,
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
