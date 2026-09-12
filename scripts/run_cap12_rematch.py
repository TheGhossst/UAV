"""8.8 MHz / 12% rematch of the four headline tests vs 25%.

Does not overwrite 25% campaigns or n100/eval.json.

  python scripts/run_cap12_rematch.py
  python scripts/run_cap12_rematch.py --skip-plot
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import SimConfig  # noqa: E402
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign  # noqa: E402
from uavdt.experiments.grids import AXES  # noqa: E402
from uavdt.experiments.n100 import evaluate_bank, write_eval  # noqa: E402
from uavdt.experiments.scenario_bank import load_bank  # noqa: E402
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402

PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
}
SRC_100M = ROOT / "results" / "bw_fine_7p1_8p8" / "campaign_8p8mhz_cap12.json"
OUT_100M = ROOT / "results" / "campaign_8.8mhz_cap12_n20.json"
OUT_500M = ROOT / "results" / "campaign_8.8mhz_cap12_n20_500m.json"
METHODS = ("random", "kmeans", "pso", "sca")


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


def _campaign_complete(path: Path, *, area: float, share: float, n_runs: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("checkpoint"):
        return False
    if int(payload.get("n_runs", -1)) != n_runs:
        return False
    if abs(float(payload.get("b_sys_hz", -1)) - 8.8e6) > 1.0:
        return False
    if payload.get("max_bw_share") != share:
        return False
    area_m = payload.get("area_m") or []
    if list(area_m) != [area, area]:
        return False
    axes = {pt["axis"] for pt in payload.get("points", [])}
    return set(AXES).issubset(axes)


def _n100_complete(path: Path, n: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    by_m = payload.get("by_method") or {}
    if set(METHODS) - set(by_m):
        return False
    return all(int(by_m[m].get("n") or 0) == n for m in METHODS)


def copy_100m() -> None:
    _refuse(OUT_100M)
    if _campaign_complete(OUT_100M, area=100.0, share=0.12, n_runs=20):
        _log(f"skip complete 100m campaign {OUT_100M.name}")
        return
    if not SRC_100M.exists():
        raise SystemExit(f"missing fine-search 12% campaign {SRC_100M}")
    src = json.loads(SRC_100M.read_text(encoding="utf-8"))
    if src.get("max_bw_share") != 0.12:
        raise SystemExit(f"{SRC_100M} is not cap 12%")
    shutil.copy2(SRC_100M, OUT_100M)
    csv_src = SRC_100M.with_suffix(".csv")
    if csv_src.exists():
        shutil.copy2(csv_src, OUT_100M.with_suffix(".csv"))
    _log(f"copied {SRC_100M.name} -> {OUT_100M}")


def run_500m() -> None:
    _refuse(OUT_500M)
    if _campaign_complete(OUT_500M, area=500.0, share=0.12, n_runs=20):
        _log(f"skip complete 500m campaign {OUT_500M.name}")
        return
    cfg = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.12).with_square_area_m(500.0)
    settings = CampaignSettings(
        n_runs=20,
        seed_start=1,
        methods=METHODS,
        sca_settings=SCASettings(solver=None, max_iterations=30, step_size_m=20.0),
        pso_settings=PSOSettings(),
    )
    ckpt = OUT_500M.with_name(OUT_500M.stem + ".checkpoint.json")
    _log(f"=== 500x500 campaign 8.8 MHz cap=12% -> {OUT_500M.name} ===")
    t0 = perf_counter()
    payload = run_campaign(AXES, cfg, settings, checkpoint_path=ckpt)
    write_campaign(payload, OUT_500M)
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {OUT_500M}  ({perf_counter() - t0:.1f}s)")


def run_n100(
    *,
    bank_path: Path,
    out: Path,
    checkpoint: Path,
    fig_dir: Path,
    skip_plot: bool,
) -> None:
    _refuse(out)
    bank = load_bank(bank_path)
    n = int(bank["n_scenarios"])
    if _n100_complete(out, n):
        _log(f"skip complete n100 {out}")
        return
    overlay = SimConfig(b_sys_hz=8.8e6, max_bw_share=0.12)
    _log(f"=== n100 bank={bank_path.name} cap=12% -> {out} ===")
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        overlay,
        METHODS,
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
        checkpoint_path=checkpoint,
        resume=True,
        bank_path=bank_path,
    )
    write_eval(payload, out)
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")
    for method, stats in payload["by_method"].items():
        _log(
            f"  {method:8s}  mean={stats['mean_sum_rate_Mbps']:.4f}  "
            f"std={stats['std_sum_rate_Mbps']:.4f}  "
            f"feas={100.0 * stats['feasible_fraction']:.1f}%"
        )
    if skip_plot:
        return
    try:
        from uavdt.experiments.n100_plot import plot_n100_figures

        for p in plot_n100_figures(payload, bank, fig_dir):
            _log(f"wrote {p}")
    except ImportError:
        _log("matplotlib missing; skip n100 figures")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--only",
        default="",
        help="Comma list: camp100,camp500,n100_100,n100_500",
    )
    args = parser.parse_args(argv)
    wanted = {x.strip() for x in str(args.only).split(",") if x.strip()} or {
        "camp100",
        "camp500",
        "n100_100",
        "n100_500",
    }
    t0 = perf_counter()
    if "camp100" in wanted:
        copy_100m()
    if "camp500" in wanted:
        run_500m()
    if "n100_100" in wanted:
        run_n100(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
            out=ROOT / "results" / "n100_cap12" / "eval.json",
            checkpoint=ROOT / "results" / "n100_cap12" / "eval.checkpoint.json",
            fig_dir=ROOT / "results" / "figures" / "n100_cap12",
            skip_plot=bool(args.skip_plot),
        )
    if "n100_500" in wanted:
        run_n100(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            out=ROOT / "results" / "n100_500m_cap12" / "eval.json",
            checkpoint=ROOT / "results" / "n100_500m_cap12" / "eval.checkpoint.json",
            fig_dir=ROOT / "results" / "figures" / "n100_500m_cap12",
            skip_plot=bool(args.skip_plot),
        )
    _log(f"cap12 rematch done in {perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
