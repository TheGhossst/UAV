"""Run keep-best multi-start SCA on the four default-point geometries.

Cases (8.8 MHz, 25% cap, I=10, J=3, T_k=2.8 s):
  n20_100m   — 20-seed campaign point, 100×100 m
  n20_500m   — 20-seed campaign point, 500×500 m
  n100_100m  — 100 frozen layouts, 100×100 m
  n100_500m  — 100 frozen layouts, 500×500 m  (this is "n500" in the request:
               there is no 500-layout bank)
  j_sweep_100m — UAV axis J=1..5, 20 seeds (merge into campaign_sca_multistart_uavs.json)
  i_sweep_100m — IoT axis I=10..32 at 100 m (merge into campaign_sca_multistart_iots.json)

Reuses existing random/k-means/PSO/SCA n100 checkpoints. Does not overwrite
headline campaigns or n100/eval.json.

Usage:
  python scripts/run_sca_multistart_cases.py
  python scripts/run_sca_multistart_cases.py --skip-plot
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_sca_multistart_eval import run_n20_eval  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign  # noqa: E402
from uavdt.experiments.n100 import evaluate_bank, write_eval  # noqa: E402
from uavdt.experiments.scenario_bank import load_bank  # noqa: E402
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_multistart import MultiStartSettings  # noqa: E402

PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
    ROOT / "results" / "residual_on_sca" / "multistart_n20.json",
}
METHODS = ("random", "kmeans", "pso", "sca", "sca_multistart")


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


def _n100_complete(path: Path, n_scenarios: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    by_m = payload.get("by_method") or {}
    ms = by_m.get("sca_multistart") or {}
    return int(ms.get("n") or 0) == int(n_scenarios)


def _prepare_n100_checkpoint(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if not src.exists():
        _log(f"warning: missing baseline checkpoint {src}; n100 will re-run all methods")
        return
    shutil.copy2(src, dest)
    _log(f"copied checkpoint {src.name} -> {dest}")


def run_n100_case(
    *,
    bank_path: Path,
    src_ckpt: Path,
    out: Path,
    checkpoint: Path,
    fig_dir: Path,
    skip_plot: bool,
) -> dict:
    _refuse(out)
    bank = load_bank(bank_path)
    n = int(bank["n_scenarios"])
    if _n100_complete(out, n):
        _log(f"skip complete n100 {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    _prepare_n100_checkpoint(src_ckpt, checkpoint)
    overlay = SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)
    _log(
        f"n100 bank={bank_path.name}  methods={list(METHODS)}  "
        f"out={out.relative_to(ROOT)}"
    )
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        overlay,
        METHODS,
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
        multistart_settings=MultiStartSettings(),
        checkpoint_path=checkpoint,
        resume=True,
        bank_path=bank_path,
    )
    write_eval(payload, out)
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")
    for method, stats in payload["by_method"].items():
        _log(
            f"  {method:16s}  mean={stats['mean_sum_rate_Mbps']:.4f} Mbps  "
            f"std={stats['std_sum_rate_Mbps']:.4f}  "
            f"feas={100.0 * stats['feasible_fraction']:.1f}%  n={stats['n']}"
        )
    if skip_plot:
        return payload
    try:
        from uavdt.experiments.n100_plot import plot_n100_figures

        for p in plot_n100_figures(payload, bank, fig_dir):
            _log(f"wrote {p}")
    except ImportError:
        _log("matplotlib missing; skip n100 figures")
    return payload


def _axis_complete(path: Path, axis: str, xs: list[int], method: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    got = sorted(
        int(float(p["x"]))
        for p in payload.get("points") or []
        if p.get("axis") == axis and method in (p.get("by_method") or {})
    )
    return got == sorted(int(x) for x in xs)


def _j_sweep_complete(path: Path) -> bool:
    return _axis_complete(path, "uavs", [1, 2, 3, 4, 5], "sca_multistart")


def _merge_axis_campaign(base_path: Path, extra: dict, axis: str) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    extra_by_x = {
        float(p["x"]): p
        for p in extra.get("points") or []
        if p.get("axis") == axis
    }
    merged_points = []
    for pt in base.get("points") or []:
        if pt.get("axis") != axis:
            continue
        copy = json.loads(json.dumps(pt))
        other = extra_by_x.get(float(pt["x"]))
        if other is not None:
            copy.setdefault("by_method", {})
            copy["by_method"].update(other.get("by_method") or {})
        merged_points.append(copy)
    methods = list(base.get("methods") or [])
    if "sca_multistart" not in methods:
        methods.append("sca_multistart")
    header = dict(base)
    header["methods"] = methods
    header["sca_multistart"] = True
    header["points"] = merged_points
    header["note"] = (
        f"Merged headline campaign {axis} axis with opt-in sca_multistart. "
        "Does not replace frozen SCA. " + str(base.get("note") or "")
    )
    return header


def run_j_sweep(*, area_m: float, out: Path, campaign_path: Path) -> dict:
    _refuse(out)
    if _j_sweep_complete(out):
        _log(f"skip complete J-sweep {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    cfg = SimConfig(
        b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE
    ).with_square_area_m(float(area_m))
    settings = CampaignSettings(
        n_runs=20,
        seed_start=1,
        methods=("sca_multistart",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        multistart_settings=MultiStartSettings(),
    )
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    _log(
        f"J-sweep sca_multistart  area={area_m:g}x{area_m:g} m  "
        f"J=1..5  n_runs=20  out={out.relative_to(ROOT)}"
    )
    t0 = perf_counter()
    extra = run_campaign(("uavs",), cfg, settings, checkpoint_path=ckpt)
    payload = _merge_axis_campaign(campaign_path, extra, "uavs")
    write_campaign(payload, out)
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")
    for pt in payload["points"]:
        if pt.get("axis") != "uavs":
            continue
        ms = (pt.get("by_method") or {}).get("sca_multistart") or {}
        sca = (pt.get("by_method") or {}).get("sca") or {}
        _log(
            f"  J={int(float(pt['x']))}  "
            f"multistart={ms.get('mean_sum_rate_Mbps', float('nan')):.4f}  "
            f"SCA={sca.get('mean_sum_rate_Mbps', float('nan')):.4f}"
        )
    return payload


def run_i_sweep(*, area_m: float, out: Path, campaign_path: Path) -> dict:
    _refuse(out)
    want = [10, 16, 20, 24, 28, 32]
    if _axis_complete(out, "iots", want, "sca_multistart"):
        _log(f"skip complete I-sweep {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    cfg = SimConfig(
        b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE
    ).with_square_area_m(float(area_m))
    settings = CampaignSettings(
        n_runs=20,
        seed_start=1,
        methods=("sca_multistart",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        multistart_settings=MultiStartSettings(),
    )
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    _log(
        f"I-sweep sca_multistart  area={area_m:g}x{area_m:g} m  "
        f"I=10..32  n_runs=20  out={out.relative_to(ROOT)}"
    )
    t0 = perf_counter()
    extra = run_campaign(("iots",), cfg, settings, checkpoint_path=ckpt)
    payload = _merge_axis_campaign(campaign_path, extra, "iots")
    write_campaign(payload, out)
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")
    for pt in payload["points"]:
        if pt.get("axis") != "iots":
            continue
        ms = (pt.get("by_method") or {}).get("sca_multistart") or {}
        sca = (pt.get("by_method") or {}).get("sca") or {}
        _log(
            f"  I={int(float(pt['x']))}  "
            f"multistart={ms.get('mean_sum_rate_Mbps', float('nan')):.4f}  "
            f"SCA={sca.get('mean_sum_rate_Mbps', float('nan')):.4f}"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma list of case ids: n20_100m,n20_500m,n100_100m,n100_500m,j_sweep_100m,i_sweep_100m",
    )
    args = parser.parse_args(argv)
    wanted = {
        x.strip()
        for x in str(args.only).split(",")
        if x.strip()
    } or {"n20_100m", "n20_500m", "n100_100m", "n100_500m"}

    t_all = perf_counter()
    if "n20_100m" in wanted:
        _log("=== n20 100x100 m ===")
        run_n20_eval(
            out=ROOT / "results" / "sca_multistart_n20.json",
            area_m=100.0,
            skip_if_complete=True,
        )
    if "n20_500m" in wanted:
        _log("=== n20 500x500 m ===")
        run_n20_eval(
            out=ROOT / "results" / "sca_multistart_n20_500m.json",
            area_m=500.0,
            skip_if_complete=True,
        )
    if "n100_100m" in wanted:
        _log("=== n100 100x100 m ===")
        run_n100_case(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
            src_ckpt=ROOT / "results" / "n100" / "eval.checkpoint.json",
            out=ROOT / "results" / "n100" / "eval_multistart.json",
            checkpoint=ROOT / "results" / "n100" / "eval_multistart.checkpoint.json",
            fig_dir=ROOT / "results" / "figures" / "n100_multistart",
            skip_plot=bool(args.skip_plot),
        )
    if "n100_500m" in wanted:
        _log("=== n100 500x500 m (n500 = this 100-layout 500 m bank) ===")
        run_n100_case(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            src_ckpt=ROOT / "results" / "n100_500m_cap25" / "eval.checkpoint.json",
            out=ROOT / "results" / "n100_500m_cap25" / "eval_multistart.json",
            checkpoint=(
                ROOT / "results" / "n100_500m_cap25" / "eval_multistart.checkpoint.json"
            ),
            fig_dir=ROOT / "results" / "figures" / "n100_500m_multistart",
            skip_plot=bool(args.skip_plot),
        )
    if "j_sweep_100m" in wanted:
        _log("=== J-sweep 100x100 m ===")
        run_j_sweep(
            area_m=100.0,
            out=ROOT / "results" / "campaign_sca_multistart_uavs.json",
            campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
        )
    if "i_sweep_100m" in wanted:
        _log("=== I-sweep 100x100 m ===")
        run_i_sweep(
            area_m=100.0,
            out=ROOT / "results" / "campaign_sca_multistart_iots.json",
            campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
        )
    _log(f"all requested cases done in {perf_counter() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
