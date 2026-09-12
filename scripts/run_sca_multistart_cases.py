"""Run keep-best multi-start SCA on the four default-point geometries.

Cases (8.8 MHz, 25% cap, I=10, J=3, T_k=2.8 s):
  n20_100m   — 20-seed campaign point, 100×100 m
  n20_500m   — 20-seed campaign point, 500×500 m
  n100_100m  — 100 frozen layouts, 100×100 m
  n100_500m  — 100 frozen layouts, 500×500 m  (this is "n500" in the request:
               there is no 500-layout bank)

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma list of case ids: n20_100m,n20_500m,n100_100m,n100_500m",
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
    _log(f"all requested cases done in {perf_counter() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
