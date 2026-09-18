"""200-scenario Monte Carlo for PPT default figures (100 m and 500 m fields).

Generates frozen banks, resumes from existing n100 checkpoints for seeds 1–100,
runs anchor/multi-start baselines, then TD3. Re-plots PPT figures when done.

Usage:
  python scripts/run_n200_ppt_eval.py
  python scripts/run_n200_ppt_eval.py --only 100m
  python scripts/run_n200_ppt_eval.py --only 500m --skip-td3
  python scripts/run_n200_ppt_eval.py --plot-only
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

from run_sca_anchor_cases import METHODS, run_n100_case  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.n100 import evaluate_bank, write_eval  # noqa: E402
from uavdt.experiments.scenario_bank import generate_bank, load_bank, write_bank  # noqa: E402
from uavdt.experiments.grids import config_for_counts  # noqa: E402
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings  # noqa: E402
from uavdt.td3.settings import TD3Settings  # noqa: E402

N_SCENARIOS = 200
TD3_METHODS = ("random", "kmeans", "pso", "sca", "td3")

CASES = {
    "100m": {
        "area_m": 100.0,
        "bank": ROOT / "data" / "scenario_bank" / "n200_i10_j3_100m.json",
        "anchor_src_ckpt": ROOT / "results" / "n100" / "eval_anchor.checkpoint.json",
        "anchor_out": ROOT / "results" / "n200" / "eval_anchor.json",
        "anchor_ckpt": ROOT / "results" / "n200" / "eval_anchor.checkpoint.json",
        "anchor_fig": ROOT / "results" / "figures" / "n200_anchor",
        "td3_src_ckpt": ROOT / "results" / "n100" / "eval_td3.checkpoint.json",
        "td3_out": ROOT / "results" / "n200" / "eval_td3.json",
        "td3_ckpt": ROOT / "results" / "n200" / "eval_td3.checkpoint.json",
        "ppt_stem": "n200_mean_sum_rate",
        "field_m": 100.0,
    },
    "500m": {
        "area_m": 500.0,
        "bank": ROOT / "data" / "scenario_bank" / "n200_i10_j3_500m.json",
        "anchor_src_ckpt": (
            ROOT / "results" / "n100_500m_cap25" / "eval_anchor.checkpoint.json"
        ),
        "anchor_out": ROOT / "results" / "n200_500m_cap25" / "eval_anchor.json",
        "anchor_ckpt": ROOT / "results" / "n200_500m_cap25" / "eval_anchor.checkpoint.json",
        "anchor_fig": ROOT / "results" / "figures" / "n200_500m_anchor",
        "td3_src_ckpt": ROOT / "results" / "n100_500m_cap25" / "eval_td3.checkpoint.json",
        "td3_out": ROOT / "results" / "n200_500m_cap25" / "eval_td3.json",
        "td3_ckpt": ROOT / "results" / "n200_500m_cap25" / "eval_td3.checkpoint.json",
        "ppt_stem": "n200_500m_mean_sum_rate",
        "field_m": 500.0,
    },
}


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _ensure_bank(area_m: float, bank_path: Path, *, force: bool = False) -> dict:
    if not force and bank_path.exists():
        bank = load_bank(bank_path)
        if int(bank["n_scenarios"]) == N_SCENARIOS:
            _log(f"loaded bank {bank_path.name}  n={bank['n_scenarios']}")
            return bank
    if force and bank_path.exists():
        bank_path.unlink()
        _log(f"removed bank for regenerate {bank_path.name}")
    cfg = config_for_counts(
        10,
        3,
        SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE),
    ).with_square_area_m(float(area_m))
    bank = generate_bank(
        N_SCENARIOS,
        cfg,
        seed_start=1,
        num_iot=10,
        num_uav=3,
    )
    write_bank(bank, bank_path)
    _log(f"wrote {bank_path}  n={bank['n_scenarios']}  area={area_m:g} m")
    return bank


def _seed_checkpoint(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if not src.exists():
        _log(f"warning: no checkpoint to seed from {src}")
        return
    shutil.copy2(src, dest)
    _log(f"seeded checkpoint {src.name} -> {dest}")


def _td3_settings() -> TD3Settings:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"TD3 device            {device}")
    return TD3Settings(device=device, log_every=250)


def run_td3_case(case: dict) -> dict:
    bank_path = case["bank"]
    bank = load_bank(bank_path)
    out = case["td3_out"]
    checkpoint = case["td3_ckpt"]
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        td3_n = int((payload.get("by_method") or {}).get("td3", {}).get("n") or 0)
        if td3_n >= N_SCENARIOS:
            _log(f"skip complete TD3 {out}")
            return payload
    _seed_checkpoint(case["td3_src_ckpt"], checkpoint)
    overlay = SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)
    _log(f"TD3 bank={bank_path.name}  out={out.relative_to(ROOT)}")
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        overlay,
        TD3_METHODS,
        sca_settings=SCASettings(solver=None),
        pso_settings=PSOSettings(),
        td3_settings=_td3_settings(),
        checkpoint_path=checkpoint,
        resume=True,
        bank_path=bank_path,
    )
    write_eval(payload, out)
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")
    stats = payload["by_method"]["td3"]
    _log(
        f"  td3              mean={stats['mean_sum_rate_Mbps']:.4f} Mbps  "
        f"n={stats['n']}"
    )
    return payload


def plot_ppt(case: dict) -> None:
    from plot_ppt_all_methods_bank import plot_delta, plot_mean_bars, _load_by_method

    out_dir = ROOT / "results" / "figures" / "ppt_all_methods"
    td3_path = case["td3_out"] if case["td3_out"].exists() else case["anchor_out"]
    by = _load_by_method(case["anchor_out"], td3_path)
    stem = out_dir / case["ppt_stem"]
    plot_delta(by, out_stem=stem, field_m=case["field_m"], n_layouts=N_SCENARIOS)
    plot_mean_bars(
        by,
        out_stem=out_dir / f"{case['ppt_stem']}_bars",
        field_m=case["field_m"],
        n_layouts=N_SCENARIOS,
    )
    _log(f"PPT figures -> {stem}.pdf")


def run_case(
    key: str,
    *,
    skip_td3: bool,
    skip_plot: bool,
    force_banks: bool = False,
    force_eval: bool = False,
) -> None:
    case = CASES[key]
    _ensure_bank(case["area_m"], case["bank"], force=force_banks)
    if force_eval:
        for p in (case["anchor_out"], case["anchor_ckpt"]):
            if p.exists():
                p.unlink()
                _log(f"removed {p.relative_to(ROOT)}")
    run_n100_case(
        bank_path=case["bank"],
        src_ckpt=case["anchor_src_ckpt"],
        out=case["anchor_out"],
        checkpoint=case["anchor_ckpt"],
        fig_dir=case["anchor_fig"],
        skip_plot=skip_plot,
    )
    if not skip_td3:
        run_td3_case(case)
    if not skip_plot:
        plot_ppt(case)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma list: 100m,500m (default: both)",
    )
    parser.add_argument("--skip-td3", action="store_true")
    parser.add_argument(
        "--force-banks",
        action="store_true",
        help="Regenerate frozen banks (new random UAV placement)",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Delete anchor checkpoints/outputs and re-run bank eval",
    )
    parser.add_argument(
        "--only-td3",
        action="store_true",
        help="Skip anchor/baselines; run TD3 + PPT plots only",
    )
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Re-render PPT figures from existing JSON only",
    )
    args = parser.parse_args(argv)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()} or {"100m", "500m"}
    t0 = perf_counter()
    if args.plot_only:
        for key in sorted(wanted):
            if key not in CASES:
                raise SystemExit(f"unknown case {key}")
            plot_ppt(CASES[key])
        return 0
    for key in ("100m", "500m"):
        if key not in wanted:
            continue
        _log(f"======== {key} ({N_SCENARIOS} scenarios) ========")
        if args.only_td3:
            _ensure_bank(
                CASES[key]["area_m"],
                CASES[key]["bank"],
                force=bool(args.force_banks),
            )
            if not args.skip_td3:
                run_td3_case(CASES[key])
            if not args.skip_plot:
                plot_ppt(CASES[key])
            continue
        run_case(
            key,
            skip_td3=bool(args.skip_td3),
            skip_plot=bool(args.skip_plot),
            force_banks=bool(args.force_banks),
            force_eval=bool(args.force_eval),
        )
    _log(f"done in {perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
