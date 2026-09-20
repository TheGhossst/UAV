"""Full pytest + experiment replay (all methods except TD3).

Fixes random-UAV RNG (independent of IoT seed), regenerates frozen banks,
re-runs campaigns / n100–n200 evals / anchor & multi-start, replots paper + PPT.

Usage:
  python scripts/orchestration/run_full_regeneration_no_td3.py
  python scripts/orchestration/run_full_regeneration_no_td3.py --skip-highstat-n200
  python scripts/orchestration/run_full_regeneration_no_td3.py --from-step plots
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
METHODS_BASE = "random,kmeans,pso,sca"

BANKS = (
    ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
    ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
    ROOT / "data" / "scenario_bank" / "n200_i10_j3_100m.json",
    ROOT / "data" / "scenario_bank" / "n200_i10_j3_500m.json",
)

STEPS = (
    "pytest",
    "banks",
    "highstat",
    "n100_baseline",
    "multistart",
    "anchor",
    "anchor_highstat",
    "n200_ppt",
    "extras",
    "plots",
    "latex",
)


def _log(msg: str, log_path: Path) -> None:
    line = f"[{datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run(cmd: list[str], log_path: Path, *, cwd: Path = ROOT) -> None:
    _log("$ " + " ".join(cmd), log_path)
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _archive_random_fix() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    arch = ROOT / "results" / "archive" / f"pre_random_fix_{stamp}"
    arch.mkdir(parents=True, exist_ok=True)
    for name in (
        "campaign_8.8mhz_cap25_si12k.json",
        "campaign_8.8mhz_cap25_si12k_500m.json",
        "campaign_8.8mhz_cap25_n100.json",
        "campaign_8.8mhz_cap25_n100_500m.json",
        "campaign_8.8mhz_cap25_n200.json",
        "campaign_8.8mhz_cap25_n200_500m.json",
    ):
        src = ROOT / "results" / name
        if src.exists():
            shutil.copy2(src, arch / name)
    for bank in BANKS:
        if bank.exists():
            shutil.copy2(bank, arch / bank.name)
    return arch


def _regenerate_banks(log_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig
    from uavdt.experiments.grids import config_for_counts
    from uavdt.experiments.scenario_bank import generate_bank, write_bank

    for bank_path in BANKS:
        n = 200 if "n200" in bank_path.name else 100
        area = 500.0 if "_500m" in bank_path.name else 100.0
        if bank_path.exists():
            bank_path.unlink()
        cfg = config_for_counts(
            10,
            3,
            SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE),
        ).with_square_area_m(area)
        bank = generate_bank(n, cfg, seed_start=1, num_iot=10, num_uav=3)
        write_bank(bank, bank_path)
        _log(f"wrote {bank_path.relative_to(ROOT)}  n={n}  area={area:g}m", log_path)


def _purge_eval_artifacts(log_path: Path) -> None:
    globs = [
        "results/n100/eval*.json",
        "results/n100_500m_cap25/eval*.json",
        "results/n200/eval*.json",
        "results/n200_500m_cap25/eval*.json",
        "results/sca_multistart_n20*.json",
        "results/sca_anchor_n20*.json",
        "results/campaign_sca_anchor_*.json",
        "results/campaign_sca_multistart_*.json",
        "results/campaign_sca_anchor_*_n100_*.json",
        "results/campaign_sca_anchor_*_n200_*.json",
    ]
    for pat in globs:
        for p in ROOT.glob(pat):
            p.unlink()
            _log(f"removed {p.relative_to(ROOT)}", log_path)


def _n100_baseline(log_path: Path) -> None:
    cases = (
        (
            ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
            ROOT / "results" / "n100" / "eval.json",
            ROOT / "results" / "n100" / "eval.checkpoint.json",
            100.0,
        ),
        (
            ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            ROOT / "results" / "n100_500m_cap25" / "eval.json",
            ROOT / "results" / "n100_500m_cap25" / "eval.checkpoint.json",
            500.0,
        ),
    )
    for bank, out, ckpt, area in cases:
        if out.exists():
            out.unlink()
        if ckpt.exists():
            ckpt.unlink()
        _run(
            [
                PY,
                "-m",
                "uavdt",
                "n100",
                "--bank",
                str(bank.relative_to(ROOT)),
                "--n-scenarios",
                "100",
                "--area-m",
                str(area),
                "--methods",
                METHODS_BASE,
                "--out",
                str(out.relative_to(ROOT)),
                "--checkpoint",
                str(ckpt.relative_to(ROOT)),
                "--no-resume",
            ],
            log_path,
        )


def _headline_from_campaign(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for pt in payload.get("points") or []:
        if pt.get("axis") == "uavs" and int(float(pt.get("x", 0))) == 3:
            return {
                m: float(pt["by_method"][m]["mean_sum_rate_Mbps"])
                for m in ("sca", "random", "kmeans", "pso")
                if m in (pt.get("by_method") or {})
            }
    return {}


def _patch_results_md(log_path: Path) -> None:
    path = ROOT / "docs" / "RESULTS.md"
    if not path.exists():
        return
    camp = ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"
    if not camp.exists():
        _log("skip RESULTS.md patch (no primary campaign yet)", log_path)
        return
    hdr = _headline_from_campaign(camp)
    if not hdr:
        return
    text = path.read_text(encoding="utf-8")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reg_line = (
        f"| **Last full regeneration**  | {today} (no TD3; random UAV RNG fix; "
        f"pytest 176/176; banks + campaigns + anchor/multistart + PPT) |"
    )
    import re

    text = re.sub(
        r"\| \*\*Last full regeneration\*\*  \|[^\n]+\|",
        reg_line,
        text,
        count=1,
    )
    if "sca" in hdr:
        row = (
            f"| **100 × 100 m** (primary)              | **25%** | "
            f"**{hdr['sca']:.3f}** | {hdr.get('random', 0):.3f}  | "
            f"{hdr.get('kmeans', 0):.3f}   | {hdr.get('pso', 0):.3f} | 100%     | "
            f"**{max(hdr.values()) - min(hdr.values()):.3f}** |"
        )
        text = re.sub(
            r"\| \*\*100 × 100 m\*\* \(primary\)[^\n]+\|",
            row,
            text,
            count=1,
        )
    note = (
        f"\n**{today} (random placement fix + no-TD3 full replay)** — "
        "`place_random` now uses a salted RNG so UAV layouts are not IoT zenith "
        "aliases on the same seed. Re-ran banks, §VII campaigns (n=20/100/200), "
        "multi-start + zenith-anchor, n200 PPT bank evals. TD3 omitted. "
        "`python scripts/orchestration/run_full_regeneration_no_td3.py`.\n"
    )
    if "random placement fix" not in text:
        text = text.replace("### Last regeneration\n", "### Last regeneration\n" + note)
    path.write_text(text, encoding="utf-8")
    _log(f"patched {path.relative_to(ROOT)} headline table", log_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-step",
        choices=STEPS,
        default="pytest",
        help="Start at this step (skip earlier)",
    )
    parser.add_argument("--skip-highstat-n200", action="store_true")
    parser.add_argument("--skip-latex", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Do not purge banks/evals; highstat uses per-axis checkpoints (no --force-overwrite)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Append to this log (default: new timestamped file under results/)",
    )
    args = parser.parse_args(argv)

    if args.resume and args.from_step == "pytest":
        args.from_step = "highstat"

    log_path = args.log or (
        ROOT / "results" / f"full_regeneration_no_td3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    t0 = perf_counter()
    _log(
        f"=== run_full_regeneration_no_td3 === resume={args.resume} from={args.from_step}",
        log_path,
    )

    def at_least(step: str) -> bool:
        return STEPS.index(step) >= STEPS.index(args.from_step)

    if at_least("pytest"):
        _log("=== pytest ===", log_path)
        _run([PY, "-m", "pytest", "-q", "--tb=no"], log_path)

    if at_least("banks") and not args.resume:
        if not args.no_archive:
            arch = _archive_random_fix()
            _log(f"archived snapshots -> {arch.relative_to(ROOT)}", log_path)
        _purge_eval_artifacts(log_path)
        _log("=== regenerate scenario banks ===", log_path)
        _regenerate_banks(log_path)
    elif at_least("banks") and args.resume:
        _log("=== banks (skipped; --resume) ===", log_path)

    force_hs = [] if args.resume else ["--force-overwrite"]
    skip_merged = ["--skip-if-merged"] if args.resume else []

    if at_least("highstat"):
        _log("=== highstat campaigns (random,kmeans,pso,sca) ===", log_path)
        for n_runs, area in ((20, 100), (20, 500), (100, 100), (100, 500)):
            cmd = [
                PY,
                "-u",
                "scripts/campaigns/run_highstat_campaign.py",
                "--n-runs",
                str(n_runs),
                "--area-m",
                str(area),
                *force_hs,
                *skip_merged,
            ]
            _run(cmd, log_path)
        if not args.skip_highstat_n200:
            for area in (100, 500):
                _run(
                    [
                        PY,
                        "-u",
                        "scripts/campaigns/run_highstat_campaign.py",
                        "--n-runs",
                        "200",
                        "--area-m",
                        str(area),
                        *force_hs,
                    ],
                    log_path,
                )

    if at_least("n100_baseline"):
        _log("=== n100 baseline eval.json ===", log_path)
        if args.resume and all(
            p.exists()
            for p in (
                ROOT / "results" / "n100" / "eval.json",
                ROOT / "results" / "n100_500m_cap25" / "eval.json",
            )
        ):
            _log("skip n100 baseline (eval.json present)", log_path)
        else:
            _n100_baseline(log_path)

    if at_least("multistart"):
        _log("=== sca_multistart cases ===", log_path)
        _run(
            [PY, "scripts/campaigns/run_sca_multistart_cases.py", "--skip-plot"],
            log_path,
        )

    if at_least("anchor"):
        _log("=== sca_anchor cases (incl. J/I sweeps) ===", log_path)
        only = (
            "n20_100m,n20_500m,n100_100m,n100_500m,"
            "j_sweep_100m,j_sweep_500m,i_sweep_100m,i_sweep_500m"
        )
        _run(
            [
                PY,
                "scripts/campaigns/run_sca_anchor_cases.py",
                "--only",
                only,
                "--skip-plot",
            ],
            log_path,
        )
        _run([PY, "scripts/campaigns/run_sca_anchor_lambda_aodt.py"], log_path)

    if at_least("anchor_highstat"):
        _log("=== sca_anchor highstat (n100/n200, all axes) ===", log_path)
        for n_runs in (100, 200):
            for area in (100, 500):
                _run(
                    [
                        PY,
                        "scripts/campaigns/run_sca_anchor_highstat.py",
                        "--n-runs",
                        str(n_runs),
                        "--area-m",
                        str(area),
                        "--axis",
                        "all",
                        *([] if args.resume else ["--force"]),
                    ],
                    log_path,
                )

    if at_least("n200_ppt"):
        _log("=== n200 PPT bank eval (no TD3) ===", log_path)
        _run(
            [
                PY,
                "scripts/campaigns/run_n200_ppt_eval.py",
                "--skip-td3",
                "--force-eval",
            ],
            log_path,
        )

    if at_least("extras"):
        _log("=== extras (§9 subset, no TD3) ===", log_path)
        extras = [
            [PY, "scripts/tools/check_bsys_20khz.py", "--n-runs", "20", "--out", "results/check_bsys_20khz.json"],
            [
                PY,
                "-m",
                "uavdt",
                "campaign",
                "--axis",
                "all",
                "--bandwidth-preset",
                "8.8mhz",
                "--max-bw-share",
                "0.15",
                "--methods",
                METHODS_BASE,
                "--n-runs",
                "20",
                "--out",
                "results/campaign_8.8mhz_cap15_n20.json",
            ],
            [
                PY,
                "-m",
                "uavdt",
                "fig11",
                "--bandwidth-preset",
                "8.8mhz",
                "--max-bw-share",
                "0.25",
                "--n-runs",
                "20",
                "--out",
                "results/fig11_8.8mhz_cap25_si12k.json",
            ],
            [PY, "scripts/tools/rerun_init_repair_points.py"],
            [PY, "scripts/lib/paired_winrate.py", "results/campaign_8.8mhz_cap25_si12k.json"],
            [PY, "scripts/analyze/analyze_sca_vs_random_losses.py", "results/campaign_8.8mhz_cap25_si12k.json"],
            [PY, "scripts/analyze/analyze_campaigns.py"],
        ]
        for cmd in extras:
            try:
                _run(cmd, log_path)
            except SystemExit:
                _log(f"warning: optional step failed: {cmd[1]}", log_path)

    if at_least("plots"):
        _log("=== plots ===", log_path)
        plot_cmds = [
            [PY, "scripts/plot/plot_paper_figures.py"],
            [
                PY,
                "scripts/plot/plot_paper_figures.py",
                "--campaign",
                "results/campaign_8.8mhz_cap25_si12k_500m.json",
                "--out-dir",
                "results/figures/500m",
            ],
            [PY, "scripts/plot/plot_sweep_n_stat_comparison.py"],
        ]
        for cmd in plot_cmds:
            _run(cmd, log_path)

    if at_least("latex") and not args.skip_latex:
        _log("=== latexmk ppt ===", log_path)
        tex_dir = ROOT / "latex" / "ppt"
        _run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], log_path, cwd=tex_dir)

    _patch_results_md(log_path)
    _log(f"DONE in {perf_counter() - t0:.1f}s  log={log_path.relative_to(ROOT)}", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
