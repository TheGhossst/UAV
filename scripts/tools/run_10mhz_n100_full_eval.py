"""10 MHz / 25% cap / 100×100 m / 100 seeds — all five methods + sweep anchor + plots.

Writes under results/eval_10mhz_cap25_100m_n100/ (does not overwrite headline JSON).

Usage (repo root):
  $env:PYTHONPATH = "src"
  python scripts/tools/run_10mhz_n100_full_eval.py
  python scripts/tools/run_10mhz_n100_full_eval.py --skip-sweeps
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "campaigns"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from run_sca_anchor_cases import _merge_axis_campaign  # noqa: E402

from uavdt.config import (  # noqa: E402
    PRIMARY_B_SYS_HZ,
    PRIMARY_CAMPAIGN_REL,
    PRIMARY_MAX_BW_SHARE,
    headline_sim_config,
)
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign  # noqa: E402
from uavdt.experiments.n100 import evaluate_bank, write_eval  # noqa: E402
from uavdt.experiments.scenario_bank import load_bank  # noqa: E402
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings  # noqa: E402

OUT = ROOT / "results" / "eval_10mhz_cap25_100m_n100"
FIG = OUT / "figures"
BASE_CAMP = ROOT / PRIMARY_CAMPAIGN_REL
BANK = ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json"
N_RUNS = 100
AXES = ("uavs", "iots", "lambda", "aodt")
METHODS = ("random", "kmeans", "pso", "sca", "sca_anchor")
PY = sys.executable


def _log(msg: str) -> None:
    print(msg, flush=True)


def _anchor_axis_complete(path: Path, axis: str) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if float(payload.get("b_sys_hz") or 0) != float(PRIMARY_B_SYS_HZ):
        return False
    if int(payload.get("n_runs") or 0) != N_RUNS:
        return False
    pts = [p for p in payload.get("points") or [] if p.get("axis") == axis]
    return bool(pts) and all("sca_anchor" in (p.get("by_method") or {}) for p in pts)


def run_anchor_sweeps(force: bool) -> None:
    cfg = headline_sim_config(area_m=100.0)
    for axis in AXES:
        out = OUT / f"anchor_{axis}_n100_100m.json"
        if not force and _anchor_axis_complete(out, axis):
            _log(f"skip complete {out.name}")
            continue
        if force and out.exists():
            out.unlink()
        anc = (
            AnchorSettings(max_enumerate=10_000)
            if axis == "iots"
            else AnchorSettings()
        )
        settings = CampaignSettings(
            n_runs=N_RUNS,
            seed_start=1,
            methods=("sca_anchor",),
            sca_settings=SCASettings(solver=None, max_iterations=30),
            anchor_settings=anc,
        )
        ckpt = out.with_name(out.stem + ".checkpoint.json")
        _log(f"=== anchor sweep {axis} (n={N_RUNS}) ===")
        t0 = perf_counter()
        payload = run_campaign((axis,), cfg, settings, checkpoint_path=ckpt)
        write_campaign(payload, out)
        if ckpt.exists():
            ckpt.unlink()
        _log(f"wrote {out} ({perf_counter() - t0:.1f}s)")


def merge_campaign_with_anchor() -> Path:
    if not BASE_CAMP.exists():
        raise SystemExit(f"missing baseline campaign {BASE_CAMP}")
    merged_path = OUT / "campaign_10mhz_cap25_n100_with_anchor.json"
    base = json.loads(BASE_CAMP.read_text(encoding="utf-8"))
    header = dict(base)
    methods = list(header.get("methods") or [])
    if "sca_anchor" not in methods:
        methods.append("sca_anchor")
    header["methods"] = methods
    header["sca_anchor"] = True
    points = []
    for pt in base.get("points") or []:
        copy = json.loads(json.dumps(pt))
        axis = copy.get("axis")
        if axis not in AXES:
            points.append(copy)
            continue
        ax_path = OUT / f"anchor_{axis}_n100_100m.json"
        if not ax_path.exists():
            points.append(copy)
            continue
        extra = json.loads(ax_path.read_text(encoding="utf-8"))
        merged_axis = _merge_axis_campaign(BASE_CAMP, extra, axis)
        by_x = {float(p["x"]): p for p in merged_axis["points"] if p.get("axis") == axis}
        other = by_x.get(float(copy["x"]))
        if other is not None:
            copy.setdefault("by_method", {})
            copy["by_method"].update((other.get("by_method") or {}))
        points.append(copy)
    header["points"] = points
    header["note"] = (
        "Merged headline 10 MHz campaign with sca_anchor sweeps from "
        f"{OUT.name}/anchor_*_n100_100m.json. "
        + str(base.get("note") or "")
    )
    write_campaign(header, merged_path)
    _log(f"wrote {merged_path}")
    return merged_path


def run_n100_bank() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    eval_out = OUT / "eval_all_methods.json"
    ckpt = OUT / "eval.checkpoint.json"
    src_ckpt = ROOT / "results" / "n100" / "eval.checkpoint.json"
    if not ckpt.exists() and src_ckpt.exists():
        shutil.copy2(src_ckpt, ckpt)
        _log(f"copied baseline checkpoint -> {ckpt.name}")
    bank = load_bank(BANK)
    cfg = headline_sim_config(area_m=100.0)
    _log("=== n100 bank eval (5 methods) ===")
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        cfg,
        METHODS,
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
        anchor_settings=AnchorSettings(),
        checkpoint_path=ckpt,
        resume=True,
        bank_path=BANK,
    )
    write_eval(payload, eval_out)
    _log(f"wrote {eval_out} ({perf_counter() - t0:.1f}s)")
    return eval_out


def run_default_j3_anchor_eval() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "default_j3_paired_eval.json"
    if out.exists():
        d = json.loads(out.read_text(encoding="utf-8"))
        if int(d.get("n_runs") or 0) == N_RUNS and float(d.get("b_sys_hz") or 0) == float(
            PRIMARY_B_SYS_HZ
        ):
            _log(f"skip complete {out.name}")
            return out
    cmd = [
        PY,
        str(ROOT / "scripts" / "campaigns" / "run_sca_anchor_eval.py"),
        "--n-runs",
        str(N_RUNS),
        "--area-m",
        "100",
        "--out",
        str(out),
        "--campaign",
        str(BASE_CAMP),
        "--max-bw-share",
        str(PRIMARY_MAX_BW_SHARE),
    ]
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    return out


def plot_all(merged_camp: Path) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(ROOT / "scripts" / "plot" / "plot_paper_figures.py"),
        "--campaign",
        str(merged_camp.relative_to(ROOT)),
        "--out-dir",
        str(FIG.relative_to(ROOT)),
    ]
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    eval_json = OUT / "eval_all_methods.json"
    if eval_json.exists():
        try:
            from uavdt.experiments.n100_plot import plot_n100_figures

            bank = load_bank(BANK)
            payload = json.loads(eval_json.read_text(encoding="utf-8"))
            bank_fig = FIG / "n100_bank"
            bank_fig.mkdir(parents=True, exist_ok=True)
            for p in plot_n100_figures(payload, bank, bank_fig):
                _log(f"wrote {p}")
        except ImportError:
            _log("matplotlib missing; skip n100 bank figures")


def write_summary(eval_path: Path, default_path: Path, merged_camp: Path) -> Path:
    from paired_winrate import wilcoxon_signed_rank  # noqa: E402

    lines: list[str] = []
    lines.append("# 10 MHz / 25% cap / 100×100 m / 100 seeds\n")
    lines.append(f"- Campaign (4 baselines): `{BASE_CAMP.relative_to(ROOT)}`")
    lines.append(f"- Merged sweeps + anchor: `{merged_camp.relative_to(ROOT)}`")
    lines.append(f"- Default J=3 bank: `{eval_path.relative_to(ROOT)}`")
    lines.append(f"- Default J=3 paired: `{default_path.relative_to(ROOT)}`")
    lines.append(f"- Figures: `{FIG.relative_to(ROOT)}/`\n")

    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        lines.append("## Default point (frozen bank I=10, J=3)\n")
        lines.append("| Method | Mean Mbps | Std | Feasible % |")
        lines.append("|--------|-----------|-----|------------|")
        for m in METHODS:
            st = (ev.get("by_method") or {}).get(m)
            if not st:
                continue
            lines.append(
                f"| {m} | {st['mean_sum_rate_Mbps']:.4f} | "
                f"{st['std_sum_rate_Mbps']:.4f} | "
                f"{100 * st['feasible_fraction']:.1f}% |"
            )
        sca = (ev.get("by_method") or {}).get("sca")
        anc = (ev.get("by_method") or {}).get("sca_anchor")
        if sca and anc:
            d = [
                float(a) - float(s)
                for a, s in zip(
                    anc["per_seed_Mbps"],
                    sca["per_seed_Mbps"],
                )
            ]
            import numpy as np

            dd = np.array(d, dtype=float)
            w = wilcoxon_signed_rank(dd)
            lines.append(
                f"\n**sca_anchor vs sca:** mean Δ = {dd.mean():+.4f} Mbps, "
                f"wins = {int(np.sum(dd > 1e-9))}/{len(dd)}, "
                f"p = {w['p_two_sided']:.4g}\n"
            )

    if default_path.exists():
        dj = json.loads(default_path.read_text(encoding="utf-8"))
        lines.append("## Default J=3 (campaign baselines + fresh anchor)\n")
        for m, v in (dj.get("mean_Mbps") or {}).items():
            lines.append(f"- **{m}**: {v:.4f} Mbps")
        for key in ("vs_sca", "vs_pso", "vs_random", "vs_kmeans"):
            p = dj.get(key)
            if p:
                lines.append(
                    f"- {key}: Δ = {p['mean_delta_Mbps']:+.4f} Mbps, "
                    f"{p['wins']}/{p['n']} wins, p = {p['wilcoxon']['p_two_sided']:.4g}"
                )

    if merged_camp.exists():
        camp = json.loads(merged_camp.read_text(encoding="utf-8"))
        lines.append("\n## Sweep means (J=3 column where applicable)\n")
        for axis in AXES + ("cpu",):
            pts = [p for p in camp.get("points") or [] if p.get("axis") == axis]
            if not pts:
                continue
            lines.append(f"### {axis}\n")
            for pt in sorted(pts, key=lambda p: float(p["x"])):
                bm = pt.get("by_method") or {}
                parts = []
                for m in METHODS:
                    if m in bm:
                        parts.append(f"{m}={bm[m].get('mean_sum_rate_Mbps', 0):.3f}")
                lines.append(f"- x={pt['x']}: " + ", ".join(parts))

    summary = OUT / "SUMMARY.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"wrote {summary}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--skip-n100", action="store_true")
    parser.add_argument("--skip-default-anchor", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--force-sweeps", action="store_true")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    t_all = perf_counter()

    if not args.skip_default_anchor:
        default_path = run_default_j3_anchor_eval()
    else:
        default_path = OUT / "default_j3_paired_eval.json"

    if not args.skip_sweeps:
        run_anchor_sweeps(force=bool(args.force_sweeps))

    merged = merge_campaign_with_anchor()

    if not args.skip_n100:
        eval_path = run_n100_bank()
    else:
        eval_path = OUT / "eval_all_methods.json"

    if not args.skip_plots:
        plot_all(merged)

    write_summary(eval_path, default_path, merged)
    _log(f"done in {perf_counter() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
