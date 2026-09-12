"""Run zenith-anchor SCA on default-point cells, J/I sweeps, 15% and T_k=0.8.

Cases (8.8 MHz, I=10, T_k=2.8 s unless noted):
  n20_100m / n20_500m / n100_100m / n100_500m — 25% default-point cells
  j_sweep_100m / j_sweep_500m — UAV axis J=1..5, 20 seeds
  i_sweep_500m — IoT axis I=10..32 at 500 m, full enum (max_enumerate=10000)
  n20_500m_cap15 / n100_500m_cap15 — tighter 15% cap at 500 m
  tk08_100m — T_k=0.8 s + process-cohesive candidate per zenith set

Reuses existing random/k-means/PSO/SCA/multi-start checkpoints. Does not
overwrite headline campaigns or n100/eval.json.

Usage:
  python scripts/run_sca_anchor_cases.py
  python scripts/run_sca_anchor_cases.py --skip-plot
  python scripts/run_sca_anchor_cases.py --only n20_100m,n20_500m
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

from run_sca_anchor_eval import run_n20_eval  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SENSITIVITY_MAX_BW_SHARE, SimConfig  # noqa: E402
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign  # noqa: E402
from uavdt.experiments.n100 import evaluate_bank, write_eval  # noqa: E402
from uavdt.experiments.scenario_bank import load_bank  # noqa: E402
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings  # noqa: E402

PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
    ROOT / "results" / "n100_500m_cap15" / "eval.json",
    ROOT / "results" / "residual_on_sca" / "multistart_n20.json",
}
METHODS = ("random", "kmeans", "pso", "sca", "sca_multistart", "sca_anchor")


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
    ms = by_m.get("sca_anchor") or {}
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
    max_bw_share: float = PRIMARY_MAX_BW_SHARE,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    _refuse(out)
    bank = load_bank(bank_path)
    n = int(bank["n_scenarios"])
    if _n100_complete(out, n):
        _log(f"skip complete n100 {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    _prepare_n100_checkpoint(src_ckpt, checkpoint)
    overlay = SimConfig(b_sys_hz=8.8e6, max_bw_share=float(max_bw_share))
    _log(
        f"n100 bank={bank_path.name}  methods={list(methods)}  "
        f"cap={float(max_bw_share):.0%}  out={out.relative_to(ROOT)}"
    )
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        overlay,
        methods,
        sca_settings=SCASettings(solver=None, max_iterations=30),
        pso_settings=PSOSettings(),
        anchor_settings=AnchorSettings(),
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


def _j_sweep_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    points = payload.get("points") or []
    js = sorted(int(float(p["x"])) for p in points if p.get("axis") == "uavs")
    if js != [1, 2, 3, 4, 5]:
        return False
    return all("sca_anchor" in (p.get("by_method") or {}) for p in points)


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
    if "sca_anchor" not in methods:
        methods.append("sca_anchor")
    header = dict(base)
    header["methods"] = methods
    header["sca_anchor"] = True
    header["points"] = merged_points
    header["note"] = (
        f"Merged headline campaign {axis} axis with opt-in sca_anchor. "
        "Does not replace frozen SCA. " + str(base.get("note") or "")
    )
    return header


def _merge_uavs_campaign(base_path: Path, extra: dict) -> dict:
    return _merge_axis_campaign(base_path, extra, "uavs")


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
        methods=("sca_anchor",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        anchor_settings=AnchorSettings(),
    )
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    _log(
        f"J-sweep sca_anchor  area={area_m:g}x{area_m:g} m  "
        f"J=1..5  n_runs=20  out={out.relative_to(ROOT)}"
    )
    t0 = perf_counter()
    extra = run_campaign(("uavs",), cfg, settings, checkpoint_path=ckpt)
    payload = _merge_uavs_campaign(campaign_path, extra)
    write_campaign(payload, out)
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")
    for pt in payload["points"]:
        anc = (pt.get("by_method") or {}).get("sca_anchor") or {}
        sca = (pt.get("by_method") or {}).get("sca") or {}
        _log(
            f"  J={int(float(pt['x']))}  "
            f"anchor={anc.get('mean_sum_rate_Mbps', float('nan')):.4f}  "
            f"SCA={sca.get('mean_sum_rate_Mbps', float('nan')):.4f}"
        )
    return payload


def _axis_complete(path: Path, axis: str, xs: list[int]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    got = sorted(
        int(float(p["x"]))
        for p in payload.get("points") or []
        if p.get("axis") == axis and "sca_anchor" in (p.get("by_method") or {})
    )
    return got == sorted(int(x) for x in xs)


def run_i_sweep(*, area_m: float, out: Path, campaign_path: Path) -> dict:
    _refuse(out)
    want = [10, 16, 20, 24, 28, 32]
    if _axis_complete(out, "iots", want):
        _log(f"skip complete I-sweep {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    cfg = SimConfig(
        b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE
    ).with_square_area_m(float(area_m))
    settings = CampaignSettings(
        n_runs=20,
        seed_start=1,
        methods=("sca_anchor",),
        sca_settings=SCASettings(solver=None, max_iterations=30),
        anchor_settings=AnchorSettings(max_enumerate=10_000),
    )
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    _log(
        f"I-sweep sca_anchor  area={area_m:g}x{area_m:g} m  "
        f"I=10..32  full enum  n_runs=20  out={out.relative_to(ROOT)}"
    )
    t0 = perf_counter()
    extra = run_campaign(("iots",), cfg, settings, checkpoint_path=ckpt)
    payload = _merge_axis_campaign(campaign_path, extra, "iots")
    write_campaign(payload, out)
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out}  ({perf_counter() - t0:.1f}s)")
    for pt in payload["points"]:
        anc = (pt.get("by_method") or {}).get("sca_anchor") or {}
        sca = (pt.get("by_method") or {}).get("sca") or {}
        _log(
            f"  I={int(float(pt['x']))}  "
            f"anchor={anc.get('mean_sum_rate_Mbps', float('nan')):.4f}  "
            f"SCA={sca.get('mean_sum_rate_Mbps', float('nan')):.4f}"
        )
    return payload


def run_tk08(*, out: Path) -> dict:
    _refuse(out)
    if out.exists():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = {}
        if int((old.get("n_ok") or 0)) == 20:
            _log(f"skip complete T_k=0.8 {out}")
            return old
    from dataclasses import replace

    from uavdt.experiments.grids import config_for_counts
    from uavdt.experiments.methods import run_method
    from uavdt.scenario import generate_scenario

    cfg = replace(
        config_for_counts(10, 3, SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)),
        aodt_threshold_s=0.8,
    )
    anc = AnchorSettings(process_cohesive_candidate=True)
    sca_settings = SCASettings(solver=None, max_iterations=30)
    rows = []
    _log("T_k=0.8 s  sca_anchor + process-cohesive per zenith set  n=20")
    t0 = perf_counter()
    for seed in range(1, 21):
        sc = generate_scenario(seed, cfg)
        run = run_method(
            sc,
            "sca_anchor",
            seed,
            sca_settings=sca_settings,
            anchor_settings=anc,
        )
        rec = {
            "seed": seed,
            "feasible": bool(run.feasible),
            "sum_rate_Mbps": float(run.sum_rate_mbps),
            "winner_kind": run.diagnostics.get("winner_kind"),
            "winner_assoc_kind": run.diagnostics.get("winner_assoc_kind"),
            "winner_combo": run.diagnostics.get("winner_combo"),
            "n_lp": run.diagnostics.get("n_lp"),
        }
        rows.append(rec)
        _log(
            f"  seed {seed:2d}  {run.sum_rate_mbps:.4f} Mbps  "
            f"feas={int(run.feasible)}  "
            f"kind={rec['winner_kind']}  assoc={rec['winner_assoc_kind']}"
        )
    feas = [r for r in rows if r["feasible"]]
    payload = {
        "label": "sca_anchor T_k=0.8 with process-cohesive candidate",
        "b_sys_hz": cfg.b_sys_hz,
        "max_bw_share": cfg.max_bw_share,
        "aodt_threshold_s": 0.8,
        "area_m": [cfg.area_x_m, cfg.area_y_m],
        "n_runs": 20,
        "n_ok": len(feas),
        "feasible_fraction": len(feas) / 20.0,
        "mean_Mbps_all": float(sum(r["sum_rate_Mbps"] for r in rows) / 20.0),
        "mean_Mbps_feasible": (
            float(sum(r["sum_rate_Mbps"] for r in feas) / len(feas)) if feas else None
        ),
        "sca_joint_cohesive_ref_Mbps": 8.393,
        "per_seed": rows,
        "wall_s": perf_counter() - t0,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(
        f"wrote {out}  feas={len(feas)}/20  "
        f"mean_feas={payload['mean_Mbps_feasible']}  "
        f"ref cohesive SCA-joint=8.393"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-j-sweep", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=(
            "Comma list of case ids: n20_100m,n20_500m,n100_100m,n100_500m,"
            "j_sweep_100m,j_sweep_500m,i_sweep_500m,n20_500m_cap15,"
            "n100_500m_cap15,tk08_100m"
        ),
    )
    args = parser.parse_args(argv)
    wanted = {
        x.strip()
        for x in str(args.only).split(",")
        if x.strip()
    } or {
        "n20_100m",
        "n20_500m",
        "n100_100m",
        "n100_500m",
        "j_sweep_100m",
        "j_sweep_500m",
    }
    if args.skip_j_sweep:
        wanted.discard("j_sweep_100m")
        wanted.discard("j_sweep_500m")

    t_all = perf_counter()
    if "n20_100m" in wanted:
        _log("=== n20 100x100 m ===")
        run_n20_eval(
            out=ROOT / "results" / "sca_anchor_n20.json",
            area_m=100.0,
            skip_if_complete=True,
            campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
            multistart_path=ROOT / "results" / "sca_multistart_n20.json",
        )
    if "n20_500m" in wanted:
        _log("=== n20 500x500 m ===")
        run_n20_eval(
            out=ROOT / "results" / "sca_anchor_n20_500m.json",
            area_m=500.0,
            skip_if_complete=True,
            campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
            multistart_path=ROOT / "results" / "sca_multistart_n20_500m.json",
        )
    if "n100_100m" in wanted:
        _log("=== n100 100x100 m ===")
        run_n100_case(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
            src_ckpt=ROOT / "results" / "n100" / "eval_multistart.checkpoint.json",
            out=ROOT / "results" / "n100" / "eval_anchor.json",
            checkpoint=ROOT / "results" / "n100" / "eval_anchor.checkpoint.json",
            fig_dir=ROOT / "results" / "figures" / "n100_anchor",
            skip_plot=bool(args.skip_plot),
        )
    if "n100_500m" in wanted:
        _log("=== n100 500x500 m ===")
        run_n100_case(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            src_ckpt=(
                ROOT / "results" / "n100_500m_cap25" / "eval_multistart.checkpoint.json"
            ),
            out=ROOT / "results" / "n100_500m_cap25" / "eval_anchor.json",
            checkpoint=(
                ROOT / "results" / "n100_500m_cap25" / "eval_anchor.checkpoint.json"
            ),
            fig_dir=ROOT / "results" / "figures" / "n100_500m_anchor",
            skip_plot=bool(args.skip_plot),
        )
    if "j_sweep_100m" in wanted:
        _log("=== J-sweep 100x100 m ===")
        run_j_sweep(
            area_m=100.0,
            out=ROOT / "results" / "campaign_sca_anchor_uavs.json",
            campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
        )
    if "j_sweep_500m" in wanted:
        _log("=== J-sweep 500x500 m ===")
        run_j_sweep(
            area_m=500.0,
            out=ROOT / "results" / "campaign_sca_anchor_uavs_500m.json",
            campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
        )
    if "i_sweep_500m" in wanted:
        _log("=== I-sweep 500x500 m (full enum) ===")
        run_i_sweep(
            area_m=500.0,
            out=ROOT / "results" / "campaign_sca_anchor_iots_500m.json",
            campaign_path=ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
        )
        if not args.skip_plot:
            try:
                from plot_paper_figures import plot_sum_rate_figure

                fig_dir = ROOT / "results" / "figures" / "anchor_iots_500m"
                fig_dir.mkdir(parents=True, exist_ok=True)
                payload = json.loads(
                    (ROOT / "results" / "campaign_sca_anchor_iots_500m.json").read_text(
                        encoding="utf-8"
                    )
                )
                p = plot_sum_rate_figure(
                    payload,
                    "iots",
                    "Number of IoT devices ($I$)",
                    "Fig. 7 analogue — sum rate vs IoT count ($J=3$)",
                    7,
                    fig_dir,
                )
                _log(f"wrote {p}")
            except Exception as exc:
                _log(f"I-sweep plot skipped: {exc}")
    if "n20_500m_cap15" in wanted:
        _log("=== n20 500x500 m 15% cap ===")
        run_n20_eval(
            out=ROOT / "results" / "sca_anchor_n20_500m_cap15.json",
            area_m=500.0,
            max_bw_share=SENSITIVITY_MAX_BW_SHARE,
            skip_if_complete=True,
        )
    if "n100_500m_cap15" in wanted:
        _log("=== n100 500x500 m 15% cap ===")
        cap15_ckpt = ROOT / "results" / "n100_500m_cap15" / "eval.checkpoint.json"
        cap15_eval = ROOT / "results" / "n100_500m_cap15" / "eval.json"
        run_n100_case(
            bank_path=ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
            src_ckpt=cap15_ckpt if cap15_ckpt.exists() else cap15_eval,
            out=ROOT / "results" / "n100_500m_cap15" / "eval_anchor.json",
            checkpoint=ROOT / "results" / "n100_500m_cap15" / "eval_anchor.checkpoint.json",
            fig_dir=ROOT / "results" / "figures" / "n100_500m_cap15_anchor",
            skip_plot=bool(args.skip_plot),
            max_bw_share=SENSITIVITY_MAX_BW_SHARE,
            methods=("random", "kmeans", "pso", "sca", "sca_anchor"),
        )
    if "tk08_100m" in wanted:
        _log("=== T_k=0.8 s + process-cohesive per zenith set ===")
        run_tk08(out=ROOT / "results" / "sca_anchor_tk08.json")
    _log(f"all requested cases done in {perf_counter() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
