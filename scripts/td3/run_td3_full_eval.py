"""Full TD3 eval: 100-scenario bank plus §VII campaign, compared to other methods.

Uses CUDA when available. Resumes from checkpoints. Does not change SimConfig.
Campaign baselines (random/k-means/PSO/SCA) are reused from the primary
20-seed artifact; only TD3 is trained.
"""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from uavdt.config import BANDWIDTH_PRESETS, PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.campaign import CampaignSettings, write_campaign
from uavdt.experiments.grids import AXES, iter_axis
from uavdt.experiments.methods import run_method
from uavdt.experiments.n100 import evaluate_bank, write_eval
from uavdt.experiments.n100_plot import plot_n100_figures
from uavdt.experiments.scenario_bank import load_bank
from uavdt.placement.pso import PSOSettings
from uavdt.sca.settings import SCASettings
from uavdt.scenario import generate_scenario
from uavdt.td3.settings import TD3Settings
from uavdt.td3.solve import format_eta

BANK = Path("data/scenario_bank/n100_i10_j3_100m.json")
SRC_CKPT = Path("results/n100/eval.checkpoint.json")
N100_CKPT = Path("results/n100/eval_td3.checkpoint.json")
N100_OUT = Path("results/n100/eval_td3.json")
N100_FIG = Path("results/figures/n100_td3")
SRC_CAMP = Path("results/campaign_8.8mhz_cap25_si12k.json")
CAMP_CKPT = Path("results/campaign_8.8mhz_cap25_td3.checkpoint.json")
CAMP_OUT = Path("results/campaign_8.8mhz_cap25_td3.json")
CAMP_FIG = Path("results/figures/td3_campaign")
ANALYSIS = Path("results/td3/full_eval_analysis.txt")
EVAL_LOG = Path("results/td3/full_eval.log")
METHODS = ("random", "kmeans", "pso", "sca", "td3")
BASELINE_METHODS = ("random", "kmeans", "pso", "sca")


class _Tee:
    """Console + UTF-8 file. Do not also redirect with PowerShell *>> (UTF-16)."""

    def __init__(self, stream, file) -> None:
        self.stream = stream
        self.file = file
        self.encoding = "utf-8"
        self.errors = "replace"

    def write(self, data) -> int:
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = str(data)
        try:
            n = self.stream.write(text)
        except UnicodeEncodeError:
            safe = text.encode("ascii", "replace").decode("ascii")
            n = self.stream.write(safe)
        self.file.write(text)
        self.flush()
        return n if isinstance(n, int) else len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self):
        return self.stream.fileno()


def _open_utf8_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        head = path.read_bytes()[:4]
        if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
            path = path.with_name(path.stem + ".utf8.log")
    return path.open(
        "a", encoding="utf-8", newline="\n", errors="replace", buffering=1
    )


def _install_tee(path: Path) -> None:
    if isinstance(sys.stdout, _Tee):
        return
    fh = _open_utf8_log(path)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _td3_settings() -> TD3Settings:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"TD3 device            {device}  torch={torch.__version__}")
    return TD3Settings(device=device, log_every=250)


def _cfg() -> SimConfig:
    return SimConfig(
        b_sys_hz=BANDWIDTH_PRESETS["8.8mhz"],
        max_bw_share=PRIMARY_MAX_BW_SHARE,
    )


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _point_key(axis: str, x: float) -> str:
    return f"{axis}:{float(x):.10g}"


def _td3_completed(payload: dict, todo, n_runs: int) -> int:
    by_key = {(p["axis"], float(p["x"])): p for p in payload.get("points", [])}
    partial = payload.get("td3_partial") or {}
    done = 0
    for point in todo:
        key = (point.axis, float(point.x_value))
        pkey = _point_key(point.axis, point.x_value)
        row = by_key.get(key) or {}
        stats = (row.get("by_method") or {}).get("td3") or {}
        n = int(stats.get("n", 0))
        if n >= n_runs:
            done += n_runs
        else:
            done += len(partial.get(pkey, []))
    return done


def _summarize_td3_rows(rows: list[dict]) -> dict:
    rates = np.array([r["sum_rate_bit_per_s"] for r in rows], dtype=float)
    feas = np.array([r["feasible"] for r in rows], dtype=bool)
    max_aodt = np.array([r["max_AoDT_s"] for r in rows], dtype=float)
    std = float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0
    return {
        "n": len(rows),
        "mean_sum_rate_bit_per_s": float(np.mean(rates)),
        "std_sum_rate_bit_per_s": std,
        "mean_sum_rate_Mbps": float(np.mean(rates) / 1e6),
        "std_sum_rate_Mbps": std / 1e6,
        "feasible_fraction": float(np.mean(feas)),
        "mean_max_AoDT_s": float(np.mean(max_aodt)),
        "seeds": [int(r["seed"]) for r in rows],
        "per_seed_Mbps": [float(r["sum_rate_Mbps"]) for r in rows],
        "per_seed_feasible": [bool(r["feasible"]) for r in rows],
        "diagnostics": [r.get("diagnostics", {}) for r in rows],
    }


def _paired(by_method: dict, champion: str) -> dict:
    if champion not in by_method:
        return {}
    champ = np.asarray(by_method[champion]["per_seed_Mbps"], dtype=float)
    out = {}
    for name, stats in by_method.items():
        if name == champion:
            continue
        other = np.asarray(stats["per_seed_Mbps"], dtype=float)
        if other.shape != champ.shape:
            continue
        delta = champ - other
        out[name] = {
            "mean_delta_Mbps": float(np.mean(delta)),
            "std_delta_Mbps": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0,
            "win_fraction": float(np.mean(delta > 0.0)),
            "tie_fraction": float(np.mean(np.abs(delta) < 1e-9)),
            "n": int(delta.size),
        }
    return out


def run_n100(td3: TD3Settings, cfg: SimConfig) -> dict:
    bank = load_bank(BANK)
    if SRC_CKPT.exists() and not N100_CKPT.exists():
        N100_CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC_CKPT, N100_CKPT)
        _log(f"copied checkpoint     {SRC_CKPT} -> {N100_CKPT}")
    _log("=== n100  methods=" + ",".join(METHODS) + " ===")
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        cfg,
        METHODS,
        sca_settings=SCASettings(solver=None),
        pso_settings=PSOSettings(),
        td3_settings=td3,
        checkpoint_path=N100_CKPT,
        resume=True,
        bank_path=BANK,
    )
    write_eval(payload, N100_OUT)
    _log(f"wrote {N100_OUT}  ({perf_counter() - t0:.1f}s)")
    try:
        paths = plot_n100_figures(payload, bank, N100_FIG)
        for p in paths:
            _log(f"wrote {p}")
    except ImportError:
        _log("matplotlib missing; skip n100 figures")
    return payload


def _campaign_base(cfg: SimConfig, settings: CampaignSettings) -> dict:
    if not SRC_CAMP.exists():
        raise FileNotFoundError(
            f"missing primary campaign {SRC_CAMP}; cannot reuse baselines"
        )
    src = json.loads(SRC_CAMP.read_text(encoding="utf-8"))
    payload = deepcopy(src)
    payload["td3_opt_in"] = True
    payload["methods"] = list(METHODS)
    payload["n_runs"] = settings.n_runs
    payload["seed_start"] = settings.seed_start
    payload["note"] = (
        "TD3 trained on the same 20 seeds as the primary campaign. "
        "random/k-means/PSO/SCA numbers are reused from "
        f"{SRC_CAMP.as_posix()}. Area is {cfg.area_x_m:g}×{cfg.area_y_m:g} m. "
        f"B_sys={cfg.b_sys_hz:g} Hz, max_bw_share={cfg.max_bw_share}. "
        "TD3Settings, not Table II."
    )
    payload["td3_partial"] = {}
    return payload


def run_campaign_resume(td3: TD3Settings, cfg: SimConfig) -> dict:
    settings = CampaignSettings(
        n_runs=20,
        seed_start=1,
        methods=METHODS,
        sca_settings=SCASettings(solver=None),
        pso_settings=PSOSettings(),
        td3_settings=td3,
    )
    seeds = tuple(range(settings.seed_start, settings.seed_start + settings.n_runs))
    if CAMP_CKPT.exists():
        payload = json.loads(CAMP_CKPT.read_text(encoding="utf-8"))
        n_pts = sum(
            1
            for p in payload.get("points", [])
            if int((p.get("by_method") or {}).get("td3", {}).get("n", 0)) >= 20
        )
        n_part = sum(len(v) for v in (payload.get("td3_partial") or {}).values())
        _log(f"resume campaign       {n_pts} points complete, {n_part} partial TD3 seeds")
    else:
        payload = _campaign_base(cfg, settings)
        _log(f"seeded campaign from  {SRC_CAMP}")

    payload.setdefault("td3_partial", {})
    by_key = {(p["axis"], float(p["x"])): p for p in payload["points"]}
    todo = []
    for axis in AXES:
        todo.extend(iter_axis(axis, cfg))
    _log(f"=== campaign TD3-only  n_runs=20  {len(todo)} points ===")
    t0 = perf_counter()
    n_total = len(todo) * settings.n_runs
    n_session = 0
    for i, point in enumerate(todo, start=1):
        key = (point.axis, float(point.x_value))
        pkey = _point_key(point.axis, point.x_value)
        row = by_key.get(key)
        if row is None:
            raise KeyError(f"primary campaign missing point {point.label}")
        missing = [m for m in BASELINE_METHODS if m not in row.get("by_method", {})]
        if missing:
            raise KeyError(f"{point.label} missing baselines {missing}")
        td3_stats = row.get("by_method", {}).get("td3")
        if td3_stats and int(td3_stats.get("n", 0)) >= settings.n_runs:
            _log(f"  [{i}/{len(todo)}] {point.label}  (checkpoint)")
            continue
        done_rows = list(payload["td3_partial"].get(pkey, []))
        done_seeds = {int(r["seed"]) for r in done_rows}
        _log(
            f"  [{i}/{len(todo)}] {point.label}  "
            f"td3 seeds {len(done_seeds)}/{settings.n_runs}"
        )
        for seed in seeds:
            if seed in done_seeds:
                continue
            sc = generate_scenario(seed, point.cfg)
            n_done = _td3_completed(payload, todo, settings.n_runs)
            n_left = n_total - n_done
            mean_s = (perf_counter() - t0) / n_session if n_session else None
            eta_s = (mean_s * n_left) if mean_s is not None else None
            eta = format_eta(eta_s) if eta_s is not None else "first-train"
            _log(
                f"    train seed={seed}/{settings.n_runs}  "
                f"campaign {n_done}/{n_total} done  left={n_left}  eta={eta}"
            )
            t_seed = perf_counter()
            run = run_method(
                sc,
                "td3",
                seed,
                sca_settings=settings.sca_settings,
                pso_settings=settings.pso_settings,
                td3_settings=settings.td3_settings,
            )
            rec = {
                "seed": int(seed),
                "sum_rate_bit_per_s": float(run.sum_rate_bit_per_s),
                "sum_rate_Mbps": float(run.sum_rate_mbps),
                "feasible": bool(run.feasible),
                "max_AoDT_s": float(np.max(run.true_eval.aodt_s)),
                "aodt_s": [float(x) for x in run.true_eval.aodt_s],
                "diagnostics": {
                    "wall_clock_s": perf_counter() - t_seed,
                    **{
                        k: v
                        for k, v in run.diagnostics.items()
                        if isinstance(v, (int, float, bool, str))
                    },
                },
            }
            done_rows.append(rec)
            done_seeds.add(seed)
            payload["td3_partial"][pkey] = done_rows
            _atomic_write(CAMP_CKPT, payload)
            n_session += 1
            n_done = _td3_completed(payload, todo, settings.n_runs)
            n_left = n_total - n_done
            mean_s = (perf_counter() - t0) / n_session
            _log(
                f"    seed={seed}  export {run.sum_rate_mbps:.4f} Mbps  "
                f"feas={int(run.feasible)}  {perf_counter() - t_seed:.1f}s  "
                f"campaign {n_done}/{n_total}  eta={format_eta(mean_s * n_left)}"
            )
        done_rows.sort(key=lambda r: int(r["seed"]))
        row["by_method"]["td3"] = _summarize_td3_rows(done_rows)
        payload["td3_partial"].pop(pkey, None)
        _atomic_write(CAMP_CKPT, payload)
        write_campaign(payload, CAMP_OUT)
        elapsed = perf_counter() - t0
        _log(f"    checkpointed  elapsed={elapsed / 60.0:.1f} min")
    # drop in-progress map from the published artifact
    published = deepcopy(payload)
    published.pop("td3_partial", None)
    write_campaign(published, CAMP_OUT)
    _log(f"wrote {CAMP_OUT}  ({(perf_counter() - t0) / 60.0:.1f} min)")
    return published


def _fmt_paired(title: str, paired: dict) -> list[str]:
    lines = [title]
    for name, row in paired.items():
        lines.append(
            f"  vs {name:8s}  d={row['mean_delta_Mbps']:+.4f}  "
            f"win={100.0 * row['win_fraction']:.1f}%  "
            f"tie={100.0 * row.get('tie_fraction', 0.0):.1f}%"
        )
    return lines


def _print_n100(payload: dict) -> list[str]:
    lines = ["=== n100 means (100 scenarios, 8.8 MHz, 25% cap) ==="]
    lines.append(
        f"{'method':10s}  {'mean Mbps':>10s}  {'std':>8s}  "
        f"{'feas %':>7s}  {'max AoDT':>8s}"
    )
    for name, stats in payload["by_method"].items():
        lines.append(
            f"{name:10s}  {stats['mean_sum_rate_Mbps']:10.4f}  "
            f"{stats['std_sum_rate_Mbps']:8.4f}  "
            f"{100.0 * stats['feasible_fraction']:6.1f}%  "
            f"{stats['mean_max_AoDT_s']:8.3f}"
        )
    lines.extend(
        _fmt_paired(
            "SCA minus other (mean delta Mbps, win fraction):",
            _paired(payload["by_method"], "sca"),
        )
    )
    lines.extend(
        _fmt_paired(
            "TD3 minus other (mean delta Mbps, win fraction):",
            _paired(payload["by_method"], "td3"),
        )
    )
    for line in lines:
        _log(line)
    return lines


def _print_campaign(payload: dict) -> list[str]:
    lines = ["=== campaign means by axis (n_runs=20) ==="]
    for pt in payload["points"]:
        lines.append(pt["label"])
        for name in METHODS:
            stats = pt["by_method"].get(name)
            if not stats:
                continue
            lines.append(
                f"  {name:8s}  {stats['mean_sum_rate_Mbps']:8.4f} +/- "
                f"{stats['std_sum_rate_Mbps']:.4f} Mbps  "
                f"feas={100.0 * stats['feasible_fraction']:.0f}%"
            )
        if "td3" in pt["by_method"] and "sca" in pt["by_method"]:
            d = (
                pt["by_method"]["td3"]["mean_sum_rate_Mbps"]
                - pt["by_method"]["sca"]["mean_sum_rate_Mbps"]
            )
            lines.append(f"  TD3-SCA mean d={d:+.4f} Mbps")
    for line in lines:
        _log(line)
    return lines


def _write_analysis(n100: dict, camp: dict, extra: list[str]) -> None:
    ANALYSIS.parent.mkdir(parents=True, exist_ok=True)
    chunks = extra[:]
    chunks.append("")
    chunks.extend(_campaign_td3_summary(camp))
    ANALYSIS.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    _log(f"wrote {ANALYSIS}")


def _campaign_td3_summary(payload: dict) -> list[str]:
    lines = ["=== campaign TD3 vs others (mean over points) ==="]
    for other in ("sca", "random", "kmeans", "pso"):
        deltas = []
        td3_wins = 0
        n_pts = 0
        for pt in payload["points"]:
            bm = pt["by_method"]
            if "td3" not in bm or other not in bm:
                continue
            n_pts += 1
            d = bm["td3"]["mean_sum_rate_Mbps"] - bm[other]["mean_sum_rate_Mbps"]
            deltas.append(d)
            if d > 0:
                td3_wins += 1
        if not deltas:
            continue
        lines.append(
            f"  TD3 vs {other:8s}  mean-of-means d={np.mean(deltas):+.4f}  "
            f"points TD3 higher={td3_wins}/{n_pts}"
        )
    return lines


def main() -> int:
    _install_tee(EVAL_LOG)
    _log(f"log file              {EVAL_LOG}")
    cfg = _cfg()
    td3 = _td3_settings()
    n100 = run_n100(td3, cfg)
    n100_lines = _print_n100(n100)
    camp = run_campaign_resume(td3, cfg)
    camp_lines = _print_campaign(camp)
    try:
        import sys

        sys.path.insert(0, str(Path("scripts").resolve()))
        from plot_paper_figures import plot_overview_grid, plot_sum_rate_figure

        CAMP_FIG.mkdir(parents=True, exist_ok=True)
        plot_sum_rate_figure(
            camp, "uavs", "Number of UAVs", "Sum rate vs UAV count", 6, CAMP_FIG
        )
        plot_sum_rate_figure(
            camp, "iots", "Number of IoT devices", "Sum rate vs IoT count", 7, CAMP_FIG
        )
        plot_sum_rate_figure(
            camp, "lambda", "Arrival rate (1/s)", "Sum rate vs arrival rate", 8, CAMP_FIG
        )
        plot_sum_rate_figure(
            camp, "aodt", "AoDT threshold (s)", "Sum rate vs AoDT threshold", 9, CAMP_FIG
        )
        plot_sum_rate_figure(
            camp,
            "cpu",
            "UAV CPU (cycles/s)",
            "Sum rate vs computational capacity",
            10,
            CAMP_FIG,
            x_formatter=lambda x: f"{x / 1e6:.0f} MHz",
        )
        plot_overview_grid(camp, CAMP_FIG)
        _log(f"wrote figures in {CAMP_FIG}")
    except Exception as exc:
        _log(f"figure helper failed: {exc}")
    _write_analysis(n100, camp, n100_lines + [""] + camp_lines)
    _log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
