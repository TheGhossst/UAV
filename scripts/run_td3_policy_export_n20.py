"""Policy-export eval: 20 seeds at I=10, I=32, and Tk=0.8 s.

SCA-align trainer: k-means init, Alg. 2 Δ×10, leftover-dump inner B.
Does not overwrite BEFORE or AFTER hover-trainer files.

Official TD3 score is the trained policy (last-N xy, last-step a/b, frozen-q
LP). Same-train best-snapshot is A/B only. Tk=0.8 is first in the queue.

Baselines from results/campaign_8.8mhz_cap25_si12k.json. Does not change
SimConfig. Do not mix with PowerShell *>>.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from uavdt.config import BANDWIDTH_PRESETS, PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.experiments.grids import config_for_counts
from uavdt.experiments.methods import run_method
from uavdt.scenario import generate_scenario
from uavdt.td3.settings import TD3Settings
from uavdt.td3.solve import format_eta, log_td3

ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"
OLD_TD3 = (
    ROOT
    / "results"
    / "campaign_8.8mhz_cap25_td3_SNAPSHOT_INVALID_do_not_cite.json"
)
CKPT = ROOT / "results" / "td3" / "policy_export_n20_sca_align.checkpoint.json"
OUT = ROOT / "results" / "td3" / "policy_export_n20_sca_align.json"
ANALYSIS = ROOT / "results" / "td3" / "policy_export_n20_sca_align.txt"
FIG_DIR = ROOT / "results" / "figures" / "td3_policy_export_n20_sca_align"
LOG = ROOT / "results" / "td3" / "policy_export_n20_sca_align.log"

N_UAV = 3
SEEDS = tuple(range(1, 21))
BASELINES = ("sca", "random", "kmeans", "pso")
# Tk=0.8 first: the distinctive feasibility claim.
POINTS = (
    {
        "id": "tk08",
        "num_iot": 10,
        "aodt_threshold_s": 0.8,
        "axis": "aodt",
        "x": 0.8,
        "label": "Tk=0.8s I=10 J=3",
    },
    {
        "id": "i10",
        "num_iot": 10,
        "aodt_threshold_s": 2.8,
        "axis": "iots",
        "x": 10.0,
        "label": "I=10 J=3 Tk=2.8s",
    },
    {
        "id": "i32",
        "num_iot": 32,
        "aodt_threshold_s": 2.8,
        "axis": "iots",
        "x": 32.0,
        "label": "I=32 J=3 Tk=2.8s",
    },
)


class _Tee:
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
            n = self.stream.write(text.encode("ascii", "replace").decode("ascii"))
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


def _install_tee(path: Path) -> None:
    if isinstance(sys.stdout, _Tee):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8", newline="\n", errors="replace", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run_key(point_id: str, seed: int) -> str:
    return f"{point_id}:seed={seed}"


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("invalid") or data.get("do_not_cite"):
        raise SystemExit(f"refusing {path}: {data.get('reason')} -> {data.get('moved_to')}")
    return data


def _find_point(camp: dict, axis: str, x: float) -> dict:
    for pt in camp.get("points", []):
        if pt.get("axis") != axis:
            continue
        if abs(float(pt.get("x", -1)) - float(x)) < 1e-9:
            return pt
    raise KeyError(f"no campaign point axis={axis} x={x}")


def _seed_value(stats: dict, seed: int, field: str):
    seeds = [int(s) for s in stats["seeds"]]
    idx = seeds.index(int(seed))
    return stats[field][idx]


def _baselines_for(camp: dict, old_td3: dict | None, point: dict, seed: int) -> dict:
    pt = _find_point(camp, point["axis"], point["x"])
    out = {}
    for name in BASELINES:
        stats = (pt.get("by_method") or {}).get(name)
        if not stats:
            continue
        out[name] = {
            "sum_rate_Mbps": float(_seed_value(stats, seed, "per_seed_Mbps")),
            "feasible": bool(_seed_value(stats, seed, "per_seed_feasible")),
        }
    if old_td3:
        try:
            old_pt = _find_point(old_td3, point["axis"], point["x"])
            stats = (old_pt.get("by_method") or {}).get("td3")
            if stats:
                out["old_campaign_td3_snapshot"] = {
                    "sum_rate_Mbps": float(_seed_value(stats, seed, "per_seed_Mbps")),
                    "feasible": bool(_seed_value(stats, seed, "per_seed_feasible")),
                }
        except (KeyError, ValueError):
            pass
    return out


def _td3_settings() -> TD3Settings:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_td3(f"TD3 device            {device}  torch={torch.__version__}")
    return TD3Settings(
        device=device,
        log_every=250,
        export_mode="policy",
        export_avg_steps=10,
    )


def _cfg(point: dict) -> SimConfig:
    base = config_for_counts(
        int(point["num_iot"]),
        N_UAV,
        SimConfig(
            b_sys_hz=BANDWIDTH_PRESETS["8.8mhz"],
            max_bw_share=PRIMARY_MAX_BW_SHARE,
        ),
    )
    return replace(base, aodt_threshold_s=float(point["aodt_threshold_s"]))


def _summarize_group(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    policy = np.array([r["policy_Mbps"] for r in rows], dtype=float)
    snap = np.array([r["snapshot_Mbps"] for r in rows], dtype=float)
    sca = np.array([r["baselines"]["sca"]["sum_rate_Mbps"] for r in rows], dtype=float)
    rnd = np.array([r["baselines"]["random"]["sum_rate_Mbps"] for r in rows], dtype=float)
    std = lambda a: float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return {
        "n": len(rows),
        "policy_mean_Mbps": float(np.mean(policy)),
        "policy_std_Mbps": std(policy),
        "snapshot_mean_Mbps": float(np.mean(snap)),
        "snapshot_std_Mbps": std(snap),
        "sca_mean_Mbps": float(np.mean(sca)),
        "random_mean_Mbps": float(np.mean(rnd)),
        "kmeans_mean_Mbps": float(
            np.mean([r["baselines"]["kmeans"]["sum_rate_Mbps"] for r in rows])
        ),
        "pso_mean_Mbps": float(
            np.mean([r["baselines"]["pso"]["sum_rate_Mbps"] for r in rows])
        ),
        "policy_minus_sca": float(np.mean(policy - sca)),
        "snapshot_minus_sca": float(np.mean(snap - sca)),
        "policy_minus_random": float(np.mean(policy - rnd)),
        "policy_feas_frac": float(np.mean([r["policy_feasible"] for r in rows])),
        "snapshot_feas_frac": float(np.mean([r["snapshot_feasible"] for r in rows])),
        "sca_feas_frac": float(np.mean([r["baselines"]["sca"]["feasible"] for r in rows])),
        "random_feas_frac": float(
            np.mean([r["baselines"]["random"]["feasible"] for r in rows])
        ),
        "policy_win_vs_sca": float(np.mean(policy > sca)),
        "mean_inner_Mbps": float(np.mean([r["policy_inner_mean_Mbps"] for r in rows])),
        "mean_move_norm": float(
            np.mean([r.get("policy_mean_move_action_norm") or 0.0 for r in rows])
        ),
        "mean_nearest_agree": float(
            np.mean([r.get("policy_nearest_agree") or 0.0 for r in rows])
        ),
        "mean_min_pairwise_m": float(
            np.mean(
                [
                    r["policy_min_pairwise_m"]
                    for r in rows
                    if r.get("policy_min_pairwise_m") is not None
                ]
            )
        )
        if any(r.get("policy_min_pairwise_m") is not None for r in rows)
        else None,
    }


def _analysis_lines(payload: dict) -> list[str]:
    by = payload["by_point"]
    lines = [
        "TD3 policy-export n=20 -- I=10, I=32, Tk=0.8s",
        "Reproduction: 100x100 m, B_sys=8.8 MHz, 25% per-link cap. Not Table II 20 kHz.",
        "Export: noise-free rollout, last-10 UAV xy mean, last-step a/b, frozen-q LP.",
        "Actor last layer: Fujimoto uniform +/- 3e-3 (not default Linear init).",
        f"n={payload['n_seeds']} seeds {payload['seeds'][0]}-{payload['seeds'][-1]}",
        f"device={payload.get('device')}  steps={payload.get('total_steps')}",
        "",
    ]
    if "tk08" in by and by["tk08"].get("n"):
        s = by["tk08"]
        lines += [
            "=== Tk=0.8 s feasibility (I=10 J=3) ===",
            (
                f"  policy feas={100.0 * s['policy_feas_frac']:.0f}%  "
                f"same-train snapshot feas={100.0 * s['snapshot_feas_frac']:.0f}%  "
                f"SCA feas={100.0 * s['sca_feas_frac']:.0f}%  "
                f"random feas={100.0 * s['random_feas_frac']:.0f}%  n={s['n']}"
            ),
            (
                f"  policy mean={s['policy_mean_Mbps']:.4f} Mbps  "
                f"snapshot={s['snapshot_mean_Mbps']:.4f}  "
                f"SCA={s['sca_mean_Mbps']:.4f}"
            ),
            "  580-run 'TD3 40% feasible' was best-snapshot bookkeeping; this row is policy.",
            "",
        ]
    if "i10" in by and "i32" in by and by["i10"].get("n") and by["i32"].get("n"):
        i10, i32 = by["i10"], by["i32"]
        d10, d32 = i10["policy_minus_sca"], i32["policy_minus_sca"]
        lines += [
            "=== means (Mbps), policy export ===",
            (
                f"{'pt':8s}  {'SCA':>8s}  {'rand':>8s}  {'TD3-snap':>8s}  "
                f"{'TD3-pol':>8s}  {'pol-SCA':>8s}  {'pol feas':>8s}  |move|"
            ),
        ]
        for pid in ("i10", "i32"):
            s = by[pid]
            lines.append(
                f"{pid:8s}  {s['sca_mean_Mbps']:8.4f}  {s['random_mean_Mbps']:8.4f}  "
                f"{s['snapshot_mean_Mbps']:8.4f}  {s['policy_mean_Mbps']:8.4f}  "
                f"{s['policy_minus_sca']:+8.4f}  {100.0 * s['policy_feas_frac']:7.0f}%  "
                f"{s['mean_move_norm']:6.3f}"
            )
        lines += [
            "",
            f"policy TD3-SCA at I=10: {d10:+.4f} Mbps  (n={i10['n']})",
            f"policy TD3-SCA at I=32: {d32:+.4f} Mbps  (n={i32['n']})",
            f"gap shift I=32 minus I=10: {d32 - d10:+.4f} Mbps",
            "story check (default): TD3 below SCA: "
            + ("YES" if d10 < 0.0 else "NO -- policy TD3 >= SCA"),
            "story check (devices grow): TD3-SCA improves as I grows: "
            + ("YES" if d32 > d10 else "NO"),
            "story check (large I): TD3 > SCA at I=32: "
            + ("YES" if d32 > 0.0 else "NO -- policy TD3 still at or below SCA"),
            "",
        ]
    lines.append("=== per seed ===")
    for row in payload.get("runs", []):
        b = row["baselines"]
        lines.append(
            f"  {row['point_id']:4s} seed={row['seed']:2d}  "
            f"pol={row['policy_Mbps']:.4f} feas={str(row['policy_feasible']):5s}  "
            f"snap={row['snapshot_Mbps']:.4f} snap_feas={str(row['snapshot_feasible']):5s}  "
            f"SCA={b['sca']['sum_rate_Mbps']:.4f} sca_feas={str(b['sca']['feasible']):5s}  "
            f"inner={row['policy_inner_mean_Mbps']:.3f}  "
            f"|move|={row.get('policy_mean_move_action_norm')}"
        )
    return lines


def _plot(payload: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_td3("matplotlib not installed; skip figures")
        return
    by = payload["by_point"]
    if not (by.get("i10", {}).get("n") and by.get("i32", {}).get("n")):
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    i10, i32 = by["i10"], by["i32"]
    cats = ["I=10", "I=32"]
    series = [
        ("SCA", [i10["sca_mean_Mbps"], i32["sca_mean_Mbps"]]),
        ("Random", [i10["random_mean_Mbps"], i32["random_mean_Mbps"]]),
        ("TD3 snapshot", [i10["snapshot_mean_Mbps"], i32["snapshot_mean_Mbps"]]),
        ("TD3 policy", [i10["policy_mean_Mbps"], i32["policy_mean_Mbps"]]),
    ]
    x = np.arange(len(cats))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (label, vals) in enumerate(series):
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Mean sum rate (Mbps)")
    ax.set_title("Policy-export n=20 (8.8 MHz, 25% cap)")
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "means_grouped.png", dpi=140)
    fig.savefig(FIG_DIR / "means_grouped.pdf")
    plt.close(fig)

    if by.get("tk08", {}).get("n"):
        s = by["tk08"]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        labels = ["SCA", "Random", "TD3 snapshot", "TD3 policy"]
        vals = [
            100.0 * s["sca_feas_frac"],
            100.0 * s["random_feas_frac"],
            100.0 * s["snapshot_feas_frac"],
            100.0 * s["policy_feas_frac"],
        ]
        ax.bar(labels, vals)
        ax.set_ylabel("Feasible fraction (%)")
        ax.set_title("Tk=0.8 s feasibility, n=20")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "tk08_feasibility.png", dpi=140)
        fig.savefig(FIG_DIR / "tk08_feasibility.pdf")
        plt.close(fig)
    log_td3(f"wrote figures in {FIG_DIR}")


def _write_payload(ckpt: dict, settings: TD3Settings, camp_path: Path) -> dict:
    import torch

    by_point = {}
    runs = []
    for point in POINTS:
        rows = []
        for seed in SEEDS:
            key = _run_key(point["id"], seed)
            if key in ckpt["runs"]:
                rows.append(ckpt["runs"][key])
        by_point[point["id"]] = _summarize_group(rows)
        runs.extend(rows)
    payload = {
        "paper": "Khalaf et al. IEEE TNSM 2026 Algorithm 2 fill-in",
        "note": (
            "n=20 policy-export at I=10, I=32, Tk=0.8s. "
            "Official TD3 is the trained policy. 580-run snapshot files are quarantined."
        ),
        "b_sys_hz": BANDWIDTH_PRESETS["8.8mhz"],
        "max_bw_share": PRIMARY_MAX_BW_SHARE,
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "points": list(POINTS),
        "num_uav": N_UAV,
        "total_steps": int(settings.total_steps),
        "export_mode": settings.export_mode,
        "export_actor": settings.export_actor,
        "move_mode": settings.move_mode,
        "uav_init": settings.uav_init,
        "inner_bandwidth": settings.inner_bandwidth,
        "assoc_mode": settings.assoc_mode,
        "process_mode": settings.process_mode,
        "discount": settings.discount,
        "critic_use_obs": settings.critic_use_obs,
        "critic_move_only": settings.critic_move_only,
        "actor_last_layer_init": "fujimoto_uniform_3e-3",
        "device": settings.device,
        "torch": str(torch.__version__),
        "campaign_baselines": str(camp_path.as_posix()),
        "by_point": by_point,
        "runs": runs,
    }
    _atomic_write(OUT, payload)
    lines = _analysis_lines(payload)
    ANALYSIS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    _install_tee(LOG)
    if not CAMP.exists():
        raise SystemExit(f"missing {CAMP}")
    camp = _load_json(CAMP)
    old_td3 = _load_json(OLD_TD3) if OLD_TD3.exists() else None
    settings = _td3_settings()
    ckpt = {"runs": {}}
    if CKPT.exists():
        ckpt = json.loads(CKPT.read_text(encoding="utf-8"))
        ckpt.setdefault("runs", {})
        log_td3(f"resume checkpoint     {CKPT}  n={len(ckpt['runs'])}")

    todo = [(pt, seed) for pt in POINTS for seed in SEEDS]
    t_all = perf_counter()
    for idx, (point, seed) in enumerate(todo, start=1):
        key = _run_key(point["id"], seed)
        if key in ckpt["runs"] and "policy_Mbps" in ckpt["runs"][key]:
            log_td3(f"[{idx}/{len(todo)}] skip {key}")
            continue
        cfg = _cfg(point)
        sc = generate_scenario(seed, cfg)
        log_td3(f"[{idx}/{len(todo)}] train {key}  {point['label']}")
        t0 = perf_counter()
        run = run_method(sc, "td3", seed, td3_settings=settings)
        d = run.diagnostics
        row = {
            "key": key,
            "point_id": point["id"],
            "label": point["label"],
            "num_iot": int(point["num_iot"]),
            "num_uav": N_UAV,
            "aodt_threshold_s": float(point["aodt_threshold_s"]),
            "seed": int(seed),
            "policy_Mbps": float(d["policy_export_sum_rate_Mbps"]),
            "policy_feasible": bool(d["policy_export_feasible"]),
            "snapshot_Mbps": float(d["snapshot_export_sum_rate_Mbps"]),
            "snapshot_feasible": bool(d["snapshot_export_feasible"]),
            "official_Mbps": float(run.sum_rate_mbps),
            "export_rule": d.get("export_rule"),
            "policy_inner_mean_Mbps": float(d.get("policy_inner_mean_Mbps", 0.0)),
            "policy_mean_move_action_norm": d.get("policy_mean_move_action_norm"),
            "policy_nearest_agree": d.get("policy_nearest_agree"),
            "policy_min_pairwise_m": d.get("policy_min_pairwise_m"),
            "policy_uav_loads": d.get("policy_uav_loads"),
            "wall_clock_s": float(d.get("wall_clock_s", perf_counter() - t0)),
            "baselines": _baselines_for(camp, old_td3, point, seed),
        }
        ckpt["runs"][key] = row
        _atomic_write(CKPT, ckpt)
        remain = (len(todo) - idx) * (perf_counter() - t_all) / max(idx, 1)
        log_td3(
            f"  saved {key}  pol={row['policy_Mbps']:.4f} "
            f"feas={row['policy_feasible']}  "
            f"snap_feas={row['snapshot_feasible']}  "
            f"SCA_feas={row['baselines']['sca']['feasible']}  "
            f"|move|={row['policy_mean_move_action_norm']}  "
            f"{row['wall_clock_s']:.1f}s  eta={format_eta(remain)}"
        )
        if seed == SEEDS[-1] or idx == len(todo):
            payload = _write_payload(ckpt, settings, CAMP)
            for line in _analysis_lines(payload):
                log_td3(line)

    payload = _write_payload(ckpt, settings, CAMP)
    for line in _analysis_lines(payload):
        log_td3(line)
    log_td3(f"wrote {OUT}")
    log_td3(f"wrote {ANALYSIS}")
    _plot(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
