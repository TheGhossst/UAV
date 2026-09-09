"""Small policy-export rerun: I=10 vs I=32, 5 seeds, no 580 retrain.

Official TD3 score is the trained policy (last-N xy, last-step a/b, frozen-q
LP). Best-snapshot is recorded from the same train for A/B only.

Baselines are reused from the primary campaign JSON (same seeds). Does not
change SimConfig. Do not mix this script with PowerShell *>> (UTF-16).
"""

from __future__ import annotations

import json
import sys
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
CAMP_FALLBACK = (
    ROOT / "results" / "campaign_8.8mhz_cap25_td3_SNAPSHOT_INVALID_do_not_cite.json"
)
CKPT = ROOT / "results" / "td3" / "policy_export_small.checkpoint.json"
OUT = ROOT / "results" / "td3" / "policy_export_small.json"
ANALYSIS = ROOT / "results" / "td3" / "policy_export_small.txt"
FIG_DIR = ROOT / "results" / "figures" / "td3_policy_export_small"
LOG = ROOT / "results" / "td3" / "policy_export_small.log"

IOT_COUNTS = (10, 32)
N_UAV = 3
SEEDS = (1, 2, 3, 4, 5)
BASELINES = ("sca", "random", "kmeans", "pso")


class _Tee:
    """Console + UTF-8 file. Do not also redirect with PowerShell *>>."""

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


def _run_key(num_iot: int, seed: int) -> str:
    return f"I={num_iot}:J={N_UAV}:seed={seed}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_iots_point(camp: dict, num_iot: int) -> dict:
    for pt in camp.get("points", []):
        if pt.get("axis") != "iots":
            continue
        if int(pt.get("num_iot", -1)) == int(num_iot):
            return pt
        if abs(float(pt.get("x", -1)) - float(num_iot)) < 1e-9:
            return pt
    raise KeyError(f"no iots axis point with I={num_iot} in campaign JSON")


def _seed_value(stats: dict, seed: int, field: str):
    seeds = [int(s) for s in stats["seeds"]]
    try:
        idx = seeds.index(int(seed))
    except ValueError as exc:
        raise KeyError(f"seed {seed} not in campaign stats") from exc
    return stats[field][idx]


def _baselines_for(camp: dict, num_iot: int, seed: int) -> dict:
    pt = _find_iots_point(camp, num_iot)
    out = {}
    for name in BASELINES:
        stats = (pt.get("by_method") or {}).get(name)
        if not stats:
            continue
        out[name] = {
            "sum_rate_Mbps": float(_seed_value(stats, seed, "per_seed_Mbps")),
            "feasible": bool(_seed_value(stats, seed, "per_seed_feasible")),
        }
    old = (pt.get("by_method") or {}).get("td3")
    if old and "per_seed_Mbps" in old:
        try:
            out["campaign_td3_snapshot"] = {
                "sum_rate_Mbps": float(_seed_value(old, seed, "per_seed_Mbps")),
                "feasible": bool(_seed_value(old, seed, "per_seed_feasible")),
            }
        except KeyError:
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


def _cfg(num_iot: int) -> SimConfig:
    return config_for_counts(
        int(num_iot),
        N_UAV,
        SimConfig(
            b_sys_hz=BANDWIDTH_PRESETS["8.8mhz"],
            max_bw_share=PRIMARY_MAX_BW_SHARE,
        ),
    )


def _summarize_group(rows: list[dict]) -> dict:
    policy = np.array([r["policy_Mbps"] for r in rows], dtype=float)
    snap = np.array([r["snapshot_Mbps"] for r in rows], dtype=float)
    sca = np.array([r["baselines"]["sca"]["sum_rate_Mbps"] for r in rows], dtype=float)
    rnd = np.array([r["baselines"]["random"]["sum_rate_Mbps"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "policy_mean_Mbps": float(np.mean(policy)),
        "policy_std_Mbps": float(np.std(policy, ddof=1)) if len(rows) > 1 else 0.0,
        "snapshot_mean_Mbps": float(np.mean(snap)),
        "snapshot_std_Mbps": float(np.std(snap, ddof=1)) if len(rows) > 1 else 0.0,
        "sca_mean_Mbps": float(np.mean(sca)),
        "random_mean_Mbps": float(np.mean(rnd)),
        "policy_minus_sca": float(np.mean(policy - sca)),
        "snapshot_minus_sca": float(np.mean(snap - sca)),
        "policy_minus_random": float(np.mean(policy - rnd)),
        "policy_feas_frac": float(np.mean([r["policy_feasible"] for r in rows])),
        "snapshot_feas_frac": float(np.mean([r["snapshot_feasible"] for r in rows])),
        "policy_win_vs_sca": float(np.mean(policy > sca)),
        "kmeans_mean_Mbps": float(
            np.mean([r["baselines"]["kmeans"]["sum_rate_Mbps"] for r in rows])
        ),
        "pso_mean_Mbps": float(
            np.mean([r["baselines"]["pso"]["sum_rate_Mbps"] for r in rows])
        ),
    }


def _analysis_lines(payload: dict) -> list[str]:
    lines = [
        "TD3 policy-export small eval -- not the 580-run best-snapshot campaign",
        "Reproduction: 100x100 m, B_sys=8.8 MHz, 25% per-link cap. Not Table II 20 kHz.",
        "Export rule: noise-free rollout, last-10 UAV xy mean, last-step a/b, frozen-q LP.",
        "Snapshot LP from the same train is recorded only for A/B.",
        "580-run JSON had no per-step trajectories and no actor weights; cannot re-export.",
        f"n={payload['n_seeds']} seeds {payload['seeds']}  I={payload['iot_counts']}  J={N_UAV}",
        f"device={payload.get('device')}  steps={payload.get('total_steps')}",
        "",
        "Reading frame: default I=10 J=3, TD3 < SCA; gap should shrink as I grows.",
        "Story checks below use POLICY export only.",
        "",
    ]
    by_i = payload["by_iot"]
    i10 = by_i["10"]
    i32 = by_i["32"]
    lines.append("=== means (Mbps) ===")
    lines.append(
        f"{'I':>4s}  {'SCA':>8s}  {'rand':>8s}  {'kmeans':>8s}  {'PSO':>8s}  "
        f"{'TD3-snap':>8s}  {'TD3-pol':>8s}  {'pol-SCA':>8s}  {'snap-SCA':>8s}"
    )
    for key in ("10", "32"):
        s = by_i[key]
        lines.append(
            f"{int(key):4d}  {s['sca_mean_Mbps']:8.4f}  {s['random_mean_Mbps']:8.4f}  "
            f"{s['kmeans_mean_Mbps']:8.4f}  {s['pso_mean_Mbps']:8.4f}  "
            f"{s['snapshot_mean_Mbps']:8.4f}  {s['policy_mean_Mbps']:8.4f}  "
            f"{s['policy_minus_sca']:+8.4f}  {s['snapshot_minus_sca']:+8.4f}"
        )
    lines.append("")
    d10 = i10["policy_minus_sca"]
    d32 = i32["policy_minus_sca"]
    lines.append(f"policy TD3-SCA at I=10: {d10:+.4f} Mbps")
    lines.append(f"policy TD3-SCA at I=32: {d32:+.4f} Mbps")
    lines.append(f"gap shift I=32 minus I=10: {d32 - d10:+.4f} Mbps")
    lines.append(
        "story check (default): TD3 below SCA: "
        + ("YES" if d10 < 0.0 else "NO -- measured policy TD3 >= SCA")
    )
    lines.append(
        "story check (devices grow): TD3-SCA improves as I grows: "
        + ("YES" if d32 > d10 else "NO")
    )
    lines.append(
        "story check (large I): TD3 > SCA at I=32: "
        + ("YES" if d32 > 0.0 else "NO -- policy TD3 still at or below SCA")
    )
    lines.append(
        "note: snapshot-vs-SCA is bookkeeping from the same trains, not the story score."
    )
    lines.append("")
    lines.append("=== per seed ===")
    for row in payload["runs"]:
        b = row["baselines"]
        lines.append(
            f"  I={row['num_iot']:2d} seed={row['seed']}  "
            f"pol={row['policy_Mbps']:.4f} feas={str(row['policy_feasible']):5s}  "
            f"snap={row['snapshot_Mbps']:.4f}  "
            f"SCA={b['sca']['sum_rate_Mbps']:.4f}  "
            f"rand={b['random']['sum_rate_Mbps']:.4f}  "
            f"inner_lastN={row['policy_inner_mean_Mbps']:.4f}"
        )
    return lines


def _plot(payload: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_td3("matplotlib not installed; skip figures")
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    by_i = payload["by_iot"]
    cats = ["I=10", "I=32"]
    names = [
        ("SCA", [by_i["10"]["sca_mean_Mbps"], by_i["32"]["sca_mean_Mbps"]]),
        ("Random", [by_i["10"]["random_mean_Mbps"], by_i["32"]["random_mean_Mbps"]]),
        ("K-means", [by_i["10"]["kmeans_mean_Mbps"], by_i["32"]["kmeans_mean_Mbps"]]),
        ("PSO", [by_i["10"]["pso_mean_Mbps"], by_i["32"]["pso_mean_Mbps"]]),
        (
            "TD3 snapshot",
            [by_i["10"]["snapshot_mean_Mbps"], by_i["32"]["snapshot_mean_Mbps"]],
        ),
        (
            "TD3 policy",
            [by_i["10"]["policy_mean_Mbps"], by_i["32"]["policy_mean_Mbps"]],
        ),
    ]
    x = np.arange(len(cats))
    width = 0.13
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, (label, vals) in enumerate(names):
        ax.bar(x + (i - 2.5) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Mean sum rate (Mbps)")
    ax.set_title("Policy-export small eval (n=5, 8.8 MHz, 25% cap)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "means_grouped.png", dpi=140)
    fig.savefig(FIG_DIR / "means_grouped.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        [10, 32],
        [by_i["10"]["snapshot_minus_sca"], by_i["32"]["snapshot_minus_sca"]],
        marker="s",
        label="TD3 snapshot - SCA",
    )
    ax.plot(
        [10, 32],
        [by_i["10"]["policy_minus_sca"], by_i["32"]["policy_minus_sca"]],
        marker="o",
        label="TD3 policy - SCA",
    )
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("IoT count I (J=3)")
    ax.set_ylabel("TD3 - SCA (Mbps)")
    ax.set_title("Gap vs SCA: policy export vs best-snapshot")
    ax.legend(frameon=False)
    ax.set_xticks([10, 32])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "td3_minus_sca.png", dpi=140)
    fig.savefig(FIG_DIR / "td3_minus_sca.pdf")
    plt.close(fig)
    log_td3(f"wrote figures in {FIG_DIR}")


def main() -> int:
    _install_tee(LOG)
    camp_path = CAMP if CAMP.exists() else CAMP_FALLBACK
    if not camp_path.exists():
        raise SystemExit(f"missing campaign JSON: {CAMP} and {CAMP_FALLBACK}")
    camp = _load_json(camp_path)
    settings = _td3_settings()
    ckpt = {"runs": {}}
    if CKPT.exists():
        ckpt = _load_json(CKPT)
        ckpt.setdefault("runs", {})
        log_td3(f"resume checkpoint     {CKPT}  n={len(ckpt['runs'])}")

    todo = [(i, s) for i in IOT_COUNTS for s in SEEDS]
    t_all = perf_counter()
    for idx, (num_iot, seed) in enumerate(todo, start=1):
        key = _run_key(num_iot, seed)
        if key in ckpt["runs"] and "policy_Mbps" in ckpt["runs"][key]:
            log_td3(f"[{idx}/{len(todo)}] skip {key}")
            continue
        cfg = _cfg(num_iot)
        sc = generate_scenario(seed, cfg)
        log_td3(f"[{idx}/{len(todo)}] train {key}")
        t0 = perf_counter()
        run = run_method(sc, "td3", seed, td3_settings=settings)
        d = run.diagnostics
        row = {
            "key": key,
            "num_iot": int(num_iot),
            "num_uav": N_UAV,
            "seed": int(seed),
            "policy_Mbps": float(d["policy_export_sum_rate_Mbps"]),
            "policy_feasible": bool(d["policy_export_feasible"]),
            "snapshot_Mbps": float(d["snapshot_export_sum_rate_Mbps"]),
            "snapshot_feasible": bool(d["snapshot_export_feasible"]),
            "official_Mbps": float(run.sum_rate_mbps),
            "export_rule": d.get("export_rule"),
            "policy_inner_mean_Mbps": float(d.get("policy_inner_mean_Mbps", 0.0)),
            "policy_inner_last_Mbps": float(d.get("policy_inner_last_Mbps", 0.0)),
            "best_inner_sum_rate_Mbps": float(d.get("best_inner_sum_rate_Mbps", 0.0)),
            "wall_clock_s": float(d.get("wall_clock_s", perf_counter() - t0)),
            "n_updates": int(d.get("n_updates", 0)),
            "baselines": _baselines_for(camp, num_iot, seed),
        }
        ckpt["runs"][key] = row
        _atomic_write(CKPT, ckpt)
        remain = (len(todo) - idx) * (perf_counter() - t_all) / idx
        log_td3(
            f"  saved {key}  pol={row['policy_Mbps']:.4f}  "
            f"snap={row['snapshot_Mbps']:.4f}  "
            f"SCA={row['baselines']['sca']['sum_rate_Mbps']:.4f}  "
            f"{row['wall_clock_s']:.1f}s  eta={format_eta(remain)}"
        )

    runs = [ckpt["runs"][_run_key(i, s)] for i in IOT_COUNTS for s in SEEDS]
    by_iot = {
        str(i): _summarize_group([r for r in runs if r["num_iot"] == i])
        for i in IOT_COUNTS
    }
    import torch

    payload = {
        "paper": "Khalaf et al. IEEE TNSM 2026 Algorithm 2 fill-in",
        "note": (
            "Small policy-export eval. Official TD3 is the trained policy, "
            "not best-snapshot. 580-run artifacts cannot be re-exported."
        ),
        "b_sys_hz": BANDWIDTH_PRESETS["8.8mhz"],
        "max_bw_share": PRIMARY_MAX_BW_SHARE,
        "area_m": [100.0, 100.0],
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "iot_counts": list(IOT_COUNTS),
        "num_uav": N_UAV,
        "total_steps": int(settings.total_steps),
        "export_mode": settings.export_mode,
        "export_avg_steps": int(settings.export_avg_steps),
        "device": settings.device,
        "torch": str(torch.__version__),
        "campaign_baselines": str(camp_path.as_posix()),
        "by_iot": by_iot,
        "runs": runs,
    }
    _atomic_write(OUT, payload)
    lines = _analysis_lines(payload)
    ANALYSIS.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        log_td3(line)
    log_td3(f"wrote {OUT}")
    log_td3(f"wrote {ANALYSIS}")
    _plot(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
