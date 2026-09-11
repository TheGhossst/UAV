"""Fine B_sys x cap search: 7.1-8.8 MHz, full campaign axes, no TD3.

Why this exists
---------------
At 8.8 MHz / 25% the leftover-dump LP lets random sit within ~0.02 Mbps of
SCA at J=3. That is expected once B_sys is large: extra Hertz piles onto
the best-SE links, so placement barely moves the objective.

This script still sweeps B_sys from 7.1 to 8.8 MHz at 0.1 MHz and every
listed cap, on the full §VII axis grid (J, I, lambda, T_k, CPU), 20 seeds,
methods random / k-means / PSO / SCA.

Null prediction (would be surprising if false): at a *fixed* cap fraction,
J=3 method spread scales ~linearly with B_sys. The lever that actually
separates methods is the cap, not 7.4 vs 8.1 MHz.

"Perfect" here means: 100% feasible at default J=3, SCA uniquely best,
SCA-vs-random Wilcoxon p<0.05, and large method spread. Rank by that, not
by raw Mbps.

Usage
-----
  python scripts/run_bw_fine_search.py --estimate
  python scripts/run_bw_fine_search.py --run
  python scripts/run_bw_fine_search.py --run --resume
  python scripts/run_bw_fine_search.py --report

Outputs (does not overwrite campaign_8.8mhz_cap25_si12k.json):
  results/bw_fine_7p1_8p8/campaign_<mhz>_capXX.json
  results/bw_fine_7p1_8p8/index.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uavdt.config import SimConfig  # noqa: E402
from uavdt.experiments.campaign import (  # noqa: E402
    CampaignSettings,
    run_campaign,
    write_campaign,
)
from uavdt.experiments.grids import AXES  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from paired_winrate import wilcoxon_signed_rank  # noqa: E402

OUT_DIR = ROOT / "results" / "bw_fine_7p1_8p8"
INDEX_PATH = OUT_DIR / "index.json"
METHODS = ("random", "kmeans", "pso", "sca")

# 7.1, 7.2, ..., 8.8 MHz
B_MHZ = tuple(round(7.1 + 0.1 * i, 1) for i in range(18))

# None = Problem (P) (26)-(27) only. Fractions are EXTERNAL leftover-dump caps.
CAPS: tuple[float | None, ...] = (
    None,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.22,
    0.25,
)

# Already-run full-axis 20-seed campaigns. Reused, never overwritten.
REUSE: dict[tuple[float, float | None], Path] = {
    (8_800_000.0, None): ROOT / "results" / "campaign_8.8mhz_n20.json",
    (8_800_000.0, 0.15): ROOT / "results" / "campaign_8.8mhz_cap15_n20.json",
    (8_800_000.0, 0.18): ROOT / "results" / "campaign_8.8mhz_cap18_n20.json",
    (8_800_000.0, 0.20): ROOT / "results" / "campaign_8.8mhz_cap20_n20.json",
    (8_800_000.0, 0.25): ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    (7_000_000.0, None): ROOT / "results" / "campaign_7mhz_n20.json",
    (7_000_000.0, 0.15): ROOT / "results" / "campaign_7mhz_cap15_n20.json",
    (7_000_000.0, 0.25): ROOT / "results" / "campaign_7mhz_cap25_n20.json",
}

# Calibrated: one 7 MHz full-axis 20-seed campaign ~8.5 min on this machine.
SEC_PER_CAMPAIGN = 510.0


def _mhz_tag(mhz: float) -> str:
    return f"{mhz:.1f}".replace(".", "p")


def _cap_tag(share: float | None) -> str:
    if share is None:
        return "nocap"
    return f"cap{int(round(share * 100)):02d}"


def cell_key(mhz: float, share: float | None) -> str:
    return f"{mhz:.1f}:{_cap_tag(share)}"


def cell_path(mhz: float, share: float | None) -> Path:
    return OUT_DIR / f"campaign_{_mhz_tag(mhz)}mhz_{_cap_tag(share)}.json"


def b_hz_of(mhz: float) -> float:
    return float(round(mhz * 1e6))


def parse_caps(raw: str) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for part in raw.split(","):
        p = part.strip().lower()
        if p in {"none", "nocap", "0", "0.0"}:
            out.append(None)
        else:
            out.append(float(p))
    if not out:
        raise ValueError("empty --caps")
    return tuple(out)


def parse_mhz_list(raw: str) -> tuple[float, ...]:
    return tuple(round(float(x.strip()), 1) for x in raw.split(",") if x.strip())


def grid_cells(
    b_mhz: tuple[float, ...],
    caps: tuple[float | None, ...],
) -> list[tuple[float, float | None]]:
    return [(mhz, cap) for mhz in b_mhz for cap in caps]


def _point(payload: dict, axis: str, x: float) -> dict | None:
    for pt in payload.get("points", []):
        if pt["axis"] == axis and abs(float(pt["x"]) - x) < 1e-9:
            return pt
    return None


def _paired(pt: dict, baseline: str) -> dict[str, float | int]:
    sca = pt["by_method"]["sca"]["per_seed_Mbps"]
    base = pt["by_method"][baseline]["per_seed_Mbps"]
    deltas = [a - b for a, b in zip(sca, base)]
    n = len(deltas)
    w = wilcoxon_signed_rank(deltas)
    return {
        "delta_mbps": sum(deltas) / n,
        "wins": int(sum(1 for d in deltas if d > 1e-12)),
        "n": n,
        "p": float(w["p_two_sided"]),
    }


def summarize_payload(payload: dict) -> dict[str, Any]:
    methods = [m for m in METHODS if m in payload.get("methods", METHODS)]
    j3 = _point(payload, "uavs", 3.0)
    j1 = _point(payload, "uavs", 1.0)
    i32 = _point(payload, "iots", 32.0)
    tk08 = _point(payload, "aodt", 0.8)
    if j3 is None:
        raise ValueError("campaign missing uavs J=3")

    means = {m: float(j3["by_method"][m]["mean_sum_rate_Mbps"]) for m in methods}
    feas = {m: float(j3["by_method"][m]["feasible_fraction"]) for m in methods}
    spread = max(means.values()) - min(means.values())
    ranked = sorted(means, key=lambda m: means[m], reverse=True)
    vs_random = _paired(j3, "random") if "random" in methods else None
    vs_kmeans = _paired(j3, "kmeans") if "kmeans" in methods else None
    vs_pso = _paired(j3, "pso") if "pso" in methods else None

    j1_spread = None
    if j1 is not None:
        j1m = [float(j1["by_method"][m]["mean_sum_rate_Mbps"]) for m in methods]
        j1_spread = max(j1m) - min(j1m)

    i32_spread = None
    if i32 is not None:
        i32m = [float(i32["by_method"][m]["mean_sum_rate_Mbps"]) for m in methods]
        i32_spread = max(i32m) - min(i32m)

    sca = means["sca"]
    rnd = means.get("random", float("nan"))
    p_rand = float(vs_random["p"]) if vs_random else 1.0
    d_rand = float(vs_random["delta_mbps"]) if vs_random else 0.0
    feas_ok = feas.get("sca", 0.0) >= 1.0
    sca_best = ranked[0] == "sca"
    sig = bool(p_rand < 0.05 and d_rand > 0)
    # User goal: methods actually differ. Mbps is secondary.
    score = (
        (20.0 if feas_ok else -50.0)
        + (8.0 if sca_best else -8.0)
        + (6.0 if sig else 0.0)
        + 40.0 * spread
        + 8.0 * d_rand
        + (5.0 * j1_spread if j1_spread is not None else 0.0)
    )
    perfect = bool(feas_ok and sca_best and sig and spread >= 0.08)
    return {
        "b_sys_hz": payload.get("b_sys_hz"),
        "max_bw_share": payload.get("max_bw_share"),
        "n_runs": payload.get("n_runs"),
        "n_points": len(payload.get("points", [])),
        "j3_means_mbps": means,
        "j3_feas": feas,
        "j3_spread_mbps": spread,
        "j3_rank": ranked,
        "j3_vs_random": vs_random,
        "j3_vs_kmeans": vs_kmeans,
        "j3_vs_pso": vs_pso,
        "j1_spread_mbps": j1_spread,
        "i32_spread_mbps": i32_spread,
        "tk08_sca_feas": (
            None
            if tk08 is None
            else float(tk08["by_method"]["sca"]["feasible_fraction"])
        ),
        "sca_best": sca_best,
        "sig_vs_random": sig,
        "perfect": perfect,
        "score": score,
        "sca_minus_random_mbps": sca - rnd,
    }


def campaign_complete(path: Path, mhz: float, share: float | None, n_runs: int) -> bool:
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
    if abs(float(payload.get("b_sys_hz", -1)) - b_hz_of(mhz)) > 1.0:
        return False
    if payload.get("max_bw_share") != share:
        return False
    if "td3" in payload.get("methods", []):
        return False
    axes = {pt["axis"] for pt in payload.get("points", [])}
    return set(AXES).issubset(axes)


def load_index() -> dict[str, Any]:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return {"cells": {}, "meta": {}}


def save_index(index: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def print_leaderboard(index: dict[str, Any], n: int = 15) -> None:
    cells = list(index.get("cells", {}).values())
    if not cells:
        print("No completed cells yet.")
        return
    ranked = sorted(cells, key=lambda c: c["summary"]["score"], reverse=True)
    perfect = [c for c in ranked if c["summary"]["perfect"]]
    print()
    print(f"Completed {len(cells)} cells. Perfect (feas + SCA best + p<0.05 + spread>=0.08): {len(perfect)}")
    print(
        f"{'MHz':>5} {'cap':>6} {'SCA':>7} {'rnd':>7} {'spread':>7} "
        f"{'d-rnd':>7} {'p':>8} {'J1sp':>6} {'perf':>4} {'score':>7}"
    )
    for c in ranked[:n]:
        s = c["summary"]
        cap = "none" if c["max_bw_share"] is None else f"{c['max_bw_share']:.0%}"
        p = s["j3_vs_random"]["p"] if s["j3_vs_random"] else float("nan")
        j1 = s["j1_spread_mbps"]
        j1s = f"{j1:6.3f}" if j1 is not None else "   n/a"
        print(
            f"{c['b_mhz']:5.1f} {cap:>6} {s['j3_means_mbps']['sca']:7.3f} "
            f"{s['j3_means_mbps']['random']:7.3f} {s['j3_spread_mbps']:7.3f} "
            f"{s['sca_minus_random_mbps']:+7.3f} {p:8.1e} {j1s} "
            f"{'yes' if s['perfect'] else 'no':>4} {s['score']:7.2f}"
        )
    if perfect:
        best = perfect[0]
        cap = "none" if best["max_bw_share"] is None else f"{best['max_bw_share']:.0%}"
        print(
            f"\nBest perfect: {best['b_mhz']:.1f} MHz cap={cap}  "
            f"spread={best['summary']['j3_spread_mbps']:.3f} Mbps  "
            f"SCA={best['summary']['j3_means_mbps']['sca']:.3f}"
        )


def _record_cell(
    index: dict[str, Any],
    mhz: float,
    share: float | None,
    path: Path,
    payload: dict,
    *,
    reused: bool,
) -> None:
    summary = summarize_payload(payload)
    index["cells"][cell_key(mhz, share)] = {
        "b_mhz": mhz,
        "b_sys_hz": b_hz_of(mhz),
        "max_bw_share": share,
        "path": str(path),
        "reused": reused,
        "summary": summary,
    }
    save_index(index)


def cmd_estimate(
    b_mhz: tuple[float, ...],
    caps: tuple[float | None, ...],
    n_runs: int,
) -> int:
    cells = grid_cells(b_mhz, caps)
    reuse_ok = 0
    need = 0
    for mhz, share in cells:
        reuse = REUSE.get((b_hz_of(mhz), share))
        if reuse is not None and reuse.exists():
            reuse_ok += 1
        elif campaign_complete(cell_path(mhz, share), mhz, share, n_runs):
            reuse_ok += 1
        else:
            need += 1
    eta = need * SEC_PER_CAMPAIGN
    print("7.1-8.8 MHz x cap full-axis search")
    print(f"  B_sys:     {b_mhz[0]:.1f} .. {b_mhz[-1]:.1f} MHz  ({len(b_mhz)} values)")
    print(f"  caps:      {[_cap_tag(c) for c in caps]}")
    print(f"  cells:     {len(cells)}")
    print(f"  reusable:  {reuse_ok}")
    print(f"  to run:    {need}")
    print(f"  n_runs:    {n_runs}  methods={list(METHODS)}  axes={list(AXES)}")
    print(f"  no TD3")
    print(f"  ETA:       {fmt_eta(eta)}  ({need} x {SEC_PER_CAMPAIGN:.0f}s)")
    print(f"  out dir:   {OUT_DIR}")
    return 0


def cmd_run(
    *,
    b_mhz: tuple[float, ...],
    caps: tuple[float | None, ...],
    n_runs: int,
    seed_start: int,
    resume: bool,
) -> int:
    cells = grid_cells(b_mhz, caps)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index() if resume else {"cells": {}, "meta": {}}
    settings = CampaignSettings(
        n_runs=n_runs,
        seed_start=seed_start,
        methods=METHODS,
        sca_settings=SCASettings(solver=None, max_iterations=30, step_size_m=20.0),
    )
    t0 = time.time()
    n_skip = 0
    n_run = 0
    for i, (mhz, share) in enumerate(cells, 1):
        key = cell_key(mhz, share)
        out = cell_path(mhz, share)
        hz = b_hz_of(mhz)

        reuse = REUSE.get((hz, share))
        if reuse is not None and reuse.exists():
            if resume and key in index.get("cells", {}):
                n_skip += 1
                continue
            try:
                payload = json.loads(reuse.read_text(encoding="utf-8"))
                summarize_payload(payload)
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                print(f"[{i}/{len(cells)}] reuse failed {reuse.name}: {exc}", flush=True)
            else:
                print(f"[{i}/{len(cells)}] reuse {reuse.name}", flush=True)
                _record_cell(index, mhz, share, reuse, payload, reused=True)
                n_skip += 1
                continue

        if resume and campaign_complete(out, mhz, share, n_runs):
            if key not in index.get("cells", {}):
                payload = json.loads(out.read_text(encoding="utf-8"))
                _record_cell(index, mhz, share, out, payload, reused=False)
            n_skip += 1
            continue

        cap_s = "none" if share is None else f"{share:.0%}"
        print(
            f"[{i}/{len(cells)}] {mhz:.1f} MHz cap={cap_s} -> {out.name}",
            flush=True,
        )
        cfg = SimConfig(b_sys_hz=hz, max_bw_share=share)
        ckpt = out.with_name(out.stem + ".checkpoint.json")
        payload = run_campaign(
            AXES,
            cfg,
            settings,
            checkpoint_path=ckpt,
        )
        write_campaign(payload, out)
        if ckpt.exists():
            ckpt.unlink()
        _record_cell(index, mhz, share, out, payload, reused=False)
        n_run += 1
        elapsed = time.time() - t0
        left = len(cells) - i
        print(
            f"  done {out.name}  elapsed={fmt_eta(elapsed)}  "
            f"remaining~{fmt_eta(left * SEC_PER_CAMPAIGN)}",
            flush=True,
        )
        print_leaderboard(index, n=8)

    index["meta"] = {
        "b_mhz": list(b_mhz),
        "caps": list(caps),
        "n_runs": n_runs,
        "seed_start": seed_start,
        "methods": list(METHODS),
        "axes": list(AXES),
        "td3": False,
        "elapsed_s": time.time() - t0,
        "n_run": n_run,
        "n_skip": n_skip,
        "sec_per_campaign_assumed": SEC_PER_CAMPAIGN,
        "note": (
            "At fixed cap fraction, spread vs B_sys is expected to be nearly "
            "linear. Cap fraction is the ranking lever. No TD3."
        ),
    }
    save_index(index)
    print(f"\nwrote {INDEX_PATH}")
    print_leaderboard(index, n=20)
    return 0


def cmd_report() -> int:
    if not INDEX_PATH.exists():
        print(f"missing {INDEX_PATH}", file=sys.stderr)
        return 1
    index = load_index()
    print_leaderboard(index, n=40)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument(
        "--caps",
        default="none,0.10,0.12,0.15,0.18,0.20,0.22,0.25",
        help="Comma list. none = no per-link cap.",
    )
    ap.add_argument(
        "--b-mhz",
        default=None,
        help="Comma list of MHz values. Default 7.1,7.2,...,8.8",
    )
    args = ap.parse_args()
    caps = parse_caps(args.caps)
    b_mhz = parse_mhz_list(args.b_mhz) if args.b_mhz else B_MHZ
    if args.estimate:
        return cmd_estimate(b_mhz, caps, args.n_runs)
    if args.report:
        return cmd_report()
    if args.run:
        return cmd_run(
            b_mhz=b_mhz,
            caps=caps,
            n_runs=args.n_runs,
            seed_start=args.seed_start,
            resume=args.resume,
        )
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
