"""Copy post-fix random (and PSO) into TD3 artifacts for pairing.

Algorithm 2 TD3 uses a k-means origin, so TD3 Mbps do not change with the
place_random salt. The TD3 campaign JSON still carries the pre-fix random
column it copied at train time. This script replaces random and pso from the
current 25% campaign / n100 banks. Does not overwrite TD3 scores or protected
headline JSON.

Usage:
  python scripts/campaigns/repair_td3_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402

from uavdt.experiments.n100 import _paired_stats  # noqa: E402

PROTECTED = {
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json",
    ROOT / "results" / "n100" / "eval.json",
    ROOT / "results" / "n100_500m_cap25" / "eval.json",
}
TAG = (
    "random and pso columns copied from post-fix 25% artifacts 2026-09-20; "
    "TD3 scores unchanged (k-means origin)."
)
COPY_METHODS = ("random", "pso")


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


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pair(a: list[float], b: list[float]) -> dict:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    d = x - y
    w = wilcoxon_signed_rank(d)
    return {
        "mean_delta": float(np.mean(d)),
        "std_delta": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "n_higher": int(np.sum(d > 1e-12)),
        "n": int(d.size),
        "p_two_sided": float(w["p_two_sided"]),
    }


def _pt_key(pt: dict) -> tuple:
    return (
        str(pt.get("axis")),
        round(float(pt.get("x", 0.0)), 9),
        int(pt.get("num_iot") or 0),
        int(pt.get("num_uav") or 0),
    )


def repair_campaign(*, dst: Path, src: Path) -> None:
    _refuse(dst)
    if not dst.exists() or not src.exists():
        _log(f"skip missing campaign {dst.name} or {src.name}")
        return
    td3 = _load(dst)
    camp = _load(src)
    src_pts = {_pt_key(pt): pt for pt in camp.get("points") or []}
    n_copy = 0
    for pt in td3.get("points") or []:
        match = src_pts.get(_pt_key(pt))
        if match is None:
            continue
        bm = dict(pt.get("by_method") or {})
        src_bm = match.get("by_method") or {}
        for method in COPY_METHODS:
            if method in src_bm:
                bm[method] = src_bm[method]
                n_copy += 1
        pt["by_method"] = bm
    note = str(td3.get("note") or "")
    if TAG not in note:
        td3["note"] = (note + " " + TAG).strip()
    _dump(dst, td3)
    _log(f"wrote {dst}  copied {n_copy} method-blocks from {src.name}")
    for pt in td3.get("points") or []:
        if int(pt.get("num_iot") or 0) != 10 or int(pt.get("num_uav") or 0) != 3:
            continue
        if str(pt.get("axis")) != "uavs":
            continue
        bm = pt.get("by_method") or {}
        if "td3" not in bm or "random" not in bm:
            continue
        pr = _pair(bm["td3"]["per_seed_Mbps"], bm["random"]["per_seed_Mbps"])
        _log(
            f"  default J=3  TD3={bm['td3']['mean_sum_rate_Mbps']:.4f}  "
            f"random={bm['random']['mean_sum_rate_Mbps']:.4f}  "
            f"TD3-rand={pr['mean_delta']:+.4f} +/- {pr['std_delta']:.3f}  "
            f"{pr['n_higher']}/{pr['n']}  p={pr['p_two_sided']:.4g}"
        )
        break


def repair_bank(*, dst: Path, src: Path) -> None:
    _refuse(dst)
    if not dst.exists() or not src.exists():
        _log(f"skip missing bank {dst} or {src}")
        return
    td3 = _load(dst)
    bank = _load(src)
    src_bm = bank.get("by_method") or {}
    dst_bm = dict(td3.get("by_method") or {})
    for method in COPY_METHODS:
        if method not in src_bm:
            continue
        dst_bm[method] = src_bm[method]
    td3["by_method"] = dst_bm
    new_rows = {
        (r["seed"], r["method"]): r
        for r in bank.get("runs") or []
        if r.get("method") in COPY_METHODS
    }
    runs = []
    for r in td3.get("runs") or []:
        key = (r.get("seed"), r.get("method"))
        runs.append(new_rows.get(key, r))
    present = {(r.get("seed"), r.get("method")) for r in runs}
    for key, row in new_rows.items():
        if key not in present:
            runs.append(row)
    td3["runs"] = runs
    if "sca" in dst_bm:
        td3["sca_minus_baseline_Mbps"] = _paired_stats(dst_bm)
    note = str(td3.get("note") or "")
    if TAG not in note:
        td3["note"] = (note + " " + TAG).strip()
    _dump(dst, td3)
    _log(f"wrote {dst}")
    bm = td3["by_method"]
    if "td3" in bm and "random" in bm:
        pr = _pair(bm["td3"]["per_seed_Mbps"], bm["random"]["per_seed_Mbps"])
        _log(
            f"  TD3={bm['td3']['mean_sum_rate_Mbps']:.4f}  "
            f"random={bm['random']['mean_sum_rate_Mbps']:.4f}  "
            f"TD3-rand={pr['mean_delta']:+.4f} +/- {pr['std_delta']:.3f}  "
            f"{pr['n_higher']}/{pr['n']}  p={pr['p_two_sided']:.4g}"
        )
        if "kmeans" in bm:
            pk = _pair(bm["td3"]["per_seed_Mbps"], bm["kmeans"]["per_seed_Mbps"])
            _log(
                f"  TD3-kmeans={pk['mean_delta']:+.4f}  "
                f"{pk['n_higher']}/{pk['n']}  p={pk['p_two_sided']:.4g}"
            )
        if "pso" in bm:
            pp = _pair(bm["td3"]["per_seed_Mbps"], bm["pso"]["per_seed_Mbps"])
            _log(
                f"  TD3-pso={pp['mean_delta']:+.4f}  "
                f"{pp['n_higher']}/{pp['n']}  p={pp['p_two_sided']:.4g}"
            )
        if "sca" in bm:
            ps = _pair(bm["td3"]["per_seed_Mbps"], bm["sca"]["per_seed_Mbps"])
            _log(
                f"  TD3-sca={ps['mean_delta']:+.4f}  "
                f"{ps['n_higher']}/{ps['n']}  p={ps['p_two_sided']:.4g}"
            )


def main() -> int:
    repair_campaign(
        dst=ROOT / "results" / "campaign_8.8mhz_cap25_td3.json",
        src=ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json",
    )
    repair_bank(
        dst=ROOT / "results" / "n100" / "eval_td3.json",
        src=ROOT / "results" / "n100" / "eval.json",
    )
    repair_bank(
        dst=ROOT / "results" / "n100_500m_cap25" / "eval_td3.json",
        src=ROOT / "results" / "n100_500m_cap25" / "eval.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
