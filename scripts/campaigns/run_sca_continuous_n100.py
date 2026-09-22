"""Continuous-candidate SCA on the n100 500 m bank.

~120 uniform UAV layouts, leftover-dump LP, top-3 polish, keep-best.
Reuses the zenith-anchor checkpoint so only ``sca_continuous`` is new.
Does not overwrite ``eval.json`` or ``eval_anchor.json``.

Usage:
  python scripts/campaigns/run_sca_continuous_n100.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig  # noqa: E402
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
    ROOT / "results" / "n100_500m_cap25" / "eval_anchor.json",
}
BANK = ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json"
SRC_CKPT = ROOT / "results" / "n100_500m_cap25" / "eval_anchor.checkpoint.json"
OUT = ROOT / "results" / "n100_500m_cap25" / "eval_continuous.json"
CKPT = ROOT / "results" / "n100_500m_cap25" / "eval_continuous.checkpoint.json"
N_CONTINUOUS = 120
METHODS = (
    "random",
    "kmeans",
    "pso",
    "sca",
    "sca_multistart",
    "sca_anchor",
    "sca_continuous",
)
PRACTICAL_MBPS = 0.05


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


def _complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stats = (payload.get("by_method") or {}).get("sca_continuous") or {}
    return int(stats.get("n") or 0) == 100


def _pair(by_method: dict, left: str, right: str) -> dict:
    a = (by_method.get(left) or {}).get("per_seed_Mbps") or []
    b = (by_method.get(right) or {}).get("per_seed_Mbps") or []
    n = min(len(a), len(b))
    import numpy as np

    d = np.array([float(a[i]) - float(b[i]) for i in range(n)], dtype=float)
    if d.size == 0:
        return {"n": 0}
    return {
        "n": int(d.size),
        "mean_delta_Mbps": float(d.mean()),
        "std_delta_Mbps": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "n_higher": int((d > 1e-4).sum()),
        "n_lower": int((d < -1e-4).sum()),
        "n_practical": int((d > PRACTICAL_MBPS).sum()),
        "wilcoxon": wilcoxon_signed_rank(d, alternative="greater"),
        "paired_t": paired_t(d),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-skip-complete", action="store_true")
    args = p.parse_args()
    _refuse(OUT)
    if not args.no_skip_complete and _complete(OUT):
        _log(f"skip complete {OUT}")
        payload = json.loads(OUT.read_text(encoding="utf-8"))
    else:
        if not CKPT.exists():
            if not SRC_CKPT.exists():
                raise SystemExit(f"missing baseline checkpoint {SRC_CKPT}")
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC_CKPT, CKPT)
            _log(f"copied {SRC_CKPT.name} -> {CKPT.name}")
        bank = load_bank(BANK)
        overlay = SimConfig(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)
        _log(
            f"n100 500 m continuous  n={N_CONTINUOUS}  "
            f"methods={list(METHODS)}  out={OUT.relative_to(ROOT)}"
        )
        t0 = perf_counter()
        payload = evaluate_bank(
            bank,
            overlay,
            METHODS,
            sca_settings=SCASettings(solver=None, max_iterations=30),
            pso_settings=PSOSettings(),
            anchor_settings=AnchorSettings(
                selection="continuous", n_continuous=N_CONTINUOUS, top_k=3
            ),
            checkpoint_path=CKPT,
            resume=True,
            bank_path=BANK,
        )
        write_eval(payload, OUT)
        _log(f"wrote {OUT}  ({perf_counter() - t0:.1f}s)")
    by_m = payload.get("by_method") or {}
    for method, stats in by_m.items():
        _log(
            f"  {method:16s}  mean={stats['mean_sum_rate_Mbps']:.4f} Mbps  "
            f"std={stats['std_sum_rate_Mbps']:.4f}  n={stats['n']}"
        )
    pairs = {
        "continuous_vs_anchor": _pair(by_m, "sca_continuous", "sca_anchor"),
        "continuous_vs_sca": _pair(by_m, "sca_continuous", "sca"),
        "continuous_vs_multistart": _pair(by_m, "sca_continuous", "sca_multistart"),
        "anchor_vs_sca": _pair(by_m, "sca_anchor", "sca"),
        "anchor_vs_multistart": _pair(by_m, "sca_anchor", "sca_multistart"),
    }
    side = OUT.with_name("eval_continuous_pairs.json")
    side.write_text(json.dumps(pairs, indent=2, default=str), encoding="utf-8")
    for name, stats in pairs.items():
        _log(
            f"  {name}: {stats.get('mean_delta_Mbps', float('nan')):+.4f} Mbps  "
            f"{stats.get('n_higher', 0)}/{stats.get('n', 0)} higher  "
            f"{stats.get('n_practical', 0)}/{stats.get('n', 0)} practical"
        )
    _log(f"wrote {side}")


if __name__ == "__main__":
    main()
