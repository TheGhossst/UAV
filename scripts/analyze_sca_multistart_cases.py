"""Paired readout for the four multi-start default-point cases.

Usage:
  python scripts/analyze_sca_multistart_cases.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paired_winrate import paired_t, wilcoxon_signed_rank  # noqa: E402

PRACTICAL_MBPS = 0.05
OUT_JSON = ROOT / "results" / "sca_multistart_cases_analysis.json"
OUT_TXT = ROOT / "results" / "sca_multistart_cases_analysis.txt"

CASES = {
    "n20_100m": {
        "kind": "n20",
        "path": ROOT / "results" / "sca_multistart_n20.json",
        "title": "20-seed campaign point, 100×100 m",
    },
    "n20_500m": {
        "kind": "n20",
        "path": ROOT / "results" / "sca_multistart_n20_500m.json",
        "title": "20-seed campaign point, 500×500 m",
    },
    "n100_100m": {
        "kind": "n100",
        "path": ROOT / "results" / "n100" / "eval_multistart.json",
        "title": "n100 frozen bank, 100×100 m",
    },
    "n100_500m": {
        "kind": "n100",
        "path": ROOT / "results" / "n100_500m_cap25" / "eval_multistart.json",
        "title": "n100 frozen bank, 500×500 m",
    },
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_from_arrays(left: np.ndarray, right: np.ndarray, seeds: list[int]) -> dict:
    d = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    wins = int(np.sum(d > 1e-9))
    losses = int(np.sum(d < -1e-9))
    w = wilcoxon_signed_rank(d)
    tstat = paired_t(d)
    loss_seeds = [int(s) for s, delta in zip(seeds, d) if delta < -1e-9]
    return {
        "mean_delta_Mbps": float(np.mean(d)),
        "std_delta_Mbps": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "wins": wins,
        "losses": losses,
        "n": int(d.size),
        "loss_seeds": loss_seeds,
        "n_practical": int(np.sum(d > PRACTICAL_MBPS)),
        "wilcoxon": w,
        "paired_t": tstat,
    }


def _fmt_pair(label: str, p: dict) -> str:
    w = p["wilcoxon"]
    p_g = w.get("p_greater")
    p_s = "n/a" if p_g is None else f"{p_g:.4g}"
    return (
        f"  vs {label:16s}  {p['mean_delta_Mbps']:+.4f} ± {p['std_delta_Mbps']:.4f}  "
        f"{p['wins']}/{p['n']} wins  {p['losses']} losses  "
        f"practical>{PRACTICAL_MBPS:g}: {p['n_practical']}  p_greater={p_s}"
    )


def _n20_rows(payload: dict) -> tuple[list[int], dict[str, np.ndarray], Counter]:
    rows = payload["per_seed"]
    seeds = [int(r["seed"]) for r in rows]
    by: dict[str, np.ndarray] = {}
    for m in ("random", "sca", "sca_multistart"):
        by[m] = np.array([float(r[m]["sum_rate_Mbps"]) for r in rows], dtype=float)
    kinds = Counter(
        str(r["sca_multistart"]["diagnostics"].get("winner_kind") or "unknown")
        for r in rows
    )
    return seeds, by, kinds


def _n100_rows(payload: dict) -> tuple[list[int], dict[str, np.ndarray], Counter]:
    by_m = payload["by_method"]
    seeds = [int(s) for s in by_m["sca"]["seeds"]]
    by = {
        m: np.asarray(by_m[m]["per_seed_Mbps"], dtype=float)
        for m in by_m
    }
    kinds: Counter = Counter()
    for run in payload.get("runs") or []:
        if run.get("method") != "sca_multistart":
            continue
        kind = (run.get("diagnostics") or {}).get("winner_kind") or "unknown"
        kinds[str(kind)] += 1
    return seeds, by, kinds


def summarize_case(case_id: str, spec: dict) -> dict | None:
    path = spec["path"]
    if not path.exists():
        return None
    payload = _load(path)
    if spec["kind"] == "n20":
        seeds, by, kinds = _n20_rows(payload)
        walls = payload.get("mean_wall_clock_s") or {}
        feas = {
            m: float(np.mean([r[m]["feasible"] for r in payload["per_seed"]]))
            for m in by
        }
    else:
        seeds, by, kinds = _n100_rows(payload)
        walls = {}
        for run in payload.get("runs") or []:
            if run.get("method") != "sca_multistart":
                continue
            w = (run.get("diagnostics") or {}).get("wall_clock_s")
            if w is not None:
                walls.setdefault("sca_multistart", []).append(float(w))
        if "sca_multistart" in walls:
            walls = {"sca_multistart": float(np.mean(walls["sca_multistart"]))}
        feas = {
            m: float(payload["by_method"][m]["feasible_fraction"]) for m in by
        }
    means = {m: float(np.mean(v)) for m, v in by.items()}
    stds = {
        m: float(np.std(v, ddof=1)) if v.size > 1 else 0.0 for m, v in by.items()
    }
    ms = by["sca_multistart"]
    paired = {
        other: _pair_from_arrays(ms, by[other], seeds)
        for other in by
        if other != "sca_multistart"
    }
    random_beat_sca = [
        int(s)
        for s, a, b in zip(seeds, by["sca"], by["random"])
        if float(a) + 1e-9 < float(b)
    ]
    still_lose = [
        int(s)
        for s, a, b in zip(seeds, ms, by["random"])
        if int(s) in random_beat_sca and float(a) + 1e-9 < float(b)
    ]
    closed = [s for s in random_beat_sca if s not in still_lose]
    return {
        "id": case_id,
        "title": spec["title"],
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "n": int(ms.size),
        "area_m": payload.get("area_m"),
        "max_bw_share": payload.get("max_bw_share"),
        "means_Mbps": means,
        "std_Mbps": stds,
        "feasible_fraction": feas,
        "winner_kinds": dict(kinds),
        "paired": paired,
        "random_beat_sca_seeds": random_beat_sca,
        "closed_random_losses": closed,
        "still_lose_to_random": still_lose,
        "mean_wall_clock_s": walls,
    }


def _lines(summary: dict) -> list[str]:
    lines = [
        f"=== {summary['id']}: {summary['title']} ===",
        f"n={summary['n']}  artifact={summary['path']}",
        "means (Mbps):",
    ]
    order = ("sca_multistart", "sca", "random", "pso", "kmeans", "td3")
    for m in order:
        if m not in summary["means_Mbps"]:
            continue
        lines.append(
            f"  {m:16s}  {summary['means_Mbps'][m]:7.4f}  "
            f"± {summary['std_Mbps'][m]:.4f}  "
            f"feas={100.0 * summary['feasible_fraction'][m]:.1f}%"
        )
    lines.append("paired multi-start minus other:")
    for name in ("sca", "random", "pso", "kmeans"):
        if name in summary["paired"]:
            lines.append(_fmt_pair(name, summary["paired"][name]))
    kinds = summary["winner_kinds"]
    if kinds:
        kind_s = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        lines.append(f"winner kinds: {kind_s}")
    n_beat = len(summary["random_beat_sca_seeds"])
    lines.append(
        f"SCA losses to random: {n_beat}/{summary['n']}  "
        f"closed by multi-start: {len(summary['closed_random_losses'])}  "
        f"still lose: {summary['still_lose_to_random']}"
    )
    wall = summary.get("mean_wall_clock_s") or {}
    if "sca_multistart" in wall:
        lines.append(f"mean multi-start wall: {wall['sca_multistart']:.1f} s")
    lines.append("")
    return lines


def main() -> int:
    cases: dict[str, dict] = {}
    missing: list[str] = []
    lines = [
        "Multi-start SCA default-point cases (8.8 MHz, 25% cap, I=10, J=3)",
        "Keep-best of frozen k-means SCA + 2 random + 2 k-means extra inits.",
        "Frozen SCA stays the headline solver. n100_500m is the 100-layout 500 m bank.",
        "",
    ]
    for case_id, spec in CASES.items():
        summary = summarize_case(case_id, spec)
        if summary is None:
            missing.append(case_id)
            lines.append(f"=== {case_id}: MISSING {spec['path']} ===")
            lines.append("")
            continue
        cases[case_id] = summary
        lines.extend(_lines(summary))

    if "n20_100m" in cases and "n20_500m" in cases:
        a = cases["n20_100m"]["paired"]["sca"]["mean_delta_Mbps"]
        b = cases["n20_500m"]["paired"]["sca"]["mean_delta_Mbps"]
        lines.append("=== field-size effect (20-seed, multi-start - SCA) ===")
        lines.append(f"  100 m: {a:+.4f} Mbps   500 m: {b:+.4f} Mbps")
        lines.append("")
    if "n100_100m" in cases and "n100_500m" in cases:
        a = cases["n100_100m"]["paired"]["sca"]["mean_delta_Mbps"]
        b = cases["n100_500m"]["paired"]["sca"]["mean_delta_Mbps"]
        lines.append("=== field-size effect (n100, multi-start - SCA) ===")
        lines.append(f"  100 m: {a:+.4f} Mbps   500 m: {b:+.4f} Mbps")
        lines.append("")

    payload = {
        "practical_bar_Mbps": PRACTICAL_MBPS,
        "missing": missing,
        "cases": cases,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = "\n".join(lines).rstrip() + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    try:
        print(text, end="")
        print(f"wrote {OUT_JSON}")
        print(f"wrote {OUT_TXT}")
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), end="")
        print(f"wrote {OUT_JSON}")
        print(f"wrote {OUT_TXT}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
