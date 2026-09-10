"""Analyze 500x500 m n100 runs at 25% and 15% caps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("random", "kmeans", "pso", "sca")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(champion: str, by_method: dict) -> dict[str, dict]:
    champ = np.asarray(by_method[champion]["per_seed_Mbps"], dtype=float)
    out: dict[str, dict] = {}
    for name in METHODS:
        if name == champion:
            continue
        other = np.asarray(by_method[name]["per_seed_Mbps"], dtype=float)
        delta = champ - other
        _, p = stats.wilcoxon(delta)
        out[name] = {
            "mean_delta_Mbps": float(delta.mean()),
            "std_delta_Mbps": float(delta.std(ddof=1)),
            "win_fraction": float(np.mean(delta > 0.0)),
            "wilcoxon_p": float(p),
        }
    return out


def print_block(title: str, payload: dict, champion: str = "sca") -> None:
    area = payload.get("area_m", [500, 500])
    cap = payload.get("max_bw_share")
    print(f"\n=== {title} ===")
    print(f"Field {area[0]:g}x{area[1]:g} m  cap={cap:g}  n={payload['n_scenarios']}")
    print(f"{'method':8s}  {'mean':>8s}  {'std':>8s}  {'feas':>6s}")
    for m in METHODS:
        s = payload["by_method"][m]
        print(
            f"{m:8s}  {s['mean_sum_rate_Mbps']:8.4f}  "
            f"{s['std_sum_rate_Mbps']:8.4f}  {100*s['feasible_fraction']:5.0f}%"
        )
    print(f"\nPaired {champion.upper()} minus other:")
    for name, p in paired(champion, payload["by_method"]).items():
        print(
            f"  vs {name:8s}: {p['mean_delta_Mbps']:+.4f} Mbps  "
            f"std={p['std_delta_Mbps']:.4f}  wins={p['win_fraction']*100:.0f}%  "
            f"p={p['wilcoxon_p']:.2e}"
        )


def main() -> None:
    runs = {
        "500m @ 25%": ROOT / "results/n100_500m_cap25/eval.json",
        "500m @ 15%": ROOT / "results/n100_500m_cap15/eval.json",
        "100m @ 25%": ROOT / "results/n100/eval.json",
        "100m @ 15%": ROOT / "results/n100_cap15/eval.json",
    }
    payloads = {k: load(p) for k, p in runs.items() if p.exists()}

    for label, payload in payloads.items():
        print_block(label, payload)

    if "500m @ 25%" in payloads and "500m @ 15%" in payloads:
        print("\n=== 500m: cap effect (25% minus 15%) ===")
        for m in METHODS:
            a = payloads["500m @ 25%"]["by_method"][m]["mean_sum_rate_Mbps"]
            b = payloads["500m @ 15%"]["by_method"][m]["mean_sum_rate_Mbps"]
            print(f"  {m:8s}: 25%={a:.4f}  15%={b:.4f}  drop={a-b:+.4f} Mbps")

        sca25 = np.asarray(
            payloads["500m @ 25%"]["by_method"]["sca"]["per_seed_Mbps"], dtype=float
        )
        sca15 = np.asarray(
            payloads["500m @ 15%"]["by_method"]["sca"]["per_seed_Mbps"], dtype=float
        )
        pso15 = np.asarray(
            payloads["500m @ 15%"]["by_method"]["pso"]["per_seed_Mbps"], dtype=float
        )
        d = sca15 - pso15
        _, p = stats.wilcoxon(d)
        print(
            f"\n500m @ 15%: SCA vs PSO paired delta={d.mean():+.4f} Mbps  "
            f"SCA wins {100*np.mean(d>0):.0f}%  p={p:.2e}"
        )

    if "100m @ 25%" in payloads and "500m @ 25%" in payloads:
        print("\n=== Field-size effect @ 25% cap (100m minus 500m) ===")
        for m in METHODS:
            a = payloads["100m @ 25%"]["by_method"][m]["mean_sum_rate_Mbps"]
            b = payloads["500m @ 25%"]["by_method"][m]["mean_sum_rate_Mbps"]
            print(f"  {m:8s}: 100m={a:.4f}  500m={b:.4f}  gain@100m={a-b:+.4f} Mbps")


if __name__ == "__main__":
    main()
