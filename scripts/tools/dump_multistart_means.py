"""Dump multi-start n20 500 m means from the three JSON copies."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]


def ms(path: Path, key: str = "sca_multistart") -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "per_seed" in d:
        xs = [float(r[key]["sum_rate_Mbps"]) for r in d["per_seed"] if key in r]
        extra = ""
        if "mean_Mbps" in d and key in d["mean_Mbps"]:
            extra = f"  stored={d['mean_Mbps'][key]:.6f}"
        print(
            f"{path.relative_to(ROOT)}  {key}  n={len(xs)}  "
            f"mean={mean(xs):.6f}  std={stdev(xs) if len(xs)>1 else 0:.4f}{extra}"
        )
        return
    print(f"{path} no per_seed")


def main() -> None:
    for p in (
        ROOT / "results" / "sca_multistart_n20_500m.json",
        ROOT / "results" / "sca_anchor_n20_500m.json",
        ROOT / "results" / "sca_continuous_n20_500m.json",
    ):
        ms(p)
        ms(p, "sca")


if __name__ == "__main__":
    main()
