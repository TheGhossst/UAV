"""Print T_k=0.8 feasible counts from campaign JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def tk08(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    for pt in d["points"]:
        if pt.get("axis") != "aodt" or abs(float(pt.get("x", 0)) - 0.8) > 1e-9:
            continue
        print(path.name)
        for m, bm in pt["by_method"].items():
            keys = sorted(bm.keys())
            feas = bm.get("n_feasible", bm.get("n_feas", bm.get("feasible_count")))
            frac = bm.get("frac_feasible", bm.get("feasible_frac"))
            n = len(bm.get("per_seed_Mbps") or [])
            print(f"  {m} n={n} n_feasible={feas} frac={frac} keys={keys[:12]}")


def main() -> None:
    for name in (
        "campaign_8.8mhz_cap15_n20.json",
        "campaign_8.8mhz_cap12_n20.json",
        "campaign_8.8mhz_cap12_n20_500m.json",
        "campaign_8.8mhz_cap25_si12k.json",
    ):
        tk08(ROOT / "results" / name)


if __name__ == "__main__":
    main()
