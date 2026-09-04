"""Print campaign mean Mbps table."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    p = json.loads(Path("results/campaign_8.8mhz.json").read_text(encoding="utf-8"))
    print("n_runs", p["n_runs"], "methods", p["methods"], "B_sys", p["b_sys_hz"])
    print("n_points", len(p["points"]))
    cur = None
    for pt in p["points"]:
        if pt["axis"] != cur:
            cur = pt["axis"]
            print()
            print("===", cur, pt["x_name"], "===")
        bits = []
        for m in p["methods"]:
            s = pt["by_method"][m]
            bits.append(
                f"{m}={s['mean_sum_rate_Mbps']:.3f} feas={s['feasible_fraction']:.0%}"
            )
        print(f"  {pt['x_name']}={pt['x']:g}  " + "  ".join(bits))


if __name__ == "__main__":
    main()
