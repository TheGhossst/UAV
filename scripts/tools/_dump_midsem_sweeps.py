import json
from pathlib import Path

c = json.loads(Path("results/campaign_10mhz_cap25_n100.json").read_text(encoding="utf-8"))
print("b_sys", c.get("b_sys_hz"), "cap", c.get("max_bw_share"), "n", c.get("n_runs"))
methods = ("sca", "kmeans", "pso", "random")
for axis in ("uavs", "iots", "lambda", "aodt", "cpu"):
    pts = sorted(
        [p for p in c["points"] if p["axis"] == axis],
        key=lambda p: float(p["x"]),
    )
    print("===", axis, "===")
    for p in pts:
        bm = p["by_method"]
        bits = []
        for m in methods:
            if m in bm:
                bits.append(f"{m}={bm[m]['mean_sum_rate_Mbps']:.3f}")
        print(f"  x={p['x']}  " + "  ".join(bits))
