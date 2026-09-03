import json
from pathlib import Path

p = json.loads(Path("results/local_direction_diag_seed1_2p4mhz.json").read_text())
rows = [r for r in p["rows"] if r.get("true_sum_rate") is not None and r.get("gate")]
rows.sort(key=lambda r: r["delta_vs_baseline"], reverse=True)
print("top 10 deltas (bit/s):")
for r in rows[:10]:
    print(
        f"  {r.get('kind')} uav={r.get('uav')} dir={r.get('dir')} step={r.get('step_m')} "
        f"delta={r['delta_vs_baseline']:.6f} rate={r['true_sum_rate']:.3f}"
    )
pos = [r for r in rows if r["delta_vs_baseline"] > 0]
print(f"positive deltas: {len(pos)}")
infeas = [r for r in p["rows"] if r.get("true_sum_rate") is None]
print(f"infeasible: {len(infeas)}")
for r in infeas:
    print(
        f"  {r.get('kind')} uav={r.get('uav')} dir={r.get('dir')} "
        f"step={r.get('step_m')} status={r.get('bw_status')}"
    )
