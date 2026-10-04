import json
from pathlib import Path

ROOT = Path("results")
for p in [
    "campaign_10mhz_cap25_n100.json",
    "n100/eval.json",
    "n100/eval_anchor.json",
]:
    f = ROOT / p
    if not f.exists():
        print(p, "MISSING")
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    if "by_method" in d:
        bm = d["by_method"]
        print(
            p,
            "b_sys",
            d.get("b_sys_hz"),
            "cap",
            d.get("max_bw_share"),
            "methods",
            list(bm.keys()),
        )
        for m, v in bm.items():
            print(
                f"  {m}: mean={v.get('mean_sum_rate_Mbps', 0):.4f} n={v.get('n')}"
            )
    else:
        print(
            p,
            "b_sys",
            d.get("b_sys_hz"),
            "sca_anchor",
            d.get("sca_anchor"),
            "methods",
            d.get("methods"),
            "points",
            len(d.get("points", [])),
        )
for ax in ["uavs", "iots", "lambda", "aodt"]:
    f = ROOT / f"campaign_sca_anchor_{ax}_n100_100m.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        print(
            f"anchor {ax}: b_sys={d.get('b_sys_hz')} n_runs={d.get('n_runs')} "
            f"pts={len(d.get('points', []))}"
        )
