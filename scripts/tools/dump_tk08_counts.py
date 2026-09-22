import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    for name in (
        "campaign_8.8mhz_cap15_n20.json",
        "campaign_8.8mhz_cap12_n20.json",
        "campaign_8.8mhz_cap12_n20_500m.json",
        "campaign_8.8mhz_cap25_si12k.json",
    ):
        d = json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        for pt in d["points"]:
            if pt.get("axis") != "aodt" or abs(float(pt["x"]) - 0.8) > 1e-9:
                continue
            print(name)
            for m, bm in pt["by_method"].items():
                frac = bm["feasible_fraction"]
                nfeas = int(round(frac * bm["n"]))
                print(f"  {m} {nfeas}/{bm['n']} ({frac:.0%})")


if __name__ == "__main__":
    main()
