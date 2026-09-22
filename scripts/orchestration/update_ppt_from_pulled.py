"""Refresh PPT result slides from pulled eval JSON (n200 banks + campaigns).

Usage:
  python scripts/orchestration/update_ppt_from_pulled.py
  python scripts/orchestration/update_ppt_from_pulled.py --latexmk
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PPT = ROOT / "latex" / "ppt" / "sections" / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from paired_winrate import wilcoxon_signed_rank  # noqa: E402
from uavdt.experiments.method_labels import normalize_by_method  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bank_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    p = _load(path)
    bm = normalize_by_method(p.get("by_method") or {})
    out: dict = {}
    for m in (
        "sca",
        "frozen_sca",
        "random",
        "kmeans",
        "pso",
        "sca_multistart",
        "sca_anchor",
    ):
        if m not in bm:
            continue
        out[m] = {
            "mean": float(bm[m]["mean_sum_rate_Mbps"]),
            "per_seed": np.asarray(bm[m]["per_seed_Mbps"], dtype=float),
            "seeds": [int(s) for s in bm[m]["seeds"]],
        }
    if "sca" not in out or "random" not in out:
        return None
    d = out["sca"]["per_seed"] - out["random"]["per_seed"]
    w = wilcoxon_signed_rank(d)
    out["sca_vs_random"] = {
        "wins": int(np.sum(d > 1e-9)),
        "n": int(d.size),
        "delta": float(np.mean(d)),
        "p": w.get("p_greater"),
    }
    if "sca_anchor" in out:
        da = out["sca_anchor"]["per_seed"] - out["sca"]["per_seed"]
        out["anchor_vs_sca"] = {
            "wins": int(np.sum(da > 1e-9)),
            "n": int(da.size),
            "delta": float(np.mean(da)),
        }
    if "sca_multistart" in out:
        dm = out["sca_multistart"]["per_seed"] - out["sca"]["per_seed"]
        out["multistart_vs_sca"] = {
            "wins": int(np.sum(dm > 1e-9)),
            "n": int(dm.size),
            "delta": float(np.mean(dm)),
            "std": float(np.std(dm, ddof=1)),
        }
    return out


def _headline_campaign(path: Path) -> dict[str, float]:
    p = _load(path)
    for pt in p.get("points") or []:
        if pt.get("axis") == "uavs" and int(float(pt.get("x", 0))) == 3:
            return {
                m: float(pt["by_method"][m]["mean_sum_rate_Mbps"])
                for m in ("sca", "random", "kmeans", "pso")
                if m in (pt.get("by_method") or {})
            }
    return {}


def _patch_headline_j3(hdr: dict[str, float], n100: dict | None, n200: dict | None) -> None:
    path = PPT / "3_headline_j3.tex"
    text = path.read_text(encoding="utf-8")
    if hdr:
        text = re.sub(
            r"\\textbf\{8\.946\} & 8\.773",
            f"\\\\textbf{{{hdr['sca']:.3f}}} & {hdr['random']:.3f}",
            text,
            count=1,
        )
        spread = hdr["sca"] - hdr["random"]
        text = re.sub(
            r"Spread across methods \\textbf\{0\.173 Mbps\}\. SCA vs random:\s*\n\$\+0\.173",
            f"Spread across methods \\\\textbf{{{spread:.3f} Mbps}}. SCA vs random:\n$+{spread:.3f}",
            text,
            count=1,
        )
    if n200:
        p = n200["sca_vs_random"]["p"]
        ps = f"$p<{p:.3f}$" if p and p < 0.001 else f"$p={p:.3g}$" if p else "$p=n/a$"
        extra = (
            f"\n\\vspace{{0.1cm}}\n"
            f"\\textbf{{n200 bank @ 100 m / 25\\%}} (same table point, 200 layouts): "
            f"SCA {n200['sca']['mean']:.3f}, random {n200['random']['mean']:.3f}; "
            f"SCA vs random {n200['sca_vs_random']['wins']}/{n200['sca_vs_random']['n']}, {ps}.\n"
        )
        if "n200 bank @ 100 m" not in text:
            text = text.rstrip() + extra
    path.write_text(text, encoding="utf-8")


def _patch_multistart(n100_100: dict | None, n100_500: dict | None, n200_100: dict | None, n200_500: dict | None) -> None:
    path = PPT / "6_multistart.tex"
    body = [
        r"\small",
        r"",
        r"Opt-in \texttt{sca\_multistart}: keep-best over frozen k-means SCA plus",
        r"\textbf{2 random + 2 k-means} extra inits ($\texttt{seed}\times 1000+i$).",
        r"Same \texttt{solve\_sca}; \texttt{include\_frozen=True} $\Rightarrow$ never",
        r"worse than one-shot SCA \textbf{by construction}.",
        r"",
        r"\vspace{0.2cm}",
        r"",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        r"    Case & $n$ & SCA & Multi-start & $\Delta$ vs SCA \\",
        r"    \midrule",
        r"    20-seed 100 m / 25\% & 20 & 8.946 & 8.961 & $+0.014 \pm 0.020$ \\",
        r"    20-seed 500 m / 25\% & 20 & 8.300 & 8.403 & $+0.103 \pm 0.150$ \\",
    ]
    if n100_100 and "multistart_vs_sca" in n100_100:
        ms = n100_100["multistart_vs_sca"]
        body.append(
            f"    n100 bank 100 m & 100 & {n100_100['sca']['mean']:.3f} & "
            f"{n100_100['sca_multistart']['mean']:.3f} & ${ms['delta']:+.3f} \\pm {ms['std']:.3f}$ \\\\"
        )
    if n100_500 and "multistart_vs_sca" in n100_500:
        ms = n100_500["multistart_vs_sca"]
        body.append(
            f"    n100 bank 500 m & 100 & {n100_500['sca']['mean']:.3f} & "
            f"{n100_500['sca_multistart']['mean']:.3f} & ${ms['delta']:+.3f} \\pm {ms['std']:.3f}$ \\\\"
        )
    if n200_100 and "multistart_vs_sca" in n200_100:
        ms = n200_100["multistart_vs_sca"]
        body.append(
            f"    n200 bank 100 m & 200 & {n200_100['sca']['mean']:.3f} & "
            f"{n200_100['sca_multistart']['mean']:.3f} & ${ms['delta']:+.3f} \\pm {ms['std']:.3f}$ \\\\"
        )
    if n200_500 and "multistart_vs_sca" in n200_500:
        ms = n200_500["multistart_vs_sca"]
        body.append(
            f"    n200 bank 500 m & 200 & {n200_500['sca']['mean']:.3f} & "
            f"{n200_500['sca_multistart']['mean']:.3f} & ${ms['delta']:+.3f} \\pm {ms['std']:.3f}$ \\\\"
        )
    body += [
        r"    \bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{0.25cm}",
        r"",
        r"\textbf{Explanation.} At 100 m the leftover-dump LP already saturates --- gain is",
        r"a small nick ($\sim$0.01--0.02 Mbps). At \textbf{500 m} extra k-means/random",
        r"inits reach distant basins (16--72 m from frozen SCA); multi-start beats random",
        r"on \textbf{19--20/20} seeds at 100 m and \textbf{20/20} at 500 m (20-seed).",
        r"Cost $\sim 5\times$ one SCA ($\sim 5.6\,\mathrm{s/seed}$).",
        r"",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _patch_anchor(n100_100: dict | None, n100_500: dict | None, n200_100: dict | None, n200_500: dict | None) -> None:
    path = PPT / "7_anchor.tex"
    body = [
        r"\small",
        r"",
        r"Opt-in \texttt{sca\_anchor}: enumerate zenith $J$-subsets of IoTs (or beam search",
        r"if $C(I,J)>1000$), score with bandwidth LP + \texttt{evaluate}, polish",
        r"\textbf{top-3} with frozen SCA, keep-best vs k-means SCA.",
        r"",
        r"\vspace{0.15cm}",
        r"",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        r"    Test ($J=3$, 25\% cap) & Anchor & SCA & Multi-start & $\Delta$ vs SCA \\",
        r"    \midrule",
        r"    n20 @ 100 m & \textbf{8.964} & 8.946 & 8.961 & $+0.017$ (18/20 wins) \\",
        r"    n20 @ 500 m & \textbf{8.490} & 8.300 & 8.404 & $+0.190$ (20/20 wins) \\",
    ]
    for label, st in [
        ("n100 @ 100 m", n100_100),
        ("n100 @ 500 m", n100_500),
        ("n200 @ 100 m", n200_100),
        ("n200 @ 500 m", n200_500),
    ]:
        if st is None or "anchor_vs_sca" not in st:
            continue
        av = st["anchor_vs_sca"]
        bold_a = f"\\textbf{{{st['sca_anchor']['mean']:.3f}}}"
        ms_m = st.get("sca_multistart", {}).get("mean")
        ms_s = f"{ms_m:.3f}" if ms_m is not None else "---"
        body.append(
            f"    {label} & {bold_a} & {st['sca']['mean']:.3f} & {ms_s} & "
            f"${av['delta']:+.3f}$ ({av['wins']}/{av['n']} wins) \\\\"
        )
    body += [
        r"    \bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{0.25cm}",
        r"",
        r"\textbf{Explanation.} LP-only @ 500 m already yields strong placement prior;",
        r"anchor + polish largest at 500 m. At 100 m, gain vs SCA is only",
        r"$\sim +0.017\,\mathrm{Mbps}$ (saturated dump). PPT default figures use the",
        r"\textbf{n200} banks (200 layouts, corrected random UAV RNG).",
        r"",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _patch_paired_500m(n200_500: dict | None) -> None:
    if n200_500 is None:
        return
    path = PPT / "4_paired_statistics.tex"
    text = path.read_text(encoding="utf-8")
    spread = max(n200_500[m]["mean"] for m in ("sca", "random", "kmeans", "pso")) - min(
        n200_500[m]["mean"] for m in ("sca", "random", "kmeans", "pso")
    )
    new = (
        rf"At 500 m / 25\% on the \textbf{{n200}} bank, spread up to "
        rf"\textbf{{{spread:.3f} Mbps}}; SCA remains best among default four."
    )
    text = re.sub(
        r"At 500 m / 25\\%[^\n]+\n",
        new + "\n",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latexmk", action="store_true")
    args = parser.parse_args()

    n200_100 = _bank_stats(
        ROOT / "results" / "n200" / "eval_all_methods_dynamic.json"
    ) or _bank_stats(ROOT / "results" / "n200" / "eval_anchor.json")
    n200_500 = _bank_stats(
        ROOT / "results" / "n200_500m_cap25" / "eval_all_methods_dynamic.json"
    ) or _bank_stats(ROOT / "results" / "n200_500m_cap25" / "eval_anchor.json")
    n100_100 = _bank_stats(ROOT / "results" / "n100" / "eval_anchor.json")
    n100_500 = _bank_stats(ROOT / "results" / "n100_500m_cap25" / "eval_anchor.json")

    camp = ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"
    camp500 = ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json"
    hdr = _headline_campaign(camp) if camp.exists() else {}
    if camp500.exists():
        h5 = _headline_campaign(camp500)
        if h5:
            _patch_headline_500m_row(h5)

    _patch_headline_j3(hdr, n100_100, n200_100)
    _patch_multistart(n100_100, n100_500, n200_100, n200_500)
    _patch_anchor(n100_100, n100_500, n200_100, n200_500)
    _patch_paired_500m(n200_500)

    if (ROOT / "results" / "campaign_8.8mhz_cap25_td3.json").exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "orchestration" / "gen_ppt_sweep_table_tex.py"),
            ],
            check=True,
        )

    print("Updated PPT tex under latex/ppt/sections/results/")
    if n200_100:
        print(
            f"  n200 100m: SCA={n200_100['sca']['mean']:.3f} random={n200_100['random']['mean']:.3f}"
        )
    if n200_500:
        print(
            f"  n200 500m: SCA={n200_500['sca']['mean']:.3f} random={n200_500['random']['mean']:.3f}"
        )

    if args.latexmk:
        tex = ROOT / "latex" / "ppt"
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=tex, check=True)
        print(f"Built {tex / 'main.pdf'}")
    return 0


def _patch_headline_500m_row(h5: dict[str, float]) -> None:
    path = PPT / "3_headline_j3.tex"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"\\textbf\{8\.300\} & 6\.016 & 7\.474 & 8\.122",
        f"\\\\textbf{{{h5['sca']:.3f}}} & {h5['random']:.3f} & {h5['kmeans']:.3f} & {h5['pso']:.3f}",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
