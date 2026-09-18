"""Write LaTeX sweep tables for PPT from measured campaign JSON.

Outputs:
  latex/ppt/sections/results/12_j_sweep_table.tex
  latex/ppt/sections/results/13_i_sweep_table.tex
  latex/ppt/sections/results/14_runtime_sca_td3.tex
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "ppt" / "sections" / "results"

COLS = ("sca_anchor", "sca_multistart", "sca", "td3", "kmeans", "random")
HEAD = (
    "Anchor",
    "Multi-start",
    "SCA",
    "TD3",
    "K-means",
    "Random",
)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _axis_rows(campaign: dict, axis: str) -> list[dict]:
    pts = sorted(
        (p for p in campaign["points"] if p["axis"] == axis),
        key=lambda p: float(p["x"]),
    )
    rows = []
    for p in pts:
        row: dict = {"x": float(p["x"])}
        for c in COLS:
            bm = p.get("by_method", {})
            row[c] = float(bm[c]["mean_sum_rate_Mbps"]) if c in bm else None
        rows.append(row)
    return rows


def _overlay(rows: list[dict], extra: dict | None, axis: str, method: str) -> None:
    if extra is None:
        return
    by_x = {
        float(p["x"]): p
        for p in extra.get("points", [])
        if p.get("axis") == axis
    }
    for row in rows:
        bm = by_x.get(row["x"], {}).get("by_method", {})
        if method in bm:
            row[method] = float(bm[method]["mean_sum_rate_Mbps"])


def _build_j(td3: dict, anchor_u: dict | None, multistart_u: dict | None) -> list[dict]:
    rows = _axis_rows(td3, "uavs")
    _overlay(rows, anchor_u, "uavs", "sca_anchor")
    _overlay(rows, multistart_u, "uavs", "sca_multistart")
    return rows


def _build_i(td3: dict, anchor_i: dict | None, multistart_i: dict | None) -> list[dict]:
    rows = _axis_rows(td3, "iots")
    _overlay(rows, anchor_i, "iots", "sca_anchor")
    _overlay(rows, multistart_i, "iots", "sca_multistart")
    return rows


def _fmt_cell(v: float | None, best: float | None) -> str:
    if v is None:
        return "---"
    s = f"{v:.3f}"
    if best is not None and abs(v - best) < 1e-6:
        return f"\\textbf{{{s}}}"
    return s


def _row_best(row: dict) -> float | None:
    vals = [row[c] for c in COLS if row.get(c) is not None]
    return max(vals) if vals else None


def _write_j(rows: list[dict]) -> None:
    lines = [
        r"\scriptsize",
        r"Mean sum rate (Mbps), $100\times100\,\mathrm{m^2}$, 8.8 MHz, 25\% cap, 20 seeds.",
        r"\textbf{Bold} = best in row. Sources: \texttt{campaign\_8.8mhz\_cap25\_td3.json},",
        r"\texttt{campaign\_sca\_anchor\_uavs.json}, \texttt{campaign\_sca\_multistart\_uavs.json}.",
        r"",
        r"\vspace{0.08cm}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}r" + "r" * len(COLS) + r"@{}}",
        r"    \toprule",
        r"    $J$ & " + " & ".join(HEAD) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        best = _row_best(row)
        cells = [_fmt_cell(row.get(c), best) for c in COLS]
        lines.append(f"    {int(row['x'])} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"\end{tabular}",
        r"}",
    ]
    (OUT / "12_j_sweep_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_i(rows: list[dict]) -> None:
    lines = [
        r"\scriptsize",
        r"Mean sum rate (Mbps), $J=3$, 20 seeds, 100 m. Baselines + TD3 from TD3 campaign;",
        r"anchor from \texttt{campaign\_sca\_anchor\_iots.json}, multi-start from",
        r"\texttt{campaign\_sca\_multistart\_iots.json}. \textbf{Bold} = best in row.",
        r"",
        r"\vspace{0.08cm}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}r" + "r" * len(COLS) + r"@{}}",
        r"    \toprule",
        r"    $I$ & " + " & ".join(HEAD) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        best = _row_best(row)
        cells = [_fmt_cell(row.get(c), best) for c in COLS]
        xl = str(int(row["x"])) if abs(row["x"] - round(row["x"])) < 1e-9 else f"{row['x']:g}"
        lines.append(f"    {xl} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"\end{tabular}",
        r"}",
    ]
    (OUT / "13_i_sweep_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runtime() -> None:
    text = r"""\small
Per-instance wall time at default geometry ($I=10$, $J=3$, 100 m, 8.8 MHz, 25\% cap)
unless noted.

\vspace{0.12cm}
\begin{tabular}{@{}lrr@{}}
    \toprule
    Method & Mean time & vs one-shot SCA \\
    \midrule
    Frozen SCA (\texttt{solve\_sca}) & 0.89 s / seed & $1.0\times$ \\
    SCA multi-start (5 solves, keep-best) & 5.6 s / seed & $\sim 5\times$ \\
    SCA zenith-anchor (100 m) & 6.5 s / seed & $\sim 7\times$ \\
    TD3 (7000 train steps + export eval) & 97 s / seed & $\sim 110\times$ \\
    \bottomrule
\end{tabular}

\vspace{0.2cm}
\footnotesize
TD3: per-instance \textbf{training} dominates ($\sim$97 s at $J=3$);
frozen SCA is a single convex solve ($\sim$0.9 s).
On the 100-layout bank, TD3 averages $\sim$72 s / scenario vs SCA $\ll$1 s
(same train-once-per-layout protocol).
"""
    (OUT / "14_runtime_sca_td3.tex").write_text(text, encoding="utf-8")


def main() -> int:
    td3 = _load(ROOT / "results" / "campaign_8.8mhz_cap25_td3.json")
    if td3 is None:
        raise SystemExit("missing campaign_8.8mhz_cap25_td3.json")
    anchor_u = _load(ROOT / "results" / "campaign_sca_anchor_uavs.json")
    multistart_u = _load(ROOT / "results" / "campaign_sca_multistart_uavs.json")
    anchor_i = _load(ROOT / "results" / "campaign_sca_anchor_iots.json")
    multistart_i = _load(ROOT / "results" / "campaign_sca_multistart_iots.json")
    _write_j(_build_j(td3, anchor_u, multistart_u))
    _write_i(_build_i(td3, anchor_i, multistart_i))
    _write_runtime()
    print("Wrote 12_j_sweep_table.tex, 13_i_sweep_table.tex, 14_runtime_sca_td3.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
