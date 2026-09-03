"""Refresh docs/PROJECT_STATUS_AND_PAPER_ANALYSIS.md from measured result CSVs.

CLI runs call refresh_status_doc() after compare / sweeps / aodt-compare / solvers.
Standalone:

    python -m src.status_sync
    python -m src.status_sync --results results/run_20260830_rng_fixed
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DOC = REPO / "docs" / "PROJECT_STATUS_AND_PAPER_ANALYSIS.md"
DEFAULT_RESULTS = REPO / "results"
RUNS_JSONL = DEFAULT_RESULTS / "_status_runs.jsonl"

# IEEE TNSM 2026 §VII (pp. 3022–3024). Figure reads unless the paper printed the digit.
# PSO is not in the paper. SCA at Fig. 8/9 is the arXiv (no-TD3) "optimized" line.
PAPER = {
    "fig6_j3": {"random": 3.0, "kmeans": 4.0, "sca": 7.1, "td3": 5.8},
    "fig6_j5": {"random": 3.4, "kmeans": 5.6, "sca": 8.8, "td3": 7.0},
    "fig7_i32": {"random": 5.2, "kmeans": 7.8, "sca": 14.0, "td3": 11.6},
    "fig8_lam35": {"random": 4.2, "kmeans": 5.3, "td3": 7.3, "sca": "8.9 (arXiv)"},
    "fig9_tk3": {"random": 3.2, "kmeans": 4.4, "td3": 6.0, "sca": "7.8 (arXiv)"},
    "fig10_fj250": {"random": "<4.5", "kmeans": "<4.5", "sca": 7.5, "td3": 6.7},
}

METHOD_ORDER = ("random", "kmeans", "pso", "sca", "td3")
LABEL = {
    "random": "Random",
    "kmeans": "K-means",
    "pso": "PSO",
    "sca": "SCA",
    "td3": "TD3",
    "proposed": "proposed",
}

MARKER_RE = re.compile(
    r"(<!-- AUTO:([a-z0-9_]+) -->)(.*?)(<!-- /AUTO:\2 -->)",
    re.DOTALL,
)

BEGIN = "<!-- AUTO:{id} -->"
END = "<!-- /AUTO:{id} -->"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _mbps(bps: float | None) -> str:
    if bps is None:
        return "—"
    return f"{float(bps) / 1e6:.3f}"


def _num(x: Any, digits: int = 3) -> str:
    if x is None or x == "":
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(v)


def _paper(fig: str, method: str) -> str:
    block = PAPER.get(fig) or {}
    if method not in block:
        return "n/a" if method in ("pso", "proposed") else "—"
    v = block[method]
    if isinstance(v, str):
        return v if v.startswith("<") or "arXiv" in v else f"~{v}"
    return f"~{v:g}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _newest_matching(root: Path, filename: str, needles: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    found = [
        p
        for p in root.rglob(filename)
        if p.is_file() and any(n in str(p).replace("\\", "/").lower() for n in needles)
    ]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def newest(root: Path, names: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    found: list[Path] = []
    for name in names:
        found.extend(p for p in root.rglob(name) if p.is_file())
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def _newest_aodt_csv(root: Path, filename: str) -> Path | None:
    """Prefer run_* batch dirs; skip nested results/aodt/aodt/ copies."""
    if not root.exists():
        return None
    found = [
        p
        for p in root.rglob(filename)
        if p.is_file()
        and "aodt" in str(p.parent).replace("\\", "/").lower()
        and "/aodt/aodt" not in str(p).replace("\\", "/").lower()
    ]
    if not found:
        return None
    run_dirs = [p for p in found if "run_" in str(p).replace("\\", "/")]
    pool = run_dirs if run_dirs else found
    return max(pool, key=lambda p: p.stat().st_mtime)


def count_tests(repo: Path = REPO) -> int:
    n = 0
    tests = repo / "tests"
    if not tests.is_dir():
        return 0
    for p in tests.glob("test_*.py"):
        n += sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.startswith("def test_"))
    return n


def _as_float(v: Any) -> float:
    if v in (None, "", "None"):
        raise ValueError("empty")
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s in ("True", "true", "TRUE"):
            return 1.0
        if s in ("False", "false", "FALSE"):
            return 0.0
        return float(s)
    return float(v)


def summarize_raw(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    out: dict[str, dict[str, float]] = {}
    for method, sub in by.items():
        rates = [_as_float(r["sum_rate"]) for r in sub]
        feas = [_as_float(r["feasible"]) for r in sub]
        qos = [_as_float(r["qos"]) for r in sub]
        rt = [_as_float(r["runtime"]) for r in sub]
        aodt_key = "aodt_mean" if any("aodt_mean" in r for r in sub) else None
        aodts = []
        if aodt_key:
            for r in sub:
                v = r.get("aodt_mean")
                if v not in (None, "", "None"):
                    aodts.append(float(v))
        aodt_viol = []
        for r in sub:
            v = r.get("aodt_violations", r.get("aodt_viol"))
            if v not in (None, "", "None"):
                aodt_viol.append(float(v))
        n = len(rates)
        mean = sum(rates) / n
        var = sum((x - mean) ** 2 for x in rates) / (n - 1) if n > 1 else 0.0
        out[method] = {
            "n": float(n),
            "sum_rate": mean,
            "std": var ** 0.5,
            "feasible": sum(feas) / n,
            "qos": sum(qos) / n,
            "runtime": sum(rt) / n,
            "aodt": (sum(aodts) / len(aodts)) if aodts else float("nan"),
            "aodt_viol": (sum(aodt_viol) / len(aodt_viol)) if aodt_viol else float("nan"),
        }
    return out


def ranking_line(stats: dict[str, dict[str, float]], labels: dict[str, str] | None = None) -> str:
    lab = labels or LABEL
    items = [(lab.get(m, m), st["sum_rate"] / 1e6) for m, st in stats.items() if m in METHOD_ORDER]
    items.sort(key=lambda t: -t[1])
    if not items:
        return "—"
    best = items[0][0]
    body = " > ".join(f"{name} {mbps:.3f}" for name, mbps in items)
    return f"**{best}** lead: {body} Mbps"


def _src_note(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(REPO)
    except ValueError:
        rel = path
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"_Source: `{rel.as_posix()}` (mtime {mtime}). Paper columns are IEEE TNSM 2026 §VII figure reads (approximate). PSO is not in the paper._"


def table_compare(
    stats: dict[str, dict[str, float]],
    paper_fig: str,
    pso_label: str,
    paper_header: str = "Paper (Mbps)",
) -> list[str]:
    lines = [
        f"| Method | Repo (Mbps) | {paper_header} | Feasible fraction | AoDT mean (s) | QoS | Runtime (s) | Std (bit/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHOD_ORDER:
        if m not in stats:
            continue
        st = stats[m]
        name = "PSO (placement)" if m == "pso" and pso_label == "placement" else (
            "PSO joint" if m == "pso" and pso_label == "joint" else LABEL[m]
        )
        aodt = "—" if st["aodt"] != st["aodt"] else f"{st['aodt']:.3f}"
        lines.append(
            f"| {name} | {_mbps(st['sum_rate'])} | {_paper(paper_fig, m)} | "
            f"{st['feasible']:.2f} | {aodt} | {st['qos']:.2f} | {st['runtime']:.3f} | "
            f"{st['std']:.2e} |"
        )
    lines.append("| proposed | — | n/a | — | — | — | — | — |")
    return lines


def table_aodt_default(stats: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Method | Repo raw (Mbps) | Paper Fig. 6 at J=3 (Mbps) | Feasible frac | AoDT mean (s) | QoS | AoDT viol | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHOD_ORDER:
        if m not in stats:
            continue
        st = stats[m]
        name = "PSO joint" if m == "pso" else LABEL[m]
        aodt = "—" if st["aodt"] != st["aodt"] else f"{st['aodt']:.3f}"
        viol = "—" if st["aodt_viol"] != st["aodt_viol"] else f"{st['aodt_viol']:.2f}"
        lines.append(
            f"| {name} | {_mbps(st['sum_rate'])} | {_paper('fig6_j3', m)} | "
            f"{st['feasible']:.2f} | {aodt} | {st['qos']:.2f} | {viol} | {st['runtime']:.3f} |"
        )
    return lines


def _pivot(rows: list[dict[str, str]], xkey: str) -> tuple[list[float], dict[str, dict[float, dict[str, str]]]]:
    xs: set[float] = set()
    by: dict[str, dict[float, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        x = float(r[xkey])
        xs.add(x)
        by[r["method"]][x] = r
    return sorted(xs), by


def table_vs_axis(
    rows: list[dict[str, str]],
    xkey: str,
    xlabel: str,
    paper_at: dict[float, str],
    methods: tuple[str, ...] = METHOD_ORDER,
) -> list[str]:
    xs, by = _pivot(rows, xkey)
    paper_methods = ("sca", "td3", "kmeans", "random")
    header = f"| {xlabel} | " + " | ".join(LABEL[m] for m in methods if m in by) + " | " + " | ".join(
        f"Paper {LABEL[m]}" for m in paper_methods
    ) + " |"
    align = "|---:|" + "---:|" * (sum(1 for m in methods if m in by) + len(paper_methods))
    lines = [header, align]
    for x in xs:
        cells = [ _fmt_axis(x, xkey) ]
        for m in methods:
            if m not in by:
                continue
            rec = by[m].get(x)
            cells.append(_mbps(float(rec["sum_rate_mean"])) if rec else "—")
        fig = paper_at.get(x)
        for m in paper_methods:
            cells.append(_paper(fig, m) if fig else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _fmt_axis(x: float, xkey: str) -> str:
    if xkey == "uav_cpu":
        return f"{x:.1e}"
    if xkey in ("n_uav", "n_iot") or x == int(x):
        if xkey in ("n_uav", "n_iot"):
            return str(int(x))
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:g}"


def patch_sections(text: str, updates: dict[str, str]) -> str:
    found: set[str] = set()

    def repl(match: re.Match[str]) -> str:
        sid = match.group(2)
        found.add(sid)
        if sid not in updates:
            return match.group(0)
        body = updates[sid].rstrip() + "\n"
        return f"{match.group(1)}\n{body}{match.group(4)}"

    out = MARKER_RE.sub(repl, text)
    missing = [k for k in updates if k not in found]
    if missing:
        raise KeyError(f"status doc missing AUTO markers: {missing}")
    return out


def ensure_markers(text: str, section_ids: tuple[str, ...]) -> str:
    """Append any missing AUTO stubs at the end of Part A (before Part B)."""
    for sid in section_ids:
        token = BEGIN.format(id=sid)
        if token in text:
            continue
        stub = f"\n{token}\n_(no measured data yet)_\n{END.format(id=sid)}\n"
        needle = "# Part B"
        if needle in text:
            text = text.replace(needle, stub + "\n" + needle, 1)
        else:
            text += stub
    return text


SECTION_IDS = (
    "meta",
    "snapshot",
    "tests",
    "compare_5",
    "compare_20",
    "compare_table2",
    "aodt_default",
    "aodt_vs_j",
    "aodt_vs_i",
    "sweeps_default",
    "sweeps_lambda",
    "sweeps_tk",
    "sweeps_cpu",
    "artifacts",
)


def record_run(payload: dict[str, Any], path: Path = RUNS_JSONL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _load_raw_stats(path: Path) -> dict[str, dict[str, float]]:
    return summarize_raw(_read_csv(path))


def _newest_sweep_csv(root: Path, filename: str) -> Path | None:
    """Prefer the newest lambda/Tk/fj summary under sweeps/ or sweeps_constraints/."""
    candidates: list[Path] = []
    for needle in ("sweeps", "sweeps_constraints"):
        p = _newest_matching(root, filename, (needle,))
        if p is not None:
            candidates.append(p)
    if not candidates:
        return newest(root, (filename,))
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_updates(results: Path) -> dict[str, str]:
    n_tests = count_tests()
    updates: dict[str, str] = {}
    artifacts: list[str] = [f"- tests collected (by `def test_`): **{n_tests}**", f"- refresh: {_now()}"]

    compare20 = newest(results, ("compare_raw_20runs.csv",))
    compare5 = _newest_matching(results, "compare_raw.csv", ("compare_5seed", "/compare_5/", "\\compare_5\\"))
    table2 = _newest_matching(results, "compare_raw.csv", ("table2",))
    if table2 is None:
        table2 = _newest_matching(results, "compare_raw_20runs.csv", ("table2",))

    aodt_def = _newest_aodt_csv(results, "raw_default.csv")
    aodt_j = _newest_aodt_csv(results, "summary_vs_uav.csv")
    aodt_i = _newest_aodt_csv(results, "summary_vs_iot.csv")
    sweep_cmp = newest(results, ("raw_default_comparison.csv",))
    sweep_lam = _newest_sweep_csv(results, "sumrate_vs_lambda.csv")
    sweep_tk = _newest_sweep_csv(results, "sumrate_vs_aodt.csv")
    sweep_cpu = _newest_sweep_csv(results, "sumrate_vs_cpu.csv")

    snap_lines = [
        f"Last auto-refresh **{_now()}**. Test functions currently in `tests/`: **{n_tests}**.",
        "",
        "Rankings below are recomputed from the newest matching CSVs under `results/`. Absolute rates use the **calibrated** radio unless a table says `table2`.",
        "",
    ]

    if compare20:
        st = _load_raw_stats(compare20)
        snap_lines.append(f"- `compare --paper-runs` ({int(next(iter(st.values()))['n'])} seeds, placement PSO): {ranking_line(st)}")
        artifacts.append(f"- compare 20-seed: `{_rel(compare20)}`")
        updates["compare_20"] = "\n".join(
            table_compare(st, "fig6_j3", "placement", "Paper Fig. 6 at J=3 (Mbps)")
            + ["", _src_note(compare20)]
        )

    if compare5:
        st5 = _load_raw_stats(compare5)
        nseed = int(next(iter(st5.values()))["n"])
        if nseed <= 8:
            snap_lines.append(f"- `compare` 5-seed (placement PSO): {ranking_line(st5)}")
            artifacts.append(f"- compare 5-seed: `{_rel(compare5)}`")
            lines = table_compare(st5, "fig6_j3", "placement", "Paper Fig. 6 at J=3 (Mbps)")
            updates["compare_5"] = "\n".join(
                lines
                + [
                    "",
                    "_Paper protocol is 20 runs; this table is the 5-seed dev set. Paper column is still the Fig. 6 J=3 read._",
                    _src_note(compare5),
                ]
            )

    if table2:
        stt = _load_raw_stats(table2)
        artifacts.append(f"- table2 compare: `{_rel(table2)}`")
        lines = [
            "| Method | Repo (Mbps) | Paper Figs. 6–10 | Feasible | AoDT mean (s) | QoS | Runtime (s) |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for m in METHOD_ORDER:
            if m not in stt:
                continue
            st = stt[m]
            aodt = "—" if st["aodt"] != st["aodt"] else f"{st['aodt']:.3f}"
            lines.append(
                f"| {LABEL[m] if m != 'pso' else 'PSO'} | {_mbps(st['sum_rate'])} | "
                f"not comparable | {st['feasible']:.2f} | {aodt} | {st['qos']:.2f} | {st['runtime']:.3f} |"
            )
        updates["compare_table2"] = "\n".join(
            lines
            + [
                "",
                "Literal Table II radio sits in a **~0.08–0.11 Mbps** band. IEEE Figs. 6–10 are several Mbps; do not compare those paper numbers to this table.",
                _src_note(table2),
            ]
        )

    if aodt_def:
        st = _load_raw_stats(aodt_def)
        aodt_label = "joint PSO, pooled TD3" if "td3" in st else "joint PSO, no TD3"
        snap_lines.append(f"- `aodt-compare` default I=10 J=3 ({aodt_label}): {ranking_line(st, {**LABEL, 'pso': 'PSO-joint'})}")
        artifacts.append(f"- aodt default: `{_rel(aodt_def)}`")
        updates["aodt_default"] = "\n".join(table_aodt_default(st) + ["", _src_note(aodt_def)])

    if aodt_j:
        artifacts.append(f"- aodt vs J: `{_rel(aodt_j)}`")
        rows = _read_csv(aodt_j)
        updates["aodt_vs_j"] = "\n".join(
            table_vs_axis(rows, "n_uav", "J", {3.0: "fig6_j3", 5.0: "fig6_j5"})
            + [
                "",
                "Paper quotes **J=5** in the IEEE text. J=3 paper cells are the audit’s Fig. 6 read (not a printed table).",
                _src_note(aodt_j),
            ]
        )

    if aodt_i:
        artifacts.append(f"- aodt vs I: `{_rel(aodt_i)}`")
        rows = _read_csv(aodt_i)
        updates["aodt_vs_i"] = "\n".join(
            table_vs_axis(rows, "n_iot", "I", {32.0: "fig7_i32"})
            + [
                "",
                "Paper quotes **I=32** (Fig. 7). A shared `B_sys` simplex does not make this repo’s sum rate grow with I.",
                _src_note(aodt_i),
            ]
        )

    if sweep_cmp:
        st = _load_raw_stats(sweep_cmp)
        artifacts.append(f"- sweeps default comparison: `{_rel(sweep_cmp)}`")
        updates["sweeps_default"] = "\n".join(
            table_compare(st, "fig6_j3", "placement", "Paper Fig. 6 at J=3 (Mbps)")
            + ["", _src_note(sweep_cmp)]
        )

    if sweep_lam:
        artifacts.append(f"- λ sweep: `{_rel(sweep_lam)}`")
        rows = _read_csv(sweep_lam)
        methods = tuple(m for m in METHOD_ORDER if any(r["method"] == m for r in rows))
        updates["sweeps_lambda"] = "\n".join(
            table_vs_axis(rows, "lambda_i", r"$\lambda$", {3.5: "fig8_lam35"}, methods=methods)
            + ["", "Paper quote is at $\\lambda=3.5$. SCA 8.9 Mbps is the arXiv (no-TD3) optimized line.", _src_note(sweep_lam)]
        )

    if sweep_tk:
        artifacts.append(f"- $T_k$ sweep: `{_rel(sweep_tk)}`")
        rows = _read_csv(sweep_tk)
        methods = tuple(m for m in METHOD_ORDER if any(r["method"] == m for r in rows))
        updates["sweeps_tk"] = "\n".join(
            table_vs_axis(rows, "aodt_threshold", r"$T_k$ (s)", {3.0: "fig9_tk3"}, methods=methods)
            + ["", "Paper quote is at $T_k=3$ s. SCA 7.8 Mbps is the arXiv optimized line.", _src_note(sweep_tk)]
        )

    if sweep_cpu:
        artifacts.append(f"- $f_j$ sweep: `{_rel(sweep_cpu)}`")
        rows = _read_csv(sweep_cpu)
        methods = tuple(m for m in METHOD_ORDER if any(r["method"] == m for r in rows))
        updates["sweeps_cpu"] = "\n".join(
            table_vs_axis(rows, "uav_cpu", r"$f_j$ (Hz)", {2.5e8: "fig10_fj250"}, methods=methods)
            + ["", "Paper quote is at 250 MHz: SCA ~7.5, TD3 ~6.7, baselines <4.5.", _src_note(sweep_cpu)]
        )

    updates["snapshot"] = "\n".join(snap_lines)
    updates["tests"] = (
        f"Command: `python -m pytest -v --tb=short`  \n"
        f"Collected test functions in `tests/test_*.py`: **{n_tests}**. "
        f"Re-run pytest locally to confirm they still pass; this cell counts definitions, not a live pytest session."
    )
    updates["meta"] = (
        f"**Last auto-refresh:** {_now()}  \n"
        f"**Repository:** `{REPO}`  \n"
        f"**Paper PDF:** `docs/A_UAV-Aided_Digital_Twin_Framework_for_IoT_Networks_With_High_Accuracy_and_Synchronization.pdf`  \n"
        f"Measured tables below marked `AUTO` are rewritten from CSVs when you run `python -m src.main` "
        f"(compare / sweeps / aodt-compare / solvers) or `python -m src.status_sync`."
    )
    updates["artifacts"] = "\n".join(artifacts)
    return {k: v for k, v in updates.items() if v}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def refresh_status_doc(
    *,
    results_dir: Path | None = None,
    doc_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    results = Path(results_dir) if results_dir else DEFAULT_RESULTS
    if not results.is_absolute():
        results = (REPO / results).resolve()
    search_root = results if results_dir else (DEFAULT_RESULTS if DEFAULT_RESULTS.exists() else results)
    doc = Path(doc_path) if doc_path else DEFAULT_DOC
    if extra:
        record_run(extra)
    if not doc.exists():
        raise FileNotFoundError(doc)
    text = doc.read_text(encoding="utf-8")
    text = ensure_markers(text, SECTION_IDS)
    updates = _build_updates(search_root)
    text = patch_sections(text, updates)
    doc.write_text(text, encoding="utf-8")
    return doc


def maybe_refresh(args: Any, extra: dict[str, Any] | None = None) -> None:
    if getattr(args, "no_status_sync", False):
        return
    out = getattr(args, "out", None)
    refresh_status_doc(results_dir=Path(out) if out else DEFAULT_RESULTS, extra=extra)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rewrite AUTO tables in PROJECT_STATUS_AND_PAPER_ANALYSIS.md")
    p.add_argument("--results", type=str, default=str(DEFAULT_RESULTS))
    p.add_argument("--doc", type=str, default=str(DEFAULT_DOC))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = refresh_status_doc(results_dir=Path(args.results), doc_path=Path(args.doc))
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
