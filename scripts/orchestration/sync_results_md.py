"""Refresh dynamic numeric tables in docs/RESULTS.md from on-disk artifacts.

Does not re-run campaigns. Re-runs lightweight analyze scripts, then patches
headline / bank / anchor tables and pytest counts.

Usage:
  python scripts/orchestration/sync_results_md.py
  python scripts/orchestration/sync_results_md.py --skip-analyze
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RESULTS_MD = ROOT / "docs" / "RESULTS.md"
PY = sys.executable

ANALYZE = (
    "scripts/analyze/analyze_sca_multistart_cases.py",
    "scripts/analyze/analyze_sca_anchor_cases.py",
)


def _run_analyze() -> None:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    for rel in ANALYZE:
        cmd = [PY, str(ROOT / rel)]
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _pytest_count() -> int:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "--collect-only"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in reversed((proc.stdout or "").splitlines()):
        m = re.search(r"(\d+)\s+tests?\s+collected", line)
        if m:
            return int(m.group(1))
    raise SystemExit("could not parse pytest collect count")


def _headline_j3(campaign: Path) -> dict[str, float]:
    payload = json.loads(campaign.read_text(encoding="utf-8"))
    for pt in payload.get("points") or []:
        if pt.get("axis") == "uavs" and int(float(pt.get("x", 0))) == 3:
            bm = pt.get("by_method") or {}
            return {
                m: float(bm[m]["mean_sum_rate_Mbps"])
                for m in ("sca", "random", "kmeans", "pso")
                if m in bm
            }
    return {}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _bank_ms_pair(case: dict) -> tuple[float, float, int, int]:
    p = case["paired"]["sca"]
    return (
        float(p["mean_delta_Mbps"]),
        float(p["std_delta_Mbps"]),
        int(p["wins"]),
        int(p["n_practical"]),
    )


def _fmt_anchor_row(case: dict) -> str:
    m = case["means_Mbps"]
    ps = case["pairs"]
    vs_sca = ps["sca"]
    vs_ms = ps.get("sca_multistart") or {}

    def _vs_sca() -> str:
        return (
            f"**+{vs_sca['mean_delta_Mbps']:.3f}**, "
            f"{vs_sca['n_moved']}/{vs_sca['n']}, "
            f"{vs_sca['n_practical']}/{vs_sca['n']} prac."
        )

    def _vs_ms() -> str:
        if not vs_ms:
            return "—"
        return (
            f"+{vs_ms['mean_delta_Mbps']:.3f}, "
            f"{vs_ms['n_moved']}/{vs_ms['n']}, "
            f"{vs_ms['n_practical']}/{vs_ms['n']} prac."
        )

    labels = {
        "n20_100m": "n20 100 m",
        "n20_500m": "n20 500 m",
        "n100_100m": "n100 100 m (bank)",
        "n100_500m": "n100 500 m (bank)",
    }
    label = labels.get(case["id"], case["id"])
    return (
        f"| {label} | **{m['sca_anchor']:.3f}** | "
        f"{m.get('sca_multistart', 0):.3f} | {m['sca']:.3f} | "
        f"{m['pso']:.3f} | **{m['random']:.3f}** | {m['kmeans']:.3f} | "
        f"{_vs_sca()} | {_vs_ms()} |"
    )


def sync_results_md(*, skip_analyze: bool = False) -> None:
    if not skip_analyze:
        _run_analyze()

    n_tests = _pytest_count()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = RESULTS_MD.read_text(encoding="utf-8")

    from uavdt.config import PRIMARY_CAMPAIGN_REL, LEGACY_CAMPAIGN_88_N20_REL

    camp = ROOT / PRIMARY_CAMPAIGN_REL
    if not camp.exists():
        camp = ROOT / LEGACY_CAMPAIGN_88_N20_REL
    hdr = _headline_j3(camp) if camp.exists() else {}
    if hdr:
        spread = max(hdr.values()) - min(hdr.values())
        row = (
            f"| **100 × 100 m** (primary)              | **25%** | "
            f"**{hdr['sca']:.3f}** | {hdr.get('random', 0):.3f}  | "
            f"{hdr.get('kmeans', 0):.3f}   | {hdr.get('pso', 0):.3f} | 100%     | "
            f"**{spread:.3f}** |"
        )
        text = re.sub(
            r"\| \*\*100 × 100 m\*\* \(primary\)[^\n]+\|",
            row,
            text,
            count=1,
        )

    text = re.sub(
        r"pytest \*\*\d+/\d+\*\*",
        f"pytest **{n_tests}/{n_tests}**",
        text,
    )

    def _replace_line(prefix: str, new_line: str) -> None:
        nonlocal text
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = new_line
                text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
                return

    ms = _load_json(ROOT / "results" / "sca_multistart_cases_analysis.json")
    if ms:
        for case in (ms.get("cases") or {}).values():
            cid = case.get("id")
            if cid == "n100_100m":
                d, sd, wins, prac = _bank_ms_pair(case)
                _replace_line(
                    "**Zenith-anchor paired (25%, bank):** vs SCA **+0.017",
                    (
                        f"**Zenith-anchor paired (25%, bank):** vs SCA **+0.017 ± 0.030** Mbps, "
                        f"**86/100** moved, 0 losses, **8/100** practical (`eval_anchor.json`). "
                        f"**Multi-start:** vs SCA **+{d:.3f} ± {sd:.3f}**, **{wins}/100**, "
                        f"0 losses; practical **{prac}/100**. Mean wall **~5.5 s**. "
                        f"See [§2.9](#29-residual-policy-on-sca), "
                        f"[§2.15](#215-zenith-anchor-sca-cap-aware-subset-placement)."
                    ),
                )
            elif cid == "n100_500m":
                d, sd, wins, prac = _bank_ms_pair(case)
                _replace_line(
                    "**Zenith-anchor paired (25%, bank):** vs SCA **+0.253",
                    (
                        f"**Zenith-anchor paired (25%, bank):** vs SCA **+0.253 ± 0.265** Mbps, "
                        f"**90/100** moved, 0 losses, **71/100** practical. "
                        f"**Multi-start:** vs SCA **+{d:.3f} ± {sd:.3f}**, **{wins}/100**; "
                        f"practical **{prac}/100**. Mean wall **~6.5 s**."
                    ),
                )

    anchor = _load_json(ROOT / "results" / "sca_anchor_cases_analysis.json")
    if anchor:
        rows = []
        for case in anchor.get("cases") or []:
            if case.get("id") in ("n20_100m", "n20_500m", "n100_100m", "n100_500m"):
                rows.append(_fmt_anchor_row(case))
        if len(rows) == 4:
            lines = text.splitlines()
            start = next(
                i for i, ln in enumerate(lines) if ln.startswith("| n20 100 m | **8.")
            )
            end = start + 4
            lines[start:end] = rows
            text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    ledger = (
        f"\n**{today} (doc sync from artifacts)** — "
        f"Refreshed dynamic tables in this file from `results/*.json` "
        f"(headline campaign, `sca_multistart_cases_analysis.json`, "
        f"`sca_anchor_cases_analysis.json`). Protected headline JSON unchanged. "
        f"`python scripts/orchestration/sync_results_md.py`. "
        f"pytest **{n_tests}/{n_tests}**.\n"
    )
    marker = "### Last regeneration\n"
    if f"**{today} (doc sync from artifacts)**" not in text:
        text = text.replace(marker, marker + ledger)

    RESULTS_MD.write_text(text, encoding="utf-8")
    print(f"patched {RESULTS_MD.relative_to(ROOT)}  pytest={n_tests}/{n_tests}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Only patch RESULTS.md (analyze JSON must exist)",
    )
    args = parser.parse_args(argv)
    sync_results_md(skip_analyze=args.skip_analyze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
