"""Patch docs/results_new.md zenith-anchor tables from cap25 artifacts (dynamic only)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAP25 = ROOT / "results" / "run_10mhz_cap25"
RESULTS_MD = ROOT / "docs" / "results_new.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(payload: dict, method: str = "sca_anchor") -> tuple[float, float]:
    b = payload["by_method"][method]
    return float(b["mean_sum_rate_Mbps"]), float(b["std_sum_rate_Mbps"])


def campaign_means(path: Path) -> dict[tuple[str, float], float]:
    out: dict[tuple[str, float], float] = {}
    for pt in load(path).get("points") or []:
        bm = (pt.get("by_method") or {}).get("sca_anchor")
        if bm is None:
            continue
        out[(str(pt["axis"]), float(pt["x"]))] = float(bm["mean_sum_rate_Mbps"])
    return out


def cap_anchor_mean(cap_slug: str, field: str) -> float:
    if cap_slug == "cap25":
        return stats(load(CAP25 / field / "n100_sca_anchor.json"))[0]
    summ = (
        ROOT
        / "results"
        / "run_10mhz_cap_sweep"
        / cap_slug
        / field
        / "n100_all_summary.csv"
    )
    with summ.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["method"] == "sca_anchor":
                return float(row["mean_Mbps"])
    raise FileNotFoundError(summ)


def sweep_rows(field: str, axis: str, anchor: dict) -> list[str]:
    rows = list(
        csv.DictReader(
            (CAP25 / field / "campaign_all.csv").open(newline="", encoding="utf-8")
        )
    )
    xs = sorted(
        float(r["x"])
        for r in rows
        if r["axis"] == axis and r["method"] == "sca"
    )
    body: list[str] = []
    for x in xs:
        key = (axis, x)
        base = {
            m: float(
                next(
                    r["mean_Mbps"]
                    for r in rows
                    if r["axis"] == axis and float(r["x"]) == x and r["method"] == m
                )
            )
            for m in ("random", "kmeans", "sca", "sca_multistart")
        }
        xlab = f"{int(x)}" if axis in {"uavs", "iots"} else f"{x:g}"
        if axis == "cpu":
            xlab = f"{x / 1e8:g}"
        body.append(
            f"| {xlab} | {base['random']:.3f} | {base['kmeans']:.3f} | "
            f"{base['sca']:.3f} | {base['sca_multistart']:.3f} | {anchor[key]:.3f} |"
        )
    return body


def sweep_rows_aodt(field: str, anchor: dict) -> list[str]:
    rows = list(
        csv.DictReader(
            (CAP25 / field / "campaign_all.csv").open(newline="", encoding="utf-8")
        )
    )
    xs = sorted(
        float(r["x"])
        for r in rows
        if r["axis"] == "aodt" and r["method"] == "sca"
    )
    body: list[str] = []
    for x in xs:
        key = ("aodt", x)
        base = {
            m: float(
                next(
                    r["mean_Mbps"]
                    for r in rows
                    if r["axis"] == "aodt" and float(r["x"]) == x and r["method"] == m
                )
            )
            for m in ("random", "kmeans", "sca", "sca_multistart")
        }
        fracs = {
            m: float(
                next(
                    r["feasible_fraction"]
                    for r in rows
                    if r["axis"] == "aodt"
                    and float(r["x"]) == x
                    and r["method"] == m
                )
            )
            for m in ("random", "kmeans", "sca", "sca_multistart")
        }
        if all(v <= 0 for v in fracs.values()):
            note = "**0%** all methods"
        elif all(v >= 1 for v in fracs.values()):
            note = "100%"
        else:
            short = {"random": "random", "kmeans": "k-means", "sca": "SCA", "sca_multistart": "multi-start"}
            partial = [
                f"{short[m]} **{fracs[m] * 100:.0f}%**"
                for m in fracs
                if fracs[m] < 1
            ]
            note = "; ".join(partial) + "; others 100%"
        body.append(
            f"| {x:g} | {base['random']:.3f} | {base['kmeans']:.3f} | "
            f"{base['sca']:.3f} | {base['sca_multistart']:.3f} | "
            f"{anchor[key]:.3f} | {note} |"
        )
    return body


def replace_table_after(text: str, section_title: str, header: str, body: list[str]) -> str:
    pos = text.find(section_title)
    if pos < 0:
        raise ValueError(section_title)
    sub = text[pos:]
    m = re.search(r"\n(\|[^\n]+\|\n\| --- \|[^\n]*\n)", sub)
    if not m:
        raise ValueError(f"table in {section_title}")
    start = pos + m.start() + 1
    end = text.find("\n\n", start)
    if end < 0:
        end = len(text)
    table = header + "\n" + "\n".join(body) + "\n"
    return text[:start] + table + text[end:]


def main() -> None:
    text = RESULTS_MD.read_text(encoding="utf-8")
    anchor_c = {
        "100m": campaign_means(CAP25 / "100m" / "campaign_sca_anchor.json"),
        "500m": campaign_means(CAP25 / "500m" / "campaign_sca_anchor.json"),
    }

    hdr = (
        "| J | Random | K-means | SCA | Multi-start | Anchor |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    sections = [
        ("### 100 × 100 m — UAV count J", "uavs", "100m", hdr),
        ("### 100 × 100 m — IoT count I", "iots", "100m", hdr.replace("| J |", "| I |")),
        (
            "### 100 × 100 m — arrival rate λ",
            "lambda",
            "100m",
            hdr.replace("| J |", "| λ (/s) |"),
        ),
        (
            "### 100 × 100 m — AoDT threshold T_k",
            "aodt",
            "100m",
            "| T_k (s) | Random | K-means | SCA | Multi-start | Anchor | Feasible |\n"
            "| --- | --- | --- | --- | --- | --- | --- |",
        ),
        (
            "### 100 × 100 m — UAV CPU f_j",
            "cpu",
            "100m",
            hdr.replace("| J |", "| f_j (×10⁸) |"),
        ),
        ("### 500 × 500 m — UAV count J", "uavs", "500m", hdr),
        ("### 500 × 500 m — IoT count I", "iots", "500m", hdr.replace("| J |", "| I |")),
        (
            "### 500 × 500 m — arrival rate λ",
            "lambda",
            "500m",
            hdr.replace("| J |", "| λ (/s) |"),
        ),
        (
            "### 500 × 500 m — AoDT threshold T_k",
            "aodt",
            "500m",
            "| T_k (s) | Random | K-means | SCA | Multi-start | Anchor | Feasible |\n"
            "| --- | --- | --- | --- | --- | --- | --- |",
        ),
        (
            "### 500 × 500 m — UAV CPU f_j",
            "cpu",
            "500m",
            hdr.replace("| J |", "| f_j (×10⁸) |"),
        ),
    ]
    for title, axis, field, header in sections:
        if axis == "aodt":
            body = sweep_rows_aodt(field, anchor_c[field])
        else:
            body = sweep_rows(field, axis, anchor_c[field])
        text = replace_table_after(text, title, header, body)

    cap_hdr = (
        "| Cap | Random | K-means | SCA | Multi-start | Anchor |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    caps = (
        ("none", "nocap"),
        ("10%", "cap10"),
        ("12%", "cap12"),
        ("15%", "cap15"),
        ("18%", "cap18"),
        ("20%", "cap20"),
        ("22%", "cap22"),
        ("25%", "cap25"),
    )

    def cap_body(slug: str) -> list[str]:
        body: list[str] = []
        for label, cap_slug in caps:
            cap_rows = list(
                csv.DictReader(
                    (
                        ROOT
                        / "results"
                        / "run_10mhz_cap_sweep"
                        / cap_slug
                        / slug
                        / "n100_all_summary.csv"
                    ).open(newline="", encoding="utf-8")
                )
            )
            base = {
                m: float(next(r["mean_Mbps"] for r in cap_rows if r["method"] == m))
                for m in ("random", "kmeans", "sca", "sca_multistart")
            }
            am = cap_anchor_mean(cap_slug, slug)
            body.append(
                f"| {label} | {base['random']:.3f} | {base['kmeans']:.3f} | "
                f"{base['sca']:.3f} | {base['sca_multistart']:.3f} | {am:.3f} |"
            )
        return body

    cap_start = text.index("## Cap family")
    cap_chunk = text[cap_start:]
    for slug, marker in (
        ("100m", "### 100 × 100 m\n\n\n"),
        ("500m", "### 500 × 500 m\n\n\n"),
    ):
        rel = cap_chunk.index(marker) + len(marker)
        sub = cap_chunk[rel:]
        end = sub.index("\n\n")
        new_block = cap_hdr + "\n" + "\n".join(cap_body(slug)) + "\n"
        cap_chunk = cap_chunk[:rel] + new_block + sub[end + 2 :]
    text = text[:cap_start] + cap_chunk

    RESULTS_MD.write_text(text, encoding="utf-8")
    print("updated", RESULTS_MD)


if __name__ == "__main__":
    main()
