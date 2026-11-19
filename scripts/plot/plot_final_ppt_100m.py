"""16:10 line graphs for the end-semester deck. 100 m, 10 MHz, 25% bound.

Reads ``results/run_10mhz_cap25/100m/campaign_all.csv``.
The AoDT axis starts at 1.2 s.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inside_legend import place_inside_legend

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "run_10mhz_cap25" / "100m" / "campaign_all.csv"
OUT = ROOT / "latex" / "final_ppt" / "images"

SERIES = (
    ("random", "Random", "#2ca02c", "o"),
    ("kmeans", "K-means", "#ff7f0e", "s"),
    ("sca", "SCA", "#1f77b4", "^"),
    ("sca_multistart", "Multi-start", "#8c564b", "D"),
    ("sca_anchor", "Zenith-subset", "#c51b8a", "*"),
)

SWEEPS = (
    ("uavs", "Number of UAVs", "uavs_100m"),
    ("iots", "Number of IoT devices", "iots_100m"),
    ("lambda", "Task arrival rate (tasks/s)", "tasks_100m"),
    ("aodt", "AoDT threshold (s)", "aodt_100m"),
    ("cpu", r"CPU cycles ($\times 10^{8}$/s)", "cpu_100m"),
)


def _read() -> list[dict]:
    with CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _x(axis: str, raw: float) -> float:
    if axis == "cpu":
        return raw / 1e8
    return raw


def _series(
    rows: list[dict],
    axis: str,
    method: str,
    ycol: str = "mean_Mbps",
) -> tuple[list[float], list[float]]:
    pts = []
    for row in rows:
        if row["axis"] != axis or row["method"] != method:
            continue
        x = _x(axis, float(row["x"]))
        if axis == "aodt" and x < 1.2 - 1e-9:
            continue
        pts.append((x, float(row[ycol])))
    pts.sort()
    if not pts:
        raise ValueError(f"no points for {axis} / {method}")
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def _plot(
    rows: list[dict],
    axis: str,
    xlabel: str,
    stem: str,
    ycol: str = "mean_Mbps",
    ylabel: str = "Sum Rate (Mbps)",
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.6, 5.15))
    ys_all: list[float] = []
    xs_ref: list[float] | None = None
    for method, label, color, marker in SERIES:
        xs, ys = _series(rows, axis, method, ycol)
        xs_ref = xs
        ys_all.extend(ys)
        ax.plot(
            xs,
            ys,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.0,
            markersize=9 if marker != "*" else 12,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        print(stem, label, [(round(x, 3), round(y, 3)) for x, y in zip(xs, ys)])
    lo, hi = min(ys_all), max(ys_all)
    span = hi - lo
    if span <= 0.0:
        span = max(abs(lo) * 0.02, 0.01)
    ax.set_ylim(lo - 0.04 * span, hi + 0.14 * span)
    if xs_ref is not None:
        ax.set_xticks(xs_ref)
        xspan = xs_ref[-1] - xs_ref[0]
        if xspan <= 0.0:
            xspan = 1.0
        ax.set_xlim(xs_ref[0] - 0.06 * xspan, xs_ref[-1] + 0.06 * xspan)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", color="0.75", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True, length=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    place_inside_legend(
        fig,
        ax,
        ncol_options=(3, 2),
        fontsize=11,
        markerscale=0.85,
        handlelength=1.6,
        columnspacing=0.8,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
        frame_linewidth=0.7,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    rows = _read()
    for axis, xlabel, stem in SWEEPS:
        _plot(rows, axis, xlabel, stem)
    _plot(rows, "uavs", "Number of Relays", "uavs_relays_100m")


if __name__ == "__main__":
    main()
