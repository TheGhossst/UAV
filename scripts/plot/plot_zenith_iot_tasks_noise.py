"""Zenith-anchor sum rate versus IoT count under dynamic bound and dynamic noise.

Six task-arrival lines. The equal-share bound does not move with arrival
rate, and spot checks at the extreme rates match, so each line is the
100-seed mean at 2 tasks/s.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from inside_legend import place_inside_legend

ROOT = Path(__file__).resolve().parents[2]
CSV = (
    ROOT
    / "results"
    / "run_10mhz_dynamic_noise"
    / "100m"
    / "zenith_iot_tasks"
    / "summary.csv"
)

LAMBDAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
MARKERS = ("o", "s", "^", "D", "v", "*")


def _curve() -> tuple[list[float], list[float]]:
    pts = []
    with CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pts.append((float(row["num_iot"]), float(row["mean_Mbps"])))
    pts.sort()
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def _draw(figsize: tuple[float, float], fontsize: float, legend: float) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.labelsize": fontsize + 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=figsize)
    xs, ys = _curve()
    for lam, color, marker in zip(LAMBDAS, COLORS, MARKERS):
        ax.plot(
            xs,
            ys,
            label=f"{lam:.1f} tasks/s",
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.8,
            markersize=8 if marker != "*" else 11,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    lo, hi = min(ys), max(ys)
    span = hi - lo
    ax.set_ylim(lo - 0.08 * span, hi + 0.22 * span)
    xspan = xs[-1] - xs[0]
    ax.set_xlim(xs[0] - 0.06 * xspan, xs[-1] + 0.06 * xspan)
    ax.set_xticks(xs)
    ax.set_xlabel("Number of IoT devices")
    ax.set_ylabel("Sum Rate (Mbps)")
    ax.grid(True, axis="y", linestyle=":", color="0.75", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    fig.tight_layout()
    place_inside_legend(
        fig,
        ax,
        ncol_options=(2, 1),
        fontsize=legend,
        markerscale=0.8,
        handlelength=1.5,
        columnspacing=0.7,
        handletextpad=0.35,
        borderpad=0.3,
        labelspacing=0.25,
        frame_linewidth=0.6,
    )
    return fig


def main() -> None:
    journal = ROOT / "results" / "figures" / "results_new_lines" / "100m" / "zenith_iot_tasks_noise"
    slides = ROOT / "latex" / "final_ppt" / "images" / "zenith_iot_tasks_noise"
    for path, size, font, legend in (
        (journal, (3.5, 2.55), 8, 6),
        (slides, (10.6, 5.15), 13, 11),
    ):
        fig = _draw(size, font, legend)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
