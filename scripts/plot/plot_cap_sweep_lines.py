"""Line graphs of the 10 MHz cap sweep: one figure per method, every cap on it.

Reads ``results/run_10mhz_cap_sweep/{nocap,cap10,...,cap25}/{100m,500m}/campaign_all.csv``.
The y-axis starts at the minimum plotted sum rate. No title; the method and field
are the output path.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inside_legend import place_inside_legend

FIGSIZE = (3.5, 2.7)

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "run_10mhz_cap_sweep"
OUT = ROOT / "results" / "figures" / "cap_sweep"

FIELDS = ("100m", "500m")

# Legend order. Short labels so the box can sit inside the axes.
CAPS = (
    ("nocap", "No bound", "#222222", "o", "--", 1.25),
    ("cap10", "10%", "#d62728", "s", "-", 1.2),
    ("cap12", "12%", "#ff7f0e", "^", "-", 1.2),
    ("cap15", "15%", "#2ca02c", "D", "-", 1.15),
    ("cap18", "18%", "#17becf", "v", "-", 1.2),
    ("cap20", "20%", "#1f77b4", "P", "-", 1.2),
    ("cap22", "22%", "#9467bd", "X", "-", 1.2),
    ("cap25", "25%", "#8c564b", "*", "-", 1.25),
)

METHODS = (
    "random",
    "kmeans",
    "sca",
    "sca_multistart",
    "sca_anchor",
)

SWEEPS = (
    ("uavs", "Number of UAVs", False),
    ("iots", "Number of IoT devices", False),
    ("lambda", "Task arrival rate (tasks/s)", False),
    ("aodt", "AoDT threshold (s)", True),
    ("cpu", r"UAV CPU ($\times 10^{8}$ cycles/s)", False),
)


def _read(cap: str, field: str) -> list[dict]:
    path = DATA / cap / field / "campaign_all.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _x_value(axis: str, raw: float) -> float:
    if axis == "cpu":
        return raw / 1e8
    return raw


def _series(rows: list[dict], axis: str, method: str) -> tuple[list[float], list[float]]:
    pts = []
    for row in rows:
        if row["axis"] != axis or row["method"] != method:
            continue
        # The 0.8 s AoDT point is infeasible on every cap. Leave it off the axis.
        if axis == "aodt" and float(row["feasible_fraction"]) <= 0.0:
            continue
        pts.append((_x_value(axis, float(row["x"])), float(row["mean_Mbps"])))
    pts.sort(key=lambda item: item[0])
    if not pts:
        raise ValueError(f"no points for {axis} / {method}")
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def _style_ax(ax: plt.Axes) -> None:
    ax.tick_params(
        labelsize=7,
        direction="in",
        top=True,
        right=True,
        length=3.0,
        width=0.6,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)


def _frame_xlim(ax: plt.Axes, xs: list[float]) -> None:
    lo, hi = xs[0], xs[-1]
    span = hi - lo
    if span <= 0.0:
        span = 1.0
    pad = 0.08 * span
    ax.set_xlim(lo - pad, hi + pad)


def _legend(fig: plt.Figure, ax: plt.Axes) -> None:
    place_inside_legend(
        fig,
        ax,
        ncol_options=(4, 2),
        fontsize=6.0,
        markerscale=0.7,
        handlelength=1.2,
        columnspacing=0.55,
        handletextpad=0.25,
        borderpad=0.2,
        labelspacing=0.15,
        frame_linewidth=0.5,
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot(field: str, by_cap: dict[str, list[dict]], axis: str, xlabel: str, method: str) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ys_all: list[float] = []
    xs_ref: list[float] | None = None
    for cap, label, color, marker, linestyle, lw in CAPS:
        xs, ys = _series(by_cap[cap], axis, method)
        xs_ref = xs
        ys_all.extend(ys)
        ax.plot(
            xs,
            ys,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=lw,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=3,
        )
    lo = min(ys_all)
    hi = max(ys_all)
    span = hi - lo
    if span <= 0.0:
        span = max(abs(lo) * 0.02, 0.01)
    # Opens at the minimum sum rate. A few percent keeps the lowest and highest markers inside the box.
    ax.set_ylim(lo - 0.06 * span, hi + 0.10 * span)
    if xs_ref is not None:
        ax.set_xticks(xs_ref)
        _frame_xlim(ax, xs_ref)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel("Sum rate (Mbps)", labelpad=2)
    _style_ax(ax)
    _legend(fig, ax)
    _save(fig, OUT / field / axis / method)


def main() -> None:
    for field in FIELDS:
        by_cap = {cap: _read(cap, field) for cap, *_rest in CAPS}
        for axis, xlabel, _aodt in SWEEPS:
            for method in METHODS:
                _plot(field, by_cap, axis, xlabel, method)
        print(f"wrote {OUT / field}")


if __name__ == "__main__":
    main()
