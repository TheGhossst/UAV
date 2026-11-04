"""Journal-column line graphs for the 10 MHz, 25% cap sweeps in docs/results_new.md.

Reads ``results/run_10mhz_cap25/{100m,500m}/campaign_all.csv``. The difference
panel is sum rate minus SCA along the UAV sweep. p-median is the J=3 control
in ``results/run_10mhz_controls/cap25/``; it is not a sweep.

The AoDT axis starts at 1.2 s. Points below that are not plotted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inside_legend import place_inside_legend

# IEEE two-column journals: one column is 3.5 in. Fonts are sized for that print width.
FIGSIZE = (3.5, 2.55)

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
OUT = ROOT / "results" / "figures" / "results_new_lines"

FIELDS = ("100m", "500m")

# Draw low curves first so the ranking curves sit on top.
SERIES = (
    ("random", "Random", "#2ca02c", "^", "-", 1.15, 4.2),
    ("kmeans", "K-means", "#ff7f0e", "s", "-", 1.15, 3.6),
    ("sca", "SCA", "#1f77b4", "o", "-", 1.35, 4.2),
    ("sca_multistart", "SCA multi-start", "#8c564b", "v", "-", 1.25, 4.2),
    ("sca_anchor", "Zenith anchor", "#c51b8a", "*", "-", 1.35, 6.5),
)

SWEEPS = (
    ("iots", "Number of IoT devices", "Sum rate vs IoT count", False),
    ("uavs", "Number of UAVs", "Sum rate vs UAV count", False),
    ("lambda", "Task arrival rate (tasks/s)", "Sum rate vs task arrival rate", False),
    ("aodt", "AoDT threshold (s)", "Sum rate vs AoDT threshold", True),
    ("cpu", r"UAV CPU ($\times 10^{8}$ cycles/s)", "Sum rate vs UAV CPU", False),
)

DELTA_SERIES = (
    ("sca_multistart", "SCA multi-start", "#8c564b", "v", 4.2),
    ("sca_anchor", "Zenith anchor", "#c51b8a", "*", 6.5),
)


def _read_campaign(field: str) -> list[dict]:
    path = ROOT / "results" / "run_10mhz_cap25" / field / "campaign_all.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_medoid_j3(field: str) -> float:
    path = (
        ROOT
        / "results"
        / "run_10mhz_controls"
        / "cap25"
        / field
        / "n100_medoid_summary.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    if row["method"] != "sca_medoid":
        raise ValueError(f"unexpected medoid summary in {path}")
    return float(row["mean_Mbps"])


def _x_value(axis: str, raw: float) -> float:
    if axis == "cpu":
        return raw / 1e8
    return raw


def _series(rows: list[dict], axis: str, method: str) -> tuple[list[float], list[float]]:
    pts = []
    for row in rows:
        if row["axis"] != axis or row["method"] != method:
            continue
        if axis == "aodt" and float(row["feasible_fraction"]) <= 0.0:
            continue
        pts.append((_x_value(axis, float(row["x"])), float(row["mean_Mbps"])))
    pts.sort(key=lambda item: item[0])
    if not pts:
        raise ValueError(f"no points for {axis} / {method}")
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def _ylim_from_smallest(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0.0:
        span = max(abs(lo) * 0.02, 0.01)
    # A few percent keeps markers inside the closed frame.
    return lo - 0.07 * span, hi + 0.10 * span


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle=":", color="0.75", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
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
        ncol_options=(2, 1),
        fontsize=5.5,
        markerscale=0.6,
        handlelength=1.15,
        columnspacing=0.45,
        handletextpad=0.25,
        borderpad=0.2,
        labelspacing=0.15,
        frame_linewidth=0.45,
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_sweep(field: str, rows: list[dict], axis: str, xlabel: str, title: str) -> None:
    del title  # captions live outside the figure
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ys_all: list[float] = []
    xs_ref: list[float] | None = None
    for method, label, color, marker, linestyle, lw, ms in SERIES:
        xs, ys = _series(rows, axis, method)
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
            markersize=ms,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=3,
        )
    ax.set_ylim(*_ylim_from_smallest(ys_all))
    if xs_ref is not None:
        ax.set_xticks(xs_ref)
        _frame_xlim(ax, xs_ref)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel("Sum rate (Mbps)", labelpad=2)
    _style_ax(ax)
    _legend(fig, ax)
    _save(fig, OUT / field / f"{axis}_sum_rate")


def _plot_delta(field: str, rows: list[dict], medoid_mbps: float) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    xs_ref, sca = _series(rows, "uavs", "sca")
    sca_at = dict(zip(xs_ref, sca))
    plotted: list[float] = [0.0]
    ax.axhline(
        0.0,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.1,
        label="SCA",
        zorder=2,
    )
    for method, label, color, marker, ms in DELTA_SERIES:
        xs, ys = _series(rows, "uavs", method)
        delta = [y - sca_at[x] for x, y in zip(xs, ys)]
        plotted.extend(delta)
        ax.plot(
            xs,
            delta,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.35,
            markersize=ms,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=3,
        )
    if 3.0 not in sca_at:
        raise ValueError(f"{field}: UAV sweep has no J = 3 SCA point")
    medoid_delta = medoid_mbps - sca_at[3.0]
    plotted.append(medoid_delta)
    ax.scatter(
        [3.0],
        [medoid_delta],
        s=28,
        marker="D",
        color="#17becf",
        edgecolors="black",
        linewidths=0.45,
        zorder=4,
        label="p-median (J = 3 only)",
    )
    lo, hi = _ylim_from_smallest(plotted)
    # Keep the SCA line (0) inside the frame when every other curve is above it.
    ax.set_ylim(min(lo, -0.04 * (hi - lo if hi > lo else 0.05)), hi)
    ax.set_xticks(xs_ref)
    _frame_xlim(ax, xs_ref)
    ax.set_xlabel("Number of UAVs", labelpad=2)
    ax.set_ylabel("Sum rate minus SCA (Mbps)", labelpad=2)
    _style_ax(ax)
    _legend(fig, ax)
    _save(fig, OUT / field / "delta_vs_sca")


def main() -> None:
    for field in FIELDS:
        rows = _read_campaign(field)
        for axis, xlabel, title, _aodt in SWEEPS:
            _plot_sweep(field, rows, axis, xlabel, title)
        _plot_delta(field, rows, _read_medoid_j3(field))
        print(f"wrote {OUT / field}")


if __name__ == "__main__":
    main()
