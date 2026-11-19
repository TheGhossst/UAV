"""Temporary line graphs for the unfinished 10 MHz dynamic-noise campaign.

Reads finished rows from results/run_10mhz_dynamic_noise/100m/campaign_all.csv
and any completed multi-start points in the campaign checkpoint. Zenith is
included only when its campaign file exists.

Writes only under results/run_10mhz_dynamic_noise/tmp_graphs/.
Does not touch results/figures or any existing campaign plot.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from inside_legend import place_inside_legend

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
RUN = ROOT / "results" / "run_10mhz_dynamic_noise"
OUT = RUN / "tmp_graphs"

# Low curves first so the ranking curves sit on top.
SERIES = (
    ("random", "Random", "#2ca02c", "^", "-", 1.15, 4.2),
    ("kmeans", "K-means", "#ff7f0e", "s", "-", 1.15, 3.6),
    ("sca", "SCA", "#1f77b4", "o", "-", 1.35, 4.2),
    ("pso", "PSO", "#9467bd", "D", "-", 1.15, 3.6),
    ("sca_multistart_dyn", "SCA multi-start", "#8c564b", "v", "-", 1.25, 4.2),
    ("sca_anchor", "Zenith anchor", "#c51b8a", "*", "-", 1.35, 6.5),
)

SWEEPS = (
    ("iots", "Number of IoT devices", "sum_rate"),
    ("uavs", "Number of UAVs", "sum_rate"),
    ("lambda", "Task arrival rate (tasks/s)", "sum_rate"),
    ("aodt", "AoDT threshold (s)", "sum_rate"),
    ("cpu", r"CPU cycles ($\times 10^{8}$/s)", "sum_rate"),
)


def _campaign_rows(field: str) -> list[dict]:
    path = RUN / field / "campaign_all.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    present = {row["method"] for row in rows}
    extra = _checkpoint_rows(field, "sca_multistart_dyn")
    if extra and "sca_multistart_dyn" not in present:
        rows.extend(extra)
    anchor = _finished_method_rows(field, "sca_anchor")
    if anchor and "sca_anchor" not in present:
        rows.extend(anchor)
    return rows


def _finished_method_rows(field: str, method: str) -> list[dict]:
    path = RUN / field / f"campaign_{method}.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_rows(field: str, method: str) -> list[dict]:
    path = RUN / field / f"campaign_{method}.checkpoint.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for pt in payload.get("points") or []:
        stats = (pt.get("by_method") or {}).get(method)
        if not stats:
            continue
        rows.append(
            {
                "axis": pt["axis"],
                "x_name": pt["x_name"],
                "x": pt["x"],
                "method": method,
                "mean_Mbps": stats["mean_sum_rate_Mbps"],
                "std_Mbps": stats["std_sum_rate_Mbps"],
                "feasible_fraction": stats["feasible_fraction"],
                "mean_max_AoDT_s": stats["mean_max_AoDT_s"],
                "n": stats["n"],
            }
        )
    return rows


def _x_value(axis: str, raw: float) -> float:
    if axis == "cpu":
        return raw / 1e8
    return raw


def _series(
    rows: list[dict], axis: str, method: str
) -> tuple[list[float], list[float]]:
    pts = []
    for row in rows:
        if row["axis"] != axis or row["method"] != method:
            continue
        if float(row["feasible_fraction"]) <= 0.0:
            continue
        pts.append((_x_value(axis, float(row["x"])), float(row["mean_Mbps"])))
    pts.sort(key=lambda item: item[0])
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def _ylim_from_smallest(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0.0:
        span = max(abs(lo) * 0.02, 0.01)
    return lo - 0.04 * span, hi + 0.12 * span


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


def _plot_sweep(
    field: str,
    rows: list[dict],
    axis: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ys_all: list[float] = []
    xs_ref: list[float] | None = None
    drawn = 0
    for method, label, color, marker, linestyle, lw, ms in SERIES:
        xs, ys = _series(rows, axis, method)
        if not xs:
            continue
        if xs_ref is None or len(xs) > len(xs_ref):
            xs_ref = xs
        ys_all.extend(ys)
        drawn += 1
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
    if drawn == 0 or xs_ref is None:
        plt.close(fig)
        raise ValueError(f"no points for {axis}")
    ax.set_ylim(*_ylim_from_smallest(ys_all))
    ax.set_xticks(xs_ref)
    _frame_xlim(ax, xs_ref)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel("Sum Rate (Mbps)", labelpad=2)
    _style_ax(ax)
    _legend(fig, ax)
    _save(fig, OUT / field / f"{axis}_sum_rate")
    print(f"wrote {OUT / field / f'{axis}_sum_rate'}.png ({drawn} series)")


def main() -> None:
    field = "100m"
    rows = _campaign_rows(field)
    methods = sorted({row["method"] for row in rows})
    print("methods:", ", ".join(methods))
    for axis, xlabel, _suffix in SWEEPS:
        counts: dict[str, int] = {}
        for row in rows:
            if row["axis"] == axis:
                counts[row["method"]] = counts.get(row["method"], 0) + 1
        detail = ", ".join(f"{name}={n}" for name, n in sorted(counts.items()))
        print(f"{axis}: {detail}")
        _plot_sweep(field, rows, axis, xlabel)


if __name__ == "__main__":
    main()
