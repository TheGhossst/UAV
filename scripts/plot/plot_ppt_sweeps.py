"""PPT 2x2 sweep figure: mean sum rate, tight y-axis, legend on each panel.

Highest curve wins. No error bars (seed σ hides ranking). No CPU axis.
UAV panel overlays zenith-anchor from campaign_sca_anchor_uavs.json.

Writes results/figures/ppt_all_methods/sweeps_overview.{png,pdf}.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METHOD_ORDER = ("sca_anchor", "sca", "td3", "kmeans", "random")
LABELS = {
    "sca_anchor": "SCA zenith-anchor",
    "sca": "SCA",
    "td3": "TD3",
    "kmeans": "K-means",
    "random": "Random",
}
STYLES = {
    "sca_anchor": {
        "color": "#c51b8a",
        "marker": "*",
        "markersize": 10,
        "linewidth": 2.4,
        "zorder": 6,
    },
    "sca": {
        "color": "#08519c",
        "marker": "o",
        "markersize": 7,
        "linewidth": 2.6,
        "zorder": 5,
    },
    "td3": {
        "color": "#2ca6c9",
        "marker": "P",
        "markersize": 7,
        "linewidth": 2.0,
        "zorder": 4,
    },
    "kmeans": {
        "color": "#e6550d",
        "marker": "s",
        "markersize": 6,
        "linewidth": 1.9,
        "zorder": 3,
    },
    "random": {
        "color": "#31a354",
        "marker": "^",
        "markersize": 7,
        "linewidth": 1.9,
        "zorder": 2,
    },
}

PANELS = (
    ("uavs", "Number of UAVs ($J$)", r"$J$ sweep  ($I=10$)", True),
    ("iots", "Number of IoT devices ($I$)", r"$I$ sweep  ($J=3$)", False),
    ("lambda", r"Arrival rate $\lambda$ (tasks/s)", r"$\lambda$ sweep  ($I=10$, $J=3$)", False),
    ("aodt", r"AoDT threshold $T_k$ (s)", r"$T_k$ sweep  ($I=10$, $J=3$)", False),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _axis_points(campaign: dict, axis: str) -> list[dict]:
    pts = [p for p in campaign["points"] if p["axis"] == axis]
    pts.sort(key=lambda p: p["x"])
    return pts


def _means_by_x(points: list[dict], method: str) -> dict[float, float]:
    out: dict[float, float] = {}
    for p in points:
        if method in p["by_method"]:
            out[float(p["x"])] = float(p["by_method"][method]["mean_sum_rate_Mbps"])
    return out


def _panel_series(
    td3: dict,
    anchor: dict,
    axis: str,
    include_anchor: bool,
) -> tuple[list[float], dict[str, list[float]]]:
    pts = _axis_points(td3, axis)
    xs = [float(p["x"]) for p in pts]
    series: dict[str, list[float]] = {}
    for method in METHOD_ORDER:
        if method == "sca_anchor":
            if not include_anchor:
                continue
            by_x = _means_by_x(_axis_points(anchor, "uavs"), "sca_anchor")
            if not all(x in by_x for x in xs):
                continue
            series[method] = [by_x[x] for x in xs]
            continue
        if method not in td3.get("methods", []):
            continue
        series[method] = [
            float(p["by_method"][method]["mean_sum_rate_Mbps"]) for p in pts
        ]
    return xs, series


def _set_xticks(ax, xs: list[float], axis: str) -> None:
    ax.set_xticks(xs)
    if axis in ("uavs", "iots"):
        ax.set_xticklabels([str(int(round(x))) for x in xs])
    elif axis in ("lambda", "aodt"):
        ax.set_xticklabels([f"{x:g}" for x in xs])


def _xlim_padded(xs: list[float]) -> tuple[float, float]:
    if len(xs) <= 1:
        return xs[0] - 0.5, xs[0] + 0.5
    span = xs[-1] - xs[0]
    pad = max(0.10 * span, 0.12)
    return xs[0] - pad, xs[-1] + pad


def _legend_loc(axis: str) -> str:
    """Keep legend away from the right edge (avoids clip in Beamer)."""
    if axis in ("uavs", "aodt"):
        return "lower right"
    return "upper left"


def main() -> int:
    td3 = _load(ROOT / "results" / "campaign_8.8mhz_cap25_td3.json")
    anchor = _load(ROOT / "results" / "campaign_sca_anchor_uavs.json")
    out_dir = ROOT / "results" / "figures" / "ppt_all_methods"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 5.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.11, hspace=0.40, wspace=0.32)
    axes_flat = axes.flatten()

    for ax, (axis, xlabel, title, include_anchor) in zip(axes_flat, PANELS):
        xs, series = _panel_series(td3, anchor, axis, include_anchor)
        all_y: list[float] = []
        for method, ys in series.items():
            all_y.extend(ys)
            st = STYLES[method]
            ax.plot(xs, ys, label=LABELS[method], **st)

        y_min = min(all_y)
        y_max = max(all_y)
        span = max(y_max - y_min, 0.012)
        ax.set_ylim(y_min - 0.04 * span, y_max + 0.16 * span)
        x0, x1 = _xlim_padded(xs)
        ax.set_xlim(x0, x1)
        ax.margins(x=0)
        _set_xticks(ax, xs, axis)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Sum rate (Mbps)", fontsize=9)
        ax.set_title(title, fontsize=10, pad=4)
        ax.grid(True, alpha=0.28, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(
            loc=_legend_loc(axis),
            framealpha=0.95,
            edgecolor="0.85",
            fontsize=6.8,
            handlelength=1.5,
            borderpad=0.22,
            labelspacing=0.18,
        )

    stem = out_dir / "sweeps_overview"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(stem.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
