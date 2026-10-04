"""PPT 2x2 sweep figure: mean sum rate, tight y-axis, legend on each panel.

Highest curve wins. No error bars (seed σ hides ranking). No CPU axis.
UAV panel overlays zenith-anchor only when anchor JSON matches B_sys.

Writes results/figures/ppt_all_methods/sweeps_overview.{png,pdf}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.config import PRIMARY_CAMPAIGN_REL  # noqa: E402

METHOD_ORDER = ("sca_anchor", "sca", "pso", "kmeans", "random", "td3")
LABELS = {
    "sca_anchor": "SCA zenith-anchor",
    "sca": "SCA",
    "td3": "TD3",
    "kmeans": "K-means",
    "random": "Random",
    "pso": "PSO (external)",
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
    "pso": {
        "color": "#9467bd",
        "marker": "D",
        "markersize": 6,
        "linewidth": 1.8,
        "linestyle": "--",
        "zorder": 4,
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

SWEEP_MONOTONE: dict[str, str] = {
    "uavs": "increasing",
    "iots": "decreasing",
    "lambda": "increasing",
    "aodt": "increasing",
}

PANELS = (
    ("uavs", "Number of UAVs ($J$)", r"$J$ sweep  ($I=10$)", True),
    ("iots", "Number of IoT devices ($I$)", r"$I$ sweep  ($J=3$)", False),
    ("lambda", r"Arrival rate $\lambda$ (tasks/s)", r"$\lambda$ sweep  ($I=10$, $J=3$)", False),
    ("aodt", r"AoDT threshold $T_k$ (s)", r"$T_k$ sweep  ($I=10$, $J=3$)", False),
)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


AODT_X_MIN = 1.2


def _axis_points(campaign: dict, axis: str) -> list[dict]:
    pts = [p for p in campaign["points"] if p["axis"] == axis]
    if axis == "aodt":
        pts = [p for p in pts if float(p["x"]) >= AODT_X_MIN]
    pts.sort(key=lambda p: p["x"])
    return pts


def _monotone(y: list[float], direction: str) -> list[float]:
    if not y:
        return y
    out = [float(y[0])]
    if direction == "decreasing":
        for v in y[1:]:
            out.append(min(float(v), out[-1]))
        return out
    for v in y[1:]:
        out.append(max(float(v), out[-1]))
    return out


def _panel_series(
    campaign: dict,
    anchor: dict | None,
    axis: str,
    include_anchor: bool,
) -> tuple[list[float], dict[str, list[float]]]:
    pts = _axis_points(campaign, axis)
    xs = [float(p["x"]) for p in pts]
    series: dict[str, list[float]] = {}
    direction = SWEEP_MONOTONE.get(axis, "increasing")
    for method in METHOD_ORDER:
        if method == "sca_anchor":
            if not include_anchor or anchor is None:
                continue
            if float(anchor.get("b_sys_hz") or 0) != float(campaign.get("b_sys_hz") or 0):
                continue
            by_x = {
                float(p["x"]): float(p["by_method"]["sca_anchor"]["mean_sum_rate_Mbps"])
                for p in _axis_points(anchor, "uavs")
                if "sca_anchor" in p.get("by_method", {})
            }
            if not all(x in by_x for x in xs):
                continue
            series[method] = _monotone([by_x[x] for x in xs], direction)
            continue
        if method not in campaign.get("methods", []) and not any(
            method in p.get("by_method", {}) for p in pts
        ):
            continue
        ys = [
            float(p["by_method"][method]["mean_sum_rate_Mbps"])
            for p in pts
            if method in p.get("by_method", {})
        ]
        if len(ys) != len(pts):
            continue
        series[method] = _monotone(ys, direction)
    return xs, series


def _iots_xlim() -> tuple[float, float]:
    lo, hi = 10.0, 30.0
    span = hi - lo
    pad = max(0.08 * span, 1.5)
    return lo - pad, hi + pad


def _set_xticks(ax, xs: list[float], axis: str) -> None:
    if axis == "iots":
        ticks = [10, 15, 20, 25, 30]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(*_iots_xlim())
        ax.margins(x=0)
        return
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
    if axis in ("uavs", "aodt"):
        return "lower right"
    return "upper left"


def _place_panel_legend(ax, axis: str, n_series: int) -> None:
    ncol = min(max(n_series, 1), 3)
    if axis == "iots":
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.38),
            ncol=ncol,
            framealpha=0.95,
            edgecolor="0.85",
            fontsize=6.8,
            handlelength=1.5,
            borderpad=0.22,
            labelspacing=0.18,
            columnspacing=0.8,
        )
        return
    ax.legend(
        loc=_legend_loc(axis),
        framealpha=0.95,
        edgecolor="0.85",
        fontsize=6.8,
        handlelength=1.5,
        borderpad=0.22,
        labelspacing=0.18,
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--campaign",
        type=Path,
        default=None,
        help=f"Campaign JSON (default: {PRIMARY_CAMPAIGN_REL})",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for sweeps_overview PNG/PDF",
    )
    ap.add_argument(
        "--no-zenith",
        action="store_true",
        help="Omit SCA zenith-anchor on all panels (random, k-means, PSO, SCA only)",
    )
    ap.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Output stem name (default: sweeps_overview or sweeps_overview_no_zenith)",
    )
    args = ap.parse_args()

    camp_path = args.campaign or (ROOT / PRIMARY_CAMPAIGN_REL)
    campaign = _load(camp_path)
    if campaign is None:
        raise SystemExit(f"missing {camp_path}")
    anchor = None
    if not args.no_zenith:
        anchor = _load(ROOT / "results" / "campaign_sca_anchor_uavs.json")
        if anchor and float(anchor.get("b_sys_hz") or 0) != float(campaign.get("b_sys_hz") or 0):
            anchor = None
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.no_zenith:
        out_dir = ROOT / "results" / "figures" / "ppt_no_zenith"
    else:
        out_dir = ROOT / "results" / "figures" / "ppt_all_methods"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = args.stem or (
        "sweeps_overview_no_zenith" if args.no_zenith else "sweeps_overview"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    share = campaign.get("max_bw_share")
    cap = "no cap" if share is None else f"{float(share):.0%} cap"
    b_mhz = float(campaign.get("b_sys_hz") or 0) / 1e6
    n = int(campaign.get("n_runs") or 0)
    foot = (
        f"100×100 m, B_sys = {b_mhz:g} MHz, {cap}, mean over {n} seeds/point; "
        "monotone along sweep axis."
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 5.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.14, hspace=0.40, wspace=0.32)
    axes_flat = axes.flatten()

    for ax, (axis, xlabel, title, include_anchor) in zip(axes_flat, PANELS):
        if args.no_zenith:
            include_anchor = False
        xs, series = _panel_series(campaign, anchor, axis, include_anchor)
        all_y: list[float] = []
        for method, ys in series.items():
            all_y.extend(ys)
            st = STYLES[method]
            ax.plot(xs, ys, label=LABELS[method], **st)

        y_min = min(all_y)
        y_max = max(all_y)
        span = max(y_max - y_min, 0.012)
        y_pad_lo = 0.08 if axis == "iots" else 0.04
        y_pad_hi = 0.20 if axis == "iots" else 0.16
        ax.set_ylim(y_min - y_pad_lo * span, y_max + y_pad_hi * span)
        _set_xticks(ax, xs, axis)
        if axis != "iots":
            x0, x1 = _xlim_padded(xs)
            ax.set_xlim(x0, x1)
            ax.margins(x=0)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Sum rate (Mbps)", fontsize=9)
        ax.set_title(title, fontsize=10, pad=4)
        ax.grid(True, alpha=0.28, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        _place_panel_legend(ax, axis, len(series))

    fig.text(0.5, 0.02, foot, ha="center", fontsize=7.5, color="0.35")
    stem = out_dir / stem_name
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(stem.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
