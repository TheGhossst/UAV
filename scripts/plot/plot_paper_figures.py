"""Plot paper-style Figs. 6–11 from campaign JSON artifacts.

Reads results/campaign_8.8mhz_cap25_si12k.json (Figs. 6–10) and
results/fig11_8.8mhz_cap25_si12k.json (Fig. 11). Writes PNG + PDF to
results/figures/.

Usage (from repo root):
    pip install matplotlib
    python scripts/plot/plot_paper_figures.py
    python scripts/plot/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap25_si12k.json
    python scripts/plot/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap15_n20.json --fig11 results/fig11_8.8mhz_cap15.json --out-dir results/figures/cap15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_LIB = Path(__file__).resolve().parents[1] / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from aodt_plot_utils import (  # noqa: E402
    ZERO_EPS,
    aodt_xticks,
    aodt_ylim,
    collect_aodt_series,
    draw_infeasible_ticks,
    infeasible_xs_for_method,
)

# Paper methods first (SCA), then opt-in solvers, then k-means/random, then external PSO.
METHOD_ORDER = ("sca", "sca_anchor", "sca_multistart", "td3", "kmeans", "random", "pso")
METHOD_LABELS = {
    "sca": "SCA",
    "sca_anchor": "SCA zenith-anchor",
    "sca_multistart": "SCA multi-start",
    "td3": "TD3",
    "kmeans": "K-means",
    "random": "Random",
    "pso": "PSO (external)",
}
METHOD_STYLES = {
    "sca": {"color": "#1f77b4", "marker": "o", "linewidth": 2.2, "zorder": 5},
    "sca_anchor": {"color": "#e377c2", "marker": "*", "linewidth": 2.1, "zorder": 4.5},
    "sca_multistart": {"color": "#8c564b", "marker": "v", "linewidth": 2.0, "zorder": 4.2},
    "td3": {"color": "#17becf", "marker": "P", "linewidth": 2.0, "zorder": 4},
    "kmeans": {"color": "#ff7f0e", "marker": "s", "linewidth": 1.8, "zorder": 3},
    "random": {"color": "#2ca02c", "marker": "^", "linewidth": 1.8, "zorder": 2},
    "pso": {"color": "#9467bd", "marker": "D", "linewidth": 1.6, "linestyle": "--", "zorder": 1},
}

FIG11_PATTERNS = ("uniform_fast", "heterogeneous", "uniform_slow")
FIG11_LABELS = {
    "uniform_fast": "Uniform fast",
    "heterogeneous": "Heterogeneous",
    "uniform_slow": "Uniform slow",
}
FIG11_STYLES = {
    "uniform_fast": {"color": "#1f77b4", "marker": "o"},
    "heterogeneous": {"color": "#d62728", "marker": "s"},
    "uniform_slow": {"color": "#2ca02c", "marker": "^"},
}

def _repro_note(campaign: dict | None = None) -> str:
    if campaign is None:
        return (
            "Reproduction: 100×100 m, B_sys = 8.8 MHz, 25% per-link cap, "
            "20 seeds/point. Not Table II 20 kHz."
        )
    share = campaign.get("max_bw_share")
    cap = "no per-link cap" if share is None else f"{float(share):.0%} per-link cap"
    b_mhz = float(campaign.get("b_sys_hz") or 8_800_000.0) / 1e6
    n = int(campaign.get("n_runs") or 20)
    area = campaign.get("area_m") or [100.0, 100.0]
    return (
        f"Reproduction: {float(area[0]):g}×{float(area[1]):g} m, "
        f"B_sys = {b_mhz:g} MHz, {cap}, {n} seeds/point. Not Table II 20 kHz."
    )


REPRO_NOTE = _repro_note()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _axis_points(campaign: dict, axis: str) -> list[dict]:
    pts = [p for p in campaign["points"] if p["axis"] == axis]
    pts.sort(key=lambda p: p["x"])
    return pts


def _methods_present(campaign: dict) -> tuple[str, ...]:
    return tuple(m for m in METHOD_ORDER if m in campaign["methods"])


def _series(point: dict, method: str) -> tuple[float, float]:
    s = point["by_method"][method]
    return s["mean_sum_rate_Mbps"], s["std_sum_rate_Mbps"]


def _paired_delta_series(
    point: dict,
    method: str,
    baseline: str = "sca",
) -> tuple[float, float]:
    """Mean and sample std of per-seed (method − baseline) sum rate (Mbps)."""
    if method == baseline:
        return 0.0, 0.0
    base = np.asarray(point["by_method"][baseline]["per_seed_Mbps"], dtype=float)
    cur = np.asarray(point["by_method"][method]["per_seed_Mbps"], dtype=float)
    if base.shape != cur.shape:
        raise ValueError(f"seed mismatch for {method} vs {baseline}")
    delta = cur - base
    if delta.size <= 1:
        return float(np.mean(delta)), 0.0
    return float(np.mean(delta)), float(np.std(delta, ddof=1))


def _delta_ylim(means: list[float], stds: list[float]) -> tuple[float, float]:
    lows = [m - s for m, s in zip(means, stds)]
    highs = [m + s for m, s in zip(means, stds)]
    if not lows:
        return -0.05, 0.05
    y_min = float(min(lows))
    y_max = float(max(highs))
    span = max(y_max - y_min, 0.005)
    pad = max(0.002, 0.12 * span)
    return y_min - pad, y_max + pad


def _decorate_delta_axis(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="-", zorder=1)
    ax.set_ylabel(r"$\Delta$ sum rate vs SCA (Mbps)")


def _plot_delta_sweep_on_axes(
    ax: plt.Axes,
    campaign: dict,
    axis: str,
    xlabel: str,
    subtitle: str,
    *,
    baseline: str = "sca",
    legend: bool = False,
    labelsize: int = 8,
) -> None:
    if baseline not in campaign.get("methods", []):
        raise ValueError(f"baseline {baseline!r} not in campaign methods")
    points = _axis_points(campaign, axis)
    xs = np.array([p["x"] for p in points], dtype=float)
    methods = [m for m in _methods_present(campaign) if m != baseline]

    all_means: list[float] = []
    all_stds: list[float] = []
    for method in methods:
        means = []
        stds = []
        for p in points:
            m, s = _paired_delta_series(p, method, baseline)
            means.append(m)
            stds.append(s)
        all_means.extend(means)
        all_stds.extend(stds)
        style = METHOD_STYLES[method]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            label=METHOD_LABELS[method],
            capsize=2 if labelsize <= 8 else 3,
            markersize=4 if labelsize <= 8 else 6,
            **style,
        )

    y0, y1 = _delta_ylim(all_means, all_stds)
    ax.set_ylim(y0, y1)
    if axis in ("uavs", "iots", "cpu"):
        _set_sweep_xticks(ax, xs, axis)
    else:
        ax.set_xticks(xs)
    ax.set_xlabel(xlabel, fontsize=9 if labelsize <= 8 else 10)
    _decorate_delta_axis(ax)
    ax.set_title(subtitle, fontsize=10 if labelsize <= 8 else 11)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=labelsize)
    if legend:
        ax.legend(loc="best", framealpha=0.9, fontsize=9)


def _sum_rate_ylim(
    means: list[float],
    stds: list[float],
    *,
    extra_lows: list[float] | None = None,
) -> tuple[float, float]:
    """Y limits for sum-rate plots: floor at data minimum (no padding below)."""
    lows = [m - s for m, s in zip(means, stds)]
    highs = [m + s for m, s in zip(means, stds)]
    if extra_lows:
        lows.extend(extra_lows)
    if not lows:
        return 0.0, 1.0
    y_min = float(min(lows))
    y_max = float(max(highs))
    span = max(y_max - y_min, 0.02)
    pad_top = max(0.008, 0.06 * span)
    return y_min, y_max + pad_top


def _set_sweep_xticks(ax: plt.Axes, xs: np.ndarray, axis: str) -> None:
    ax.set_xticks(xs)
    if axis in ("uavs", "iots"):
        ax.set_xticklabels([str(int(round(x))) for x in xs])
    elif axis == "cpu":
        ax.set_xticklabels([f"{x / 1e8:g}" for x in xs])


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str = "Sum rate (Mbps)", title: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_aodt_sum_rate(ax: plt.Axes, campaign: dict, xlabel: str, title: str) -> None:
    points = _axis_points(campaign, "aodt")
    methods = _methods_present(campaign)

    def mean_at(p: dict, method: str) -> float | None:
        if method not in (p.get("by_method") or {}):
            return None
        m, _ = _series(p, method)
        return float(m) if m > ZERO_EPS else 0.0

    all_ys: list[float] = []
    y0: float | None = None
    for method in methods:
        xs_ok, ys_ok, xs_zero = collect_aodt_series(points, method, mean_at)
        all_ys.extend(ys_ok)
        if not xs_ok and not xs_zero:
            continue
        style = METHOD_STYLES[method]
        stds_ok: list[float] = []
        for x in xs_ok:
            for p in points:
                if abs(float(p["x"]) - x) > 1e-9:
                    continue
                m, s = _series(p, method)
                if m > ZERO_EPS:
                    stds_ok.append(s)
                break
        ax.errorbar(
            np.asarray(xs_ok, dtype=float),
            np.asarray(ys_ok, dtype=float),
            yerr=np.asarray(stds_ok, dtype=float) if len(stds_ok) == len(ys_ok) else None,
            label=METHOD_LABELS[method],
            capsize=3,
            markersize=6,
            **style,
        )
    if all_ys:
        y0, y1 = aodt_ylim(all_ys)
        ax.set_ylim(y0, y1)
        for method in methods:
            _, _, xs_zero = collect_aodt_series(points, method, mean_at)
            xz = infeasible_xs_for_method(xs_zero, method)
            if xz:
                color = METHOD_STYLES.get(method, {}).get("color", "0.5")
                draw_infeasible_ticks(ax, xz, y0, color=color)
    xs_ticks = aodt_xticks(points, methods)
    ax.set_xticks(xs_ticks)
    ax.set_xticklabels([f"{x:g}" for x in xs_ticks])
    _style_axes(ax, xlabel, title=title)
    ax.legend(loc="best", framealpha=0.9)
    ax.text(
        0.02,
        0.02,
        r"$T_k\geq1.2$ only ($0.8$ omitted — QoS-infeasible for all methods).",
        transform=ax.transAxes,
        fontsize=7,
        color="0.4",
        va="bottom",
    )


def plot_sum_rate_figure(
    campaign: dict,
    axis: str,
    xlabel: str,
    title: str,
    fig_num: int,
    out_dir: Path,
    x_formatter=None,
    *,
    delta_vs_baseline: str | None = None,
) -> Path:
    points = _axis_points(campaign, axis)
    if not points:
        raise ValueError(f"campaign has no points for axis {axis!r}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if axis == "aodt" and not delta_vs_baseline:
        _plot_aodt_sum_rate(ax, campaign, xlabel, title)
        fig.text(
            0.5,
            0.01,
            REPRO_NOTE + "  Y-axis zoomed to feasible sum rates.",
            ha="center",
            fontsize=7.5,
            color="0.35",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        stem = out_dir / f"fig{fig_num:02d}_sum_rate"
        fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return stem.with_suffix(".png")
    if delta_vs_baseline:
        _plot_delta_sweep_on_axes(
            ax,
            campaign,
            axis,
            xlabel,
            title,
            baseline=delta_vs_baseline,
            legend=True,
            labelsize=9,
        )
    else:
        methods = _methods_present(campaign)
        xs = np.array([p["x"] for p in points], dtype=float)
        for method in methods:
            means = []
            stds = []
            for p in points:
                m, s = _series(p, method)
                means.append(m)
                stds.append(s)
            style = METHOD_STYLES[method]
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                label=METHOD_LABELS[method],
                capsize=3,
                markersize=6,
                **style,
            )

        all_means: list[float] = []
        all_stds: list[float] = []
        for p in points:
            for method in methods:
                m, s = _series(p, method)
                all_means.append(m)
                all_stds.append(s)
        y0, y1 = _sum_rate_ylim(all_means, all_stds)
        ax.set_ylim(y0, y1)

        if x_formatter is not None:
            ax.set_xticks(np.array([p["x"] for p in points], dtype=float))
            ax.set_xticklabels([x_formatter(x) for x in np.array([p["x"] for p in points], dtype=float)])
        else:
            _set_sweep_xticks(ax, np.array([p["x"] for p in points], dtype=float), axis)
        _style_axes(ax, xlabel, title=title)
        ax.legend(loc="best", framealpha=0.9)

    if delta_vs_baseline:
        fig.text(
            0.5,
            0.01,
            REPRO_NOTE + "  Positive Δ: beats frozen SCA on paired seeds.",
            ha="center",
            fontsize=7.5,
            color="0.35",
        )
    else:
        fig.text(0.5, 0.01, REPRO_NOTE, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    stem = out_dir / f"fig{fig_num:02d}_sum_rate"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def plot_fig11(fig11: dict, out_dir: Path, metric: str = "fcfs_sim") -> Path:
    """Fig. 11: AoDT vs J for three arrival-rate patterns.

    metric:
      - 'fcfs_sim': FCFS simulator process-max age (matches paper narrative)
      - 'eq17': Eq. (17) max age (Problem P score)
    """
    points = sorted(fig11["points"], key=lambda p: p["num_uav"])
    js = np.array([p["num_uav"] for p in points], dtype=float)

    ylabel = "Max process AoDT (s)"
    if metric == "eq17":
        ylabel = "Eq. (17) max AoDT (s)"
        getter = lambda pat, pt: (
            pt["by_pattern"][pat]["mean_eq17_max_s"],
            pt["by_pattern"][pat]["std_eq17_max_s"],
        )
        title = "Fig. 11 analogue — arrival patterns (Eq. 17)"
        stem_name = "fig11_eq17_aodt"
    else:
        getter = lambda pat, pt: (
            pt["by_pattern"][pat]["sim_mean_max_process_age_s"]["fcfs"],
            0.0,
        )
        title = "Fig. 11 analogue — arrival patterns (FCFS simulator)"
        stem_name = "fig11_fcfs_sim_aodt"

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for pat in FIG11_PATTERNS:
        means = []
        stds = []
        for pt in points:
            m, s = getter(pat, pt)
            means.append(m)
            stds.append(s)
        style = FIG11_STYLES[pat]
        if metric == "eq17":
            ax.errorbar(
                js,
                means,
                yerr=stds,
                label=FIG11_LABELS[pat],
                capsize=3,
                markersize=6,
                linewidth=2.0,
                **style,
            )
        else:
            ax.plot(js, means, label=FIG11_LABELS[pat], linewidth=2.0, markersize=6, **style)

    ax.set_xticks(js)
    ax.set_xticklabels([str(int(j)) for j in js])
    _style_axes(ax, "Number of UAVs ($J$)", ylabel=ylabel, title=title)
    ax.legend(loc="best", framealpha=0.9)
    fig.text(
        0.5,
        0.01,
        REPRO_NOTE + "  k-means placement; I = 10.",
        ha="center",
        fontsize=7.5,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    stem = out_dir / stem_name
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def plot_overview_grid(campaign: dict, out_dir: Path, *, delta_vs_baseline: str = "sca") -> Path:
    """Single-page overview: Figs. 6–9 as Δ sum rate vs SCA (no CPU panel)."""
    specs = [
        ("uavs", "Number of UAVs ($J$)", "Fig. 6 — $I=10$"),
        ("iots", "Number of IoT devices ($I$)", "Fig. 7 — $J=3$"),
        ("lambda", r"Arrival rate $\lambda$ (tasks/s)", "Fig. 8 — $I=10$, $J=3$"),
        ("aodt", r"AoDT threshold $T_k$ (s)", "Fig. 9 — $I=10$, $J=3$"),
    ]
    specs = [(a, xl, st) for a, xl, st in specs if _axis_points(campaign, a)]
    if not specs:
        raise ValueError("campaign has no sweep points to plot")

    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 7.5))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (axis, xlabel, subtitle) in zip(axes_flat, specs):
        _plot_delta_sweep_on_axes(
            ax,
            campaign,
            axis,
            xlabel,
            subtitle,
            baseline=delta_vs_baseline,
            legend=False,
        )

    for ax in axes_flat[len(specs) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    fig.suptitle(
        r"Axis sweeps — paired $\Delta$ sum rate vs frozen SCA (20 seeds/point)",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        REPRO_NOTE + "  Above zero: method beats SCA on mean paired Δ.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    stem = out_dir / "overview_figs06_10"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot paper-style Figs. 6–11.")
    ap.add_argument(
        "--campaign",
        default="results/campaign_8.8mhz_cap25_si12k.json",
        help="Campaign JSON for Figs. 6–10",
    )
    ap.add_argument(
        "--fig11",
        default="results/fig11_8.8mhz_cap25_si12k.json",
        help="Fig. 11 JSON (skipped if the file is missing)",
    )
    ap.add_argument(
        "--out-dir",
        default="results/figures",
        help="Output directory for PNG/PDF figures",
    )
    ap.add_argument(
        "--skip-fig11",
        action="store_true",
        help="Do not plot Fig. 11 even if the JSON exists",
    )
    ap.add_argument(
        "--ppt-sweeps",
        action="store_true",
        help="Fig. 6–9 single panels as Δ vs SCA (for slides); overview always Δ, no CPU",
    )
    args = ap.parse_args()

    campaign_path = Path(args.campaign)
    fig11_path = Path(args.fig11)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    campaign = _load(campaign_path)
    global REPRO_NOTE
    REPRO_NOTE = _repro_note(campaign)

    written: list[Path] = []
    fig_specs = [
        ("uavs", "Number of UAVs ($J$)", "Fig. 6 analogue — sum rate vs UAV count ($I=10$)", 6, None),
        ("iots", "Number of IoT devices ($I$)", "Fig. 7 analogue — sum rate vs IoT count ($J=3$)", 7, None),
        (
            "lambda",
            r"Task arrival rate $\lambda$ (tasks/s)",
            "Fig. 8 analogue — sum rate vs arrival rate ($I=10$, $J=3$)",
            8,
            None,
        ),
        (
            "aodt",
            r"AoDT threshold $T_k$ (s)",
            "Fig. 9 analogue — sum rate vs AoDT threshold ($I=10$, $J=3$)",
            9,
            None,
        ),
        (
            "cpu",
            r"UAV CPU capacity $f_j$ ($\times 10^8$ cycles/s)",
            "Fig. 10 analogue — sum rate vs UAV CPU ($I=10$, $J=3$)",
            10,
            lambda x: f"{x / 1e8:g}",
        ),
    ]
    delta = "sca" if args.ppt_sweeps else None
    for axis, xlabel, title, fig_num, x_formatter in fig_specs:
        if not _axis_points(campaign, axis):
            continue
        if axis == "cpu" and args.ppt_sweeps:
            continue
        written.append(
            plot_sum_rate_figure(
                campaign,
                axis,
                xlabel,
                title,
                fig_num,
                out_dir,
                x_formatter=x_formatter,
                delta_vs_baseline=delta,
            )
        )
    if not args.skip_fig11 and fig11_path.exists():
        fig11 = _load(fig11_path)
        written.append(plot_fig11(fig11, out_dir, metric="fcfs_sim"))
        written.append(plot_fig11(fig11, out_dir, metric="eq17"))
    written.append(plot_overview_grid(campaign, out_dir))

    print(f"Campaign: {campaign_path}")
    print(f"Fig. 11:  {fig11_path}")
    print(f"Wrote {len(written)} figure files to {out_dir.resolve()}:")
    for p in written:
        print(f"  {p.name}")
        pdf = p.with_suffix(".pdf")
        if pdf.exists():
            print(f"  {pdf.name}")


if __name__ == "__main__":
    main()
