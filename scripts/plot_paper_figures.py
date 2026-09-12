"""Plot paper-style Figs. 6–11 from campaign JSON artifacts.

Reads results/campaign_8.8mhz_cap25_si12k.json (Figs. 6–10) and
results/fig11_8.8mhz_cap25_si12k.json (Fig. 11). Writes PNG + PDF to
results/figures/.

Usage (from repo root):
    pip install matplotlib
    python scripts/plot_paper_figures.py
    python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap25_si12k.json
    python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap15_n20.json --fig11 results/fig11_8.8mhz_cap15.json --out-dir results/figures/cap15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str = "Sum rate (Mbps)", title: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_sum_rate_figure(
    campaign: dict,
    axis: str,
    xlabel: str,
    title: str,
    fig_num: int,
    out_dir: Path,
    x_formatter=None,
) -> Path:
    points = _axis_points(campaign, axis)
    methods = _methods_present(campaign)
    xs = np.array([p["x"] for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
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

    if x_formatter is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels([x_formatter(x) for x in xs])
    _style_axes(ax, xlabel, title=title)
    ax.legend(loc="best", framealpha=0.9)
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


def plot_overview_grid(campaign: dict, out_dir: Path) -> Path:
    """Single-page overview of Figs. 6–10."""
    specs = [
        ("uavs", "Number of UAVs ($J$)", "Fig. 6 — $I=10$"),
        ("iots", "Number of IoT devices ($I$)", "Fig. 7 — $J=3$"),
        ("lambda", r"Arrival rate $\lambda$ (tasks/s)", "Fig. 8 — $I=10$, $J=3$"),
        ("aodt", r"AoDT threshold $T_k$ (s)", "Fig. 9 — $I=10$, $J=3$"),
        ("cpu", r"UAV CPU $f_j$ ($\times 10^8$ cycles/s)", "Fig. 10 — $I=10$, $J=3$"),
    ]
    methods = _methods_present(campaign)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, (axis, xlabel, subtitle) in zip(axes_flat, specs):
        points = _axis_points(campaign, axis)
        xs = np.array([p["x"] for p in points], dtype=float)
        for method in methods:
            means = [_series(p, method)[0] for p in points]
            stds = [_series(p, method)[1] for p in points]
            style = METHOD_STYLES[method]
            ax.errorbar(xs, means, yerr=stds, label=METHOD_LABELS[method], capsize=2, markersize=4, **style)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Mbps", fontsize=9)
        ax.set_title(subtitle, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=8)

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02), ncol=2, fontsize=9)
    fig.suptitle("Khalaf et al. (2026) §VII reproduction — sum rate sweeps", fontsize=12, y=0.98)
    fig.text(0.5, 0.01, REPRO_NOTE, ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

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
    args = ap.parse_args()

    campaign_path = Path(args.campaign)
    fig11_path = Path(args.fig11)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    campaign = _load(campaign_path)
    global REPRO_NOTE
    REPRO_NOTE = _repro_note(campaign)

    written: list[Path] = []
    written.append(
        plot_sum_rate_figure(
            campaign,
            "uavs",
            "Number of UAVs ($J$)",
            "Fig. 6 analogue — sum rate vs UAV count ($I=10$)",
            6,
            out_dir,
        )
    )
    written.append(
        plot_sum_rate_figure(
            campaign,
            "iots",
            "Number of IoT devices ($I$)",
            "Fig. 7 analogue — sum rate vs IoT count ($J=3$)",
            7,
            out_dir,
        )
    )
    written.append(
        plot_sum_rate_figure(
            campaign,
            "lambda",
            r"Task arrival rate $\lambda$ (tasks/s)",
            "Fig. 8 analogue — sum rate vs arrival rate ($I=10$, $J=3$)",
            8,
            out_dir,
        )
    )
    written.append(
        plot_sum_rate_figure(
            campaign,
            "aodt",
            r"AoDT threshold $T_k$ (s)",
            "Fig. 9 analogue — sum rate vs AoDT threshold ($I=10$, $J=3$)",
            9,
            out_dir,
        )
    )
    written.append(
        plot_sum_rate_figure(
            campaign,
            "cpu",
            r"UAV CPU capacity $f_j$ ($\times 10^8$ cycles/s)",
            "Fig. 10 analogue — sum rate vs UAV CPU ($I=10$, $J=3$)",
            10,
            out_dir,
            x_formatter=lambda x: f"{x / 1e8:g}",
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
