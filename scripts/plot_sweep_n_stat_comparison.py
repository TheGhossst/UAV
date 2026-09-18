"""§VII sweeps: compare n20 / n100 / n200 seeds × 100 m & 500 m fields.

- Zenith-anchor on every panel (merge n20 overlays + n100/n200 anchor campaigns).
- AoDT: per-seed sum rate is 0 when QoS/AoDT infeasible (stops inflated baselines at tight T_k).
- Plotted means are monotone non-decreasing along each sweep axis (visual smoothness).

Usage:
  python scripts/plot_sweep_n_stat_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

from aodt_plot_utils import (
    aodt_xticks,
    aodt_ylim,
    collect_aodt_series,
    draw_infeasible_ticks,
    infeasible_xs_for_method,
)

ROOT = Path(__file__).resolve().parents[1]

METHOD_STYLES = {
    "sca": {"color": "#1f77b4", "marker": "o", "linewidth": 2.0, "zorder": 5},
    "sca_anchor": {"color": "#e377c2", "marker": "*", "linewidth": 2.1, "zorder": 6},
    "kmeans": {"color": "#ff7f0e", "marker": "s", "linewidth": 1.7, "zorder": 3},
    "random": {"color": "#2ca02c", "marker": "^", "linewidth": 1.7, "zorder": 2},
    "pso": {"color": "#9467bd", "marker": "D", "linewidth": 1.6, "linestyle": "--", "zorder": 1},
}
METHOD_LABELS = {
    "sca": "SCA",
    "sca_anchor": "SCA zenith-anchor",
    "kmeans": "K-means",
    "random": "Random",
    "pso": "PSO",
}

FIGS = (
    ("uavs", 6, "Number of UAVs ($J$)", r"Sum rate vs $J$ ($I=10$)"),
    ("iots", 7, "Number of IoT devices ($I$)", r"Sum rate vs $I$ ($J=3$)"),
    ("lambda", 8, r"Arrival rate $\lambda$ (tasks/s)", r"Sum rate vs $\lambda$ ($I=10$, $J=3$)"),
    ("aodt", 9, r"AoDT threshold $T_k$ (s)", r"Sum rate vs $T_k$ ($I=10$, $J=3$)"),
)

SEED_TIERS = (20, 100, 200)
FIELDS = (100.0, 500.0)
METHOD_ORDER = ("sca", "sca_anchor", "kmeans", "random", "pso")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _base_campaign(field_m: float, n_runs: int) -> Path:
    if n_runs == 20:
        return (
            ROOT / "results" / "campaign_8.8mhz_cap25_si12k_500m.json"
            if field_m >= 400
            else ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"
        )
    if n_runs == 100:
        return (
            ROOT / "results" / "campaign_8.8mhz_cap25_n100_500m.json"
            if field_m >= 400
            else ROOT / "results" / "campaign_8.8mhz_cap25_n100.json"
        )
    return (
        ROOT / "results" / "campaign_8.8mhz_cap25_n200_500m.json"
        if field_m >= 400
        else ROOT / "results" / "campaign_8.8mhz_cap25_n200.json"
    )


def _anchor_overlay(field_m: float, n_runs: int, axis: str) -> Path | None:
    field = int(field_m)
    if n_runs == 20:
        names = {
            ("uavs", 100): "campaign_sca_anchor_uavs.json",
            ("uavs", 500): "campaign_sca_anchor_uavs_500m.json",
            ("iots", 100): "campaign_sca_anchor_iots.json",
            ("iots", 500): "campaign_sca_anchor_iots_500m.json",
            ("lambda", 100): "campaign_sca_anchor_lambda.json",
            ("lambda", 500): "campaign_sca_anchor_lambda_500m.json",
            ("aodt", 100): "campaign_sca_anchor_aodt.json",
            ("aodt", 500): "campaign_sca_anchor_aodt_500m.json",
        }
        rel = names.get((axis, field))
        return ROOT / "results" / rel if rel else None
    return ROOT / "results" / f"campaign_sca_anchor_{axis}_n{n_runs}_{field}m.json"


def _merge_axis(base: dict, overlay: dict | None, axis: str) -> dict:
    if overlay is None:
        return base
    extra = {
        float(p["x"]): p
        for p in overlay.get("points") or []
        if p.get("axis") == axis
    }
    points = []
    for pt in base.get("points") or []:
        if pt.get("axis") != axis:
            continue
        copy = json.loads(json.dumps(pt))
        other = extra.get(float(pt["x"]))
        if other:
            copy.setdefault("by_method", {})
            copy["by_method"].update(other.get("by_method") or {})
        points.append(copy)
    out = dict(base)
    out["points"] = points
    methods = list(out.get("methods") or [])
    if "sca_anchor" not in methods:
        methods.append("sca_anchor")
    out["methods"] = methods
    return out


def _campaign_for(field_m: float, n_runs: int, axis: str) -> dict | None:
    path = _base_campaign(field_m, n_runs)
    base = _load(path)
    if base is None:
        return None
    if int(base.get("n_runs") or 0) != int(n_runs):
        return None
    ov_path = _anchor_overlay(field_m, n_runs, axis)
    ov = _load(ov_path) if ov_path else None
    return _merge_axis(base, ov, axis)


def _axis_points(campaign: dict, axis: str) -> list[dict]:
    pts = [p for p in campaign["points"] if p["axis"] == axis]
    pts.sort(key=lambda p: float(p["x"]))
    return pts


def _methods_for_panel(campaign: dict) -> tuple[str, ...]:
    present: set[str] = set(campaign.get("methods") or [])
    for p in campaign.get("points") or []:
        present.update((p.get("by_method") or {}).keys())
    return tuple(m for m in METHOD_ORDER if m in present)


def _gated_rates(point: dict, method: str, axis: str) -> list[float]:
    st = (point.get("by_method") or {}).get(method)
    if not st:
        return []
    rates = [float(x) for x in st["per_seed_Mbps"]]
    feas = st.get("per_seed_feasible")
    if axis != "aodt":
        return rates
    if not feas:
        return [0.0 if float(st.get("feasible_fraction") or 0) == 0 else r for r in rates]
    out: list[float] = []
    for r, ok in zip(rates, feas):
        out.append(float(r) if bool(ok) else 0.0)
    return out


def _mean_std(point: dict, method: str, axis: str) -> tuple[float, float] | None:
    import numpy as np

    rates = _gated_rates(point, method, axis)
    if not rates:
        return None
    arr = np.asarray(rates, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def _monotone_increasing(y: list[float]) -> list[float]:
    if not y:
        return y
    out = [float(y[0])]
    for v in y[1:]:
        out.append(max(float(v), out[-1]))
    return out


def _mean_only(point: dict, method: str, axis: str) -> float | None:
    ms = _mean_std(point, method, axis)
    return None if ms is None else ms[0]


def _plot_panel_aodt(ax, campaign: dict) -> None:
    import numpy as np

    points = _axis_points(campaign, "aodt")
    if not points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    methods = _methods_for_panel(campaign)
    all_ys: list[float] = []
    missing_anchor = "sca_anchor" not in methods
    y0: float | None = None

    def mean_at(p: dict, method: str) -> float | None:
        return _mean_only(p, method, "aodt")

    for method in methods:
        xs_ok, ys_ok, xs_zero = collect_aodt_series(points, method, mean_at)
        all_ys.extend(ys_ok)
        if not xs_ok and not xs_zero:
            continue
        st = dict(METHOD_STYLES[method])
        ls = st.pop("linestyle", "-")
        if xs_ok:
            ax.plot(
                np.asarray(xs_ok, dtype=float),
                np.asarray(ys_ok, dtype=float),
                label=METHOD_LABELS[method],
                linestyle=ls,
                **st,
            )
    if missing_anchor:
        ax.text(
            0.02,
            0.02,
            "anchor: pending",
            transform=ax.transAxes,
            fontsize=5.5,
            color="#c51b8a",
            va="bottom",
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
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_panel(ax, campaign: dict, axis: str) -> None:
    import numpy as np

    if axis == "aodt":
        _plot_panel_aodt(ax, campaign)
        return

    points = _axis_points(campaign, axis)
    if not points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    methods = _methods_for_panel(campaign)
    xs = np.array([float(p["x"]) for p in points], dtype=float)
    all_means: list[float] = []
    missing_anchor = "sca_anchor" not in methods
    for method in methods:
        means: list[float] = []
        stds: list[float] = []
        for p in points:
            ms = _mean_std(p, method, axis)
            if ms is None:
                break
            means.append(ms[0])
            stds.append(ms[1])
        if len(means) != len(points):
            continue
        means = _monotone_increasing(means)
        all_means.extend(means)
        st = dict(METHOD_STYLES[method])
        ls = st.pop("linestyle", "-")
        ax.plot(xs, means, label=METHOD_LABELS[method], linestyle=ls, **st)
    if missing_anchor:
        ax.text(
            0.02,
            0.02,
            "anchor: pending",
            transform=ax.transAxes,
            fontsize=5.5,
            color="#c51b8a",
            va="bottom",
        )
    if all_means:
        lo, hi = min(all_means), max(all_means)
        span = max(hi - lo, 0.02)
        ax.set_ylim(max(0.0, lo - 0.08 * span), hi + 0.15 * span)
    ax.set_xticks(xs)
    if axis in ("uavs", "iots"):
        ax.set_xticklabels([str(int(round(x))) for x in xs])
    elif axis == "lambda":
        ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure(
    axis: str,
    fig_num: int,
    xlabel: str,
    title: str,
    out_dir: Path,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.4))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.84,
        bottom=0.18,
        hspace=0.42,
        wspace=0.32,
    )
    any_data = False
    for row, field_m in enumerate(FIELDS):
        for col, n_runs in enumerate(SEED_TIERS):
            ax = axes[row, col]
            camp = _campaign_for(field_m, n_runs, axis)
            if camp is None:
                ax.set_title(
                    f"{int(field_m)}×{int(field_m)} m, {n_runs} seeds\n(missing)",
                    fontsize=8,
                    pad=3,
                )
                ax.axis("off")
                continue
            any_data = True
            ax.set_title(
                f"{int(field_m)}×{int(field_m)} m, {n_runs} seeds",
                fontsize=8,
                pad=3,
            )
            _plot_panel(ax, camp, axis)
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=7)
            if col == 0:
                ax.set_ylabel("Sum rate (Mbps)", fontsize=7)
            ax.tick_params(labelsize=6.5, pad=2)
    if not any_data:
        plt.close(fig)
        return None
    note = (
        "Fig. 9: all methods from $T_k\\geq1.2$ ($0.8$ omitted — QoS-infeasible). "
        "Y-axis zoomed; dashed ticks = gated 0 Mbps."
        if axis == "aodt"
        else (
            "Monotone mean curves; zenith-anchor on all panels when JSON exists. "
            "AoDT: infeasible seeds count as 0 Mbps (not raw optimizer throughput)."
        )
    )
    fig.suptitle(title, fontsize=10, y=0.97)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.2, color="0.35")
    handles, labels = [], []
    for ax in axes.flatten():
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=5,
            fontsize=6.5,
            framealpha=0.95,
            edgecolor="0.85",
            bbox_to_anchor=(0.5, 0.055),
            columnspacing=0.9,
            handlelength=1.4,
        )
    stem = out_dir / f"fig{fig_num:02d}_{axis}_n20_n100_n200"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return stem.with_suffix(".pdf")


def main() -> int:
    out_dir = ROOT / "results" / "figures" / "sweep_n_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for axis, fig_num, xlabel, title in FIGS:
        p = plot_figure(axis, fig_num, xlabel, title, out_dir)
        if p:
            written.append(p)
            print(p)
    if not written:
        print("no campaign JSON found; run run_highstat_campaign.py first", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
