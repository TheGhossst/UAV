"""Replacement panels for the three flat dynamic-noise sum-rate sweeps.

Checks Eq. (17) on the campaign CSV before drawing. Writes only
eq17_split, occupancy, and cliff_markers under
results/run_10mhz_dynamic_noise/tmp_graphs/100m/.
Does not write the UAV-count or IoT-count sum-rate graphs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from inside_legend import place_inside_legend
from plot_dynamic_noise_tmp_lines import OUT, SERIES, _campaign_rows, _style_ax

# Lambda / CPU / threshold sweeps hold I=10, K=2, so |N_k|=5.
N_K = 5
TASK_CYCLES = 3.75e6
LAMBDA_DEFAULT = 2.0
CPU_DEFAULT = 2.0e8
THRESHOLD_DEFAULT = 2.8
RESIDUAL_LO = 0.45
RESIDUAL_HI = 0.55
RESIDUAL_SPAN_MAX = 0.01

AXES = (
    ("lambda", "Task arrival rate (tasks/s)"),
    ("cpu", r"CPU cycles ($\times 10^{8}$/s)"),
    ("aodt", "AoDT threshold (s)"),
)


def _mu(cpu: float) -> float:
    return cpu / TASK_CYCLES


def _queue(lam: float, cpu: float) -> float:
    """Eq. (17) queue term for equal arrivals: 1/λ + |N_k|/μ."""
    return (1.0 / lam) + (N_K / _mu(cpu))


def _held(axis: str, x: float) -> tuple[float, float, float]:
    lam = x if axis == "lambda" else LAMBDA_DEFAULT
    cpu = x if axis == "cpu" else CPU_DEFAULT
    threshold = x if axis == "aodt" else THRESHOLD_DEFAULT
    return lam, cpu, threshold


def _rows() -> list[dict]:
    out = []
    for row in _campaign_rows("100m"):
        if row["axis"] not in {"lambda", "cpu", "aodt"}:
            continue
        out.append(
            {
                "axis": row["axis"],
                "x": float(row["x"]),
                "method": row["method"],
                "aodt": float(row["mean_max_AoDT_s"]),
                "n": int(float(row["n"])),
            }
        )
    return out


def _enrich(rows: list[dict]) -> None:
    for row in rows:
        lam, cpu, threshold = _held(row["axis"], row["x"])
        q = _queue(lam, cpu)
        row["lambda"] = lam
        row["cpu"] = cpu
        row["threshold"] = threshold
        row["q"] = q
        row["clock"] = 1.0 / lam
        row["service"] = N_K / _mu(cpu)
        row["residual"] = row["aodt"] - row["clock"]
        row["D"] = row["aodt"] - q
        window = threshold - q
        row["window"] = window
        row["occupancy"] = row["D"] / window if window != 0.0 else float("nan")


def _methods(rows: list[dict]) -> list[str]:
    present = {row["method"] for row in rows}
    return [method for method, *_rest in SERIES if method in present]


def _print_table(title: str, rows: list[dict], fields: tuple[str, ...]) -> None:
    print(title)
    header = ("method", "axis", "x") + fields
    print("  " + " ".join(f"{name:>14}" for name in header))
    for row in rows:
        cells = [f"{row['method']:>14}", f"{row['axis']:>14}", f"{row['x']:14.6g}"]
        for name in fields:
            cells.append(f"{row[name]:14.6f}")
        print("  " + " ".join(cells))


def _audit(rows: list[dict]) -> dict[str, dict]:
    print("columns used: axis, x, method, mean_max_AoDT_s, n")
    print("held fixed off-axis: lambda=2 /s, CPU=2e8 cycles/s, threshold=2.8 s, |N_k|=5")
    lam_rows = [row for row in rows if row["axis"] == "lambda"]
    _print_table(
        "arrival-rate residual = mean_max_AoDT_s - 1/lambda",
        lam_rows,
        ("aodt", "clock", "residual"),
    )
    by_method: dict[str, dict] = {}
    failed = False
    for method in _methods(rows):
        vals = [row["residual"] for row in lam_rows if row["method"] == method]
        if not vals:
            continue
        span = max(vals) - min(vals)
        mean = float(np.mean(vals))
        ok = RESIDUAL_LO <= mean <= RESIDUAL_HI and span <= RESIDUAL_SPAN_MAX
        failed = failed or not ok
        by_method[method] = {"residual_mean": mean, "residual_span": span}
        print(
            f"  {method}: residual mean {mean:.6f} s, span {span:.3e} s "
            f"{'PASS' if ok else 'FAIL'}"
        )
    if failed:
        raise SystemExit(
            "arrival-rate residual is not a flat ~0.50 s; Eq. (17) split not plotted"
        )

    _print_table(
        "upload delay D = AoDT - Q and occupancy D/(T-Q)",
        rows,
        ("q", "D", "window", "occupancy"),
    )
    bad = [
        row
        for row in rows
        if not (0.0 < row["occupancy"] < 1.0)
    ]
    if bad:
        print(f"occupancy outside (0, 1): {len(bad)} rows")
        for row in bad:
            print(
                f"  {row['method']} {row['axis']} x={row['x']:g} "
                f"occupancy={row['occupancy']}"
            )
        raise SystemExit("occupancy figure rejected")
    print(f"occupancy inside (0, 1): PASS ({len(rows)} rows)")

    for method in by_method:
        dvals = [row["D"] for row in rows if row["method"] == method]
        by_method[method]["D"] = float(np.mean(dvals))
    print("cliff roots from CSV means, beside the collected grid")
    print("  grid lambda [1, 3.5] tasks/s; CPU [0.5, 2.5] x 1e8 cycles/s; threshold [1.2, 3.2] s")
    for method, stats in by_method.items():
        d = stats["D"]
        service_room = THRESHOLD_DEFAULT - d - (1.0 / LAMBDA_DEFAULT)
        lam_star = 1.0 / (THRESHOLD_DEFAULT - d - (N_K / _mu(CPU_DEFAULT)))
        cpu_star = (N_K * TASK_CYCLES) / service_room
        t_rows = [row["aodt"] for row in rows if row["method"] == method and row["axis"] == "aodt"]
        t_star = float(np.mean(t_rows)) if t_rows else float("nan")
        stats["lambda_star"] = lam_star
        stats["cpu_star"] = cpu_star
        stats["t_star"] = t_star
        print(
            f"  {method}: lambda*={lam_star:.4f} /s  "
            f"CPU*={cpu_star:.4e} cycles/s  T*={t_star:.4f} s  "
            f"D={d:.4f} s"
        )
    return by_method


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


def _series_xy(rows: list[dict], axis: str, method: str, ykey: str, xscale=None):
    pts = []
    for row in rows:
        if row["axis"] != axis or row["method"] != method:
            continue
        x = row["x"] if xscale is None else xscale(row["x"])
        pts.append((x, row[ykey]))
    pts.sort(key=lambda item: item[0])
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def _plot_methods(ax, rows, axis, ykey, xscale=None) -> list[float]:
    xs_ref: list[float] = []
    for method, label, color, marker, linestyle, lw, ms in SERIES:
        xs, ys = _series_xy(rows, axis, method, ykey, xscale)
        if not xs:
            continue
        if len(xs) > len(xs_ref):
            xs_ref = xs
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
    return xs_ref


def _finish(fig: plt.Figure, stem: str) -> None:
    path = OUT / "100m" / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {path}.png")


def _eq17(rows: list[dict], stats: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), layout="constrained")

    ax = axes[0]
    clock_x = np.linspace(1.0, 3.5, 80)
    ax.plot(clock_x, 1.0 / clock_x, color="0.35", linestyle="--", linewidth=1.0, label=r"$1/\lambda$", zorder=2)
    _plot_methods(ax, rows, "lambda", "residual")
    ax.set_xlabel("Task arrival rate (tasks/s)", labelpad=2)
    ax.set_ylabel("Seconds", labelpad=2)
    ax.set_title(r"Arrival: $1/\lambda$ and residual", fontsize=8)
    _style_ax(ax)
    _legend(fig, ax)

    ax = axes[1]
    _plot_methods(ax, rows, "cpu", "residual", xscale=lambda cpu: 1e8 / cpu)
    for method, label, color, marker, linestyle, lw, ms in SERIES:
        if method not in stats:
            continue
        if not any(row["axis"] == "cpu" and row["method"] == method for row in rows):
            continue
        d = stats[method]["D"]
        grid = np.array([0.5e8, 1.0e8, 1.5e8, 2.0e8, 2.5e8])
        y = d + N_K * TASK_CYCLES / grid
        ax.plot(1e8 / grid, y, color=color, linestyle=":", linewidth=0.8, zorder=2)
    ax.set_xlabel(r"$10^{8}/f_{\mathrm{CPU}}$", labelpad=2)
    ax.set_ylabel(r"AoDT $-$ $1/\lambda$ (s)", labelpad=2)
    ax.set_title(r"CPU: residual vs $1/f$", fontsize=8)
    ax.set_xticks([0.4, 1.0, 2.0])
    _style_ax(ax)

    ax = axes[2]
    for row in rows:
        row["slack"] = row["threshold"] - row["aodt"]
    xs_ref = _plot_methods(ax, rows, "aodt", "slack")
    ax.set_xlabel("AoDT threshold (s)", labelpad=2)
    ax.set_ylabel("Slack T − AoDT (s)", labelpad=2)
    ax.set_title("Threshold: constraint slack", fontsize=8)
    if xs_ref:
        ax.set_xticks(xs_ref)
    _style_ax(ax)
    _finish(fig, "eq17_split")


def _occupancy(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), layout="constrained")
    xscales = {
        "lambda": None,
        "cpu": lambda cpu: cpu / 1e8,
        "aodt": None,
    }
    ylabels = "Upload occupancy  D / (T − Q)"
    for ax, (axis, xlabel) in zip(axes, AXES):
        xs_ref = _plot_methods(ax, rows, axis, "occupancy", xscales[axis])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.set_ylabel(ylabels, labelpad=2)
        if xs_ref:
            ax.set_xticks(xs_ref)
        _style_ax(ax)
    _legend(fig, axes[0])
    _finish(fig, "occupancy")


def _cliffs(rows: list[dict], stats: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), layout="constrained")
    specs = (
        ("lambda", "Task arrival rate (tasks/s)", "lambda_star", lambda x: x, (1.0, 3.5)),
        ("cpu", r"CPU cycles ($\times 10^{8}$/s)", "cpu_star", lambda x: x / 1e8, (0.5, 2.5)),
        ("aodt", "AoDT threshold (s)", "t_star", lambda x: x, (1.2, 3.2)),
    )
    for row in rows:
        row["slack"] = row["threshold"] - row["aodt"]
    for ax, (axis, xlabel, key, scale, grid) in zip(axes, specs):
        _plot_methods(ax, rows, axis, "slack", scale if axis == "cpu" else None)
        ax.axvspan(grid[0], grid[1], color="0.85", zorder=0, label="Collected grid")
        ax.axhline(0.0, color="black", linewidth=0.6, zorder=1)
        for method, label, color, marker, linestyle, lw, ms in SERIES:
            if method not in stats:
                continue
            if not any(row["axis"] == axis and row["method"] == method for row in rows):
                continue
            root = scale(stats[method][key])
            ax.scatter(
                [root],
                [0.0],
                marker="x",
                s=28,
                color=color,
                linewidths=0.9,
                zorder=4,
                label="Binds" if method == "random" else None,
            )
        lo = min(scale(stats[m][key]) for m in stats if any(
            row["axis"] == axis and row["method"] == m for row in rows
        ))
        ax.set_xlim(min(lo, grid[0]) - 0.08 * (grid[1] - grid[0]), grid[1] + 0.08 * (grid[1] - grid[0]))
        ax.set_xlabel(xlabel, labelpad=2)
        ax.set_ylabel("Slack (s)", labelpad=2)
        _style_ax(ax)
    _legend(fig, axes[0])
    _finish(fig, "cliff_markers")


def main() -> None:
    rows = _rows()
    if not rows:
        raise SystemExit("no lambda, cpu, or aodt rows")
    _enrich(rows)
    stats = _audit(rows)
    _eq17(rows, stats)
    _occupancy(rows)
    _cliffs(rows, stats)


if __name__ == "__main__":
    main()
