"""Average-over-bank figures for the 100-scenario Monte Carlo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

METHOD_ORDER = ("sca", "td3", "kmeans", "random", "pso", "sca_joint")
METHOD_LABELS = {
    "sca": "SCA",
    "td3": "TD3",
    "kmeans": "K-means",
    "random": "Random",
    "pso": "PSO (external)",
    "sca_joint": "SCA-joint (probe)",
}
METHOD_STYLES = {
    "sca": {"color": "#1f77b4", "marker": "o"},
    "td3": {"color": "#17becf", "marker": "P"},
    "kmeans": {"color": "#ff7f0e", "marker": "s"},
    "random": {"color": "#2ca02c", "marker": "^"},
    "pso": {"color": "#9467bd", "marker": "D"},
    "sca_joint": {"color": "#d62728", "marker": "x"},
}


def _repro_note(payload: dict) -> str:
    share = payload.get("max_bw_share")
    cap = "no per-link cap" if share is None else f"{float(share):.0%} per-link cap"
    b_mhz = float(payload.get("b_sys_hz") or 8_800_000.0) / 1e6
    area = payload.get("area_m") or [100.0, 100.0]
    n = int(payload.get("n_scenarios") or 0)
    return (
        f"{area[0]:g}×{area[1]:g} m, I={payload.get('num_iot')}, "
        f"J={payload.get('num_uav')}, B_sys={b_mhz:g} MHz, {cap}, "
        f"mean of {n} saved scenarios."
    )


def _methods(payload: dict) -> tuple[str, ...]:
    present = set(payload.get("methods") or payload.get("by_method", {}))
    ordered = tuple(m for m in METHOD_ORDER if m in present)
    extra = tuple(m for m in present if m not in METHOD_ORDER)
    return ordered + extra


def _save(fig, path: Path) -> Path:
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path.with_suffix(".png")


def plot_n100_figures(
    payload: dict,
    bank: dict | None,
    out_dir: str | Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = _methods(payload)
    note = _repro_note(payload)
    written: list[Path] = []
    written.append(_bar_sum_rate(plt, payload, methods, note, out))
    written.append(_box_sum_rate(plt, payload, methods, note, out))
    written.append(_bar_feasible(plt, payload, methods, note, out))
    written.append(_bar_aodt(plt, payload, methods, note, out))
    written.append(_running_mean(plt, payload, methods, note, out))
    written.append(_per_scenario(plt, payload, methods, note, out))
    if bank is not None:
        written.append(_scenario_maps(plt, payload, bank, note, out))
    return written


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bar_sum_rate(plt, payload, methods, note, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = np.arange(len(methods))
    means = [payload["by_method"][m]["mean_sum_rate_Mbps"] for m in methods]
    stds = [payload["by_method"][m]["std_sum_rate_Mbps"] for m in methods]
    colors = [METHOD_STYLES.get(m, {}).get("color", "#555555") for m in methods]
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, edgecolor="0.2", width=0.65)
    ax.set_xticks(xs, [METHOD_LABELS.get(m, m) for m in methods])
    lo = min(means) - 2.0 * max(stds)
    hi = max(means) + 2.0 * max(stds)
    span = max(hi - lo, 0.02)
    ax.set_ylim(lo - 0.15 * span, hi + 0.25 * span)
    for x, mean in zip(xs, means):
        ax.text(x, mean + max(stds) + 0.02 * span, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    _style(ax, "", "Sum rate (Mbps)", "Mean sum rate over saved scenarios")
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = _save(fig, out / "n100_mean_sum_rate")
    plt.close(fig)
    return png


def _box_sum_rate(plt, payload, methods, note, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    data = [payload["by_method"][m]["per_seed_Mbps"] for m in methods]
    colors = [METHOD_STYLES.get(m, {}).get("color", "#555555") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    try:
        boxes = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    except TypeError:
        boxes = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    _style(ax, "", "Sum rate (Mbps)", "Per-scenario sum-rate distribution")
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = _save(fig, out / "n100_sum_rate_box")
    plt.close(fig)
    return png


def _bar_feasible(plt, payload, methods, note, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    xs = np.arange(len(methods))
    vals = [100.0 * payload["by_method"][m]["feasible_fraction"] for m in methods]
    colors = [METHOD_STYLES.get(m, {}).get("color", "#555555") for m in methods]
    ax.bar(xs, vals, color=colors, edgecolor="0.2", width=0.65)
    ax.set_xticks(xs, [METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylim(0.0, 105.0)
    _style(ax, "", "Feasible (%)", "Feasible fraction")
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = _save(fig, out / "n100_feasible")
    plt.close(fig)
    return png


def _bar_aodt(plt, payload, methods, note, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = np.arange(len(methods))
    means = [payload["by_method"][m]["mean_max_AoDT_s"] for m in methods]
    stds = [payload["by_method"][m]["std_max_AoDT_s"] for m in methods]
    colors = [METHOD_STYLES.get(m, {}).get("color", "#555555") for m in methods]
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, edgecolor="0.2", width=0.65)
    ax.set_xticks(xs, [METHOD_LABELS.get(m, m) for m in methods])
    _style(ax, "", "Max process AoDT (s)", "Mean max AoDT over saved scenarios")
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = _save(fig, out / "n100_mean_max_aodt")
    plt.close(fig)
    return png


def _running_mean(plt, payload, methods, note, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method in methods:
        rates = np.asarray(payload["by_method"][method]["per_seed_Mbps"], dtype=float)
        n = np.arange(1, rates.size + 1)
        running = np.cumsum(rates) / n
        style = METHOD_STYLES.get(method, {})
        ax.plot(
            n,
            running,
            label=METHOD_LABELS.get(method, method),
            color=style.get("color"),
            marker=style.get("marker"),
            markevery=max(1, rates.size // 12),
            linewidth=1.8,
        )
    _style(ax, "Number of scenarios", "Running mean sum rate (Mbps)", "Sample-mean stability")
    ax.legend(loc="best", framealpha=0.9)
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = _save(fig, out / "n100_running_mean")
    plt.close(fig)
    return png


def _per_scenario(plt, payload, methods, note, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    line_styles = {
        "sca": {"linewidth": 2.4, "alpha": 1.0, "zorder": 5},
        "random": {"linewidth": 1.5, "alpha": 0.72, "zorder": 3},
        "kmeans": {"linewidth": 1.5, "alpha": 0.72, "zorder": 3},
        "pso": {"linewidth": 1.5, "alpha": 0.72, "zorder": 3},
        "td3": {"linewidth": 1.5, "alpha": 0.72, "zorder": 3},
        "sca_joint": {"linewidth": 1.5, "alpha": 0.72, "zorder": 3},
    }
    all_rates: list[float] = []
    n_scenarios = 0
    for method in methods:
        rates = np.asarray(payload["by_method"][method]["per_seed_Mbps"], dtype=float)
        n_scenarios = max(n_scenarios, rates.size)
        all_rates.extend(rates.tolist())
        xs = np.arange(1, rates.size + 1)
        style = METHOD_STYLES.get(method, {})
        ls = line_styles.get(method, {"linewidth": 1.5, "alpha": 0.8, "zorder": 3})
        ax.plot(
            xs,
            rates,
            label=METHOD_LABELS.get(method, method),
            color=style.get("color", "#555555"),
            linewidth=ls["linewidth"],
            alpha=ls["alpha"],
            zorder=ls["zorder"],
            solid_capstyle="round",
        )
    y_min, y_max = float(np.min(all_rates)), float(np.max(all_rates))
    pad = max(0.012, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(1, n_scenarios)
    ax.set_xticks(np.linspace(1, n_scenarios, min(6, n_scenarios), dtype=int))
    _style(ax, "Scenario", "Sum rate (Mbps)", "Per-scenario sum rate")
    ax.legend(loc="lower left", framealpha=0.95, edgecolor="0.85", fontsize=9)
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = _save(fig, out / "n100_per_scenario")
    plt.close(fig)
    return png


def plot_seed_layout_comparison(
    payload: dict,
    bank: dict,
    seed: int,
    out_dir: str | Path,
    *,
    methods: tuple[str, ...] = ("sca", "kmeans", "random", "pso"),
) -> Path:
    """Two-panel map: frozen bank layout vs per-method optimized UAV xy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    record = next((r for r in bank["scenarios"] if int(r["seed"]) == int(seed)), None)
    if record is None:
        raise ValueError(f"seed {seed} not found in scenario bank")

    scenario_id = int(record["id"])
    geo = bank["geometry"]
    area_x = float(geo["area_x_m"])
    area_y = float(geo["area_y_m"])

    iot = np.asarray(record["iot_xyz_m"], dtype=float)
    proc = np.asarray(record["process_id_of_iot"], dtype=int)
    init_uav = np.asarray(record["uav_xyz_m"], dtype=float)

    runs: dict[str, dict] = {}
    for method in methods:
        match = [
            r
            for r in payload.get("runs", [])
            if int(r["seed"]) == int(seed) and r["method"] == method
        ]
        if not match:
            raise ValueError(f"method {method!r} missing for seed {seed}")
        runs[method] = match[0]

    proc_colors = ("#4c78a8", "#f58518")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=True, sharey=True)
    panels = (
        (axes[0], init_uav, "Initial placement (saved bank UAVs)", None),
        (axes[1], None, "Optimized UAV positions by method", methods),
    )

    def _draw_iot(ax) -> None:
        for idx, (xy, p) in enumerate(zip(iot, proc, strict=True)):
            color = proc_colors[int(p)]
            ax.scatter(
                xy[0],
                xy[1],
                c=color,
                s=52,
                zorder=3,
                edgecolors="white",
                linewidths=0.6,
            )
            ax.annotate(
                str(idx + 1),
                (xy[0], xy[1]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="0.15",
                zorder=6,
            )
        for k, color in enumerate(proc_colors):
            ax.scatter(
                [],
                [],
                c=color,
                s=52,
                edgecolors="white",
                linewidths=0.6,
                label=f"IoT $N_{{{k + 1}}}$ ({int(np.sum(proc == k))} devices)",
            )

    for ax, uav, title, panel_methods in panels:
        _draw_iot(ax)
        if panel_methods is None:
            ax.scatter(
                uav[:, 0],
                uav[:, 1],
                marker="^",
                s=110,
                c="#54a24b",
                edgecolors="0.15",
                linewidths=0.8,
                label="UAV (initial)",
                zorder=4,
            )
        else:
            for method in panel_methods:
                if method not in runs:
                    continue
                run = runs[method]
                uav_opt = np.asarray(run["uav_xyz_m"], dtype=float)
                style = METHOD_STYLES.get(method, {})
                label = (
                    f"{METHOD_LABELS.get(method, method)} "
                    f"({run['sum_rate_Mbps']:.3f} Mbps)"
                )
                ax.scatter(
                    uav_opt[:, 0],
                    uav_opt[:, 1],
                    marker=style.get("marker", "o"),
                    s=95,
                    c=style.get("color", "#555555"),
                    edgecolors="0.15",
                    linewidths=0.8,
                    label=label,
                    zorder=4,
                )
        ax.set_xlim(-0.02 * area_x, 1.02 * area_x)
        ax.set_ylim(-0.02 * area_y, 1.02 * area_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("x (m)")
        if ax is axes[0]:
            ax.set_ylabel("y (m)")

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.95, edgecolor="0.85")
    axes[1].legend(loc="upper left", fontsize=7.5, framealpha=0.95, edgecolor="0.85")
    fig.suptitle(
        f"Seed {seed} (scenario {scenario_id}) — "
        f"$I={len(iot)}$ IoT, $J={init_uav.shape[0]}$ UAV",
        fontsize=12,
        y=1.02,
    )
    note = _repro_note(payload)
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))

    stem = out / f"n100_seed{seed}_layout_comparison"
    return _save(fig, stem)


def _scenario_maps(plt, payload, bank, note, out: Path) -> Path:
    records = bank["scenarios"]
    n = len(records)
    pick = [0]
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        idx = min(n - 1, int(round(frac * (n - 1))))
        if idx not in pick:
            pick.append(idx)
    pick = pick[:6]
    geo = bank["geometry"]
    area_x = float(geo["area_x_m"])
    area_y = float(geo["area_y_m"])
    sca_uav = {}
    for run in payload.get("runs", []):
        if run["method"] == "sca":
            sca_uav[int(run["scenario_id"])] = np.asarray(run["uav_xyz_m"], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.6), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    for ax, idx in zip(axes, pick):
        rec = records[idx]
        iot = np.asarray(rec["iot_xyz_m"], dtype=float)
        proc = np.asarray(rec["process_id_of_iot"], dtype=int)
        uav = np.asarray(rec["uav_xyz_m"], dtype=float)
        for k, color in enumerate(("#4c78a8", "#f58518")):
            mask = proc == k
            ax.scatter(
                iot[mask, 0],
                iot[mask, 1],
                c=color,
                s=28,
                label=f"IoT N_{k+1}" if idx == pick[0] else None,
                zorder=3,
            )
        ax.scatter(
            uav[:, 0],
            uav[:, 1],
            marker="^",
            s=70,
            c="#54a24b",
            edgecolors="0.15",
            label="UAV (saved)" if idx == pick[0] else None,
            zorder=4,
        )
        trained = sca_uav.get(int(rec["id"]))
        if trained is not None:
            ax.scatter(
                trained[:, 0],
                trained[:, 1],
                marker="*",
                s=90,
                c="#e45756",
                edgecolors="0.15",
                label="UAV after SCA" if idx == pick[0] else None,
                zorder=5,
            )
        ax.set_xlim(-0.02 * area_x, 1.02 * area_x)
        ax.set_ylim(-0.02 * area_y, 1.02 * area_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"scenario {rec['id']}  seed={rec['seed']}", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(pick) :]:
        ax.set_visible(False)
    axes[0].legend(loc="best", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Saved area + IoT + UAV layouts", fontsize=12)
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    png = _save(fig, out / "n100_scenario_maps")
    plt.close(fig)
    return png
