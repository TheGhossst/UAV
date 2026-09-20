"""PPT figures: per-scenario Δ vs SCA and mean sum-rate bars for a scenario bank.

Default paths target the n200 100 m bank; pass --field-m 500 for the 500 m field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "sca"

METHODS_DELTA = (
    "sca_multistart",
    "sca_anchor",
    "td3",
    "kmeans",
    "random",
)
METHODS_BAR = (
    "sca",
    "sca_multistart",
    "sca_anchor",
    "td3",
    "kmeans",
    "random",
)
LABELS = {
    "sca": "SCA",
    "sca_multistart": "SCA multi-start",
    "sca_anchor": "SCA zenith-anchor",
    "td3": "TD3",
    "kmeans": "K-means",
    "random": "Random",
}
STYLES = {
    "sca": {"color": "#1f77b4"},
    "sca_multistart": {"color": "#8c564b", "linewidth": 2.0, "alpha": 0.92, "zorder": 4},
    "sca_anchor": {"color": "#e377c2", "linewidth": 2.2, "alpha": 0.95, "zorder": 4.5},
    "td3": {"color": "#17becf", "linewidth": 1.8, "alpha": 0.9, "zorder": 3},
    "kmeans": {"color": "#ff7f0e", "linewidth": 1.6, "alpha": 0.85, "zorder": 3},
    "random": {"color": "#2ca02c", "linewidth": 1.6, "alpha": 0.85, "zorder": 3},
}


def _load_by_method(anchor_path: Path, td3_path: Path) -> dict:
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    by = dict(anchor["by_method"])
    if "td3" not in by and td3_path.exists() and td3_path != anchor_path:
        td3 = json.loads(td3_path.read_text(encoding="utf-8"))
        td3_m = (td3.get("by_method") or {}).get("td3")
        if td3_m is not None:
            by["td3"] = td3_m
    return by


def plot_delta(
    by: dict,
    *,
    out_stem: Path,
    field_m: float,
    n_layouts: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline = np.asarray(by[BASELINE]["per_seed_Mbps"], dtype=float)
    n_scenarios = int(baseline.size)
    if n_scenarios != n_layouts:
        n_layouts = n_scenarios
    xs = np.arange(1, n_scenarios + 1)

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.15)
    all_delta: list[float] = []

    for method in METHODS_DELTA:
        if method not in by:
            continue
        rates = np.asarray(by[method]["per_seed_Mbps"], dtype=float)
        if rates.shape != baseline.shape:
            raise ValueError(f"{method} seed count != {BASELINE}")
        delta = rates - baseline
        all_delta.extend(delta.tolist())
        st = STYLES[method]
        ax.plot(
            xs,
            delta,
            label=LABELS[method],
            color=st["color"],
            linewidth=st["linewidth"],
            alpha=st["alpha"],
            zorder=st["zorder"],
            solid_capstyle="round",
        )

    ax.axhline(
        0.0,
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        zorder=2,
        label="SCA (reference)",
    )

    y_min = float(np.min(all_delta))
    y_max = float(np.max(all_delta))
    span = max(y_max - y_min, 0.005)
    pad = max(0.002, 0.15 * span)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(0, n_scenarios + 1)
    ax.margins(x=0.02)
    ax.set_xticks(np.linspace(1, n_scenarios, min(6, n_scenarios), dtype=int))
    ax.set_xlabel("Scenario index")
    ax.set_ylabel(r"$\Delta$ sum rate vs SCA (Mbps)")
    ax.set_title(
        rf"Paired $\Delta$ vs SCA ($I=10$, $J=3$, {int(field_m)}$\times${int(field_m)} m, "
        rf"{n_layouts} layouts)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        framealpha=0.95,
        edgecolor="0.85",
        fontsize=7.0,
        handlelength=1.5,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_stem.with_suffix(".pdf")


def plot_mean_bars(
    by: dict,
    *,
    out_stem: Path,
    field_m: float,
    n_layouts: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = [m for m in METHODS_BAR if m in by]
    xs = np.arange(len(methods))
    means = [by[m]["mean_sum_rate_Mbps"] for m in methods]
    stds = [by[m]["std_sum_rate_Mbps"] for m in methods]
    colors = [STYLES.get(m, {}).get("color", "#555555") for m in methods]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, edgecolor="0.2", width=0.62)
    ax.set_xticks(xs, [LABELS.get(m, m) for m in methods], rotation=18, ha="right")
    lo = min(means) - 2.0 * max(stds)
    hi = max(means) + 2.0 * max(stds)
    span = max(hi - lo, 0.02)
    ax.set_ylim(lo - 0.12 * span, hi + 0.22 * span)
    for x, mean, std in zip(xs, means, stds):
        ax.text(
            x,
            mean + std + 0.02 * span,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Mean sum rate (Mbps)")
    ax.set_title(
        rf"Bank average ($I=10$, $J=3$, {int(field_m)}$\times${int(field_m)} m, "
        rf"{n_layouts} layouts)",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_stem.with_suffix(".pdf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor",
        type=Path,
        default=ROOT / "results" / "n200" / "eval_anchor.json",
    )
    parser.add_argument(
        "--td3",
        type=Path,
        default=ROOT / "results" / "n200" / "eval_td3.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "figures" / "ppt_all_methods",
    )
    parser.add_argument("--field-m", type=float, default=100.0)
    parser.add_argument("--n-layouts", type=int, default=200)
    parser.add_argument(
        "--stem",
        type=str,
        default="",
        help="Output basename (default: n200_mean_sum_rate or n200_500m_mean_sum_rate)",
    )
    parser.add_argument("--no-bars", action="store_true")
    args = parser.parse_args(argv)

    field = float(args.field_m)
    stem = args.stem.strip() or (
        "n200_mean_sum_rate" if field <= 150 else "n200_500m_mean_sum_rate"
    )
    by = _load_by_method(args.anchor, args.td3)
    n = int(by[BASELINE]["n"])
    plot_delta(
        by,
        out_stem=args.out_dir / stem,
        field_m=field,
        n_layouts=args.n_layouts or n,
    )
    if not args.no_bars:
        bar_stem = args.out_dir / f"{stem}_bars"
        plot_mean_bars(
            by,
            out_stem=bar_stem,
            field_m=field,
            n_layouts=args.n_layouts or n,
        )
        print(bar_stem.with_suffix(".pdf"))
    print((args.out_dir / stem).with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
