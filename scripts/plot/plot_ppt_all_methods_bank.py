"""PPT figures: per-scenario Delta vs SCA and mean sum-rate bars for a scenario bank.

Loads the n200 ``eval_all_methods_dynamic.json`` bank by default (SCA = dynamic
assignment; frozen-SCA = former frozen ``sca``). Legacy JSON with
``sca`` + ``sca_dynamic`` is normalized on load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.experiments.method_labels import label_for, normalize_by_method

BASELINE = "sca"

METHODS_DELTA = (
    "frozen_sca",
    "pso",
    "kmeans",
    "random",
    "sca_multistart",
    "sca_anchor",
    "td3",
)
METHODS_BAR = (
    "sca",
    "frozen_sca",
    "pso",
    "sca_multistart",
    "sca_anchor",
    "td3",
    "kmeans",
    "random",
)
STYLES = {
    "sca": {"color": "#1f77b4"},
    "frozen_sca": {"color": "#aec7e8", "linewidth": 1.8, "alpha": 0.9, "zorder": 3.5},
    "pso": {"color": "#9467bd", "linewidth": 1.6, "alpha": 0.88, "zorder": 3},
    "sca_multistart": {"color": "#8c564b", "linewidth": 2.0, "alpha": 0.92, "zorder": 4},
    "sca_anchor": {"color": "#e377c2", "linewidth": 2.2, "alpha": 0.95, "zorder": 4.5},
    "td3": {"color": "#17becf", "linewidth": 1.8, "alpha": 0.9, "zorder": 3},
    "kmeans": {"color": "#ff7f0e", "linewidth": 1.6, "alpha": 0.85, "zorder": 3},
    "random": {"color": "#2ca02c", "linewidth": 1.6, "alpha": 0.85, "zorder": 3},
}


def _load_by_method(primary: Path, td3_path: Path | None = None) -> dict:
    payload = json.loads(primary.read_text(encoding="utf-8"))
    by = normalize_by_method(dict(payload["by_method"]))
    td3_path = td3_path or primary
    if "td3" not in by and td3_path.exists() and td3_path != primary:
        td3 = json.loads(td3_path.read_text(encoding="utf-8"))
        td3_m = normalize_by_method(td3.get("by_method") or {}).get("td3")
        if td3_m is not None:
            by["td3"] = td3_m
    anchor_path = primary.parent / "eval_anchor.json"
    if anchor_path.exists():
        anc = json.loads(anchor_path.read_text(encoding="utf-8"))
        anc_by = normalize_by_method(anc.get("by_method") or {})
        for m in ("sca_multistart", "sca_anchor"):
            if m in anc_by and m not in by:
                by[m] = anc_by[m]
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

    fig, ax = plt.subplots(figsize=(7.8, 4.35))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.30)
    all_delta: list[float] = []

    for method in METHODS_DELTA:
        if method not in by:
            continue
        rates = np.asarray(by[method]["per_seed_Mbps"], dtype=float)
        if rates.shape != baseline.shape:
            raise ValueError(f"{method} seed count != {BASELINE}")
        delta = rates - baseline
        all_delta.extend(delta.tolist())
        st = STYLES.get(method, {"color": "#555555", "linewidth": 1.5, "alpha": 0.85, "zorder": 3})
        ax.plot(
            xs,
            delta,
            label=label_for(method),
            color=st["color"],
            linewidth=st.get("linewidth", 1.5),
            alpha=st.get("alpha", 0.85),
            zorder=st.get("zorder", 3),
            solid_capstyle="round",
        )

    ax.axhline(
        0.0,
        color=STYLES["sca"]["color"],
        linewidth=2.0,
        linestyle="--",
        zorder=2,
        label=f"{label_for('sca')} (reference)",
    )

    y_min = float(np.min(all_delta)) if all_delta else -0.01
    y_max = float(np.max(all_delta)) if all_delta else 0.01
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
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=4,
        framealpha=0.95,
        edgecolor="0.85",
        fontsize=6.5,
        handlelength=1.4,
        columnspacing=0.9,
        labelspacing=0.35,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.16)
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
    ax.set_xticks(xs, [label_for(m) for m in methods], rotation=18, ha="right")
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


def plot_sca_vs_frozen(
    by: dict,
    *,
    out_stem: Path,
    field_m: float,
    n_layouts: int,
    practical_mbps: float = 0.05,
) -> Path:
    """Paired SCA vs frozen-SCA: tight sum-rate panel only (small markers)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frozen = np.asarray(by["frozen_sca"]["per_seed_Mbps"], dtype=float)
    sca = np.asarray(by["sca"]["per_seed_Mbps"], dtype=float)
    if frozen.shape != sca.shape:
        raise ValueError("frozen_sca and sca seed counts differ")
    n = int(frozen.size)
    if n_layouts != n:
        n_layouts = n
    xs = np.arange(1, n + 1)

    ms = 2.2
    # High-contrast pair (not the pale frozen-SCA bar color).
    c_frozen = "#d95f02"  # orange
    c_sca = "#1f77b4"  # blue
    lw_frozen = 0.95
    lw_sca = 1.05
    ls_frozen = (0, (4, 2))
    ls_sca = "-"

    fig, ax0 = plt.subplots(1, 1, figsize=(9.2, 2.55))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)
    ax0.plot(
        xs,
        frozen,
        linestyle=ls_frozen,
        linewidth=lw_frozen,
        color=c_frozen,
        alpha=0.92,
        zorder=2,
        label=label_for("frozen_sca"),
    )
    ax0.plot(
        xs,
        sca,
        linestyle=ls_sca,
        linewidth=lw_sca,
        color=c_sca,
        alpha=0.92,
        zorder=3,
        label=label_for("sca"),
    )
    ax0.scatter(
        xs,
        frozen,
        s=ms**2,
        c=c_frozen,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.25,
        zorder=4,
    )
    ax0.scatter(
        xs,
        sca,
        s=(ms + 0.35) ** 2,
        c=c_sca,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.25,
        zorder=5,
    )
    ax0.set_ylabel("Sum rate (Mbps)")
    rates = np.concatenate([frozen, sca])
    span = float(np.ptp(rates))
    pad = max(0.0008, 0.08 * span) if span > 1e-9 else 0.01
    ax0.set_ylim(float(rates.min()) - pad, float(rates.max()) + pad)
    ax0.set_xlim(0, n + 1)
    ax0.grid(True, alpha=0.3, linestyle=":")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(
        loc="lower right",
        fontsize=7.5,
        framealpha=0.95,
        handlelength=2.4,
        markerscale=0.85,
    )
    ax0.set_xlabel("Scenario index")
    ax0.set_title(
        rf"{int(field_m)}$\times${int(field_m)} m field ($I=10$, $J=3$, {n_layouts} layouts)",
        fontsize=10,
    )
    _ = practical_mbps  # kept for API stability; delta panel removed

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_stem.with_suffix(".pdf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Primary eval JSON (default: n200 eval_all_methods_dynamic by field)",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="Optional anchor/multistart eval (legacy default path)",
    )
    parser.add_argument(
        "--td3",
        type=Path,
        default=None,
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
    )
    parser.add_argument("--no-bars", action="store_true")
    parser.add_argument(
        "--only-sca-vs-frozen",
        action="store_true",
        help="Only write the paired SCA vs frozen-SCA figure",
    )
    args = parser.parse_args(argv)

    field = float(args.field_m)
    if args.eval_json is not None:
        primary = args.eval_json
    elif field <= 150:
        primary = ROOT / "results" / "n200" / "eval_all_methods_dynamic.json"
    else:
        primary = ROOT / "results" / "n200_500m_cap25" / "eval_all_methods_dynamic.json"
    td3 = args.td3
    if td3 is None and field > 150:
        td3 = ROOT / "results" / "n200_500m_cap25" / "eval_td3.json"
    elif td3 is None:
        td3 = ROOT / "results" / "n200" / "eval_td3.json"

    stem = args.stem.strip() or (
        "n200_mean_sum_rate" if field <= 150 else "n200_500m_mean_sum_rate"
    )
    pair_stem = (
        "n200_sca_vs_frozen" if field <= 150 else "n200_500m_sca_vs_frozen"
    )
    by = _load_by_method(primary, td3)
    if BASELINE not in by:
        raise SystemExit(f"{BASELINE} missing in {primary}; keys={list(by)}")
    if "frozen_sca" not in by:
        raise SystemExit(f"frozen_sca missing in {primary}; keys={list(by)}")
    n = int(by[BASELINE]["n"])
    plot_sca_vs_frozen(
        by,
        out_stem=args.out_dir / pair_stem,
        field_m=field,
        n_layouts=args.n_layouts or n,
    )
    print((args.out_dir / pair_stem).with_suffix(".pdf"))
    if args.only_sca_vs_frozen:
        return 0
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
