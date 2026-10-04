"""Top-down scenario map: IoT devices and relay positions before/after SCA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from uavdt.config import headline_sim_config  # noqa: E402
from uavdt.experiments.methods import run_method  # noqa: E402
from uavdt.experiments.scenario_bank import (  # noqa: E402
    eval_cfg_from_bank,
    load_bank,
    scenario_from_record,
)
from uavdt.resources import nearest_association  # noqa: E402


def _draw_panel(
    ax,
    iot_xyz: np.ndarray,
    uav_xyz: np.ndarray,
    *,
    area_x: float,
    area_y: float,
    nit_blue: str,
) -> None:
    iot_xy = iot_xyz[:, :2]
    uav_xy = uav_xyz[:, :2]
    ax.scatter(
        iot_xy[:, 0],
        iot_xy[:, 1],
        c="#4c78a8",
        s=58,
        zorder=3,
        edgecolors="white",
        linewidths=0.55,
        label="IoT devices",
    )
    assoc = nearest_association(iot_xyz, uav_xyz)
    relay_for_iot = np.argmax(assoc, axis=1)
    for i, j in enumerate(relay_for_iot):
        ax.plot(
            [iot_xy[i, 0], uav_xy[j, 0]],
            [iot_xy[i, 1], uav_xy[j, 1]],
            linestyle=":",
            color="0.65",
            linewidth=0.85,
            zorder=1,
        )
    ax.scatter(
        uav_xy[:, 0],
        uav_xy[:, 1],
        marker="^",
        s=130,
        c=nit_blue,
        edgecolors="0.15",
        linewidths=0.75,
        zorder=4,
        label="Relays",
    )
    ax.set_xlim(-0.03 * area_x, 1.03 * area_x)
    ax.set_ylim(-0.03 * area_y, 1.03 * area_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.32, linestyle=":")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _save_single_panel(
    plt,
    out_stem: Path,
    iot_xyz: np.ndarray,
    uav_xyz: np.ndarray,
    *,
    area_x: float,
    area_y: float,
    nit_blue: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(4.55, 4.45))
    _draw_panel(
        ax,
        iot_xyz,
        uav_xyz,
        area_x=area_x,
        area_y=area_y,
        nit_blue=nit_blue,
    )
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.12)
    fig.savefig(out_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_stem.with_suffix(".pdf")


def plot_before_after(
    bank_path: Path,
    out_dir: Path,
    *,
    seed: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank = load_bank(bank_path)
    cfg = eval_cfg_from_bank(bank, headline_sim_config())
    record = next((r for r in bank["scenarios"] if int(r["seed"]) == int(seed)), None)
    if record is None:
        raise ValueError(f"seed {seed} not in {bank_path}")

    scenario = scenario_from_record(record, cfg)
    kmeans = run_method(scenario, "kmeans", seed)
    sca = run_method(scenario, "sca", seed)

    iot = np.asarray(scenario.iot_xyz_m, dtype=float)
    uav_before = np.asarray(kmeans.uav_xyz_m, dtype=float)
    uav_after = np.asarray(sca.uav_xyz_m, dtype=float)
    area_x = float(bank["geometry"]["area_x_m"])
    area_y = float(bank["geometry"]["area_y_m"])
    nit_blue = "#003e7e"

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.20, wspace=0.18)

    _draw_panel(
        axes[0],
        iot,
        uav_before,
        area_x=area_x,
        area_y=area_y,
        nit_blue=nit_blue,
    )
    _draw_panel(
        axes[1],
        iot,
        uav_after,
        area_x=area_x,
        area_y=area_y,
        nit_blue=nit_blue,
    )
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", framealpha=0.9, fontsize=9)

    stem = out_dir / "scenario_layout_before_after"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    kmeans_path = _save_single_panel(
        plt,
        out_dir / "scenario_layout_kmeans",
        iot,
        uav_before,
        area_x=area_x,
        area_y=area_y,
        nit_blue=nit_blue,
    )
    sca_path = _save_single_panel(
        plt,
        out_dir / "scenario_layout_sca",
        iot,
        uav_after,
        area_x=area_x,
        area_y=area_y,
        nit_blue=nit_blue,
    )
    return sca_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bank",
        type=Path,
        default=_ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "results" / "figures" / "midsem_10mhz_n100",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    path = plot_before_after(args.bank, args.out_dir, seed=args.seed)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
