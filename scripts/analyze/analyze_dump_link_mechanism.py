"""Zenith-anchor gain vs SCA dump-link SE quality (mechanism plot).

For each bank layout, take the SCA endpoint UAV positions, count how many of
the ceil(1/cap) highest-SE associated dump slots satisfy
SE_ij >= 0.98 * SE_max (zenith ceiling), and plot anchor_gain_Mbps against
that count.

Example (500 m bank, **10 MHz** eval — do not use legacy 8.8 MHz ``n100_500m_cap25`` JSON):

    python -m uavdt n100 --bank data/scenario_bank/n100_i10_j3_500m.json \\
        --bandwidth-preset 10mhz --max-bw-share 0.25 \\
        --methods sca,sca_anchor --skip-plot \\
        --out results/test/eval_n100_500m_10mhz_cap25.json

    python scripts/analyze/analyze_dump_link_mechanism.py \\
        --bank data/scenario_bank/n100_i10_j3_500m.json \\
        --sca-eval results/test/eval_n100_500m_10mhz_cap25.json \\
        --anchor-eval results/test/eval_n100_500m_10mhz_cap25.json \\
        --out-json results/test/sca_dump_link_mechanism_10mhz_n100_500m.json \\
        --out-plot results/test/sca_dump_link_mechanism_10mhz_n100_500m.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.analysis.dump_links import dump_link_se_audit  # noqa: E402
from uavdt.config import PRIMARY_B_SYS_HZ, SimConfig  # noqa: E402
from uavdt.experiments.scenario_bank import scenario_from_record  # noqa: E402


def _require_primary_b_sys_hz(b_sys_hz: float, *, allow_legacy: bool) -> None:
    if allow_legacy:
        return
    if abs(float(b_sys_hz) - float(PRIMARY_B_SYS_HZ)) > 1.0:
        raise SystemExit(
            f"eval cfg b_sys_hz={b_sys_hz:g} is not primary 10 MHz ({PRIMARY_B_SYS_HZ:g}). "
            "Re-run n100/campaign with --bandwidth-preset 10mhz, or pass --allow-legacy-b-sys "
            "(legacy 8.8 MHz artifacts are not valid for headline mechanism plots)."
        )


def _load_runs(path: Path, method: str) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for run in payload.get("runs", []):
        if str(run.get("method")) != method:
            continue
        seed = int(run["seed"])
        out[seed] = run
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--sca-eval", type=Path, required=True)
    p.add_argument("--anchor-eval", type=Path, required=True)
    p.add_argument("--anchor-method", type=str, default="sca_anchor")
    p.add_argument("--sca-method", type=str, default="sca")
    p.add_argument("--se-frac", type=float, default=0.98)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-plot", type=Path, required=True)
    p.add_argument(
        "--allow-legacy-b-sys",
        action="store_true",
        help="Allow eval JSON with b_sys_hz != 10 MHz (not for headline plots).",
    )
    args = p.parse_args()

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    sca_payload = json.loads(args.sca_eval.read_text(encoding="utf-8"))
    b_sys = float(sca_payload.get("b_sys_hz", sca_payload["cfg"]["b_sys_hz"]))
    _require_primary_b_sys_hz(b_sys, allow_legacy=bool(args.allow_legacy_b_sys))
    cfg = SimConfig(**sca_payload["cfg"])
    if abs(float(cfg.b_sys_hz) - float(PRIMARY_B_SYS_HZ)) > 1.0 and not args.allow_legacy_b_sys:
        raise SystemExit(
            f"eval cfg block has b_sys_hz={cfg.b_sys_hz:g}; expected {PRIMARY_B_SYS_HZ:g}"
        )
    sca_runs = _load_runs(args.sca_eval, args.sca_method)
    anchor_runs = _load_runs(args.anchor_eval, args.anchor_method)
    seeds = sorted(set(sca_runs) & set(anchor_runs))
    if not seeds:
        raise SystemExit("no overlapping seeds between SCA and anchor eval files")

    rows: list[dict] = []
    for seed in seeds:
        sca = sca_runs[seed]
        anc = anchor_runs[seed]
        rec = next(r for r in bank["scenarios"] if int(r["seed"]) == int(seed))
        sc = scenario_from_record(rec, cfg)
        iot = np.asarray(rec["iot_xyz_m"], dtype=float)
        uav = np.asarray(sca["uav_xyz_m"], dtype=float)
        audit = dump_link_se_audit(sc, uav, se_quality_frac=float(args.se_frac))
        gain = float(anc["sum_rate_Mbps"]) - float(sca["sum_rate_Mbps"])
        rows.append(
            {
                "seed": int(seed),
                "scenario_id": int(rec["id"]),
                "iot_xyz_m": iot.tolist(),
                "sca_uav_xyz_m": uav.tolist(),
                "anchor_uav_xyz_m": np.asarray(anc["uav_xyz_m"], dtype=float).tolist(),
                "sca_Mbps": float(sca["sum_rate_Mbps"]),
                "anchor_Mbps": float(anc["sum_rate_Mbps"]),
                "anchor_gain_Mbps": gain,
                "k_dump": audit.k_dump,
                "se_max_bit_per_hz": audit.se_max_bit_per_hz,
                "se_threshold_bit_per_hz": audit.se_threshold_bit_per_hz,
                "n_good_dump_links": audit.n_good_dump_links,
                "frac_good_dump_links": audit.frac_good,
                "dump_links": audit.dump_links,
            }
        )

    by_count: dict[int, list[float]] = {}
    for row in rows:
        by_count.setdefault(int(row["n_good_dump_links"]), []).append(
            float(row["anchor_gain_Mbps"])
        )
    summary = {
        "n": len(rows),
        "se_quality_frac": float(args.se_frac),
        "bank": str(args.bank),
        "sca_eval": str(args.sca_eval),
        "anchor_eval": str(args.anchor_eval),
        "b_sys_hz": float(cfg.b_sys_hz),
        "max_bw_share": cfg.max_bw_share,
        "mean_gain_Mbps": float(np.mean([r["anchor_gain_Mbps"] for r in rows])),
        "gain_by_n_good_dump_links": {
            str(k): {
                "n": len(v),
                "mean_gain_Mbps": float(np.mean(v)),
                "median_gain_Mbps": float(np.median(v)),
            }
            for k, v in sorted(by_count.items())
        },
        "low_good_0_2": {
            "n": sum(1 for r in rows if r["n_good_dump_links"] <= 2),
            "mean_gain_Mbps": float(
                np.mean(
                    [
                        r["anchor_gain_Mbps"]
                        for r in rows
                        if r["n_good_dump_links"] <= 2
                    ]
                )
            )
            if any(r["n_good_dump_links"] <= 2 for r in rows)
            else float("nan"),
        },
        "high_good_3_plus": {
            "n": sum(1 for r in rows if r["n_good_dump_links"] >= 3),
            "mean_gain_Mbps": float(
                np.mean(
                    [
                        r["anchor_gain_Mbps"]
                        for r in rows
                        if r["n_good_dump_links"] >= 3
                    ]
                )
            )
            if any(r["n_good_dump_links"] >= 3 for r in rows)
            else float("nan"),
        },
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    xs = np.array([r["n_good_dump_links"] for r in rows], dtype=float)
    ys = np.array([r["anchor_gain_Mbps"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    jitter = (np.random.default_rng(0).random(len(xs)) - 0.5) * 0.08
    ax.scatter(xs + jitter, ys, alpha=0.75, s=36, edgecolors="k", linewidths=0.3)
    for k in sorted(by_count):
        vals = by_count[k]
        ax.scatter(
            [k],
            [float(np.mean(vals))],
            s=120,
            marker="D",
            color="crimson",
            zorder=5,
            label="mean per count" if k == sorted(by_count)[0] else None,
        )
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel(
        f"Good dump links (SE ≥ {args.se_frac:g}·SE_max) "
        f"among top k=⌈1/cap⌉={rows[0]['k_dump']}"
    )
    ax.set_ylabel("Zenith-anchor gain over SCA (Mbps)")
    ax.set_xticks(range(int(rows[0]["k_dump"]) + 1))
    ax.set_title(
        f"Mechanism probe: n={len(rows)}  "
        f"B_sys={cfg.b_sys_hz/1e6:g} MHz  cap={cfg.max_bw_share}"
    )
    ax.grid(True, alpha=0.25)
    if len(by_count) > 1:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=150)
    plt.close(fig)

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_plot}")
    print(
        f"mean gain {summary['mean_gain_Mbps']:.4f} Mbps; "
        f"n_good<=2: n={summary['low_good_0_2']['n']} "
        f"mean={summary['low_good_0_2']['mean_gain_Mbps']:.4f}; "
        f"n_good>=3: n={summary['high_good_3_plus']['n']} "
        f"mean={summary['high_good_3_plus']['mean_gain_Mbps']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
