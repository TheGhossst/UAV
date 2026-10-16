"""10 MHz controls that are not in the cap sweep.

PSO, p-median (``sca_medoid``), random-anchor, continuous-candidate,
and the leftover-dump upper bound. 100 scenarios, CVXPY, 30 SCA
iterations. Writes under ``results/run_10mhz_controls`` and does not
touch ``results/run_10mhz_cap25``.

Usage:
  python scripts/campaigns/run_10mhz_controls.py
  python scripts/campaigns/run_10mhz_controls.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402

from uavdt.config import headline_sim_config  # noqa: E402
from uavdt.experiments.campaign import (  # noqa: E402
    CampaignSettings,
    run_campaign,
    write_campaign,
)
from uavdt.experiments.grids import AXES  # noqa: E402
from uavdt.experiments.n100 import evaluate_bank, write_eval  # noqa: E402
from uavdt.experiments.scenario_bank import (  # noqa: E402
    eval_cfg_from_bank,
    generate_bank,
    load_bank,
    scenario_from_record,
    write_bank,
)
from uavdt.placement.pso import PSOSettings  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import AnchorSettings, leftover_dump_upper_bound  # noqa: E402

OUT = ROOT / "results" / "run_10mhz_controls"
BASE = ROOT / "results" / "run_10mhz_cap25"
BANK = {
    100: ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
    500: ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
}
J2_BANK = OUT / "bank_n100_i10_j2_500m.json"
B_SYS = 10_000_000.0
N = 100
PRACTICAL = 0.05
TIE = 1e-6  # 1 bit/s
SCA = SCASettings(solver=None, max_iterations=30, epsilon=1e-4, step_size_m=20.0)
# Repeated default point. Keep the UAV-axis J=3 row as the unique copy.
REPEAT_DEFAULT = {("iots", 10.0), ("lambda", 2.0), ("aodt", 2.8), ("cpu", 2.0e8)}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _slug(share: float | None) -> str:
    if share is None:
        return "nocap"
    return f"cap{int(round(float(share) * 100))}"


def _share_ok(got: object, want: float | None) -> bool:
    if want is None:
        return got is None
    try:
        return abs(float(got) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return False


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_n100(
    path: Path, method: str, control: str, share: float | None
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load(path)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("control") != control or payload.get("checkpoint") is True:
        return False
    cfg = payload.get("cfg") or {}
    if abs(float(cfg.get("b_sys_hz") or 0) - B_SYS) > 1.0:
        return False
    if not _share_ok(cfg.get("max_bw_share"), share):
        return False
    stats = (payload.get("by_method") or {}).get(method) or {}
    return int(stats.get("n") or 0) == N


def _complete_campaign(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load(path)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("checkpoint") is True:
        return False
    if list(payload.get("methods") or []) != ["pso"]:
        return False
    if int(payload.get("n_runs") or 0) != N:
        return False
    points = payload.get("points") or []
    if len(points) < 24:
        return False
    return all(int((pt.get("by_method") or {}).get("pso", {}).get("n") or 0) == N for pt in points)


@dataclass(frozen=True)
class N100Job:
    control: str
    area: int
    share: float | None
    method: str
    filename: str
    bank: Path
    anchor: AnchorSettings | None = None
    num_uav: int = 3


def _jobs() -> list[N100Job]:
    jobs: list[N100Job] = []
    for area in (100, 500):
        jobs.append(
            N100Job("pso", area, 0.25, "pso", "n100_pso.json", BANK[area])
        )
    for share in (None, 0.12, 0.15):
        for area in (100, 500):
            jobs.append(
                N100Job("pso", area, share, "pso", "n100_pso.json", BANK[area])
            )
    for area in (100, 500):
        jobs.append(
            N100Job(
                "medoid",
                area,
                0.25,
                "sca_medoid",
                "n100_medoid.json",
                BANK[area],
            )
        )
    jobs.append(
        N100Job(
            "medoid_j2",
            500,
            0.25,
            "sca_medoid",
            "n100_medoid_j2.json",
            J2_BANK,
            num_uav=2,
        )
    )
    jobs.append(
        N100Job(
            "random_anchor",
            500,
            0.25,
            "sca_anchor",
            "n100_random_anchor.json",
            BANK[500],
            anchor=AnchorSettings(selection="random", n_random=1, top_k=1),
        )
    )
    jobs.append(
        N100Job(
            "continuous",
            500,
            0.25,
            "sca_continuous",
            "n100_continuous.json",
            BANK[500],
        )
    )
    return jobs


def _ensure_j2_bank() -> None:
    if J2_BANK.is_file():
        bank = load_bank(J2_BANK)
        geo = bank["geometry"]
        if (
            int(bank["n_scenarios"]) == N
            and int(geo["num_uav"]) == 2
            and abs(float(geo["area_x_m"]) - 500.0) < 1e-6
        ):
            return
    cfg = headline_sim_config(area_m=500.0, max_bw_share=0.25)
    bank = generate_bank(N, cfg, seed_start=1, num_iot=10, num_uav=2)
    write_bank(bank, J2_BANK)
    _log(f"wrote {J2_BANK.relative_to(ROOT)}")


def _run_n100(job: N100Job) -> None:
    out = OUT / _slug(job.share) / f"{job.area}m" / job.filename
    if _complete_n100(out, job.method, job.control, job.share):
        _log(f"skip complete {out.relative_to(ROOT)}")
        return
    if job.num_uav == 2:
        _ensure_j2_bank()
    if not job.bank.is_file():
        raise SystemExit(f"missing bank {job.bank}")
    bank = load_bank(job.bank)
    overlay = headline_sim_config(area_m=float(job.area), max_bw_share=job.share)
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    _log(
        f"start {job.control} {job.area}m {_slug(job.share)} "
        f"method={job.method} -> {out.relative_to(ROOT)}"
    )
    t0 = perf_counter()
    payload = evaluate_bank(
        bank,
        overlay,
        (job.method,),
        sca_settings=SCA,
        pso_settings=PSOSettings(),
        anchor_settings=job.anchor or AnchorSettings(),
        checkpoint_path=ckpt,
        resume=True,
        bank_path=job.bank,
    )
    payload["control"] = job.control
    write_eval(payload, out)
    _log(f"wrote {out.relative_to(ROOT)}  ({perf_counter() - t0:.1f}s)")


def _run_pso_campaign(area: int) -> None:
    out = OUT / "cap25" / f"{area}m" / "campaign_pso.json"
    if _complete_campaign(out):
        _log(f"skip complete {out.relative_to(ROOT)}")
        return
    cfg = headline_sim_config(area_m=float(area), max_bw_share=0.25)
    ckpt = out.with_name(out.stem + ".checkpoint.json")
    settings = CampaignSettings(
        n_runs=N,
        seed_start=1,
        methods=("pso",),
        sca_settings=SCA,
        pso_settings=PSOSettings(),
    )
    _log(f"start pso campaign {area}m -> {out.relative_to(ROOT)}")
    t0 = perf_counter()
    payload = run_campaign(AXES, cfg, settings, checkpoint_path=ckpt)
    payload["control"] = "pso_campaign"
    write_campaign(payload, out)
    if ckpt.exists():
        ckpt.unlink()
    _log(f"wrote {out.relative_to(ROOT)}  ({perf_counter() - t0:.1f}s)")


def _rates(stats: dict) -> dict[int, float]:
    return {
        int(seed): float(rate)
        for seed, rate in zip(stats["seeds"], stats["per_seed_Mbps"])
    }


def _pair(left: dict[int, float], right: dict[int, float]) -> dict:
    seeds = sorted(set(left) & set(right))
    delta = np.array([left[s] - right[s] for s in seeds], dtype=float)
    if delta.size == 0:
        return {"n": 0}
    wil = wilcoxon_signed_rank(delta, alternative="two-sided")
    return {
        "n": int(delta.size),
        "mean_Mbps": float(delta.mean()),
        "std_Mbps": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
        "n_higher": int((delta > TIE).sum()),
        "n_lower": int((delta < -TIE).sum()),
        "n_tie": int((np.abs(delta) <= TIE).sum()),
        "n_practical": int((delta > PRACTICAL).sum()),
        "n_practical_loss": int((delta < -PRACTICAL).sum()),
        "p_two_sided": wil["p_two_sided"],
        "p_greater": wil["p_greater"],
    }


def _fmt_pair(name: str, stats: dict) -> str:
    if not stats or stats.get("n", 0) == 0:
        return f"{name}: no paired seeds"
    return (
        f"{name}: {stats['mean_Mbps']:+.3f} ± {stats['std_Mbps']:.3f} Mbps, "
        f"{stats['n_higher']}/{stats['n']} higher, "
        f"{stats['n_lower']} lower, {stats['n_tie']} ties, "
        f"{stats['n_practical']}/{stats['n']} above {PRACTICAL:g} Mbps, "
        f"p={stats['p_two_sided']:.2e}"
    )


def _mean_line(label: str, stats: dict) -> str:
    return (
        f"{label:16s}  {stats['mean_sum_rate_Mbps']:.3f} ± "
        f"{stats['std_sum_rate_Mbps']:.3f}  "
        f"feas {100.0 * stats['feasible_fraction']:.0f}%  n={stats['n']}"
    )


def _baseline_n100(area: int) -> dict:
    path = BASE / f"{area}m" / "n100_all.json"
    payload = _load(path)
    return payload["by_method"]


def _method_stats(path: Path, method: str) -> dict | None:
    if not path.is_file():
        return None
    payload = _load(path)
    return (payload.get("by_method") or {}).get(method)


def _wall(path: Path, method: str) -> float | None:
    if not path.is_file():
        return None
    payload = _load(path)
    times = []
    for run in payload.get("runs") or []:
        if run.get("method") != method:
            continue
        sec = (run.get("diagnostics") or {}).get("wall_clock_s")
        if sec is not None:
            times.append(float(sec))
    if not times:
        return None
    return float(np.mean(times))


def _bh(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(np.asarray(pvals, dtype=float))
    reject_upto = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= rank / m * alpha:
            reject_upto = rank
    passed = [False] * m
    if reject_upto > 0:
        for idx in order[:reject_upto]:
            passed[int(idx)] = True
    return passed


def _bound(area: int) -> dict:
    bank = load_bank(BANK[area])
    cfg = eval_cfg_from_bank(bank, headline_sim_config(area_m=float(area), max_bw_share=0.25))
    base = _baseline_n100(area)
    anchor = _rates(base["sca_anchor"])
    sca = _rates(base["sca"])
    rows = []
    for rec in bank["scenarios"]:
        seed = int(rec["seed"])
        bound = leftover_dump_upper_bound(scenario_from_record(rec, cfg))
        row = {
            "seed": seed,
            "bound_Mbps": float(bound["bound_Mbps"]),
            "ceiling_Mbps": float(bound["radio_ceiling_Mbps"]),
            "anchor_Mbps": anchor.get(seed),
            "sca_Mbps": sca.get(seed),
        }
        if row["anchor_Mbps"] is not None:
            row["gap_anchor_Mbps"] = row["bound_Mbps"] - row["anchor_Mbps"]
        if row["anchor_Mbps"] is not None and row["sca_Mbps"] is not None:
            denom = row["bound_Mbps"] - row["sca_Mbps"]
            row["closed"] = (
                None
                if abs(denom) < 1e-9
                else (row["anchor_Mbps"] - row["sca_Mbps"]) / denom
            )
        rows.append(row)
    bounds = np.array([r["bound_Mbps"] for r in rows], dtype=float)
    gaps = np.array(
        [r["gap_anchor_Mbps"] for r in rows if r.get("gap_anchor_Mbps") is not None],
        dtype=float,
    )
    pcts = np.array(
        [
            100.0 * r["gap_anchor_Mbps"] / r["bound_Mbps"]
            for r in rows
            if r.get("gap_anchor_Mbps") is not None and r["bound_Mbps"]
        ],
        dtype=float,
    )
    closed = np.array(
        [r["closed"] for r in rows if r.get("closed") is not None],
        dtype=float,
    )
    return {
        "n": len(rows),
        "mean_bound_Mbps": float(bounds.mean()),
        "mean_ceiling_Mbps": float(np.mean([r["ceiling_Mbps"] for r in rows])),
        "mean_gap_anchor_Mbps": float(gaps.mean()) if gaps.size else None,
        "mean_gap_anchor_pct": float(pcts.mean()) if pcts.size else None,
        "mean_closed": float(closed.mean()) if closed.size else None,
        "n_anchor_above_bound": int(np.sum(gaps < -1e-6)) if gaps.size else 0,
    }


def _campaign_pairs(area: int) -> list[dict]:
    pso_path = OUT / "cap25" / f"{area}m" / "campaign_pso.json"
    base_path = BASE / f"{area}m" / "campaign_all.json"
    if not pso_path.is_file() or not base_path.is_file():
        return []
    pso = _load(pso_path)
    base = _load(base_path)
    base_pts = {
        (str(pt["axis"]), float(pt["x"])): pt for pt in base.get("points") or []
    }
    rows = []
    for pt in pso.get("points") or []:
        key = (str(pt["axis"]), float(pt["x"]))
        other = base_pts.get(key)
        if other is None:
            continue
        left = _rates(pt["by_method"]["pso"])
        row = {
            "axis": key[0],
            "x": key[1],
            "pso_Mbps": pt["by_method"]["pso"]["mean_sum_rate_Mbps"],
            "pso_feas": pt["by_method"]["pso"]["feasible_fraction"],
            "repeated_default": key in REPEAT_DEFAULT,
        }
        for method in ("random", "kmeans", "sca", "sca_multistart", "sca_anchor"):
            stats = (other.get("by_method") or {}).get(method)
            if not stats:
                continue
            row[f"{method}_Mbps"] = stats["mean_sum_rate_Mbps"]
            row[f"pso_minus_{method}"] = _pair(left, _rates(stats))
        rows.append(row)
    unique = [r for r in rows if not r["repeated_default"]]
    for method in ("sca", "kmeans", "random"):
        pvals = [r[f"pso_minus_{method}"]["p_greater"] for r in unique]
        flags = _bh(pvals)
        for row, flag in zip(unique, flags):
            row[f"bh_pso_gt_{method}"] = flag
    return rows


def analyze() -> dict:
    lines: list[str] = []
    lines.append("10 MHz controls, 100 scenarios, CVXPY, 30 SCA iterations")
    lines.append("Baselines: results/run_10mhz_cap25. New runs: results/run_10mhz_controls.")
    lines.append("Practical bar: 0.05 Mbps. Tie: 1 bit/s. p is two-sided Wilcoxon.")
    lines.append("")

    report: dict = {"n100": {}, "bound": {}, "campaign": {}, "runtime_s": {}}

    for area in (100, 500):
        lines.append(f"## n100 default point, {area} x {area} m, 25% cap")
        base = _baseline_n100(area)
        block: dict = {}
        for method in ("random", "kmeans", "sca", "sca_multistart", "sca_anchor"):
            lines.append(_mean_line(method, base[method]))
            block[method] = {
                "mean_Mbps": base[method]["mean_sum_rate_Mbps"],
                "std_Mbps": base[method]["std_sum_rate_Mbps"],
                "feasible": base[method]["feasible_fraction"],
            }
        extras = [
            ("pso", OUT / "cap25" / f"{area}m" / "n100_pso.json", "pso"),
            ("sca_medoid", OUT / "cap25" / f"{area}m" / "n100_medoid.json", "sca_medoid"),
        ]
        if area == 500:
            extras.extend(
                [
                    (
                        "random_anchor",
                        OUT / "cap25" / "500m" / "n100_random_anchor.json",
                        "sca_anchor",
                    ),
                    (
                        "continuous",
                        OUT / "cap25" / "500m" / "n100_continuous.json",
                        "sca_continuous",
                    ),
                ]
            )
        fresh: dict[str, dict[int, float]] = {}
        for label, path, method in extras:
            stats = _method_stats(path, method)
            if stats is None:
                lines.append(f"{label:16s}  missing {path.relative_to(ROOT)}")
                continue
            lines.append(_mean_line(label, stats))
            fresh[label] = _rates(stats)
            block[label] = {
                "mean_Mbps": stats["mean_sum_rate_Mbps"],
                "std_Mbps": stats["std_sum_rate_Mbps"],
                "feasible": stats["feasible_fraction"],
            }
            sec = _wall(path, method)
            if sec is not None:
                report["runtime_s"][f"{label}_{area}m"] = sec
                lines.append(f"{'':16s}  mean wall {sec:.2f} s/scenario")
        base_rates = {name: _rates(base[name]) for name in base}
        lines.append("Paired, new method minus baseline:")
        comparisons = []
        if "pso" in fresh:
            for other in ("sca", "kmeans", "random", "sca_anchor", "sca_multistart"):
                comparisons.append((f"pso - {other}", fresh["pso"], base_rates[other]))
        if "sca_medoid" in fresh:
            for other in ("sca_anchor", "sca", "sca_multistart", "kmeans"):
                comparisons.append(
                    (f"medoid - {other}", fresh["sca_medoid"], base_rates[other])
                )
        if "random_anchor" in fresh:
            for other in ("sca", "sca_anchor", "sca_multistart"):
                comparisons.append(
                    (f"random-anchor - {other}", fresh["random_anchor"], base_rates[other])
                )
        if "continuous" in fresh:
            for other in ("sca_anchor", "sca_multistart", "sca"):
                comparisons.append(
                    (f"continuous - {other}", fresh["continuous"], base_rates[other])
                )
        pair_out = {}
        for name, left, right in comparisons:
            stats = _pair(left, right)
            pair_out[name] = stats
            lines.append(_fmt_pair(name, stats))
        if area == 500:
            j2 = _method_stats(OUT / "cap25" / "500m" / "n100_medoid_j2.json", "sca_medoid")
            if j2 is not None:
                lines.append(_mean_line("medoid J=2", j2))
                block["medoid_j2"] = {
                    "mean_Mbps": j2["mean_sum_rate_Mbps"],
                    "std_Mbps": j2["std_sum_rate_Mbps"],
                    "feasible": j2["feasible_fraction"],
                }
        bound = _bound(area)
        report["bound"][f"{area}m"] = bound
        lines.append(
            f"upper bound      {bound['mean_bound_Mbps']:.3f} Mbps  "
            f"ceiling {bound['mean_ceiling_Mbps']:.3f}  "
            f"anchor gap {bound['mean_gap_anchor_Mbps']:.3f} Mbps "
            f"({bound['mean_gap_anchor_pct']:.2f}%)  "
            f"closed {bound['mean_closed']:.3f} of the SCA-to-bound gap  "
            f"anchor above bound {bound['n_anchor_above_bound']}/{bound['n']}"
        )
        block["pairs"] = pair_out
        report["n100"][f"{area}m"] = block
        lines.append("")

    lines.append("## PSO n100 at other caps (no local SCA column yet; the cap sweep is still on the server)")
    for share in (None, 0.12, 0.15):
        for area in (100, 500):
            path = OUT / _slug(share) / f"{area}m" / "n100_pso.json"
            stats = _method_stats(path, "pso")
            label = f"pso {_slug(share)} {area}m"
            if stats is None:
                lines.append(f"{label:24s}  missing")
                continue
            lines.append(_mean_line(label, stats))
            report["n100"].setdefault("other_caps", {})[f"{_slug(share)}_{area}m"] = {
                "mean_Mbps": stats["mean_sum_rate_Mbps"],
                "std_Mbps": stats["std_sum_rate_Mbps"],
                "feasible": stats["feasible_fraction"],
            }
    lines.append("")

    for area in (100, 500):
        rows = _campaign_pairs(area)
        report["campaign"][f"{area}m"] = rows
        lines.append(f"## PSO campaign, {area} x {area} m, 25% cap")
        if not rows:
            lines.append("missing campaign_pso.json or baseline campaign_all.json")
            lines.append("")
            continue
        lines.append(
            f"{'axis':8s} {'x':10s} {'PSO':8s} {'SCA':8s} {'k-means':8s} {'anchor':8s}  PSO-SCA"
        )
        for row in rows:
            if row["repeated_default"]:
                continue
            gap = row.get("pso_minus_sca") or {}
            lines.append(
                f"{row['axis']:8s} {row['x']:<10g} "
                f"{row['pso_Mbps']:8.3f} "
                f"{row.get('sca_Mbps', float('nan')):8.3f} "
                f"{row.get('kmeans_Mbps', float('nan')):8.3f} "
                f"{row.get('sca_anchor_Mbps', float('nan')):8.3f}  "
                f"{gap.get('mean_Mbps', float('nan')):+.3f}  "
                f"{gap.get('n_higher', 0)}/{gap.get('n', 0)}  "
                f"p={gap.get('p_two_sided', float('nan')):.2e}"
            )
        unique = [r for r in rows if not r["repeated_default"]]
        for method, label in (("sca", "SCA"), ("kmeans", "k-means"), ("random", "random")):
            flags = [r.get(f"bh_pso_gt_{method}") for r in unique]
            n_sig = sum(1 for flag in flags if flag)
            n_mean = sum(
                1
                for r in unique
                if (r.get(f"pso_minus_{method}") or {}).get("mean_Mbps", -1) > 0
            )
            lines.append(
                f"BH-FDR 0.05, one-sided PSO > {label}: {n_sig}/{len(unique)} points; "
                f"mean PSO higher on {n_mean}/{len(unique)}"
            )
        j2_rows = [r for r in rows if r["axis"] == "uavs" and abs(r["x"] - 2.0) < 1e-9]
        med = _method_stats(OUT / "cap25" / "500m" / "n100_medoid_j2.json", "sca_medoid")
        if area == 500 and j2_rows and med is not None:
            camp = _load(BASE / "500m" / "campaign_all.json")
            pt = next(
                p
                for p in camp["points"]
                if p["axis"] == "uavs" and abs(float(p["x"]) - 2.0) < 1e-9
            )
            left = _rates(med)
            lines.append("J=2, 500 m, p-median versus the campaign seeds:")
            lines.append(_mean_line("medoid J=2", med))
            for method in ("sca", "sca_anchor", "kmeans", "random"):
                lines.append(_fmt_pair(f"medoid - {method}", _pair(left, _rates(pt["by_method"][method]))))
            pso_path = OUT / "cap25" / "500m" / "campaign_pso.json"
            if pso_path.is_file():
                pso_payload = _load(pso_path)
                pso_pt = next(
                    p
                    for p in pso_payload["points"]
                    if p["axis"] == "uavs" and abs(float(p["x"]) - 2.0) < 1e-9
                )
                lines.append(
                    _fmt_pair(
                        "medoid - pso",
                        _pair(left, _rates(pso_pt["by_method"]["pso"])),
                    )
                )
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.txt").write_text(text, encoding="utf-8")
    (OUT / "analysis.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _log(text)
    _log(f"wrote {(OUT / 'analysis.txt').relative_to(ROOT)}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    failures: list[str] = []
    if not args.analyze_only:
        for job in _jobs():
            try:
                _run_n100(job)
            except Exception:
                failures.append(f"n100 {job.control} {job.area} {_slug(job.share)}")
                _log(traceback.format_exc())
        for area in (100, 500):
            try:
                _run_pso_campaign(area)
            except Exception:
                failures.append(f"campaign pso {area}")
                _log(traceback.format_exc())
    analyze()
    if failures:
        _log("FAILED: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
