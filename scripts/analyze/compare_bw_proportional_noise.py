"""Does zenith-subset still beat p-median and k-means SCA when noise scales with B?

Frozen Eq. (6) uses SNR = p g / sigma^2, so rate is linear in bandwidth and
leftover Hertz is dumped onto the highest-SE links. This check swaps only
the inner allocation to

    SNR(B) = p g / (N0 B),    N0 = sigma^2 / (B_sys / I)

so an equal share of B_sys reproduces the frozen SNR, then re-solves that
concave program (CVXPY rel_entr). Frozen Algorithm 1 is not edited and is
not re-run under the new radio: k-means-started SCA positions and its
association stay the constant-sigma^2 solution. Zenith-subset is re-ranked
with the inner solver of the model being scored. p-median is the Euclidean
optimum on IoT xy; geometric ties are broken by that same inner solver.

A second column repeats the ranking under frozen sigma^2 (linear LP, and
SCA's own evaluate() score) on the same seeds.

    python scripts/analyze/compare_bw_proportional_noise.py --n-runs 20 --areas 100 500

Verdicts use the 0.05 Mbps practical bar already used for anchor gains,
applied to the zenith-minus-SCA paired mean. ``zenith_still_wins`` means
that gap and the gap over p-median both clear the bar. ``zenith_beats_sca_not_pmedian``
means zenith still clears the bar against k-means SCA but not against
Euclidean p-median. ``zenith_ahead_of_sca_but_small`` means the SCA gap is
positive and under the bar. ``zenith_advantage_gone`` means zenith does not
lead k-means SCA.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from paired_winrate import wilcoxon_signed_rank  # noqa: E402
from uavdt.aodt import average_aodt_s  # noqa: E402
from uavdt.computation import offered_load, queue_unstable, service_rate_per_s  # noqa: E402
from uavdt.constraints import check_constraints  # noqa: E402
from uavdt.analysis.bw_proportional import (  # noqa: E402
    calibrated_noise_density_w_per_hz,
    equal_share_reference_hz,
    solve_bandwidth_proportional,
)
from uavdt.config import headline_sim_config  # noqa: E402
from uavdt.evaluator import evaluate  # noqa: E402
from uavdt.models import Allocation  # noqa: E402
from uavdt.placement.kmedoids import exact_pmedian_combos  # noqa: E402
from uavdt.resources import cpu_stable_processing, nearest_association  # noqa: E402
from uavdt.sca.algorithm import solve_sca  # noqa: E402
from uavdt.sca.cvx_problem import solve_bandwidth_at_fixed_q  # noqa: E402
from uavdt.sca.settings import SCASettings  # noqa: E402
from uavdt.sca_anchor import n_anchor_combos, uav_for_combo  # noqa: E402
from uavdt.scenario import generate_scenario  # noqa: E402

PRACTICAL_MBPS = 0.05
METHODS = ("zenith", "pmedian", "sca")


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _associate(scenario, uav: np.ndarray):
    a = nearest_association(scenario.iot_xyz_m, uav)
    b = cpu_stable_processing(scenario, a)
    return a, b


def _pack(rate_mbps: float, feasible: bool, **extra) -> dict:
    finite = bool(np.isfinite(rate_mbps))
    rate = float(rate_mbps) if finite else float("nan")
    row = {
        "rate_mbps": rate,
        "feasible": bool(feasible) and finite,
    }
    row.update(extra)
    return row


def _score_proportional(scenario, uav, association, processing) -> dict:
    res = solve_bandwidth_proportional(scenario, uav, association, processing)
    share = float("nan")
    if res.bandwidth_hz.size:
        share = float(np.max(res.bandwidth_hz) / scenario.cfg.b_sys_hz)
    if res.infeasible or not np.isfinite(res.objective_bit_per_s):
        return _pack(
            float("nan"),
            False,
            max_link_share=share,
            status=res.status,
            solver=res.solver_name,
        )
    cfg = scenario.cfg
    a = np.asarray(association, dtype=float)
    b = np.asarray(processing, dtype=float)
    mu = service_rate_per_s(cfg)
    rho = offered_load(b, scenario.lambdas_per_s, mu)
    stable = ~queue_unstable(b, scenario.lambdas_per_s, mu)
    rates = res.rates_bit_per_s
    aodt = average_aodt_s(
        scenario,
        a,
        b,
        rates,
        np.full(uav.shape[0], mu),
        uav_stable=stable,
    )
    assoc_rates = (a * rates).sum(axis=1)
    report = check_constraints(scenario, uav, a, b, res.bandwidth_hz, assoc_rates, rho, aodt)
    cap_ok = bool(np.all(res.bandwidth_hz <= cfg.link_bandwidth_cap_hz + 1.0))
    return _pack(
        res.objective_mbps,
        bool(report.feasible and cap_ok),
        max_link_share=share,
        status=res.status,
        solver=res.solver_name,
    )


def _score_constant(scenario, uav, association, processing) -> dict:
    res = solve_bandwidth_at_fixed_q(
        scenario,
        uav,
        association,
        processing,
        SCASettings(solver=None),
    )
    if res.infeasible:
        return _pack(float("nan"), False, max_link_share=float("nan"), status=res.status)
    alloc = Allocation(association, processing, res.bandwidth_hz)
    ev = evaluate(scenario, uav, alloc)
    share = float(np.max(res.bandwidth_hz) / scenario.cfg.b_sys_hz)
    return _pack(
        ev.sum_rate_mbps,
        ev.feasible,
        max_link_share=share,
        status=res.status,
    )


def _rank_key(row: dict) -> tuple[int, float]:
    rate = row["rate_mbps"]
    if not np.isfinite(rate):
        rate = -1.0
    return (int(bool(row["feasible"])), float(rate))


def _consider(best: dict | None, row: dict) -> dict:
    if best is None or _rank_key(row) > _rank_key(best):
        return row
    return best


def _layouts(scenario, seed: int) -> list[tuple[tuple[int, ...], np.ndarray]]:
    cfg = scenario.cfg
    n = n_anchor_combos(cfg.num_iot, cfg.num_uav)
    if n <= 0 or n > 1000:
        raise RuntimeError(
            f"C(I,J)={n} is outside the full-enum cutoff (1..1000); "
            "this check is the I=10, J=3 placement comparison"
        )
    out = []
    for combo in itertools.combinations(range(int(cfg.num_iot)), int(cfg.num_uav)):
        uav = uav_for_combo(scenario, combo, seed)
        if uav is None:
            continue
        out.append((combo, uav))
    return out


def _best_of(scenario, layouts, score_fn) -> dict:
    best: dict | None = None
    for combo, uav in layouts:
        a, b = _associate(scenario, uav)
        row = score_fn(scenario, uav, a, b)
        row["combo"] = [int(i) for i in combo]
        best = _consider(best, row)
    if best is None:
        return _pack(float("nan"), False, combo=[], max_link_share=float("nan"), status="no_layout")
    return best


def _pmedian_layouts(scenario, seed: int) -> list[tuple[tuple[int, ...], np.ndarray]]:
    cfg = scenario.cfg
    _primary, ties, _cost = exact_pmedian_combos(
        scenario.iot_xyz_m[:, :2], int(cfg.num_uav)
    )
    out = []
    for combo in ties:
        uav = uav_for_combo(scenario, combo, seed)
        if uav is None:
            continue
        out.append((combo, uav))
    return out


def _run_sca(scenario, seed: int, settings: SCASettings) -> dict:
    try:
        result = solve_sca(scenario, seed, settings=settings)
    except RuntimeError as exc:
        blank = _pack(float("nan"), False, status="init_cpu_unstable", message=str(exc))
        return {"prop": blank, "const": dict(blank), "n_iterations": 0}

    a = result.allocation.hard_association()
    b = result.allocation.hard_processing()
    prop = _score_proportional(scenario, result.uav_xyz_m, a, b)
    bw = np.asarray(result.allocation.bandwidth_hz, dtype=float)
    const = _pack(
        result.true_eval.sum_rate_mbps,
        result.true_eval.feasible,
        max_link_share=float(np.max(bw) / scenario.cfg.b_sys_hz),
        status=str(result.solver_status),
        n_iterations=int(result.n_iterations),
    )
    prop["n_iterations"] = int(result.n_iterations)
    return {"prop": prop, "const": const, "n_iterations": int(result.n_iterations)}


def _one_seed(scenario, seed: int, settings: SCASettings) -> dict:
    t0 = time.perf_counter()
    layouts = _layouts(scenario, seed)
    # Score each zenith subset once per radio and keep each radio's winner.
    # Also keep the proportional score of the constant-noise winner (transfer).
    best_prop = None
    best_const = None
    transferred = None
    for combo, uav in layouts:
        a, b = _associate(scenario, uav)
        prop = _score_proportional(scenario, uav, a, b)
        const = _score_constant(scenario, uav, a, b)
        prop["combo"] = [int(i) for i in combo]
        const["combo"] = [int(i) for i in combo]
        if best_prop is None or _rank_key(prop) > _rank_key(best_prop):
            best_prop = prop
        if best_const is None or _rank_key(const) > _rank_key(best_const):
            best_const = const
            transferred = dict(prop)
    zenith_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    p_layouts = _pmedian_layouts(scenario, seed)
    p_prop = _best_of(scenario, p_layouts, _score_proportional)
    p_const = _best_of(scenario, p_layouts, _score_constant)
    p_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    sca = _run_sca(scenario, seed, settings)
    sca_s = time.perf_counter() - t2

    if best_prop is None:
        best_prop = _pack(float("nan"), False, combo=[], status="no_layout")
    if best_const is None:
        best_const = _pack(float("nan"), False, combo=[], status="no_layout")
    if transferred is None:
        transferred = _pack(float("nan"), False, combo=[], status="no_layout")

    return {
        "seed": int(seed),
        "proportional": {
            "zenith": best_prop,
            "pmedian": p_prop,
            "sca": sca["prop"],
            "zenith_transferred": transferred,
        },
        "constant_sigma2": {
            "zenith": best_const,
            "pmedian": p_const,
            "sca": sca["const"],
        },
        "seconds": {
            "zenith_enum": zenith_s,
            "pmedian": p_s,
            "sca": sca_s,
        },
    }


def _finite_rates(rows: list[dict], model: str, method: str) -> np.ndarray:
    vals = []
    for row in rows:
        cell = row[model][method]
        if cell["feasible"] and np.isfinite(cell["rate_mbps"]):
            vals.append(float(cell["rate_mbps"]))
    return np.asarray(vals, dtype=float)


def _paired(rows: list[dict], model: str, left: str, right: str) -> dict:
    deltas = []
    wins = losses = ties = practical = 0
    n_skip = 0
    for row in rows:
        a = row[model][left]
        b = row[model][right]
        if not (a["feasible"] and b["feasible"]):
            n_skip += 1
            continue
        if not (np.isfinite(a["rate_mbps"]) and np.isfinite(b["rate_mbps"])):
            n_skip += 1
            continue
        d = float(a["rate_mbps"]) - float(b["rate_mbps"])
        deltas.append(d)
        if d > 1e-6:
            wins += 1
        elif d < -1e-6:
            losses += 1
        else:
            ties += 1
        if d >= PRACTICAL_MBPS:
            practical += 1
    arr = np.asarray(deltas, dtype=float)
    wil = wilcoxon_signed_rank(arr) if arr.size else {
        "n_nonzero": 0,
        "p_two_sided": 1.0,
        "p_greater": 1.0,
        "method": "empty",
    }
    return {
        "left": left,
        "right": right,
        "n_paired": int(arr.size),
        "n_skipped": int(n_skip),
        "mean_delta_mbps": float(arr.mean()) if arr.size else float("nan"),
        "std_delta_mbps": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "practical_wins_0.05": practical,
        "wilcoxon_p_two_sided": wil["p_two_sided"],
        "wilcoxon_p_greater": wil["p_greater"],
    }


def _mean_share(rows: list[dict], model: str, method: str) -> float:
    vals = []
    for row in rows:
        cell = row[model][method]
        share = cell.get("max_link_share")
        if cell["feasible"] and share is not None and np.isfinite(share):
            vals.append(float(share))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _verdict(rows: list[dict], model: str) -> dict:
    z_sca = _paired(rows, model, "zenith", "sca")
    z_p = _paired(rows, model, "zenith", "pmedian")
    ahead_sca = (
        z_sca["n_paired"] > 0
        and z_sca["mean_delta_mbps"] > 0.0
        and z_sca["wins"] > z_sca["losses"]
    )
    ahead_p = (
        z_p["n_paired"] > 0
        and z_p["mean_delta_mbps"] > 0.0
        and z_p["wins"] > z_p["losses"]
    )
    sca_practical = ahead_sca and z_sca["mean_delta_mbps"] >= PRACTICAL_MBPS
    p_practical = ahead_p and z_p["mean_delta_mbps"] >= PRACTICAL_MBPS
    p_tied = (
        z_p["n_paired"] > 0
        and z_p["losses"] == 0
        and z_p["mean_delta_mbps"] < PRACTICAL_MBPS
    )
    if z_sca["n_paired"] == 0:
        code = "no_paired_feasible"
        text = "No seed had both zenith-subset and k-means SCA feasible."
    elif sca_practical and p_practical:
        code = "zenith_still_wins"
        text = (
            "Zenith-subset still leads p-median and k-means SCA by at least "
            f"{PRACTICAL_MBPS:.2f} Mbps under this radio."
        )
    elif sca_practical and p_tied:
        code = "zenith_beats_sca_not_pmedian"
        text = (
            "Zenith-subset still beats k-means SCA by at least "
            f"{PRACTICAL_MBPS:.2f} Mbps. It does not separate from Euclidean "
            "p-median: the radio-best IoT subset is usually the geometric one."
        )
    elif ahead_sca:
        code = "zenith_ahead_of_sca_but_small"
        text = (
            "Zenith-subset still beats k-means SCA on paired seeds, but the "
            f"mean gap is under the {PRACTICAL_MBPS:.2f} Mbps practical bar."
        )
    else:
        code = "zenith_advantage_gone"
        text = (
            "The zenith-subset lead over k-means SCA does not survive this radio."
        )
    means = {}
    for method in METHODS:
        vals = _finite_rates(rows, model, method)
        means[method] = float(vals.mean()) if vals.size else float("nan")
    return {
        "code": code,
        "text": text,
        "mean_mbps": means,
        "mean_max_link_share": {m: _mean_share(rows, model, m) for m in METHODS},
        "n_feasible": {
            m: int(sum(1 for row in rows if row[model][m]["feasible"])) for m in METHODS
        },
        "zenith_minus_sca": z_sca,
        "zenith_minus_pmedian": z_p,
        "pmedian_minus_sca": _paired(rows, model, "pmedian", "sca"),
    }


def _fmt_pair(pair: dict) -> str:
    mean = pair["mean_delta_mbps"]
    mean_s = "nan" if not np.isfinite(mean) else f"{mean:+.3f}"
    return (
        f"{pair['left']}-{pair['right']} {mean_s} Mbps, "
        f"{pair['wins']}/{pair['n_paired']} wins, "
        f"{pair['losses']} losses, {pair['ties']} ties, "
        f"p={pair['wilcoxon_p_two_sided']:.3g}, "
        f"practical(>={PRACTICAL_MBPS:.2f}) {pair['practical_wins_0.05']}"
    )


def _print_model(title: str, summary: dict) -> None:
    print(f"  {title}: {summary['text']}")
    means = summary["mean_mbps"]
    shares = summary["mean_max_link_share"]
    feas = summary["n_feasible"]
    for method in METHODS:
        m = means[method]
        m_s = "nan" if not np.isfinite(m) else f"{m:.3f}"
        sh = shares[method]
        sh_s = "nan" if not np.isfinite(sh) else f"{sh:.3f}"
        print(
            f"    {method:8} {m_s} Mbps  feasible {feas[method]}  "
            f"mean max-link share {sh_s}"
        )
    print(f"    {_fmt_pair(summary['zenith_minus_sca'])}")
    print(f"    {_fmt_pair(summary['zenith_minus_pmedian'])}")
    print(f"    {_fmt_pair(summary['pmedian_minus_sca'])}")


def run_area(area_m: float, seeds: list[int], share: float, settings: SCASettings) -> dict:
    cfg = headline_sim_config(area_m=area_m, max_bw_share=share)
    rows = []
    for k, seed in enumerate(seeds, start=1):
        scenario = generate_scenario(seed, cfg)
        print(f"area {area_m:.0f} m  seed {seed} ({k}/{len(seeds)})", flush=True)
        row = _one_seed(scenario, seed, settings)
        rows.append(row)
        prop = row["proportional"]
        const = row["constant_sigma2"]
        print(
            "  prop  zenith {z:.3f}  pmedian {p:.3f}  sca {s:.3f}   "
            "const zenith {zc:.3f}  pmedian {pc:.3f}  sca {sc:.3f}".format(
                z=prop["zenith"]["rate_mbps"],
                p=prop["pmedian"]["rate_mbps"],
                s=prop["sca"]["rate_mbps"],
                zc=const["zenith"]["rate_mbps"],
                pc=const["pmedian"]["rate_mbps"],
                sc=const["sca"]["rate_mbps"],
            ),
            flush=True,
        )
    prop_summary = _verdict(rows, "proportional")
    const_summary = _verdict(rows, "constant_sigma2")
    # Transfer: constant-noise zenith subset, re-scored with proportional B.
    transfer_rows = []
    for row in rows:
        cloned = {
            "proportional": {
                "zenith": row["proportional"]["zenith_transferred"],
                "pmedian": row["proportional"]["pmedian"],
                "sca": row["proportional"]["sca"],
            }
        }
        transfer_rows.append(cloned)
    transfer = _paired(transfer_rows, "proportional", "zenith", "sca")
    return {
        "area_m": float(area_m),
        "b_sys_hz": float(cfg.b_sys_hz),
        "max_bw_share": cfg.max_bw_share,
        "n0_w_per_hz": calibrated_noise_density_w_per_hz(cfg),
        "b_eq_hz": equal_share_reference_hz(cfg),
        "sigma2_w": float(cfg.noise_power_w),
        "seeds": rows,
        "proportional": prop_summary,
        "constant_sigma2": const_summary,
        "zenith_transferred_minus_sca": transfer,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--areas", type=float, nargs="+", default=[100.0, 500.0])
    p.add_argument("--max-bw-share", type=float, default=0.25)
    p.add_argument("--sca-iterations", type=int, default=30)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "bw_proportional_noise.json",
    )
    args = p.parse_args(argv)
    if args.n_runs <= 0:
        raise SystemExit("--n-runs must be positive")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_runs)))
    settings = SCASettings(
        solver=None,
        max_iterations=int(args.sca_iterations),
        dynamic_assignment=True,
    )
    cfg0 = headline_sim_config(max_bw_share=float(args.max_bw_share))
    print(
        "N0 = sigma^2 / (B_sys/I) = "
        f"{calibrated_noise_density_w_per_hz(cfg0):.6e} W/Hz  "
        f"B_eq = {equal_share_reference_hz(cfg0):.6g} Hz  "
        f"sigma^2 = {cfg0.noise_power_w:.6g} W"
    )
    print(
        "k-means SCA is frozen Algorithm 1 (dynamic assignment, constant sigma^2). "
        "Only its inner bandwidth is re-solved."
    )
    areas = []
    for area in args.areas:
        block = run_area(float(area), seeds, float(args.max_bw_share), settings)
        areas.append(block)
        print(f"\narea {area:.0f} m")
        _print_model("proportional noise", block["proportional"])
        _print_model("constant sigma^2", block["constant_sigma2"])
        tr = block["zenith_transferred_minus_sca"]
        print(f"  transferred zenith subset (chosen under sigma^2, scored under N0): {_fmt_pair(tr)}")
        print()

    payload = {
        "model": "snr = p*g / (N0*B)",
        "n0_rule": "N0 = sigma**2 / (B_sys/I); SNR(B_eq) = p*g/sigma**2",
        "practical_bar_mbps": PRACTICAL_MBPS,
        "sca": {
            "code": "frozen solve_sca",
            "init": "k-means",
            "dynamic_assignment": True,
            "max_iterations": int(args.sca_iterations),
            "inner_realloc": "proportional rel_entr at the frozen (q, a, b)",
        },
        "areas": areas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
