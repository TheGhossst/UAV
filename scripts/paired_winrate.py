"""Paired-difference + per-seed win-rate vs a champion method.

Reads a campaign JSON (per-seed Mbps + seed ids) and, for each sweep
point, pairs SCA (default) with every baseline on the *same* seed.

Prints writeup-ready lines:

    SCA wins by 0.014+/-0.018 Mbps vs random, p=0.002, 16/20 seeds

Primary test is Wilcoxon signed-rank on paired deltas (exact when there
are no |delta| ties; normal approximation with tie correction otherwise).
Paired t-test is reported as a secondary check. No extra dependencies.

uavs J=3, iots I=10, lambda=2, aodt Tk=2.8, and cpu 2e8 are the same
default scenario — do not pool those rows. The script marks them.

Usage:

    python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
    python scripts/paired_winrate.py results/campaign_8.8mhz_cap15_n20.json --champion sca
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Default scenario ticks that repeat I=10, J=3, lambda=2, Tk=2.8, fj=2e8.
_DEFAULT_TICK = {
    "uavs": 3.0,
    "iots": 10.0,
    "lambda": 2.0,
    "aodt": 2.8,
    "cpu": 2.0e8,
}


def _norm_sf(z: float) -> float:
    """P(Z > z) for standard normal Z."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta (Lentz)."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    c = 1.0
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        return float("nan")
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
        return front * _betacf(a, b, x)
    front = math.exp(b * math.log(1.0 - x) + a * math.log(x) - lbeta) / b
    return 1.0 - front * _betacf(b, a, 1.0 - x)


def student_t_sf_two_sided(t: float, df: float) -> float:
    """P(|T_df| >= |t|)."""
    if df <= 0.0 or not math.isfinite(t):
        return float("nan")
    tt = abs(float(t))
    if tt == 0.0:
        return 1.0
    x = df / (df + tt * tt)
    return betainc_reg(0.5 * df, 0.5, x)


def student_t_sf_greater(t: float, df: float) -> float:
    """P(T_df >= t)."""
    if df <= 0.0 or not math.isfinite(t):
        return float("nan")
    two = student_t_sf_two_sided(t, df)
    if t >= 0.0:
        return 0.5 * two
    return 1.0 - 0.5 * two


def _t_crit_approx(df: float, two_tail: float = 0.05) -> float:
    """Approximate two-sided t critical value (good for df>=10)."""
    # Newton on the survival function, start from normal quantile.
    z = 1.959963984540054  # Phi^{-1}(0.975)
    t = z
    for _ in range(20):
        p = student_t_sf_two_sided(t, df)
        # d/dt P(|T|>t) ~ -2 phi_t(t)
        x = df / (df + t * t)
        # numerical step
        dt = 1e-5
        p2 = student_t_sf_two_sided(t + dt, df)
        deriv = (p2 - p) / dt
        if abs(deriv) < 1e-18:
            break
        t -= (p - two_tail) / deriv
        if t <= 0:
            t = z
    return t


def wilcoxon_exact_p(w_plus: float, n: int, alternative: str = "two-sided") -> float:
    """Exact Wilcoxon signed-rank p-value under no |delta| ties.

    W+ is the sum of ranks of positive differences. Under H0 each of the
    2^n sign assignments is equally likely.
    """
    if n <= 0:
        return 1.0
    max_sum = n * (n + 1) // 2
    counts = [1]
    for r in range(1, n + 1):
        nxt = [0] * (len(counts) + r)
        for s, c in enumerate(counts):
            nxt[s] += c
            nxt[s + r] += c
        counts = nxt
    total = float(1 << n)
    w = int(round(w_plus))
    w = min(max(w, 0), max_sum)
    if alternative == "greater":
        # H1: median(delta) > 0  => large W+
        return sum(counts[w:]) / total
    if alternative == "less":
        return sum(counts[: w + 1]) / total
    # two-sided: P(W+ as extreme as observed toward either tail)
    tail = min(w, max_sum - w)
    return min(1.0, sum(counts[: tail + 1]) / total + sum(counts[max_sum - tail :]) / total)


def _average_ranks(abs_d: np.ndarray) -> np.ndarray:
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(abs_d.size, dtype=float)
    i = 0
    n = abs_d.size
    while i < n:
        j = i + 1
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + j)  # 1-based ranks i+1 .. j
        ranks[order[i:j]] = avg
        i = j
    return ranks


def wilcoxon_signed_rank(
    deltas: np.ndarray,
    *,
    alternative: str = "two-sided",
    zero_eps: float = 0.0,
) -> dict[str, Any]:
    """Wilcoxon signed-rank on paired differences (zeros dropped)."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > zero_eps]
    n = int(nz.size)
    if n == 0:
        return {
            "n_nonzero": 0,
            "W_plus": 0.0,
            "p_two_sided": 1.0,
            "p_greater": 1.0,
            "method": "all_zero",
        }
    ranks = _average_ranks(np.abs(nz))
    w_plus = float(np.sum(ranks[nz > 0.0]))
    # tie groups on |d|
    _, counts = np.unique(np.abs(nz), return_counts=True)
    has_ties = bool(np.any(counts > 1))
    p_two: float
    p_greater: float
    method: str
    if not has_ties:
        method = "exact"
        p_two = wilcoxon_exact_p(w_plus, n, "two-sided")
        p_greater = wilcoxon_exact_p(w_plus, n, "greater")
    else:
        method = "normal_ties"
        mean = n * (n + 1) / 4.0
        tie_adj = float(np.sum(counts**3 - counts)) / 48.0
        var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_adj
        sd = math.sqrt(max(var, 0.0))
        if sd == 0.0:
            p_two, p_greater = 1.0, 1.0
        else:
            # continuity correction toward the mean
            z_two = (abs(w_plus - mean) - 0.5) / sd
            p_two = 2.0 * _norm_sf(z_two)
            z_g = (w_plus - mean - 0.5) / sd
            p_greater = _norm_sf(z_g)
            p_two = min(1.0, max(0.0, p_two))
            p_greater = min(1.0, max(0.0, p_greater))
    return {
        "n_nonzero": n,
        "W_plus": w_plus,
        "p_two_sided": p_two,
        "p_greater": p_greater,
        "method": method,
    }


def paired_t(deltas: np.ndarray, alternative: str = "two-sided") -> dict[str, Any]:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 2:
        return {
            "n": n,
            "t": float("nan"),
            "df": max(n - 1, 0),
            "p_two_sided": float("nan"),
            "p_greater": float("nan"),
        }
    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1))
    sem = std / math.sqrt(n)
    df = n - 1
    if sem == 0.0:
        t_stat = float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0)
        if mean == 0.0:
            p_two, p_g = 1.0, 1.0
        elif mean > 0:
            p_two, p_g = 0.0, 0.0
        else:
            p_two, p_g = 0.0, 1.0
    else:
        t_stat = mean / sem
        p_two = student_t_sf_two_sided(t_stat, df)
        p_g = student_t_sf_greater(t_stat, df)
    _ = alternative
    return {
        "n": n,
        "t": t_stat,
        "df": df,
        "p_two_sided": p_two,
        "p_greater": p_g,
        "mean": mean,
        "std": std,
        "sem": sem,
    }


def align_by_seed(champ: dict, base: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (seeds, champ_Mbps, base_Mbps, jointly_feasible)."""
    c_map = {int(s): float(r) for s, r in zip(champ["seeds"], champ["per_seed_Mbps"])}
    b_map = {int(s): float(r) for s, r in zip(base["seeds"], base["per_seed_Mbps"])}
    c_feas = {int(s): bool(f) for s, f in zip(champ["seeds"], champ["per_seed_feasible"])}
    b_feas = {int(s): bool(f) for s, f in zip(base["seeds"], base["per_seed_feasible"])}
    seeds = sorted(set(c_map) & set(b_map))
    if not seeds:
        raise ValueError("no overlapping seeds between champion and baseline")
    c = np.array([c_map[s] for s in seeds], dtype=float)
    b = np.array([b_map[s] for s in seeds], dtype=float)
    joint = np.array([c_feas[s] and b_feas[s] for s in seeds], dtype=bool)
    return np.array(seeds, dtype=int), c, b, joint


def compare_pair(
    champ: dict,
    base: dict,
    *,
    tie_eps_mbps: float = 1e-9,
    jointly_feasible_only: bool = False,
) -> dict[str, Any]:
    seeds, c, b, joint = align_by_seed(champ, base)
    if jointly_feasible_only:
        mask = joint
        seeds, c, b, joint = seeds[mask], c[mask], b[mask], joint[mask]
    deltas = c - b
    n = int(deltas.size)
    wins = int(np.sum(deltas > tie_eps_mbps))
    losses = int(np.sum(deltas < -tie_eps_mbps))
    ties = n - wins - losses
    wx = wilcoxon_signed_rank(deltas)
    tt = paired_t(deltas)
    mean = float(np.mean(deltas)) if n else float("nan")
    std = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n else float("nan")
    df = max(n - 1, 1)
    tcrit = _t_crit_approx(df) if n > 1 else float("nan")
    ci_lo = mean - tcrit * sem if n > 1 else mean
    ci_hi = mean + tcrit * sem if n > 1 else mean
    return {
        "n_seeds": n,
        "n_jointly_feasible": int(np.sum(joint)),
        "mean_delta_Mbps": mean,
        "std_delta_Mbps": std,
        "sem_delta_Mbps": sem,
        "ci95_lo_Mbps": ci_lo,
        "ci95_hi_Mbps": ci_hi,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": (wins / n) if n else float("nan"),
        "wilcoxon_W_plus": wx["W_plus"],
        "wilcoxon_n_nonzero": wx["n_nonzero"],
        "wilcoxon_method": wx["method"],
        "wilcoxon_p_two_sided": wx["p_two_sided"],
        "wilcoxon_p_greater": wx["p_greater"],
        "t_stat": tt["t"],
        "t_df": tt["df"],
        "t_p_two_sided": tt["p_two_sided"],
        "t_p_greater": tt["p_greater"],
        "champion_mean_Mbps": float(np.mean(c)) if n else float("nan"),
        "baseline_mean_Mbps": float(np.mean(b)) if n else float("nan"),
    }


def is_repeated_default(pt: dict) -> bool:
    """Tick-based: I=10 / lambda=2 / Tk=2.8 / fj=2e8 copy uavs J=3."""
    axis = pt["axis"]
    if axis not in _DEFAULT_TICK:
        return False
    return abs(float(pt["x"]) - _DEFAULT_TICK[axis]) < 1e-9 and axis != "uavs"


def _rate_key(pt: dict, methods: list[str]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    seeds = tuple(int(s) for s in pt["by_method"][methods[0]]["seeds"])
    rates: list[float] = []
    for m in methods:
        rates.extend(float(x) for x in pt["by_method"][m]["per_seed_Mbps"])
    return seeds, tuple(rates)


def mark_identical_to_uavs_j3(payload: dict, rows: list[dict]) -> None:
    """Flag points whose per-seed vectors match uavs J=3 (comm-limited copies).

    lambda and CPU sweeps are identical to the default scenario whenever the
    point stays feasible: the published score does not depend on those axes.
    """
    methods = list(payload["methods"])
    ref = None
    for pt in payload["points"]:
        if pt["axis"] == "uavs" and abs(float(pt["x"]) - 3.0) < 1e-9:
            ref = _rate_key(pt, methods)
            break
    if ref is None:
        return
    identical_labels = set()
    for pt in payload["points"]:
        if pt["axis"] == "uavs" and abs(float(pt["x"]) - 3.0) < 1e-9:
            continue
        if _rate_key(pt, methods) == ref:
            identical_labels.add((pt["axis"], float(pt["x"])))
    for row in rows:
        if (row["axis"], float(row["x"])) in identical_labels:
            row["repeated_default"] = True
            row["identical_to_uavs_j3"] = True
        else:
            row.setdefault("identical_to_uavs_j3", False)


def fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    if p < 0.05:
        return f"p={p:.3g}"
    return f"p={p:.3g} (n.s.)"


def quote_line(
    champion: str,
    baseline: str,
    stats: dict[str, Any],
    *,
    p_key: str = "wilcoxon_p_two_sided",
) -> str:
    mean = stats["mean_delta_Mbps"]
    std = stats["std_delta_Mbps"]
    n = stats["n_seeds"]
    wins = stats["wins"]
    verb = "wins" if mean >= 0 else "loses"
    p = fmt_p(stats[p_key])
    return (
        f"{champion.upper()} {verb} by {abs(mean):.3f}+/-{std:.3f} Mbps vs {baseline}, "
        f"{p}, {wins}/{n} seeds"
    )


def bh_qvalues(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values over one family of tests."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.asarray(pvals, dtype=float)[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, n * ranked[i] / float(i + 1))
        q[i] = min(1.0, prev)
    out = np.empty(n, dtype=float)
    out[order] = q
    return [float(x) for x in out]


def attach_fdr(rows: list[dict[str, Any]]) -> None:
    """BH-FDR within each baseline, unique (non-repeated) sweep points only."""
    baselines = sorted({r["baseline"] for r in rows})
    for baseline in baselines:
        idxs = [
            i
            for i, r in enumerate(rows)
            if r["baseline"] == baseline and not r.get("repeated_default")
        ]
        qs = bh_qvalues([rows[i]["wilcoxon_p_two_sided"] for i in idxs])
        for i, q in zip(idxs, qs):
            rows[i]["bh_q_unique_family"] = q
            rows[i]["bh_family_size"] = len(idxs)


def _print_fdr(rows: list[dict[str, Any]], champion: str) -> None:
    print()
    print("--- BH-FDR on unique points (one family per baseline) ---")
    print(
        "Repeated default-scenario copies excluded. "
        "J=3 vs-random Bonferroni over 3 baselines is separate (m=3)."
    )
    j3 = [
        r
        for r in rows
        if r["axis"] == "uavs" and abs(float(r["x"]) - 3.0) < 1e-9
    ]
    if len(j3) >= 2:
        m = len(j3)
        print(f"  Bonferroni at uavs J=3 (m={m} baselines):")
        for r in j3:
            padj = min(1.0, r["wilcoxon_p_two_sided"] * m)
            flag = "sig" if padj < 0.05 else "n.s."
            print(
                f"    vs {r['baseline']:8}  p={r['wilcoxon_p_two_sided']:.4g}  "
                f"p_adj={padj:.4g}  {flag}"
            )
    baselines = sorted({r["baseline"] for r in rows})
    for baseline in baselines:
        fam = [
            r
            for r in rows
            if r["baseline"] == baseline and not r.get("repeated_default")
        ]
        n_sig = sum(1 for r in fam if r.get("bh_q_unique_family", 1.0) < 0.05)
        print(f"  {champion} vs {baseline}: {n_sig}/{len(fam)} unique points FDR q<0.05")
        for r in fam:
            q = r.get("bh_q_unique_family", float("nan"))
            mark = "*" if q < 0.05 else " "
            print(
                f"   {mark} {r['axis']:6} x={r['x']:<8g}  "
                f"d={r['mean_delta_Mbps']:+.4f}  W={r['wins']}/{r['n_seeds']}  "
                f"p={r['wilcoxon_p_two_sided']:.4g}  q={q:.4g}"
            )


def analyze_campaign(
    payload: dict,
    *,
    champion: str = "sca",
    tie_eps_mbps: float = 1e-9,
    jointly_feasible_only: bool = False,
) -> dict[str, Any]:
    methods = list(payload["methods"])
    if champion not in methods:
        raise ValueError(f"champion {champion!r} not in {methods}")
    baselines = [m for m in methods if m != champion]
    rows: list[dict[str, Any]] = []
    for pt in payload["points"]:
        champ_stats = pt["by_method"][champion]
        for baseline in baselines:
            stats = compare_pair(
                champ_stats,
                pt["by_method"][baseline],
                tie_eps_mbps=tie_eps_mbps,
                jointly_feasible_only=jointly_feasible_only,
            )
            row = {
                "axis": pt["axis"],
                "x_name": pt["x_name"],
                "x": pt["x"],
                "label": pt["label"],
                "repeated_default": is_repeated_default(pt),
                "identical_to_uavs_j3": False,
                "champion": champion,
                "baseline": baseline,
                "quote": quote_line(champion, baseline, stats),
                **stats,
            }
            rows.append(row)
    mark_identical_to_uavs_j3(payload, rows)
    attach_fdr(rows)
    return {
        "source_n_runs": payload.get("n_runs"),
        "b_sys_hz": payload.get("b_sys_hz"),
        "max_bw_share": payload.get("max_bw_share"),
        "champion": champion,
        "baselines": baselines,
        "tie_eps_Mbps": tie_eps_mbps,
        "jointly_feasible_only": jointly_feasible_only,
        "note": (
            "delta = champion - baseline on the same seed. "
            "Primary p-value is Wilcoxon signed-rank two-sided. "
            "+/- in quotes is sample std of the 20 paired deltas, not SEM. "
            "repeated_default / identical_to_uavs_j3 copy I=10 J=3 "
            "(whole lambda and CPU sweeps when the comm score does not move). "
            "Do not pool them with uavs J=3."
        ),
        "rows": rows,
    }


def _print_table(rows: list[dict[str, Any]], *, skip_repeated: bool) -> None:
    shown = [r for r in rows if not (skip_repeated and r["repeated_default"])]
    cur = None
    for r in shown:
        key = (r["axis"], r["x"])
        if key != cur:
            cur = key
            flag = ""
            if r.get("identical_to_uavs_j3") or r["repeated_default"]:
                flag = "  [duplicate of uavs J=3]"
            print()
            print(f"=== {r['axis']}  {r['x_name']}={r['x']:g}{flag} ===")
        sig = "*" if r["wilcoxon_p_two_sided"] < 0.05 else " "
        print(
            f"  {sig} vs {r['baseline']:8}  "
            f"d={r['mean_delta_Mbps']:+.4f}+/-{r['std_delta_Mbps']:.4f}  "
            f"CI95=[{r['ci95_lo_Mbps']:+.4f},{r['ci95_hi_Mbps']:+.4f}]  "
            f"W={r['wins']}/{r['n_seeds']}  "
            f"wilcox {fmt_p(r['wilcoxon_p_two_sided'])}  "
            f"t {fmt_p(r['t_p_two_sided'])}"
        )
        print(f"      {r['quote']}")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "axis",
        "x_name",
        "x",
        "repeated_default",
        "bh_q_unique_family",
        "bh_family_size",
        "champion",
        "baseline",
        "n_seeds",
        "n_jointly_feasible",
        "mean_delta_Mbps",
        "std_delta_Mbps",
        "sem_delta_Mbps",
        "ci95_lo_Mbps",
        "ci95_hi_Mbps",
        "wins",
        "ties",
        "losses",
        "win_rate",
        "wilcoxon_W_plus",
        "wilcoxon_n_nonzero",
        "wilcoxon_method",
        "wilcoxon_p_two_sided",
        "wilcoxon_p_greater",
        "t_stat",
        "t_df",
        "t_p_two_sided",
        "t_p_greater",
        "champion_mean_Mbps",
        "baseline_mean_Mbps",
        "quote",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def headline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fig. 6 J=3 (unique default) plus the rest of the UAV sweep."""
    return [r for r in rows if r["axis"] == "uavs"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "json_path",
        nargs="?",
        default="results/campaign_8.8mhz_cap25_si12k.json",
        help="Campaign JSON with per_seed_Mbps / seeds",
    )
    p.add_argument("--champion", default="sca")
    p.add_argument(
        "--jointly-feasible-only",
        action="store_true",
        help="Drop seeds where either method is infeasible",
    )
    p.add_argument("--tie-eps", type=float, default=1e-9, help="|delta| <= eps counts as a tie")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument(
        "--all-points",
        action="store_true",
        help="Print every sweep point, including repeated default-scenario rows",
    )
    args = p.parse_args(argv)

    path = Path(args.json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = analyze_campaign(
        payload,
        champion=args.champion,
        tie_eps_mbps=args.tie_eps,
        jointly_feasible_only=args.jointly_feasible_only,
    )

    b_mhz = float(result["b_sys_hz"] or 0.0) / 1e6
    cap = result["max_bw_share"]
    cap_s = "no cap" if cap is None else f"cap {float(cap):.0%}"
    print(f"file: {path}")
    print(
        f"B_sys={b_mhz:g} MHz  {cap_s}  n_runs={result['source_n_runs']}  "
        f"champion={result['champion']}  baselines={result['baselines']}"
    )
    print(
        "test: Wilcoxon signed-rank on paired (champion-baseline) Mbps; "
        "+/- is sample std of deltas. Do not pool repeated_default rows."
    )

    print()
    print("--- HEADLINE (Fig. 6 UAV sweep, unique default at J=3) ---")
    _print_table(headline_rows(result["rows"]), skip_repeated=False)

    print()
    print("--- ALL AXES (skipping repeated default-scenario copies) ---")
    _print_table(result["rows"], skip_repeated=not args.all_points)
    _print_fdr(result["rows"], result["champion"])

    out_json = Path(args.out_json) if args.out_json else path.with_name(path.stem + "_paired.json")
    out_csv = Path(args.out_csv) if args.out_csv else path.with_name(path.stem + "_paired.csv")
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(result["rows"], out_csv)
    print()
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
