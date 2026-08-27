"""Radio sensitivity sweep: mean sum rate (Mbps) vs J for candidate radio settings.

Every option accepts several values and the cartesian product is run, so this is
how the profile in docs/calibration.md was chosen. Defaults come from the active
profile in src/config.py.

    python -m scripts.calibrate --methods random kmeans sca --js 1 3 5
    python -m scripts.calibrate --b-sys 20000 --noise 1e-4 --scope system --cap -1
    python -m scripts.calibrate --cap 0.15 0.25 0.4 -1        # -1 means no cap
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import numpy as np

from src.config import DEFAULT
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.pso import solve_pso_placement
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca

SEEDS = (100, 101, 102, 103, 104)


def run(cfg, methods, js, seeds=SEEDS):
    out = {}
    for j, name in itertools.product(js, methods):
        cfg_j = replace(cfg, num_uav=j)
        rates, feas = [], []
        for seed in seeds:
            sc = generate_scenario(seed, cfg_j)
            if name == "random":
                _xy, res, _rt = solve_random(sc, seed=seed, n_uav=j)
            elif name == "kmeans":
                _xy, res, _rt = solve_kmeans(sc, seed=seed, n_uav=j)
            elif name == "sca":
                _xy, res, _rt = solve_sca(sc, seed=seed, n_uav=j)
            elif name == "pso":
                _xy, res, _rt, _h = solve_pso_placement(sc, seed=seed, n_uav=j, n_particles=20, n_iter=60)
            elif name == "td3":
                from src.solvers.td3 import solve_td3

                _xy, res, _rt, _l = solve_td3(sc, seed=seed, n_uav=j)
            else:
                raise ValueError(name)
            rates.append(res.sum_rate)
            feas.append(int(res.feasible))
        out[(j, name)] = (float(np.mean(rates)) / 1e6, float(np.mean(feas)))
    return out


def show(tag, table, methods, js):
    print(f"\n=== {tag} ===")
    print("  J | " + " | ".join(f"{m:>16}" for m in methods))
    for j in js:
        cells = []
        for m in methods:
            mb, fe = table[(j, m)]
            cells.append(f"{mb:9.3f} f={fe:.1f}")
        print(f"  {j} | " + " | ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-sys", type=float, nargs="*", default=[None])
    ap.add_argument("--noise", type=float, nargs="*", default=[None])
    ap.add_argument("--scope", nargs="*", default=[None])
    ap.add_argument("--cap", type=float, nargs="*", default=[None])
    ap.add_argument("--methods", nargs="*", default=["random", "kmeans", "pso", "sca"])
    ap.add_argument("--js", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    for b, nz, sc, cap in itertools.product(args.b_sys, args.noise, args.scope, args.cap):
        cfg = DEFAULT
        if b is not None:
            cfg = replace(cfg, b_sys=b)
        if nz is not None:
            cfg = replace(cfg, noise_power=nz)
        if sc is not None:
            cfg = replace(cfg, bandwidth_scope=sc)
        if cap is not None:
            cfg = replace(cfg, max_bw_share=None if cap <= 0 else cap)
        tag = f"b={cfg.b_sys:g} N={cfg.noise_power:g} scope={cfg.bandwidth_scope} cap={cfg.max_bw_share}"
        show(tag, run(cfg, args.methods, args.js), args.methods, args.js)
