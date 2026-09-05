# Experimental campaign — Khalaf et al. (IEEE TNSM, 2026) §VII axes

SCA (Algorithm 1 sequential solver) is **frozen**. This document is the
campaign ledger. Do not edit `src/uavdt/sca/` or `matlab/sca_seq.m` /
`convex_p.m` / `bandwidth_lp.m` / `run_convex_step.m` unless a campaign
run exposes a real inconsistency or bug.

Paper: Khalaf, Itani, Sharafeddine, IEEE TNSM vol. 23, 2026, §VII.

---

## Freeze

| Surface | Status |
| --- | --- |
| Core physics (`channel`, `computation`, `evaluator`, `constraints`) | Frozen |
| AoDT Eq. (17) scoring | Frozen (Problem (P) score) |
| AoDT extras (Eqs. (10), (12), (14)–(15), FCFS/LCFS-S sim, Fig. 11) | Added; does not change (17) |
| SCA solver (`uavdt.sca`, MATLAB CVX files above) | Frozen |
| New work | Baselines, sweeps, reporting |

---

## What the paper plots (PAPER)

Each plotted point is the mean of **20** random runs. This reproduction
uses 100 × 100 m and a feasible `B_sys` (see `docs/REPRODUCTION.md` §4.1).
Do not chase the paper’s 7–14 Mbps numbers. The 20 kHz Table II value,
read as the (27) cap, is bounded by `B_sys · log2(1+SNR_max)` at
**0.997 Mbps** even at `SNR_max = 10^15`; a control at 500 × 500 m is
also 0% feasible (`scripts/check_bsys_20khz.py`).

| Figure | Axis | Held fixed | Caption / text |
| --- | --- | --- | --- |
| Fig. 6 | UAV count `J` (text: five UAVs) | `I = 10` | Sum rate vs number of UAVs |
| Fig. 7 | IoT count `I` (text: up to 32) | `J = 3` | Sum rate vs IoT device count |
| Fig. 8 | Task arrival `λ` from 1 to 3.5 /s | `I = 10`, `J = 3` | Sum rate vs arrival rate |
| Fig. 9 | AoDT threshold `T_k` from 0.8 s to 3 s | `I = 10`, `J = 3` | Sum rate vs AoDT threshold |
| Fig. 10 | UAV CPU; text quotes 250 MHz | `I = 10`, `J = 3` | Sum rate vs computational capacity |
| Fig. 11 | UAV count `J` with three λ patterns | `I = 10` | Uniform fast / slow / heterogeneous λ |

Paper methods: **SCA**, **TD3**, **k-means**, **random**. TD3 is out of
scope for this campaign.

---

## This campaign (labels)

| Item | Label | Notes |
| --- | --- | --- |
| Axes and 20-run rule | PAPER | Tick lists below when the PDF omits them |
| Area 100 × 100 m | Intentional modification | Paper 500 × 500 m; 20 kHz also checked at 500 m |
| `B_sys` 8.8 MHz (headline) | Intentional modification | Table II 20 kHz as (27) cap is infeasible (ceiling 0.997 Mbps) |
| 25% per-link cap | EXTERNAL PARAMETER | Not Problem (P) |
| `S_i`, `L` | EXTERNAL PARAMETER | Settled: 12,000 bytes, `L=3.75×10⁶` |
| PSO | EXTERNAL | Extra baseline, not in the paper |
| Equal `|N_k|` when `I` grows | IMPLEMENTATION CHOICE | Keep `K = 2`, `I` even |
| Placement baselines + frozen-`q` bandwidth LP | IMPLEMENTATION CHOICE | Same B LP as SCA’s bandwidth step so the comparison is placement |
| PSO inner fitness with equal-share `B` | IMPLEMENTATION CHOICE | Final score still uses the LP + `evaluate()` |
| Campaign SCA solver default CVXPY | IMPLEMENTATION CHOICE | MATLAB CVX/MOSEK is spot-validated |

Default grids (`src/uavdt/experiments/grids.py`):

```text
J:     1, 2, 3, 4, 5
I:     10, 16, 20, 24, 28, 32
λ:     1.0, 1.5, 2.0, 2.5, 3.0, 3.5   /s
T_k:   0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0   s
f_j:   0.5e8, 1.0e8, 1.5e8, 2.0e8, 2.5e8   cycles/s
```

`n_runs` default on the CLI is **5** (dev). Paper final figures use **20**.

---

## How to run

```text
$env:PYTHONPATH="src"

python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --n-runs 5 --solver cvxpy --out results/campaign_8.8mhz.json

python scripts/check_bsys_20khz.py
python -m uavdt evaluate --seed 1 --bandwidth 20000 --area-m 500 --placement kmeans

python -m uavdt campaign --axis uavs --methods random,kmeans,pso,sca --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 5 --out results/campaign_8.8mhz_cap25_uavs.json

python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25

python -m uavdt aodt-compare --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --placement kmeans
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --out results/fig11_8.8mhz_cap25.json
```

`--solver matlab` on `campaign` runs frozen SCA in MATLAB CVX+MOSEK
(slow: one MATLAB session per SCA seed). Prefer CVXPY for the grid and
`spot-validate` for MOSEK.

---

## Outputs

- `results/campaign_*.json` — per-axis points, per-method mean/std Mbps,
  feasible fraction, per-seed rates
- `results/campaign_*.csv` — flat table for plots
- Paired writeup stats (same seed, champion vs baseline):

```text
python scripts/paired_winrate.py results/campaign_20260904_cap25.json
```

  Writes `*_paired.json` / `*_paired.csv`. Quote lines are
  `SCA wins by X+/-Y Mbps vs <baseline>, p<..., N/20 seeds`.
  `+/-Y` is the sample std of the 20 paired deltas. Primary p is
  Wilcoxon signed-rank. `uavs J=3` is the unique default scenario.
  The script flags lambda/CPU (and other) rows that are identical
  per-seed copies; do not pool those with J=3.

Published score is always Python `evaluate()` (Eq. (17) AoDT). Fig. 11
and `aodt-compare` add FCFS / FCFS-P / LCFS-S simulations beside that
score; they do not replace it.

Full analysis, audit paragraph, and tables: **`docs/RESULTS.md`**.

---

## First pass on disk (n_runs = 5, 8.8 MHz, no per-link cap)

- `results/campaign_8.8mhz.json` / `.csv` — Figs. 6–10 axes, methods
  random / k-means / PSO / SCA (CVXPY).
- `results/spot_validate_8.8mhz.json` — seed 1, no cap and 25% cap,
  CVXPY vs MATLAB. `agreement: ok`. `se_max_abs_diff ~ 1e-15`.

At 8.8 MHz with **no** per-link cap, leftover spectrum sits on the
highest-SE link, so mean sum rates sit near **~8.96–8.99 Mbps** for
every method. `λ` and `f_j` do not move the communication objective
when the point stays feasible. `T_k = 0.8 s` is infeasible for all
four methods. Differentiation is expected to show up with
`--max-bw-share 0.25` (EXTERNAL), which is the next campaign to run.
