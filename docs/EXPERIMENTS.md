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
| SCA solver (`uavdt.sca`, MATLAB CVX files above) | Frozen except `initialize.py` (CPU-stable \(b_{ij}\) repair) and `settings.py` (`process_cohesive_candidate`, ignored by frozen SCA) |
| SCA-joint (`uavdt.sca_joint`, method=`sca_joint`) | Methodology probe only. Does **not** replace frozen SCA. Writes separately named result files. Default rematch is best-SE; `--process-cohesive-candidate` is an opt-in hypothesis test. Frozen SCA ignores that flag. |
| TD3 (`uavdt.td3`, method=`td3`) | Opt-in. Default `TD3Settings` is Algorithm 2 **reproduction** (k-means residual, leftover inner \(B\), penalty reward, policy export). `TD3Settings.residual_on_sca()` / `--td3-preset residual-on-sca` is the **proposed interface to (P)**: residual \(\Delta q\) on the SCA incumbent, frozen SCA \(a,b\), inner frozen-\(q\) LP, feasible-Mbps reward, incumbent snapshot export. Does **not** change `SimConfig`. Not in default `METHODS`. |
| New work | Baselines, sweeps, reporting, SCA-joint probe, TD3 (Alg. 2 + residual-on-SCA preset), association oracle (`uavdt.assoc_search`, Experiment C) |

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

Paper methods: **SCA**, **TD3**, **k-means**, **random**. TD3 is implemented
as an opt-in method (`--methods ...,td3`); it is not in the default campaign
list. Hyperparameters are `TD3Settings`, not Table II.

---

## This campaign (labels)

| Item | Label | Notes |
| --- | --- | --- |
| Axes and 20-run rule | PAPER | Tick lists below when the PDF omits them |
| Area 100 × 100 m | Intentional modification | Paper 500 × 500 m; 20 kHz also checked at 500 m |
| `B_sys` 8.8 MHz (headline) | Intentional modification | Table II 20 kHz as (27) cap is infeasible (ceiling 0.997 Mbps) |
| `B_sys` 7 MHz | EXTERNAL PARAMETER | Extra experimental point between 2.4 and 8.8 MHz; not Table II |
| 25% per-link cap | EXTERNAL PARAMETER | **Primary** leftover-dump stress test (2.20 MHz/link) |
| 15% per-link cap | EXTERNAL PARAMETER | Tighter-cap **sensitivity** kept for experiments (1.32 MHz/link) |
| `S_i`, `L` | EXTERNAL PARAMETER | Settled: 12,000 bytes, `L=3.75×10⁶` |
| PSO | EXTERNAL | Extra baseline, not in the paper |
| Equal `|N_k|` when `I` grows | IMPLEMENTATION CHOICE | Keep `K = 2`, `I` even |
| Placement baselines + frozen-`q` bandwidth LP | IMPLEMENTATION CHOICE | Same B LP as SCA’s bandwidth step so the comparison is placement |
| PSO inner fitness with equal-share `B` | IMPLEMENTATION CHOICE | Final score still uses the LP + `evaluate()` |
| Campaign SCA solver default CVXPY | IMPLEMENTATION CHOICE | MATLAB CVX/MOSEK is spot-validated |
| TD3 | IMPLEMENTATION CHOICE | Default: Alg. 2 fill-in. Proposed: `--td3-preset residual-on-sca` (same stack, different interface to (P)) |

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

# TD3 opt-in (per-instance train; needs PyTorch). Not in the default --methods list.
python -m uavdt td3 --seed 1 --bandwidth-preset 8.8mhz --total-steps 7000
# Proposed interface to (P): residual on SCA, inner LP, feasible-rate reward.
python -m uavdt td3 --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --td3-preset residual-on-sca
python scripts/experiments/residual_on_sca/run_multistart.py
python scripts/experiments/residual_on_sca/run_residual_td3.py
python scripts/experiments/residual_on_sca/run_cmaes_polish.py

python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25

python -m uavdt aodt-compare --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --placement kmeans
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --out results/fig11_8.8mhz_cap25_si12k.json

# Tighter-cap sensitivity (not the primary campaign)
python -m uavdt campaign --axis uavs --methods random,kmeans,pso,sca --bandwidth-preset 8.8mhz --max-bw-share 0.15 --n-runs 5 --out results/campaign_8.8mhz_cap15_uavs.json

# SCA-joint methodology probe (does not overwrite frozen-SCA campaign files)
python scripts/run_sca_joint_campaign.py
python scripts/run_tk08_followup.py

# One-command replay of docs/RESULTS.md §9 (primary + cap15 + 500 m + sca-joint + analysis)
# scripts/run_full_regeneration.ps1
# Full bandwidth preset sweep (long, 7 configs): scripts/run_all_bandwidth_campaigns.ps1
# Extra 7 MHz trio (no TD3): scripts/run_7mhz_campaigns.ps1
```

`--solver matlab` on `campaign` runs frozen SCA in MATLAB CVX+MOSEK
(slow: one MATLAB session per SCA seed). Prefer CVXPY for the grid and
`spot-validate` for MOSEK.

### 100-scenario Monte Carlo (area + IoT + UAV bank)

The paper averages **20** random runs per plotted point. This extra study
freezes **100** layouts and reports the mean over that bank at the default
operating point (`I=10`, `J=3`, 100×100 m). Each record stores area bounds,
IoT \((x,y,0)\), process membership, \(\lambda_i\), and a random feasible UAV
placement. SCA / k-means / PSO still place or optimize their own UAVs; the
`random` method replays the saved UAV coordinates.

```text
python -m uavdt n100 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --solver cvxpy --n-scenarios 100
```

Writes:

- `data/scenario_bank/n100_i10_j3_100m.json` — frozen layouts (not gitignored)
- `results/n100/eval.json` / `.csv` / `_summary.csv` — per-scenario scores
- `results/figures/n100/` — mean±std bars, boxplots, running mean, maps

Checkpoint: `results/n100/eval.checkpoint.json`. Re-run the same command to
resume. `--generate-only` writes the bank without SCA.

This does **not** replace `campaign --n-runs 20` for Figs. 6–10. A 100-seed
axis sweep is `campaign --n-runs 100` and is several times the existing
primary campaign.

---

## Outputs

- `results/campaign_*.json` — per-axis points, per-method mean/std Mbps,
  feasible fraction, per-seed rates
- `results/campaign_*.csv` — flat table for plots
- `data/scenario_bank/n100_*.json` — frozen 100-layout Monte Carlo bank
- `results/n100/eval.json` — per-scenario SCA/baseline scores; figures in
  `results/figures/n100/`
- SCA-joint probe (separate files only; never overwrites frozen-SCA JSON):
  `campaign_8.8mhz_cap25_si12k_scajoint.json`, `tk08_scajoint_feasibility.json`,
  `sca_joint_vs_frozen.json`, `sca_joint_default_runtime.json`
- T_k=0.8 s follow-ups (also separate files):
  `tk08_scajoint_cohesive.json`, `tk08_cohesive_construction.json`,
  `tk08_sync_tradeoff_gap.json`
- Residual-on-SCA §2.9 experiments (`results/residual_on_sca/`; scripts in
  `scripts/experiments/residual_on_sca/`):
  `multistart_n20.json` (Experiment A),
  `residual_td3_heldout_21_40.json` (Experiment B),
  `cmaes_polish_heldout_21_40.json` (CMA-ES control),
  `assoc_oracle_n20.json` (Experiment C)
- TD3 Algorithm 2 reproduction (`results/campaign_8.8mhz_cap25_td3.json`,
  `results/n100/eval_td3.json`, `results/n100_500m_cap25/eval_td3.json`;
  analysis `scripts/analyze_td3_vs_methods.py`). Policy export. Do not cite
  `*_SNAPSHOT_INVALID_*`.
- Cap×J search: `bw_cap_by_J_grid.json`, `bw_boundary_refine_j3.json`,
  `cap_binding_diagnostic.json`
- Paired writeup stats (same seed, champion vs baseline):

```text
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
```

  Writes `*_paired.json` / `*_paired.csv`. Quote lines are
  `SCA wins by X+/-Y Mbps vs <baseline>, p<..., N/20 seeds`.
  `+/-Y` is the sample std of the 20 paired deltas. Primary p is
  Wilcoxon signed-rank. `uavs J=3` is the unique default scenario.
  The script flags lambda/CPU (and other) rows that are identical
  per-seed copies; do not pool those with J=3.

After the constraint-(24) init repair (`cpu_stable_processing`), re-run
only I=28, I=32, and `f_j=0.5e8` (other campaign rows are unchanged):

```text
python scripts/rerun_init_repair_points.py
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
```

Published score is always Python `evaluate()` (Eq. (17) AoDT). Fig. 11
and `aodt-compare` add FCFS / FCFS-P / LCFS-S simulations beside that
score; they do not replace it.

Full analysis, audit paragraph, and tables: **`docs/RESULTS.md`**.

**Figures:** `pip install -r requirements-dev.txt` then
`python scripts/plot_paper_figures.py` → `results/figures/` (25% primary).
Tighter-cap 15% plots: `--campaign results/campaign_8.8mhz_cap15_n20.json --fig11 results/fig11_8.8mhz_cap15.json --out-dir results/figures/cap15`.

**Paper field (500 × 500 m):** field-size test at 8.8 MHz, **25%** primary cap (15% not re-run at 500 m):

```text
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --solver cvxpy --area-m 500 --out results/campaign_8.8mhz_cap25_si12k_500m.json
```

20 kHz area control (both fields): `python scripts/check_bsys_20khz.py`.

### Bandwidth configuration screen (optional)

Scripts used to compare the primary **8.8 MHz / 25% cap** with a tighter **15%**
sensitivity and other `(B_sys, share)` cells. Outputs land in `results/`
(gitignored). Not required to reproduce the primary campaign.

| Script | Role |
| --- | --- |
| `scripts/sweep_bandwidth_screen.py` | Rank `(B_sys, max_bw_share)` at J=3 |
| `scripts/compare_bw_configs.py` | Side-by-side preset comparison |
| `scripts/compare_cap15_vs_cap25.py` | Primary 25% vs 15% sensitivity |
| `scripts/analyze_nocap_8p8mhz.py` | No-cap vs 25% primary (15% still a sensitivity) |
| `scripts/bw_grid_search.py` | Exhaustive B×cap grid (long; checkpointed) |
| `scripts/run_bw_fine_search.py` | 7.1–8.8 MHz × caps, full axes, no TD3 (long; resume-safe) |
| `scripts/bw_cap_by_j_grid.py` | Cap × J table with FDR |
| `scripts/bw_boundary_refine_j3.py` | J=3 cap bisection + 33-test FDR |
| `scripts/bw_threshold_refine.py` | Refine cap threshold near 25% |

---

## First pass on disk

**Headline (n_runs = 20, 8.8 MHz, 25% primary cap, 100 m):**

- `results/campaign_8.8mhz_cap25_si12k.json` / `.csv` — Figs. 6–10 axes,
  methods random / k-means / PSO / SCA (CVXPY). Last regenerated 2026-09-08.
- `results/fig11_8.8mhz_cap25_si12k.json` — Fig. 11 arrival patterns (50/50 sensibility).
- `spot-validate` (stdout) — seed 1, no cap and 25% cap, CVXPY vs MATLAB.
  `agreement: ok`. `se_max_abs_diff ~ 1e-15`.

**Dev smoke (n_runs = 5, 8.8 MHz, no per-link cap):**

- `results/campaign_8.8mhz.json` / `.csv` — quick axis sweep.

At 8.8 MHz with **no** per-link cap, leftover spectrum sits on the
highest-SE link, so mean sum rates sit near **~8.96–8.99 Mbps** for
every method. `λ` and `f_j` do not move the communication objective
when the point stays feasible. `T_k = 0.8 s` is infeasible for all
four methods. A **25%** cap (`--max-bw-share 0.25`, EXTERNAL, primary) still
lets leftover sit on a few high-SE links at default J = 3. Differentiation
widens under `--max-bw-share 0.15` (EXTERNAL, tighter-cap sensitivity).
