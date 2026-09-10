# Command reference — `uavdt`

Runnable commands for this simulator: setup, full regeneration, single-shot
evaluation, parameter sweeps, TD3, novelty experiments, SCA, analysis, and
plots.

**Shell setup (every session):**

```powershell
cd C:\code\UAV
$env:PYTHONPATH = "src"
```

On bash:

```bash
cd /path/to/UAV
export PYTHONPATH=src
```

**Install:**

```powershell
pip install -r requirements.txt          # numpy, pytest, cvxpy
pip install -e ".[dev]"                  # editable package + pytest
pip install -e ".[td3]"                  # + PyTorch (TD3 only)
```

**Entry point:** `python -m uavdt <subcommand>`. Shared flags are in
[§11](#11-shared-cli-flags).

---

## Table of contents

1. [Run everything from scratch](#1-run-everything-from-scratch)
2. [Evaluation only](#2-evaluation-only)
3. [Parameter-sweep campaigns](#3-parameter-sweep-campaigns)
4. [100-scenario Monte Carlo (`n100`)](#4-100-scenario-monte-carlo-n100)
5. [SCA](#5-sca)
6. [TD3](#6-td3)
7. [Novelty experiments](#7-novelty-experiments)
8. [SCA-joint probes](#8-sca-joint-probes)
9. [Analysis & plotting](#9-analysis--plotting)
10. [Diagnostics & bandwidth search](#10-diagnostics--bandwidth-search)
11. [Shared CLI flags](#11-shared-cli-flags)
12. [Quick recipes](#12-quick-recipes)

---

## 1. Run everything from scratch

Replays the full experiment ledger in `docs/RESULTS.md`: tests, headline
campaign, sensitivity runs, AoDT arrival-pattern study, solver cross-check,
SCA-joint probes, paired stats, and sweep plots. **Several hours on a laptop.**

```powershell
$env:PYTHONPATH = "src"
.\scripts\run_full_regeneration.ps1
```

Log: `results/full_regeneration_<timestamp>.log`.

### Manual step-by-step

```powershell
$env:PYTHONPATH = "src"

# 1. Unit tests
python -m pytest -q

# 2. 20 kHz B_sys feasibility check (model ceiling + 100 m / 500 m control)
python scripts/check_bsys_20khz.py --n-runs 20 --out results/check_bsys_20khz.json

# 3. Headline campaign: sum rate vs J, I, λ, T_k, CPU at 100 m, 8.8 MHz, 25% cap
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --methods random,kmeans,pso,sca --n-runs 20 --seed-start 1 --solver cvxpy `
  --out results/campaign_8.8mhz_cap25_si12k.json

# 4. Re-score campaign points after CPU-stable init repair
python scripts/rerun_init_repair_points.py

# 5. Tighter per-link cap sensitivity (15%)
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.15 `
  --methods random,kmeans,pso,sca --n-runs 20 --seed-start 1 --solver cvxpy `
  --out results/campaign_8.8mhz_cap15_n20.json

# 6. Larger field (500 m × 500 m), same radio settings
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --methods random,kmeans,pso,sca --n-runs 20 --seed-start 1 --solver cvxpy `
  --area-m 500 --out results/campaign_8.8mhz_cap25_si12k_500m.json

# 7. Seven bandwidth / cap combinations (long)
.\scripts\run_all_bandwidth_campaigns.ps1

# 8. AoDT vs UAV count under three arrival-rate patterns
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 `
  --out results/fig11_8.8mhz_cap25_si12k.json
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.15 --n-runs 20 `
  --out results/fig11_8.8mhz_cap15.json

# 9. CVXPY vs MATLAB SCA agreement on one seed
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-iterations 30
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --max-iterations 30

# 10. SCA-joint methodology probe + low-T_k follow-ups
python scripts/run_sca_joint_campaign.py
python scripts/run_tk08_followup.py

# 11. Stats + plots
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/paired_winrate.py results/campaign_8.8mhz_cap15_n20.json
python scripts/analyze_sca_vs_random_losses.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/analyze_campaigns.py
python scripts/plot_paper_figures.py
python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap15_n20.json `
  --fig11 results/fig11_8.8mhz_cap15.json --out-dir results/figures/cap15
python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap25_si12k_500m.json `
  --out-dir results/figures/500m
```

> **Note:** `plot_paper_figures.py` is the script filename. It plots **sum-rate
> sweep curves** (mean ± std vs swept parameter) and optional AoDT arrival-pattern
> curves from campaign JSON — not a literal figure reprint.

### Minimal smoke test (~minutes)

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python -m uavdt campaign --axis uavs --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --n-runs 5 --solver cvxpy --out results/campaign_smoke.json
python scripts/plot_paper_figures.py --campaign results/campaign_smoke.json --skip-fig11
```

---

## 2. Evaluation only

Score **one deployment** through `evaluate()`. No optimizer loop, no sweep.

**Use when:** debugging physics, checking a single layout, or comparing placement
without running SCA/PSO.

### One seed

```powershell
python -m uavdt evaluate --seed 1 --bandwidth-preset 8.8mhz --placement kmeans
python -m uavdt evaluate --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --placement random --verbose
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--seed` | `1` | RNG seed for IoT/UAV layout |
| `--placement` | `random` | `random` or `kmeans` UAV placement only |
| `--verbose` | off | Print UAV coords, per-link rates |

### Multi-seed mean/std (fixed placement rule, no SCA)

```powershell
python -m uavdt multi-seed --seed-start 1 --n-runs 20 --placement kmeans `
  --bandwidth-preset 8.8mhz --max-bw-share 0.25
```

### Same geometry, three B_sys presets

```powershell
python -m uavdt bandwidth-sweep --seed 1 --placement kmeans
```

Runs 20 kHz, 2.4 MHz, and 8.8 MHz on the same layout.

### Closed-form AoDT vs event-queue simulator

```powershell
python -m uavdt aodt-compare --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --placement kmeans --horizon 120 --warmup 24
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--horizon` | `120` | Sim horizon (s) |
| `--warmup` | `24` | Warmup discard (s) |
| `--queue-scope` | `process` | `process` (per-process queue) or `uav` (shared server) |
| `--lambda-pattern` | cfg default | `uniform_fast`, `uniform_slow`, `heterogeneous` |

---

## 3. Parameter-sweep campaigns

Run every method on a grid of operating points. Each point is averaged over
`--n-runs` random seeds (default **5** for dev; **20** for final numbers).

**Use when:** comparing SCA vs baselines across UAV count, IoT count, arrival
rate, AoDT threshold, or UAV CPU.

```powershell
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --methods random,kmeans,pso,sca --n-runs 20 --seed-start 1 --solver cvxpy `
  --out results/campaign_8.8mhz_cap25_si12k.json
```

### Sweep axes

| `--axis` | What varies | Held fixed (default grid) |
|----------|-------------|---------------------------|
| `uavs` | UAV count J = 1…5 | I = 10 |
| `iots` | IoT count I = 10…32 | J = 3 |
| `lambda` | Arrival rate λ = 1.0…3.5 /s | I = 10, J = 3 |
| `aodt` | AoDT threshold T_k = 0.8…3.0 s | I = 10, J = 3 |
| `cpu` | UAV CPU f_j = 0.5e8…2.5e8 cycles/s | I = 10, J = 3 |
| `all` | All five axes in one JSON | — |

### Single axis

```powershell
python -m uavdt campaign --axis uavs --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --n-runs 20 --out results/campaign_uavs_only.json
```

### Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--axis` | `all` | See table above |
| `--methods` | `random,kmeans,pso,sca` | Add `sca_joint` or `td3` opt-in |
| `--n-runs` | `5` | Seeds per sweep point |
| `--seed-start` | `1` | Consecutive seeds |
| `--solver` | `cvxpy` | `cvxpy` (fast) or `matlab` (MOSEK, slow) |
| `--out` | `results/campaign.json` | Writes JSON + CSV |

### High-statistics replay (100 seeds per point)

Same five axes, but `--n-runs 100` for tighter error bars. Checkpoints per axis;
resumes on crash.

```powershell
python scripts/run_n100_campaign.py
```

Output: `results/campaign_8.8mhz_cap25_n100.json` (merged from
`results/n100_campaign/*.json`).

---

## 4. 100-scenario Monte Carlo (`n100`)

**Different from a campaign:** freezes **100 layouts** once (area + IoT + bank
UAVs), then scores every method on that same bank at I = 10, J = 3.

**Use when:** you want per-scenario variance, layout maps, or a fixed geometry
bank — not a parameter sweep.

```powershell
python -m uavdt n100 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --solver cvxpy --n-scenarios 100
```

**Outputs:**

| Path | Contents |
|------|----------|
| `data/scenario_bank/n100_i10_j3_100m.json` | Frozen 100 layouts |
| `results/n100/eval.json` | Per-scenario scores |
| `results/n100/eval.csv` | Flat table |
| `results/n100/eval_summary.csv` | Method means |
| `results/n100/eval.checkpoint.json` | Resume checkpoint |
| `results/figures/n100/` | Bar, box, running-mean, per-scenario, layout maps |

### Variants

```powershell
python -m uavdt n100 --generate-only          # write bank only
python -m uavdt n100 --skip-eval              # re-plot from eval.json
python -m uavdt n100 --methods random,kmeans,pso,sca,td3 --td3-preset residual-on-sca
python -m uavdt n100 --force-generate         # overwrite bank
python -m uavdt n100 --no-resume              # ignore checkpoint
```

### Sum-rate vs UAV count from 100-seed campaign

If you ran `run_n100_campaign.py`, plot sweep curves from that JSON:

```powershell
python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap25_n100.json `
  --skip-fig11 --out-dir results/figures/n100
```

### One seed: initial vs optimized UAV layout

Side-by-side map of bank UAVs vs each method's final UAV positions.

```powershell
python -c "
import json
from pathlib import Path
from uavdt.experiments.n100_plot import plot_seed_layout_comparison
payload = json.loads(Path('results/n100/eval.json').read_text(encoding='utf-8'))
bank = json.loads(Path('data/scenario_bank/n100_i10_j3_100m.json').read_text(encoding='utf-8'))
plot_seed_layout_comparison(payload, bank, 79, 'results/figures/n100')
"
```

Writes `results/figures/n100/n100_seed79_layout_comparison.png` (+ `.pdf`).

---

## 5. SCA

Sequential convex approximation optimizer for UAV placement + association +
bandwidth. Final score always goes through `evaluate()`.

**Use when:** running or debugging the main optimizer on one seed, checking
solver backends, or logging SCA iterations.

### Single run

```powershell
python -m uavdt sca --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --solver cvxpy --max-iterations 30 --step-size 20.0
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--solver` | `matlab` | `cvxpy` or `matlab`/`mosek` |
| `--max-iterations` | `30` | Outer SCA iterations |
| `--epsilon` | `1e-4` | Stop tolerance |
| `--step-size` | `20.0` | L∞ trust region on UAV xy (m) |
| `--history-json` | `results/sca_history.json` | Iteration log |
| `--history-csv` | `results/sca_history.csv` | Flat iteration log |

### Per-iteration debug log

```powershell
python -m uavdt sca-seq-debug --seed 1 --bandwidth-preset 8.8mhz --solver cvxpy `
  --out-json results/sca_seq_debug.json
```

### CVXPY vs MATLAB cross-check

```powershell
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25
```

### Direct MATLAB bridge test

```powershell
python scripts/compare_sca_cvxpy_matlab.py
```

---

## 6. TD3

Reinforcement-learning UAV placement. Requires `pip install -e ".[td3]"` (PyTorch).

**Use when:** training the proposed residual-on-SCA policy or reproducing the
reference TD3 baseline from the source paper's Algorithm 2.

### Train one seed (CLI)

```powershell
# Reference TD3 baseline (k-means residual, leftover inner B, penalty reward)
python -m uavdt td3 --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --total-steps 7000

# Proposed method: residual Δq on SCA incumbent, inner LP, feasible-Mbps reward
python -m uavdt td3 --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --td3-preset residual-on-sca --total-steps 7000
```

### TD3 flags

| Flag | Choices | Notes |
|------|---------|-------|
| `--td3-preset` | `alg2`, `residual-on-sca` | Default `alg2` |
| `--inner-bandwidth` | `leftover`, `equal_share`, `lp` | Inner bandwidth rule |
| `--uav-init` | `kmeans`, `random`, `sca` | Starting UAV layout |
| `--reward-mode` | `alg2`, `feasible_rate` | Training reward |
| `--export-mode` | `policy`, `best_snapshot` | How final UAV is exported |
| `--total-steps` | `7000` | Training env steps |
| `--horizon` | `50` | Episode length |
| `--warmup-steps` | `256` | Random steps before learning |
| `--log-every` | `250` | Progress interval (`0` = silent) |

### TD3 inside a campaign sweep

```powershell
python -m uavdt campaign --axis uavs --methods random,kmeans,pso,sca,td3 `
  --td3-preset residual-on-sca --n-runs 5 --bandwidth-preset 8.8mhz --max-bw-share 0.25
```

### Standalone scripts

```powershell
python scripts/run_td3_train.py --seed 1 --total-steps 7000 --out results/td3/train_seed1.json
python scripts/run_td3_full_eval.py          # n100 bank + campaign; long; CUDA if available
python scripts/run_td3_policy_export_small.py
python scripts/run_td3_policy_export_n20.py
python scripts/analyze_td3_vs_methods.py
```

### Diagnostics

```powershell
python scripts/diagnose_td3_saturation.py
python scripts/diagnose_td3_rate_gap.py
python scripts/verify_td3_unsaturate.py
```

---

## 7. Novelty experiments

Pre-registered tests for **residual-on-SCA**. Never overwrite
`results/campaign_8.8mhz_cap25_si12k.json`.

### A — Multi-start SCA (is placement initialization the bottleneck?)

Runs frozen k-means-init SCA vs four extra SCA starts (2 random + 2 k-means,
keep-best).

```powershell
python scripts/run_sca_multistart.py
python scripts/run_sca_multistart.py --n-runs 20 --seed-start 1
```

Output: `results/sca_multistart_n20.json`

### B — Residual policy on SCA (TD3)

Held-out seeds 21–40. Trains `residual-on-sca` and compares to frozen SCA on the
same geometries.

```powershell
python scripts/run_residual_on_sca.py
python scripts/run_residual_on_sca.py --n-runs 20 --seed-start 21 --total-steps 7000
```

Output: `results/residual_on_sca_heldout_21_40.json`

### B control — CMA-ES + SCA polish (no neural net)

Same fitness as residual-on-SCA; checks whether RL adds anything over derivative-free search.

```powershell
python scripts/run_cmaes_sca_polish.py
python scripts/run_cmaes_sca_polish.py --n-runs 1 --max-evals 40
```

Output: `results/cmaes_sca_polish_heldout_21_40.json`

---

## 8. SCA-joint probes

SCA-joint re-optimizes discrete association `a_ij` / processing `b_ij` after
each SCA step. **Exploratory only** — does not replace frozen SCA in headline
campaigns.

```powershell
python scripts/run_sca_joint_campaign.py
python scripts/run_sca_joint_campaign.py --only tk08      # low T_k feasibility
python scripts/run_sca_joint_campaign.py --only campaign
python scripts/run_sca_joint_campaign.py --only runtime

python -m uavdt sca-joint --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 `
  --solver cvxpy --process-cohesive-candidate
```

**Outputs:** `results/campaign_8.8mhz_cap25_si12k_scajoint.json`,
`results/tk08_scajoint_feasibility.json`, `results/sca_joint_vs_frozen.json`, etc.

### Low AoDT threshold (T_k = 0.8 s) follow-ups

Tests process-cohesive rematch and construction under a tight delay bound.

```powershell
python scripts/run_tk08_followup.py
python scripts/run_tk08_followup.py --only cohesive
python scripts/run_tk08_followup.py --only construction
python scripts/run_tk08_followup.py --only tradeoff
```

---

## 9. Analysis & plotting

### Paired stats (same seed, method vs method)

Wilcoxon signed-rank and win-rate per sweep point.

```powershell
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/paired_winrate.py results/campaign_8.8mhz_cap15_n20.json --champion sca
```

Writes `*_paired.json` and `*_paired.csv` beside the campaign file.

### Text summaries

```powershell
python scripts/analyze_campaigns.py
python scripts/analyze_campaign.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/summarize_campaign.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/analyze_sca_vs_random_losses.py results/campaign_8.8mhz_cap25_si12k.json
```

### Sweep plots (`plot_paper_figures.py`)

Produces errorbar curves: **sum rate (Mbps) vs swept parameter** for each
method, plus an optional **AoDT vs UAV count** plot from `fig11` JSON, and a
2×3 overview grid.

```powershell
# Headline 100 m campaign → results/figures/
python scripts/plot_paper_figures.py

# Custom inputs
python scripts/plot_paper_figures.py `
  --campaign results/campaign_8.8mhz_cap25_si12k.json `
  --fig11 results/fig11_8.8mhz_cap25_si12k.json `
  --out-dir results/figures

# 500 m field campaign (skip AoDT arrival plot if no fig11 JSON)
python scripts/plot_paper_figures.py `
  --campaign results/campaign_8.8mhz_cap25_si12k_500m.json `
  --skip-fig11 --out-dir results/figures/500m
```

**Typical outputs** in `--out-dir`: one sum-rate curve per sweep axis (`*_sum_rate.png`
/ `.pdf`), an AoDT-vs-J plot when `--fig11` JSON is supplied, and
`overview_figs06_10` combining all sum-rate sweeps on one page. Filenames under
`results/figures/` still use legacy `fig06_`…`fig11_` prefixes from the script;
read the plot title for the swept parameter.

### Cap / bandwidth comparisons

```powershell
python scripts/compare_cap15_vs_cap25.py
python scripts/compare_bw_configs.py
python scripts/analyze_nocap_8p8mhz.py
```

### AoDT arrival-pattern study (`fig11` subcommand)

Measures max process AoDT vs UAV count for uniform-fast, uniform-slow, and
heterogeneous λ patterns.

```powershell
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 `
  --out results/fig11_8.8mhz_cap25_si12k.json
```

---

## 10. Diagnostics & bandwidth search

```powershell
# 20 kHz B_sys ceiling + feasibility at 100 m and 500 m
python scripts/check_bsys_20khz.py
python scripts/check_bsys_20khz.py --n-runs 20 --methods random,kmeans,pso,sca

python scripts/cap_binding_diagnostic.py
python scripts/se_spread_by_j.py
python scripts/bw_cap_by_j_grid.py
python scripts/bw_boundary_refine_j3.py
python scripts/bw_grid_search.py
python scripts/bw_threshold_refine.py
python scripts/sweep_bandwidth_screen.py
python scripts/run_sca_bw_matrix.py
python scripts/local_direction_diag.py
python scripts/summarize_local_diag.py
```

---

## 11. Shared CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--bandwidth-preset` | — | `20khz`, `2.4mhz`, `8.8mhz` (overrides `--bandwidth`) |
| `--bandwidth` | 8.8 MHz | `B_sys` in Hz |
| `--max-bw-share` | none | Per-link cap fraction (`0.25` headline, `0.15` sensitivity) |
| `--area-m` | `100` | Square field side (m); use `500` for large-field test |
| `--lambda-i` | `2.0` | Task arrival rate λ (tasks/s) |
| `--aodt-threshold` | `2.8` | AoDT threshold T_k (s) |
| `--task-size-bits` | external | S_i in bits |
| `--task-cycles` | external | L in cycles/task |
| `--task-size-bytes` | — | If set, S_i = bytes × 8 |
| `--los-angle-unit` | `rad` | `rad` or `deg` for LoS probability |
| `--placement` | `random` | `evaluate` only: `random` or `kmeans` |
| `--download-time` | `0` | UAV→BS download delay Z (s) |

**Methods** (campaign / n100): `random`, `kmeans`, `pso`, `sca`, optional
`sca_joint`, `td3`.

**Default sweep grids** (`src/uavdt/experiments/grids.py`):

```text
J:   1, 2, 3, 4, 5
I:   10, 16, 20, 24, 28, 32
λ:   1.0 … 3.5  /s
T_k: 0.8 … 3.0  s
f_j: 0.5e8 … 2.5e8  cycles/s
```

---

## 12. Quick recipes

| Goal | Command |
|------|---------|
| Run all tests | `python -m pytest -q` |
| Score one layout | `python -m uavdt evaluate --seed 1 --placement random --bandwidth-preset 8.8mhz` |
| Run SCA on one seed | `python -m uavdt sca --seed 1 --solver cvxpy --bandwidth-preset 8.8mhz --max-bw-share 0.25` |
| Full parameter sweep (headline settings) | `python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --solver cvxpy --out results/campaign_8.8mhz_cap25_si12k.json` |
| 100 frozen layouts | `python -m uavdt n100 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --solver cvxpy` |
| Plot sum-rate sweeps | `python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap25_si12k.json --skip-fig11` |
| SCA vs random p-values | `python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json` |
| Train proposed TD3 | `python -m uavdt td3 --td3-preset residual-on-sca --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25` |
| Multi-start SCA test | `python scripts/run_sca_multistart.py` |
| Residual-on-SCA held-out | `python scripts/run_residual_on_sca.py` |
| Full regeneration | `.\scripts\run_full_regeneration.ps1` |
| 20 kHz sanity check | `python scripts/check_bsys_20khz.py` |
| Seed 79 layout map | See [§4 — layout comparison](#one-seed-initial-vs-optimized-uav-layout) |
| AoDT vs J (three λ patterns) | `python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20` |

---

## Related docs

- [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) — campaign ledger and freeze policy
- [`docs/RESULTS.md`](RESULTS.md) — headline numbers and regeneration log
- [`docs/REPRODUCTION.md`](REPRODUCTION.md) — parameter ledger and model scope
- [`docs/param.md`](param.md) — default vs external parameters
