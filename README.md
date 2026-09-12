# UAV-aided digital twin simulator

Python implementation of the Khalaf et al. (IEEE TNSM, 2026) **system model**
for UAV-aided digital twins in IoT networks. Includes placement baselines,
Algorithm 1 SCA, Algorithm 2 TD3 (opt-in), and §VII-style experimental campaigns.

TD3 hyperparameters live in `TD3Settings` (`uavdt.td3`). They are **not** on
`SimConfig` and are **not** Table II. Default campaigns still run
`random,kmeans,pso,sca`.

| Document | Purpose |
| --- | --- |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Parameter ledger, paper vs modification vs external |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Campaign axes, methods, frozen-SCA rules |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Current experimental outcomes and audit |
| [`docs/param.md`](docs/param.md) | Quick parameter reference |

---

## Requirements

- **Python 3.11+**
- **NumPy** (required)
- **CVXPY** (required for `sca`, `campaign`, `spot-validate` with the default CVXPY backend)
- **pytest** (for tests)
- **matplotlib** (optional; only for `scripts/plot_paper_figures.py` — install via `requirements-dev.txt`)
- **PyTorch** (optional; only for `td3` — `pip install torch` or `pip install -e ".[td3]"`)
- **MATLAB R2026a + CVX + MOSEK** (optional; only needed when `--solver matlab` is used)

---

## Installation

From the repository root:

```powershell
# PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
# optional: paper-style figures
pip install -r requirements-dev.txt
```

```bash
# Bash (Linux / macOS / WSL)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# optional: paper-style figures
pip install -r requirements-dev.txt
```

`pip install -e .` registers the `uavdt` console script. If you skip it, set
`PYTHONPATH` to `src` before every command:

```powershell
# PowerShell
$env:PYTHONPATH = "src"
```

```bash
# Bash
export PYTHONPATH=src
```

All commands below assume you are in the repo root and either have run
`pip install -e .` or set `PYTHONPATH=src`.

---

## Quick start

```powershell
# Run the test suite
python -m pytest

# Evaluate one random placement at default settings (100×100 m, 20 kHz B_sys)
python -m uavdt evaluate --seed 1 --placement random

# Run SCA at the headline 8.8 MHz bandwidth
python -m uavdt sca --seed 1 --bandwidth-preset 8.8mhz --solver cvxpy
```

---

## CLI overview

Invoke the simulator as a module:

```text
python -m uavdt <command> [options]
```

Or, after `pip install -e .`:

```text
uavdt <command> [options]
```

Get built-in help for any command:

```text
python -m uavdt --help
python -m uavdt evaluate --help
python -m uavdt sca --help
```

### Commands at a glance

| Command | What it does |
| --- | --- |
| `evaluate` | One seed, one bandwidth — print sum rate, AoDT, feasibility |
| `multi-seed` | Mean/std over consecutive seeds (JSON summary) |
| `bandwidth-sweep` | Run all three `B_sys` presets on one seed |
| `sca` | Algorithm 1 SCA; writes iteration history to `results/` |
| `sca-joint` | Methodology probe: SCA plus discrete \(a_{ij}/b_{ij}\) re-match (`method="sca_joint"`). Optional `--process-cohesive-candidate`. |
| `sca-multistart` | Keep-best extra SCA inits (2 random + 2 k-means). Opt-in; does not replace frozen SCA. |
| `td3` | Algorithm 2 TD3 (opt-in). Per-instance train; knobs are `TD3Settings`, not `SimConfig`. |
| `sca-seq-debug` | Per-iteration SCA log with true-feasibility gate |
| `campaign` | §VII sweeps over UAV count, IoT count, λ, AoDT threshold, CPU |
| `spot-validate` | CVXPY vs MATLAB CVX/MOSEK spot-check on frozen SCA |
| `aodt-compare` | Closed-form AoDT vs event-driven queue simulation |
| `n100` | Generate 100 saved area+IoT+UAV layouts; run SCA/baselines; plot averages |
| `fig11` | Fig. 11 arrival-pattern sweep (uniform fast/slow/heterogeneous λ) |

---

## Shared flags

These flags are available on **every** subcommand (via `_add_shared` in the CLI).

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--bandwidth` | float | `20000` | `B_sys` in Hz. Ignored when `--bandwidth-preset` is set. |
| `--bandwidth-preset` | choice | *(none)* | Named preset: `20khz`, `2.4mhz`, `8.8mhz`. Overrides `--bandwidth`. |
| `--task-size-bits` | float | `96000` | Task size `S_i` in bits (external, not Table II). |
| `--task-size-bytes` | float | *(none)* | If set, `S_i = bytes × 8`. |
| `--task-cycles` | float | `3.75e6` | Computation load `L` in cycles/task (external). |
| `--lambda-i` | float | `2.0` | Uniform task arrival rate λ (tasks/s per IoT). |
| `--aodt-threshold` | float | `2.8` | AoDT threshold `T_k` in seconds. |
| `--max-bw-share` | float | *(none)* | Optional per-link cap as a fraction of `B_sys` (e.g. `0.25` primary, `0.15` tighter-cap sensitivity). **External parameter**, not in Problem (P). |
| `--los-angle-unit` | `rad` \| `deg` | `rad` | Unit for the LoS elevation angle in Eq. (4). |
| `--area-m` | float | `100` | Square field side in metres. Paper §VII uses `500`. |
| `--placement` | `random` \| `kmeans` | `random` | UAV placement baseline (not used by `sca`, which optimizes position). |
| `--download-time` | float | `0.0` | Eq. (12) UAV→BS download time `Z` in seconds. Paper neglects this. |

### Bandwidth presets

| Preset | `B_sys` (Hz) | Typical use |
| --- | --- | --- |
| `20khz` | 20 000 | Table II diagnostic (0% feasible under Eq. (6)+(27) cap) |
| `2.4mhz` | 2 400 000 | Headline experiment, no per-link cap |
| `8.8mhz` | 8 800 000 | Headline experiment (default for campaigns) |

---

## Command reference

### `evaluate` — single-run evaluation

Evaluate one random seed with a fixed placement and bandwidth allocation.

```powershell
python -m uavdt evaluate --seed 1 --bandwidth-preset 8.8mhz --placement kmeans --verbose
```

| Flag | Default | Description |
| --- | --- | --- |
| `--seed` | `1` | RNG seed for IoT positions and placement. |
| `--verbose` | off | Print UAV coordinates, per-link rates, and distances. |

**Example — 20 kHz at paper field size:**

```powershell
python -m uavdt evaluate --seed 1 --bandwidth 20000 --area-m 500 --placement kmeans
```

---

### `multi-seed` — aggregate over seeds

Run `evaluate` logic across consecutive seeds and print a JSON summary.

```powershell
python -m uavdt multi-seed --bandwidth-preset 8.8mhz --placement random --n-runs 20 --seed-start 1
```

| Flag | Default | Description |
| --- | --- | --- |
| `--seed-start` | `1` | First seed (inclusive). |
| `--n-runs` | `20` | Number of consecutive seeds. Paper uses 20 per plotted point. |

---

### `bandwidth-sweep` — all presets on one seed

Runs `evaluate` for `20khz`, `2.4mhz`, and `8.8mhz` in sequence.

```powershell
python -m uavdt bandwidth-sweep --seed 1 --placement random
```

| Flag | Default | Description |
| --- | --- | --- |
| `--seed` | `1` | RNG seed. |

---

### `sca` — sequential convex approximation

Run Algorithm 1 SCA of Problem (P). Writes iteration history to JSON and CSV.

```powershell
# CVXPY backend (no MATLAB required)
python -m uavdt sca --seed 1 --bandwidth-preset 2.4mhz --solver cvxpy

# MATLAB CVX + MOSEK backend (default)
python -m uavdt sca --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --solver matlab
```

| Flag | Default | Description |
| --- | --- | --- |
| `--seed` | `1` | RNG seed. |
| `--max-iterations` | `30` | Maximum SCA iterations. |
| `--epsilon` | `1e-4` | Convergence tolerance on objective improvement. |
| `--step-size` | `20.0` | L∞ trust-region radius for UAV position updates (metres). |
| `--solver` | `matlab` | `matlab` / `MOSEK` → MATLAB CVX+MOSEK; `cvxpy` / `python` / `none` → Python HiGHS LP. |
| `--history-json` | `results/sca_history.json` | Output path for iteration history (JSON). |
| `--history-csv` | `results/sca_history.csv` | Output path for iteration history (CSV). |

---

### `td3` — Algorithm 2 (opt-in)

Per-instance TD3 on one scenario. Paper Algorithm 2 gaps (network size,
\(\gamma\), \(\tau\), noise, …) are filled in `TD3Settings` from Fujimoto et al.
2018 plus the Algorithm 2 reward weights. Does **not** change `SimConfig`.
Requires PyTorch.

```powershell
python -m uavdt td3 --seed 1 --bandwidth-preset 8.8mhz --total-steps 7000
```

| Flag | Default | Description |
| --- | --- | --- |
| `--seed` | `1` | Scenario and algorithm RNG seed. |
| `--total-steps` | `7000` | Environment steps (paper Fig. 4 ≈ 7000). |
| `--horizon` | `50` | Steps per episode before a random UAV reset. |
| `--hidden` | `256` | MLP width (two ReLU layers). |
| `--batch-size` | `256` | Replay minibatch. |
| `--warmup-steps` | `256` | Random actions before critic updates. |
| `--buffer-size` | `100000` | Replay capacity. |

Campaign / n100: add `td3` to `--methods` (not in the default list).

---

### `sca-seq-debug` — iteration-level debug log

Same physics as `sca`, but prints a human-readable per-iteration table and
writes a detailed JSON log.

```powershell
python -m uavdt sca-seq-debug --seed 1 --bandwidth-preset 2.4mhz --solver cvxpy --max-iterations 10
```

| Flag | Default | Description |
| --- | --- | --- |
| `--seed` | `1` | RNG seed. |
| `--max-iterations` | `30` | Maximum SCA iterations. |
| `--step-size` | `20.0` | L∞ trust-region radius (metres). |
| `--solver` | `matlab` | Same choices as `sca --solver`. |
| `--out-json` | `results/sca_seq_debug.json` | Detailed iteration log. |

---

### `campaign` — §VII experimental sweeps

Sweep the paper's Fig. 6–10 axes (UAV count, IoT count, λ, AoDT threshold,
CPU) across placement/optimization methods.

```powershell
# Quick smoke test (5 seeds, CVXPY)
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --n-runs 5 --solver cvxpy

# Paper-faithful run count (20 seeds)
python -m uavdt campaign --axis uavs,iots --bandwidth-preset 8.8mhz --max-bw-share 0.25 --methods random,kmeans,pso,sca --n-runs 20 --seed-start 1 --solver cvxpy --out results/campaign_8.8mhz_cap25_si12k.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--axis` | `all` | Comma-separated axes or `all`. Choices: `uavs`, `iots`, `lambda`, `aodt`, `cpu`. |
| `--methods` | `random,kmeans,pso,sca` | Comma-separated methods. Opt-in: `sca_joint`, `sca_multistart`, `td3`. |
| `--n-runs` | `5` | Seeds per sweep point. Paper uses **20**; default is 5 for faster runs. |
| `--seed-start` | `1` | First seed. |
| `--max-iterations` | `30` | SCA iterations (only affects the `sca` method). |
| `--epsilon` | `1e-4` | SCA convergence tolerance. |
| `--step-size` | `20.0` | SCA trust-region radius (metres). |
| `--solver` | `cvxpy` | SCA backend: `cvxpy` (campaign default) or `matlab`. |
| `--out` | `results/campaign.json` | Output JSON (a `.csv` sibling is written automatically). |

**Batch all bandwidth presets** (PowerShell):

```powershell
.\scripts\run_all_bandwidth_campaigns.ps1
```

---

### `spot-validate` — CVXPY vs MATLAB agreement check

Run a short frozen SCA and compare Python CVXPY against MATLAB CVX/MOSEK.

```powershell
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-iterations 12
```

| Flag | Default | Description |
| --- | --- | --- |
| `--seed` | `1` | RNG seed. |
| `--max-iterations` | `12` | SCA iterations for the spot check. |
| `--step-size` | `20.0` | Trust-region radius (metres). |

Exit code `2` if objectives disagree between backends.

---

### `aodt-compare` — closed-form vs simulation

Compare Eq. (17) AoDT scoring against FCFS / FCFS-P / LCFS-S event-driven
queue simulation.

```powershell
python -m uavdt aodt-compare --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --placement kmeans --horizon 120 --warmup 24
```

| Flag | Default | Description |
| --- | --- | --- |
| `--seed` | `1` | RNG seed. |
| `--horizon` | `120.0` | Simulation horizon (seconds). |
| `--warmup` | `24.0` | Warm-up period discarded before statistics (seconds). |
| `--queue-scope` | `process` | `process` = Eq. (17) per-process isolation; `uav` = shared UAV server (constraint 24). |
| `--lambda-pattern` | *(uniform)* | Fig. 11 arrival pattern: `uniform_fast`, `uniform_slow`, `heterogeneous`. |

---

### `fig11` — arrival-pattern sensitivity

Reproduce the Fig. 11 check: three λ patterns vs UAV count `J`.

```powershell
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --seed-start 1 --out results/fig11.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--n-runs` | `20` | Seeds per (pattern, J) point. Paper uses 20. |
| `--seed-start` | `1` | First seed. |
| `--horizon` | `80.0` | Simulation horizon (seconds). |
| `--warmup` | `16.0` | Warm-up period (seconds). |
| `--out` | `results/fig11.json` | Output JSON (`.csv` sibling written automatically). |

Exit code `2` if sensibility checks fail.

---

### `n100` — 100 saved area + IoT + UAV scenarios

Generate a frozen bank of random layouts (square field, IoT at \(z=0\), random UAV xy at height \(H\)), run SCA (the per-instance optimizer) plus placement baselines on every layout, then plot **means over the bank**. This is the Monte Carlo at the default \(I=10\), \(J=3\) operating point. It is **not** the Fig. 6–10 axis sweep; for those with 100 seeds use `campaign --n-runs 100` (much longer).

Default methods are SCA plus placement baselines. TD3 is opt-in:
`--methods random,kmeans,pso,sca,td3` (slow: per-scenario training).

```powershell
# Write 100 layouts, then SCA + baselines, then average figures
python -m uavdt n100 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --solver cvxpy

# Layouts only
python -m uavdt n100 --generate-only --n-scenarios 100 --bank data/scenario_bank/n100_i10_j3_100m.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--n-scenarios` | `100` | Number of saved layouts. |
| `--seed-start` | `1` | First seed (inclusive). |
| `--num-iot` / `--num-uav` | `10` / `3` | Frozen \(I\) and \(J\). |
| `--bank` | `data/scenario_bank/n100_i10_j3_100m.json` | Saved geometry. |
| `--out` | `results/n100/eval.json` | Per-scenario scores (`_summary.csv` sibling). |
| `--fig-dir` | `results/figures/n100` | Mean/std, box, running-mean, maps. |
| `--checkpoint` | `results/n100/eval.checkpoint.json` | Resume after a crash. |
| `--methods` | `random,kmeans,pso,sca` | Same set as `campaign`. |
| `--generate-only` | off | Write the bank and stop. |
| `--skip-eval` / `--skip-plot` | off | Replot from an existing `--out`. |
| `--force-generate` | off | Overwrite an existing bank. |
| `--no-resume` | off | Ignore the checkpoint. |

If `--bandwidth-preset` is omitted, this command uses **8.8 MHz** and a **25%** per-link cap so it matches the primary campaign.

---

## Standalone scripts

Scripts in `scripts/` are not registered as console entry points. Run them
with `python scripts/<name>.py` from the repo root (set `PYTHONPATH=src` if
you did not `pip install -e .`).

### `check_bsys_20khz.py` — Table II bandwidth audit

Model-free Eq. (6)+(27) ceiling, then 20 kHz feasibility at 100 m and 500 m.

```powershell
python scripts/check_bsys_20khz.py
python scripts/check_bsys_20khz.py --n-runs 5 --methods random,kmeans --seed-start 1 --out results/check_bsys_20khz.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--n-runs` | `20` | Seeds per area/method combination. |
| `--seed-start` | `1` | First seed. |
| `--methods` | `random,kmeans` | Comma-separated placement methods. |
| `--out` | `results/check_bsys_20khz.json` | Output JSON path. |

### `analyze_campaign.py` — print campaign summary

```powershell
python scripts/analyze_campaign.py results/campaign.json
```

### `paired_winrate.py` — SCA vs baseline win-rate statistics

Paired Wilcoxon signed-rank test on per-seed Mbps deltas.

```powershell
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/paired_winrate.py results/campaign.json --champion sca --tie-eps 1e-9
```

| Flag | Default | Description |
| --- | --- | --- |
| `campaign_json` | *(required positional)* | Path to a campaign output JSON. |
| `--champion` | `sca` | Method to compare all others against. |
| `--tie-eps` | `1e-9` | Absolute delta ≤ ε counts as a tie. |
| `--out-json` | *(none)* | Optional JSON output path. |
| `--out-csv` | *(none)* | Optional CSV output path. |

### `run_all_bandwidth_campaigns.ps1` — batch campaigns

Runs full §VII campaigns for all bandwidth presets (no cap, plus 8.8 MHz
at 25% primary and 15% tighter-cap sensitivity), 20 seeds each. Requires PowerShell.

```powershell
.\scripts\run_all_bandwidth_campaigns.ps1
```

### `run_full_regeneration.ps1` — replay docs/RESULTS.md §9

Runs pytest, check_bsys, primary + cap15 + 500 m campaigns, fig11,
spot-validate, SCA-joint probe, tk08 follow-ups, and analysis/plots.
Does **not** include the long bandwidth preset sweep (run that separately).

```powershell
.\scripts\run_full_regeneration.ps1
```

---

## Running tests

```powershell
python -m pytest                  # full suite
python -m pytest tests/test_channel.py -v   # single file
python -m pytest -k "sca" -v      # name filter
```

`pytest.ini` sets `pythonpath = src`, so `PYTHONPATH` is not required for tests.

---

## Optional: MATLAB CVX + MOSEK

The `matlab` solver backend shells out to `matlab/sca_seq.m` and related
files. Requirements:

1. MATLAB installed (the bridge looks for `C:\Program Files\MATLAB\R2026a\bin\matlab.exe` on Windows; edit `src/uavdt/sca/matlab_bridge.py` if your install path differs).
2. CVX and MOSEK configured inside MATLAB (`matlab/setup_cvx_mosek.m`).

Use `--solver cvxpy` everywhere if MATLAB is not available. Campaigns default
to CVXPY; `sca` defaults to MATLAB.

---

## Common workflows

### 1. Smoke-test the install

```powershell
python -m pytest -q
python -m uavdt evaluate --seed 1 --bandwidth-preset 2.4mhz --placement random
```

### 2. Run headline SCA at 8.8 MHz with 25% per-link cap

```powershell
python -m uavdt sca --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --solver cvxpy
```

Tighter-cap experiments use `--max-bw-share 0.15` and write a separate output.

### 3. Full experimental campaign (paper run count)

```powershell
python -m uavdt campaign `
  --axis all `
  --bandwidth-preset 8.8mhz `
  --max-bw-share 0.25 `
  --methods random,kmeans,pso,sca `
  --n-runs 20 `
  --seed-start 1 `
  --solver cvxpy `
  --out results/campaign_8.8mhz_cap25_si12k.json

python scripts/analyze_campaign.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
```

### 4. Verify 20 kHz Table II infeasibility

```powershell
python scripts/check_bsys_20khz.py --n-runs 20
```

---

## Methodology note

This reproduction follows the paper's equations but does **not** tune
parameters to match published Mbps figures. Under Eq. (6) and constraint (27),
Table II's `B_sys = 20 kHz` caps the system at **~0.997 Mbps** regardless of
SNR — not the 7–14 Mbps quoted in §VII. Headline experiments use **2.4 MHz**
and **8.8 MHz** at **100 × 100 m**, with an optional **25% per-link bandwidth
cap** (`--max-bw-share 0.25`) as the primary leftover-dump stress test. A **15%**
cap (`--max-bw-share 0.15`) remains in the CLI and campaign scripts as a
tighter-cap sensitivity. Both are external parameters, not part of Problem (P).

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) §4.1 for the full 20 kHz
fork analysis and [`docs/RESULTS.md`](docs/RESULTS.md) for current numbers.
