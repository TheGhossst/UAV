# UAV-aided digital twin

Python reconstruction of the paper’s shared problem (P): maximize associated uplink sum rate under association, processing, bandwidth, QoS, UAV separation, and (when enabled) queue/AoDT constraints. All solvers call one evaluator.

Paper Table II lives in `src/config.py`. Task size `S_i` and cycles `L` are **not** in Table II; pass `--compute` to use the documented experimental defaults (`2000` bytes, `2e6` cycles).

**Radio profile.** Table II read literally (`B_sys = 20 kHz`, `sigma^2 = 1e-4 W`) caps Shannon on this channel at **0.13 Mbps**, so Eq. (6) cannot produce the Mbps-scale sum rates in Figs. 6-10 and every link sits so far into the high-SNR regime that all methods collapse into one band. The default `calibrated` profile changes three knobs (noise power, a fitted `B_sys`, and a per-link bandwidth cap) to land on the published range and ranking; `--radio-profile table2` restores the literal reading. See [`docs/calibration.md`](docs/calibration.md).

## Setup

From the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Run everything from the repo root so `src` imports resolve.

```bash
python -m pytest
```

```bash
python -m pytest tests/test_comm.py tests/test_sca.py -q
```

## CLI

```bash
python -m src.main [--mode MODE] [options]
```

`--mode` default is `single`.

| Mode | What it does |
|------|----------------|
| `single` | Frozen UAV at `(250, 250, 100)`; print per-IoT channel and rate (seed `100` by default). |
| `random` | Random UAV placement + repair. |
| `kmeans` | K-means centroids on IoT `(x, y)`. |
| `pso` | Placement-only PSO. |
| `pso-joint` | Joint PSO (positions + association + processing + bandwidth). |
| `sca` | Successive convex approximation; convexified LP via SciPy HiGHS. |
| `sca-cvx` | Same SCA loop; convexified LP via MATLAB CVX + MOSEK (paper §V stack). |
| `td3` | TD3 train + greedy eval on one scenario. |
| `proposed` | Placeholder; raises `NotImplementedError`. |
| `compare` | Random / K-means / PSO / SCA on several seeds; optional TD3. Writes CSVs + markdown under `--out`. |
| `sweeps` | Default comparison plus paper-style parameter sweeps (and plots). |
| `aodt-compare` | AoDT-on comparison (forces compute model). Always includes TD3. Writes under `--out/aodt`. |
| `aodt-param-search` | Coarse-to-fine search over undocumented `S_i` and `L` (Fig. 9 behaviour). Does not change project defaults. Default `--out` is `results/aodt_parameter_search`. |

### Global options

| Flag | Type | Default | Notes |
|------|------|---------|--------|
| `--mode` | choice | `single` | See table above. |
| `--seed` | int | `100` | Scenario / solver seed. Used by `single` and the single-solver modes. Ignored by `compare` / `sweeps` / `aodt-compare` (those use seed lists). |
| `--compute` | flag | off | Enable experimental `S_i` and `L`. Required for meaningful AoDT/CPU metrics. `aodt-compare` turns this on itself. |
| `--task-size-bytes` | float | unset | Override `S_i` (bytes) for this run only. Implies `--compute`. Does not change `EXPERIMENTAL_TASK_SIZE_BYTES`. |
| `--task-cycles` | float | unset | Override `L` (cycles/task) for this run only. Implies `--compute`. Does not change `EXPERIMENTAL_TASK_CYCLES`. |
| `--radio-profile` | `calibrated` \| `table2` | `calibrated` | `calibrated` reproduces the Mbps-scale figures; `table2` is the literal Table II reading. See [`docs/calibration.md`](docs/calibration.md). |
| `--num-uav` | int | config `3` | Override `J`. |
| `--num-iot` | int | config `10` | Override `I`. If `I` is divisible by the number of processes (`2`), `iots_per_process` is updated. |
| `--lambda-i` | float | config `2.0` | Task arrival rate (tasks/s). |
| `--aodt-threshold` | float | config `2.8` | AoDT threshold `T_k` (s). |
| `--uav-cpu` | float | config `2e8` | UAV CPU (cycles/s). |
| `--los-unit` | `rad` \| `deg` | config `deg` | LoS elevation unit in Eq. (4). |
| `--particles` | int | `20` | PSO swarm size (`pso`, `pso-joint`, `compare`, `sweeps`, `aodt-compare`). |
| `--iters` | int | `100` | PSO iterations (same modes). |
| `--td3-steps` | int | `7000` | TD3 training steps (`td3`, `compare --with-td3`, `sweeps --with-td3`, `aodt-compare`). |
| `--device` | `auto` \| `cpu` \| `cuda` | `auto` | TD3 nets: `auto` uses CUDA when available. Simulator (channel, repair, evaluator, PSO, SCA) stays on CPU. |
| `--paper-runs` | flag | off | Use 20 scenario seeds `100…119` instead of the 5-seed dev set `100…104`. Applies to `compare` and `sweeps`. |
| `--with-td3` | flag | off | Include TD3 in `compare` and `sweeps`. |
| `--aodt-short` | flag | off | `aodt-compare` only: 5 eval seeds `100…104` instead of `100…119`. |
| `--out` | str | `results` | Output directory. `aodt-compare` writes to `<out>/aodt`. |
| `--log-level` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` | `INFO` | Progress on stderr (job counters, TD3 step ETA). `DEBUG` adds PSO/SCA iteration lines. |
| `--quiet` | flag | off | Warnings and errors only. |
| `--log-file` | str | off | Also write the same log to a file. |
| `--td3-log-every` | int | `500` | TD3 training line every N steps (plus first and last). |

Config defaults are in `src/config.py` (`DEFAULT`, `PSO_*`, `TD3_TOTAL_STEPS`, `DEV_SCENARIO_SEEDS`, `PAPER_SCENARIO_SEEDS`).

## Commands

### Frozen-link check

```bash
python -m src.main
python -m src.main --mode single --seed 100
python -m src.main --mode single --compute --los-unit deg
```

### One solver, one scenario

```bash
python -m src.main --mode random --seed 100
python -m src.main --mode kmeans --seed 100
python -m src.main --mode pso --seed 100 --particles 20 --iters 100
python -m src.main --mode pso-joint --seed 100 --particles 20 --iters 100
python -m src.main --mode sca --seed 100
python -m src.main --mode sca-cvx --seed 100
python -m src.main --mode td3 --seed 100 --td3-steps 7000
```

With compute / AoDT:

```bash
python -m src.main --mode pso --compute --seed 100
python -m src.main --mode sca --compute --num-uav 4 --num-iot 16
```

### Method comparison

Dev (5 seeds), no TD3:

```bash
python -m src.main --mode compare --out results
```

Paper (20 seeds), with TD3 and compute:

```bash
python -m src.main --mode compare --paper-runs --with-td3 --compute --out results
```

Writes:

- `results/compare_raw.csv` or `compare_raw_20runs.csv`
- `results/positions_compare.csv` or `positions_compare_20runs.csv`
- `results/comparison_table.md` or `comparison_table_20runs.md`

### Paper-style sweeps

Placement-only PSO in sweeps. UAV/IoT sweeps always run. λ / AoDT / CPU sweeps run only with `--compute`.

```bash
python -m src.main --mode sweeps --out results
python -m src.main --mode sweeps --paper-runs --compute --out results
python -m src.main --mode sweeps --paper-runs --with-td3 --compute --particles 20 --iters 100 --td3-steps 7000 --out results
```

Sweep axes:

| Sweep | Values | Needs `--compute` |
|-------|--------|-------------------|
| `J` UAVs | `1, 2, 3, 4, 5` | no |
| `I` IoTs | `10, 16, 20, 24, 32` (J fixed at 3) | no |
| `λ` | `1.0 … 3.5` step `0.5` | yes |
| `T_k` | `0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0` | yes |
| UAV CPU | `1e8, 1.5e8, 2e8, 2.5e8` | yes |

Also writes default-scenario comparison, PSO convergence (`pso_convergence.csv` / `.png`), seed-100 placement plot, and `meta.json`.

### AoDT comparison

Always enables the compute model. Methods: random, k-means, **joint** PSO, SCA, TD3 (one policy trained on seeds `200…219`, greedy-eval on the listed seeds).

```bash
python -m src.main --mode aodt-compare --aodt-short --out results
python -m src.main --mode aodt-compare --out results
python -m src.main --mode aodt-compare --particles 20 --iters 100 --td3-steps 7000 --out results
```

`--aodt-short` → eval seeds `100…104`. Otherwise `100…119`.

Outputs under `results/aodt/` (tables, CSVs, position files, vs-UAV / vs-IoT plots, `meta.json`).

### Placement analysis script

Needs `results/aodt/positions_default.csv` from `aodt-compare`. No CLI flags.

```bash
python scripts/placement_analysis.py
```

Writes `results/aodt/placement_analysis.md`.

### AoDT parameter search (`S_i`, `L`)

Sweeps the paper-unspecified task size and CPU cycles against Fig. 9 (`T_k` from 0.8 s to 3 s). Calibrated radio is required (`B_sys = 8.8e6` Hz). SCA / K-means / Random first; TD3 only on the 20-seed shortlist.

```bash
python -m src.main --mode aodt-param-search --compute --out results/aodt_parameter_search
python -m src.main --mode aodt-param-search --skip-td3 --out results/aodt_parameter_search
```

`--aodt-search-stage {all,coarse,refine,shortlist,td3}` and `--resume` continue a partial run. Report: `results/aodt_parameter_search/REPORT.md`.

### Radio sensitivity

Sweeps sum rate vs `J` over any product of radio settings; this is how the
default profile was chosen.

```bash
python -m scripts.calibrate --methods random kmeans sca --js 1 3 5
python -m scripts.calibrate --b-sys 20000 --noise 1e-4 --scope system --cap -1
```

## Known limitations

- **`B_sys` is fitted, not Table II.** The `calibrated` profile's `8.8 MHz` is chosen so the `J` sweep spans the published 3-9 Mbps range. Sum rate is linear in `B_sys` only when QoS constraints are non-binding, so ranking is not guaranteed to stay the same if `B_sys` changes. Report absolute rates with the profile named (`table2` = literal Table II; `calibrated` is not a Table II constant).
- **The per-link bandwidth cap is a modelling addition.** Without it the sum-rate LP is solved at a single-link vertex and SCA's curve is flat in `J`.
- **Default SCA uses SciPy HiGHS.** `--mode sca-cvx` solves the same convexified LP with MATLAB CVX + MOSEK (the paper’s §V stack). Requires MATLAB, CVX on the MATLAB path (`cvx_setup`), and a MOSEK license. Optional env `UAV_MATLAB` points at `matlab.exe`.
- **TD3 reports greedy-policy evaluation after training**, not the best deployment visited during learning. `compare`, `sweeps`, and `aodt-compare` share that eval protocol (k-means + random restarts, noiseless rollouts). `TD3_R_MAX`, `TD3_EPISODE_LEN` and `TD3_ASSOC_ACTION_SCALE` are not paper values.

## Help

```bash
python -m src.main --help
```
