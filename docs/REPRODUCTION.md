# Reproduction notes — Khalaf et al. (IEEE TNSM, 2026)

This document is the inspection report and the parameter ledger for a
**fresh** implementation of the paper’s system model. The paper is the
source of truth. Numbers are **not** tuned to match Figs. 6–10.

Paper: Khalaf, Itani, Sharafeddine, *A UAV-Aided Digital Twin Framework
for IoT Networks With High Accuracy and Synchronization*, IEEE
Transactions on Network and Service Management, vol. 23, 2026.

---

## 0. Inspection report

### 0.1 Current project structure

This workspace is on branch `fresh`. The working tree originally
contained documentation and the PDF only (`docs/`, `_paper_extract/`).
There was **no** Python simulator on `fresh`.

An older reconstruction exists on other branches (`sco`,
`mathlabintegration`, …), roughly:

```text
src/config.py, comm.py, compute.py, aodt.py, scenario.py,
    evaluator.py, repair.py, main.py, status_sync.py
src/solvers/{random,kmeans,pso,sca,sca_cvx,td3,proposed}.py
src/experiments/{sweeps,aodt_compare,...}.py
matlab/convexified_lp*.m
tests/test_*.py
docs/{param,calibration,plan,PROJECT_STATUS_AND_PAPER_ANALYSIS}.md
```

That reconstruction is **not** copied forward. This tree rebuilds the
core model from the paper.

### 0.2 What could be reused (and what was)

| Piece | Verdict |
| --- | --- |
| NumPy + pytest layout | Reused as the language/tooling |
| Explicit process groups `N_k` | Same idea, rewritten |
| Shared evaluator for every method | Same idea, rewritten |
| `log10` for the dB path-loss terms | Kept: IEEE `20 log` in this model is dB |
| Equal-share `B_ij` as a *helper* when bandwidth is not optimized | Kept, labeled as an evaluation convention, not a paper policy |
| Lloyd k-means on IoT `(x, y)` | Same baseline concept, rewritten |
| Equations (1)–(9), (11), (13), (16), (17) as formulas | Re-derived from the PDF, not imported from old files |

### 0.3 What was discarded

Do **not** carry these forward from the old tree:

- Default area `500 × 500 m`
- Radio profile `calibrated` (`B_sys = 8.8 MHz` and `noise_power = σ` instead of `σ²`)
- Per-link cap `max_bw_share`
- Default LoS angle unit `deg` (Al-Hourani convention, not written in Eq. (4))
- Silent `S_bytes × 8` without labeling it as an interpretation
- Experimental `S_i`, `L` values presented near Table II
- Bandwidth LP / “repair” that is not in Problem (P)
- PSO, SCA surrogate, CVX/MOSEK wrapper, TD3 (later milestones)
- Any scale factor whose purpose is to hit the published Mbps figures

### 0.4 Equations implemented (core milestone)

See §2. SCA (Alg. 1) and TD3 (Alg. 2) are **out of scope** for this
milestone.

### 0.5–0.7 Parameters, unspecified values, ambiguities

See the tables in §1 and §3.

### 0.8 Proposed structure (this tree)

```text
src/uavdt/
    config.py          paper vs modification vs external
    models.py          IoT, UAV, process, allocation
    scenario.py        100×100 field, groups N_k
    channel.py         Eqs. (1)–(6)
    computation.py     Poisson + M/M/1, Eqs. (7)–(9)
    aodt.py            Eqs. (11), (13), (16), (17)
    resources.py       a_ij, b_ij, B_ij helpers
    constraints.py     Problem (P) checks
    evaluator.py       one evaluation of a deployment
    placement/         random (θ), k-means
    experiments/       CLI, multi-seed metrics
tests/                 unit tests for the 17 validation items
docs/REPRODUCTION.md   this file
```

SCA and TD3 directories are intentionally absent until the core is
validated.

### 0.9 Validation / test plan

Unit tests cover:

1. IoT positions in `[0, 100]²`, `z_i = 0`, explicit `N_1`, `N_2`
2. UAV positions `(x, y, H)`, `H = 100`
3. 3D distance Eq. (3)
4. LoS probability Eq. (4) (radians as written)
5. `J_FS`, LoS/NLoS, average path loss Eqs. (1), (2), (5)
6. Received power `p_i · 10^(-L_avg/10)`
7. SNR of Eq. (6) (`σ² = σ·σ`); no interference (orthogonal channels)
8. `B_sys` vs `B_ij`; sum constraint; unassociated `B_ij = 0`
9. Rate Eq. (6); rate ∝ `B_ij`; three bandwidth presets
10. Poisson arrivals at rate `λ_i` (sampler + mean rate)
11. `μ_j = f_j / L`
12. `λ_total,j`, `ρ_j`, stability `ρ_j < 1`
13. Upload `D_i` Eq. (11), including `T_u2u`
14. Queueing term inside Eq. (17) (not FCFS sojourn)
15. Process AoDT Eq. (17), **not** a per-IoT average
16. `AoDT_k ≤ T_k` and the rest of the checked constraints
17. Sum rate `∑_i ∑_j a_ij r_ij`

Seeds are explicit. A CLI path can average 20 runs later.

---

## 1. Parameter table

| Parameter | Value | Source |
| --- | ---: | --- |
| Area | 100 × 100 m | Intentional modification (paper: 500 × 500 m) |
| IoTs `I` | 10 | Paper §VII |
| Processes `K` | 2 | Paper §VII |
| IoTs/process | 5 | Paper §VII (`N_1`, `N_2`) |
| UAVs `J` | 3 | Paper main scenario |
| UAV altitude `H` | 100 m | Table II |
| `R_min` | 10,000 bit/s | Table II |
| UAV separation `θ` | 10 m | Table II |
| `T_u2u` | 0.3 s | Table II |
| `f_c` | 1×10^6 Hz | Table II |
| `λ_i` | 2 tasks/s | Table II (main scenario) |
| `f_j` | 2×10^8 cycles/s | Table II |
| `c` | 3×10^8 m/s | Table II |
| AoDT threshold `T_k` | 2.8 s | Table II |
| `η_LoS` | 1 | Table II |
| `η_NLoS` | 21 | Table II |
| `σ` | 0.01 | Table II (`10×10^{-3}`; see noise note) |
| Noise power in Eq. (6) | `σ² = 1×10^{-4}` | Equation uses `σ²`; not the table label alone |
| `p_i` | 0.2 W | Table II |
| `a` | 9.61 | Table II |
| `b` | 0.16 | Table II |
| Bandwidth `B_sys` | 20 kHz / 2.4 MHz / 8.8 MHz | Intentional modification (paper Table II: 20,000 Hz) |
| `S_i` (task size) | External (`task_size_bits`) | **Not specified** in Table II |
| `L` (cycles/task) | External (`task_cycles`) | **Not specified** in Table II |

Internal units are SI: m, Hz, W, bit/s, cycles/s, s. kHz/MHz/Mbps are
display only.

CLI bandwidth examples (Hz):

```text
--bandwidth 20000
--bandwidth 2400000
--bandwidth 8800000
```

or `--bandwidth-preset 20khz | 2.4mhz | 8.8mhz`.

---

## 2. Equations (as implemented)

Distance, Eq. (3):

\[
d_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2+H^2}.
\]

Free-space offset (text under Eqs. (1)–(2)), using **base-10** dB:

\[
J_{FS}=20\log_{10} f_c+20\log_{10}(4\pi/c).
\]

Path loss, Eqs. (1)–(2):

\[
L_{ij}^{\mathrm{LoS}}=J_{FS}+20\log_{10} d_{ij}+\eta_{\mathrm{LoS}},
\quad
L_{ij}^{\mathrm{NLoS}}=J_{FS}+20\log_{10} d_{ij}+\eta_{\mathrm{NLoS}}.
\]

LoS probability, Eq. (4), with \(\theta=\arcsin(H/d_{ij})\) in **radians**
as written:

\[
P_{ij}^{\mathrm{LoS}}=\frac{1}{1+a\exp\bigl(-b(\theta-a)\bigr)}.
\]

Average path loss, Eq. (5):

\[
L_{ij}^{\mathrm{avg}}=P_{ij}^{\mathrm{LoS}}L_{ij}^{\mathrm{LoS}}
+(1-P_{ij}^{\mathrm{LoS}})L_{ij}^{\mathrm{NLoS}}.
\]

Uplink rate, Eq. (6):

\[
r_{ij}=B_{ij}\log_2\Bigl(1+p_i\cdot 10^{-L_{ij}^{\mathrm{avg}}/10}\cdot\frac{1}{\sigma^2}\Bigr).
\]

Service rate and load, Eqs. (7)–(9):

\[
\mu_j=f_j/L,
\quad
\lambda_{\mathrm{total},j}=\sum_{i:\,b_{ij}=1}\lambda_i,
\quad
\rho_{\mathrm{total},j}=\lambda_{\mathrm{total},j}/\mu_j.
\]

Upload time, Eq. (11) (`S_i` in bits so that `S_i/r_{ij}` is seconds):

\[
D_i=\begin{cases}
S_i/r_{ij} & \text{processed at associated UAV}\\
S_i/r_{ij}+T_{\mathrm{u2u}} & \text{otherwise}.
\end{cases}
\]

Process rate and AoDT, Eqs. (13), (16), (17):

\[
\lambda_{N_k}=\min_{i\in N_k}\lambda_i,
\quad
D_{N_k}=\max_{i\in N_k}D_i,
\]

\[
\Delta_{\mathrm{DT}_k}
=\max_{i\in N_k}D_i
+\frac{1}{\lambda_{N_k}}\left(1+\frac{\sum_{i\in N_k}\lambda_i}{\mu}\right).
\]

Objective, Eq. (20):

\[
R_{\mathrm{sum}}=\sum_{i=1}^{I}\sum_{j=1}^{J}a_{ij}r_{ij}.
\]

---

## 3. Not specified by the paper / ambiguities

| Item | Handling |
| --- | --- |
| `S_i` (labeled bytes in the text; no Table II value) | Config `task_size_bits`. Eq. (11) is implemented as `D=S/r` with `r` in bit/s. A `--task-size-bytes` flag multiplies by 8 and **labels that conversion as an interpretation**, not as a paper equation. |
| `L` (cycles/task) | Config `task_cycles`. Required for `μ_j`. |
| Base of `20 log` in Eqs. (1)–(2) | Implemented as \(\log_{10}\) (dB). |
| Units of \(\arcsin(H/d)\) in Eq. (4) | Default **radians** (as written). `los_angle_unit="deg"` is an optional Al-Hourani-style reading, not the default. |
| Table II “Noise power, σ = 10×10^{-3} W” vs Eq. (6) `σ²` | Config stores `sigma=0.01`. Eq. (6) uses `noise_power = sigma**2`. |
| How `B_ij` is chosen (only `∑ B_ij ≤ B_sys` is written) | `B_ij` is an explicit matrix. Placement-only runs may use **equal split** among associated links; that is not claimed as the paper’s optimizer. |
| Default `a_ij` / `b_ij` when only positions are given | Nearest-UAV association; one processing UAV per process (constraint (23)). Evaluation convention. |
| `μ` in Eq. (17) has no UAV index | Use `μ_j` of the unique processing UAV of process `k` (constraint (23)). |
| Eq. (17) uses `∑_{i∈N_k} λ_i`, while constraint (24) uses all tasks on UAV `j` | Both implemented as written. |
| Eqs. (14)–(15) vs (17) (min-rate vs total-rate in the AoI term) | **(17) is the stated final form** and is what we compute. |
| Base-station coordinates | Unused: download time neglected in the paper. |
| TD3 hyperparameters (`γ`, `τ`, `d`, network sizes, …) | Not in Table II. Not implemented in this milestone. |
| Random seeds for the 20 runs | Not specified. CLI accepts `--seed` / `--n-runs`. |

Execution defaults for `task_size_bits` and `task_cycles` exist only so
the CLI can run. They are **not** paper values.

---

## 4. Intentional modifications (this reproduction)

1. Field is **100 × 100 m**, not 500 × 500 m.
2. `B_sys` is an experiment parameter: 20 kHz, 2.4 MHz, or 8.8 MHz.

No other communication parameters are replaced to chase figure Mbps.

---

## 5. How to run

```text
pip install -r requirements.txt
python -m pytest
python -m uavdt evaluate --seed 1 --bandwidth 20000 --placement random
python -m uavdt evaluate --bandwidth-preset 2.4mhz --placement kmeans
python -m uavdt evaluate --bandwidth-preset 8.8mhz --seed 1
python -m uavdt multi-seed --n-runs 5 --bandwidth 20000 --placement random
```

From the repo root, `src/` must be on `PYTHONPATH` (pytest.ini sets this
for tests). Example:

```text
$env:PYTHONPATH="src"
python -m uavdt evaluate --seed 1 --bandwidth 20000
```
