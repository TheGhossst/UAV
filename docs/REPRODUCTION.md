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
- Unlabeled per-link cap treated as part of Problem (P) (15% sensitivity /
  25% primary exist here only as **EXTERNAL PARAMETER**, see §4.1 / §6.7)
- Default LoS angle unit `deg` (Al-Hourani convention, not written in Eq. (4))
- Silent `S_bytes × 8` without labeling it as an interpretation
- Experimental `S_i`, `L` values presented near Table II
- Bandwidth LP / “repair” that is not in Problem (P)
- Treating the paper’s 7–14 Mbps figures at 20 kHz as a calibration
  target (this reproduction does not chase those Mbps)
- PSO, SCA surrogate, CVX/MOSEK wrapper, TD3 (later milestones)
- Any scale factor whose purpose is to hit the published Mbps figures

### 0.4 Equations implemented (core milestone)

See §2. **SCA (Algorithm 1)** is implemented in `src/uavdt/sca/` (CVXPY
default; MATLAB CVX+MOSEK spot-validated). **TD3 (Algorithm 2)** remains
**out of scope** for this milestone.

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
    aodt.py            Eqs. (10)–(17) closed forms; Problem (P) uses (17)
    aodt_sim.py        FCFS, FCFS-P, LCFS-S event queues
    resources.py       a_ij, b_ij, B_ij helpers
    constraints.py     Problem (P) checks
    evaluator.py       one evaluation of a deployment
    placement/         random (θ), k-means, PSO (external)
    sca/               Algorithm 1 SCA (CVXPY + MATLAB bridge)
    experiments/       CLI, campaigns, Fig. 11, spot-validate
tests/                 unit tests for the validation items in §0.9
docs/REPRODUCTION.md   this file
```

TD3 is intentionally absent. SCA was added after the core model passed
the validation plan in §0.9.

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
9. Rate Eq. (6); rate ∝ `B_ij`; three bandwidth presets; Eq. (6)+(27) sum-rate ceiling
10. Poisson arrivals at rate `λ_i` (sampler + mean rate)
11. `μ_j = f_j / L`
12. `λ_total,j`, `ρ_j`, stability `ρ_j < 1`
13. Upload `D_i` Eq. (11), including `T_u2u`
14. Queueing term inside Eq. (17) (not FCFS sojourn); Eq. (14)–(15) LCFS-S forms
15. Process AoDT Eq. (17), **not** a per-IoT average
16. `AoDT_k ≤ T_k` and the rest of the checked constraints
17. Sum rate `∑_i ∑_j a_ij r_ij`
18. Instantaneous age Eq. (10) and neglected download Eq. (12) (`Z=0`)
19. FCFS / FCFS-P / LCFS-S event queues vs closed forms; Fig. 11 λ patterns

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
| Bandwidth `B_sys` | 20 kHz / 2.4 MHz / 8.8 MHz | See §4.1: Table II 20 kHz as the (27) cap is **infeasible** (0.997 Mbps model-free ceiling); headline runs are 2.4 MHz and 8.8 MHz |
| Per-link cap `max_bw_share` | `None`, `0.25` (primary), or `0.15` (sensitivity) | **EXTERNAL PARAMETER**, not Problem (P) / Table II |
| `S_i` (task size) | 12,000 bytes (`96,000` bit) | **EXTERNAL** — settled experimental choice |
| `L` (cycles/task) | `3.75×10⁶` | **EXTERNAL** — settled experimental choice (`μ ≈ 53.3` /s) |

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
| Table II “Minimum bandwidth allocation, `B_sys` = 20{,}000 Hz” vs constraint (27) | **(27) is a sum ceiling** (`∑ B_{ij} ≤ B_sys`); the table adjective is “Minimum.” Same symbol, no second bandwidth number, no min-bandwidth constraint in Problem (P). Simulator uses 20 kHz as the (27) cap (Reading A). The alternative is that (27)’s cap is **undisclosed** (Reading B). See §4.1. |
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

Paper methodology and Table II are used **except** the items below.
Numerical figures in the paper’s plots are **not** a target (advisor
instruction: ignore the published Mbps and re-evaluate the written
model).

1. Field is **100 × 100 m**, not 500 × 500 m.
2. `B_sys` is an experiment parameter: 20 kHz (Table II diagnostic),
   2.4 MHz, or 8.8 MHz. Each of 2.4 MHz and 8.8 MHz is run **with no
   per-link cap**. Primary 8.8 MHz experiments use a **25% per-link cap**
   (leftover-dump stress test); a **15% cap** is kept as a tighter-cap
   sensitivity.
3. `S_i` (`task_size_bits`) and `L` (`task_cycles`) are **EXTERNAL**
   execution defaults; Table II does not specify them.

No other communication parameters are replaced to chase figure Mbps.

### 4.1 Paper 20 kHz / 7–14 Mbps vs this model — documented substitution

**PAPER:** Table II lists `B_sys = 20,000 Hz` under the row text
“Minimum bandwidth allocation.” Constraint (27) is
`∑_{i,j} B_{ij} ≤ B_{\mathrm{sys}}`, explained as a cap on total uplink
bandwidth. Figs. 6–10 report sum-rate on the order of **7–14 Mbps**.

**Model-free (lead with this).** Eq. (6) + (27) imply
`R_{\mathrm{sum}} \le B_{\mathrm{sys}}\log_2(1+\mathrm{SNR}_{\max})`
for any path loss, power, or noise. With `B_{\mathrm{sys}} = 20` kHz as
that cap, `SNR_{\max} = 10^{15}` still gives only **0.997 Mbps**. The
7–14 Mbps plots are about **7–14×** that fantasy ceiling. The bound does
not depend on `a`, `b`, `η`, `σ`, area, or this simulator.

**Table II vs (27).** The table says “Minimum”; (27) is a maximum. We
treat 20 kHz as the (27) cap (same symbol, only bandwidth number, no
floor constraint in Problem (P)). Figs. 6–10 **rule out Reading A**:
7–14 Mbps and Fig. 7’s growth with `I` cannot occur under a 20 kHz
sum cap (ceiling 0.997 Mbps at `SNR=10^{15}`). If the table English is
instead a per-link floor (Reading B), Fig. 7’s slope is the right
*shape*, but matching Fig. 6’s 8.8 Mbps at `I=10` on the written
channel needs **~862 kHz/link (43×** the stated 20 kHz), and Fig. 7’s
14 Mbps at `I=32` needs **~428 kHz/link (21×)**. The number (27)
actually used is then **not in the paper**, and it is not a rounding
of 20{,}000 Hz. Details: `docs/RESULTS.md` §0.2.

**This reproduction (DERIVED from Eqs. (1)–(6), (25), (31)):**

- Written-channel `SNR_{\max}\approx 1.03` tightens the ceiling to
  **~0.020 Mbps**. 20 kHz campaigns sit on that line.
- Under this reproduction’s QoS/AoDT floors, a typical 100 × 100 m
  start needs on the order of **~102 kHz** of total bandwidth
  (`∑_i R_min / SE_ij`). **20 kHz is infeasible.**
- The same 20 kHz check at the paper’s **500 × 500 m** field is also
  **0% feasible**. Realized rates drop (~0.020 → ~0.013 Mbps) because
  the infeasible QoS LP falls back to equal-share `B_sys/I`, so
  `R_sum = B_sys · mean_SE`; max SNR (best-link dump ~0.020 Mbps) is
  unchanged. Area is not a confounder (`scripts/check_bsys_20khz.py`).
- **Headline results are therefore reported at 2.4 MHz and 8.8 MHz**,
  with `max_bw_share = None` (no per-link cap), **`0.25` (8.8 MHz
  primary leftover-dump stress test)**, and `0.15` (tighter-cap
  sensitivity). The 20 kHz preset
  remains in the CLI only as a Table II diagnostic. This substitution
  is **explicit**, not implied.

**IMPLEMENTATION CHOICE / EXTERNAL:** the per-link cap
(`B_{ij} \le \texttt{max\_bw\_share}\, B_{\mathrm{sys}}`) is **not**
part of Problem (P). Primary experiments use **25%** (2.20 MHz/link) as
a leftover-dump stress test: without a per-link cap the frozen-q LP puts
~95% of `B_sys` on the best-SE link. A **15%** cap (1.32 MHz/link) remains
available as a tighter-cap sensitivity; it was not chosen independently of
a cap×J search and is not the primary ranking config.

### 4.2 Cap vs uncapped arithmetic

When comparing a no-cap run to a capped run at the **same** `B_sys`:

```text
gap_vs_uncapped     = uncapped_rate - capped_rate
gap_vs_uncapped_pct = 100 * gap_vs_uncapped / uncapped_rate
```

Both rates must already be in the **same** unit (bit/s or Mbps). Do
**not** introduce a factor of 10. Helper: `uavdt.sca.gap_vs_uncapped`.
Sweep JSON (`scripts/run_sca_bw_matrix.py`) writes `gaps_vs_uncapped`.
No stored sweep JSONs are in-tree at the time of this note; recompute
from `results/` when a campaign is run.

---

## 5. How to run

```text
pip install -r requirements.txt
python -m pytest
python -m uavdt evaluate --seed 1 --bandwidth 20000 --placement random
python -m uavdt evaluate --seed 1 --bandwidth 20000 --area-m 500 --placement kmeans
python scripts/check_bsys_20khz.py
python -m uavdt evaluate --bandwidth-preset 2.4mhz --placement kmeans
python -m uavdt evaluate --bandwidth-preset 8.8mhz --seed 1
python -m uavdt multi-seed --n-runs 5 --bandwidth 20000 --placement random
python -m uavdt sca --seed 1 --bandwidth-preset 2.4mhz
python -m uavdt sca --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25
python -m uavdt sca-seq-debug --seed 1 --bandwidth-preset 2.4mhz
```

From the repo root, `src/` must be on `PYTHONPATH` (pytest.ini sets this
for tests). Example:

```text
$env:PYTHONPATH="src"
python -m uavdt evaluate --seed 1 --bandwidth 20000
```

---

## 6. Algorithm 1 SCA of Problem (P)

**PAPER:** Algorithm 1 is successive convex approximation of Problem (P).
Each iteration solves **one convexified (P)** for UAV positions **and**
bandwidth. Section V linearizes (25) and the channel maps (3)–(5) with
first-order Taylor expansions. MATLAB CVX + MOSEK is the stated solver.
The PDF does not print the algebra of the convexified program; the map
below is the reconstruction used here.

Labels used below: **PAPER** / **DERIVED** / **IMPLEMENTATION CHOICE** /
**PAPER CORRECTION** / **EXTERNAL PARAMETER**.

**PAPER CORRECTION:** Algorithm 1 prints
`while |ObjP(n)-ObjP(n-1)| <= epsilon` (the usual **stop** test written
as a loop condition). Stop when the true objective change after an
accepted update is \(\le\varepsilon\).

**Binaries — IMPLEMENTATION CHOICE, not a paper requirement.**
Algorithm 1’s update list is UAV positions, bandwidth, and the
sum-rate matrix. This implementation **freezes** \(a_{ij}\) and
\(b_{ij}\) after initialization. That is **not** required by the paper.

### 6.1 Original Problem (P) — PAPER

Objective (20): maximize \(\sum_i\sum_j a_{ij} r_{ij}\).

| Constraint | Content | In this solver |
| --- | --- | --- |
| (21) | \(\sum_j a_{ij}=1\) | Held by fixing \(a_{ij}\) |
| (22) | one processing UAV per IoT | Held by fixing \(b_{ij}\) |
| (23) | same process → same processing UAV | Held by process-consistent \(b\) |
| (24) | \(\mu_j \ge \lambda_{\mathrm{total},j}\) | Checked at init; constant if \(b\) fixed |
| (25) | \(r_{ij}\ge a_{ij} R_{\min}\) | Exact in \(B\) at frozen \(q\): \(B_{ij}\ge R_{\min}/\mathrm{SE}_{ij}(q)\) |
| (26) | \(B_{ij}\le a_{ij} M\) | \(B_{ij}=0\) if \(a_{ij}=0\); **DERIVED** \(M=B_{\mathrm{sys}}\) under (27). Optional primary 25% cap sets \(M=0.25 B_{\mathrm{sys}}\) (**EXTERNAL PARAMETER**, not (P)); 15% is a tighter-cap sensitivity. |
| (27) | \(\sum_{i,j} B_{ij}\le B_{\mathrm{sys}}\) | Exact linear |
| (28) | \(\|q_j-q_l\|\ge\theta\) | True evaluator gate (not a supporting halfspace in the loop) |
| (29) | \(\lambda_{N_k}\le\lambda_i\) | Identity if \(\lambda_{N_k}=\min\lambda_i\) |
| (30)–(31) | process AoDT \(\le T_k\) | Exact convex constraint in \(B\) at frozen \(q\) (see 6.3) |

Altitude \(H=100\) is fixed. Box: \(x_j,y_j\in[0,100]\).

### 6.2 Variables — IMPLEMENTATION CHOICE

Problem (P) is a MINLP. **This implementation freezes \(a_{ij}\) and
\(b_{ij}\) after initialization** (nearest UAV, then majority-of-association
processing). That is **not** a paper requirement. Constraint (23)
remains satisfied. Discrete association/processing are not solved here.
If the majority vote would violate queue stability (24) and \(J\ge K\),
initialization enumerates process→UAV maps (capped; campaign \(K=2\) is
tiny) and keeps a feasible one closest to the vote; \(b_{ij}\) is still
frozen after that. Nearest \(a_{ij}\) is not rematched, so a process can
still be split across UAVs and incur \(T_{\mathrm{u2u}}\) on some IoTs.

### 6.3 Sequential phases — IMPLEMENTATION CHOICE

Two convex LPs are solved in sequence. They are **not** a joint
\((B,q)\) Taylor program and **not** unpublished paper algebra.

**Position LP (bandwidth frozen).** Maximize the first-order map of
\(\sum_i a_{ij} B_{ij}^{(n)}\mathrm{SE}_{ij}(q_j)\) using the numerical
Jacobian of the core channel. Constraints: field box,
\(\|q_j-q_j^{(n)}\|_\infty\le\) `step_size` (default **1 m**), and the
supporting halfspace for (28),
\(u_{jl}\cdot(q_j-q_l)\ge\theta\). This LP is **not** a QoS certificate.

The CVXPY fallback used by unit tests takes the same Jacobian as a
normalized gradient step instead of calling MOSEK for positions.

**True feasibility gate.** A candidate \((q,B)\) is accepted only if it
is feasible on the MATLAB-side copy of Eqs. (1)–(6)/(17) **and** the
true sum rate improves by `improvement_tolerance`. After MATLAB returns,
Python `evaluate()` is the published score and must also be clean.
Rejected candidates never become the linearization point.

**Bandwidth LP (geometry frozen).** \(\mathrm{SE}_{ij}(q)\) comes from
Eqs. (1)–(6). Then \(r_{ij}=B_{ij}\mathrm{SE}_{ij}(q)\) is exactly
linear in \(B_{ij}\). Maximize \(\sum a_{ij}\mathrm{SE}_{ij}B_{ij}\)
subject to (26)–(27) and the exact floors below.

**AoDT / QoS at fixed \(q\).** With \(a,b\) fixed, \(Q_k\) is constant.
\(S_i/(B_{ij}\mathrm{SE}_{ij})\le\mathrm{slack}_i\) is **exactly**

\[
B_{ij}\ge\max\Bigl(\frac{R_{\min}}{\mathrm{SE}_{ij}(q)},\;
\frac{S_i}{\mathrm{SE}_{ij}(q)\,\mathrm{slack}_i}\Bigr),
\]

where \(\mathrm{slack}_i=T_k-Q_k-\mathbf{1}_{\mathrm{fwd}}T_{\mathrm{u2u}}\).
That is an LP, not `inv_pos` and not a joint Taylor of \((B,q)\).
Process-level \(\max_i D_i\) is enforced by applying the floor to every
member of \(N_k\).

Leftover spectrum on the largest-\(\mathrm{SE}\) associated link is an
expected vertex of a linear sum-rate objective, not a modelling bug.

### 6.4 Initialization — IMPLEMENTATION CHOICE

Algorithm 1 does not specify the start. We use k-means UAV positions
(paper’s named initialization baseline, §VII) with the existing
\(\theta\) repair, nearest-UAV \(a_{ij}\), process-consistent \(b_{ij}\),
then the **exact** fixed-\(q\) bandwidth LP above.

If (24) already fails at this \(b\), the solver raises: positions and
bandwidth cannot repair CPU load when processing is frozen.

### 6.5 Stopping — PAPER CORRECTION / IMPLEMENTATION CHOICE

Algorithm 1 prints `while |ObjP(n)-ObjP(n-1)| <= epsilon` (the usual
**stop** test written as a loop condition). **PAPER CORRECTION:** stop
when the **true** objective change after an accepted step is
\(\le\varepsilon\).

Also stop with **`STEP_SIZE_LIMIT`** if `accepted_steps == 0` at any
non-`MAX_ITERATIONS` stop (including a stationary convex step, and
including `step_size` still above `min_step_size`). **`CONVERGED`**
requires `accepted_steps >= 1`. **`MAX_ITERATIONS`** only if the
iteration cap is hit with remaining `step_size` still above
`min_step_size`. A MOSEK `Solved` subproblem is **not** by itself
convergence.

Python `classify_stop_reason` and MATLAB `classify_stop_reason` in
`matlab/sca_seq.m` implement **the same** rule.

`improvement_tolerance` (default 1 bit/s) ignores solver jitter.

Returned point: last **accepted** true-feasible iterate (the incumbent).

\(\varepsilon\), `step_size`, and `max_iterations` are algorithm knobs,
not Table II.

### 6.6 Solver

Default CLI engine is **MATLAB CVX + MOSEK** in **one** session
(`matlab/bandwidth_lp.m`, `matlab/position_step_lp.m`,
`matlab/sca_seq.m`). Physics stay in the Python core; MATLAB solves the
two LPs. `channel_se.m` must match `uavdt.channel` (checked as
`se_max_abs_diff`). Python `evaluate()` remains the published score.

`--solver cvxpy` uses a CVXPY HiGHS/CLARABEL bandwidth LP and a
finite-difference position step (this is what `pytest` runs).

```text
python -m uavdt sca --seed 1 --bandwidth-preset 2.4mhz --solver matlab
python -m uavdt sca-seq-debug --seed 1 --bandwidth-preset 2.4mhz --solver matlab
python -m uavdt sca --seed 1 --bandwidth-preset 2.4mhz --solver cvxpy
```

Debug log writes `results/sca_seq_debug.json` and `.csv`.

Interface: `solve_sca(scenario, seed) -> SCAResult`.

### 6.7 External parameters

These are **EXTERNAL PARAMETER**, not paper Table II and not part of
Problem (P) as written:

| Quantity | Config | Role |
| --- | --- | --- |
| Task size `S_i` | `task_size_bits` | Upload delay Eq. (11) |
| Cycles/task `L` | `task_cycles` | `μ_j = f_j / L` |
| Per-link bandwidth share | `max_bw_share` (e.g. `0.25` primary, `0.15` sensitivity) | Optional `B_ij ≤ share · B_sys`. Default `None` is paper (26)–(27) only. |

### 6.8 Known gaps vs the paper

- The explicit convexified (P) is not in the PDF; this sequential split
  is an implementation choice, not unpublished paper text.
- Association/processing are not re-optimized (**IMPLEMENTATION CHOICE**,
  not a paper requirement).
- Paper Table II 20 kHz, read as the (27) cap, cannot produce the
  published 7–14 Mbps (model-free ceiling 0.997 Mbps at `SNR=10^{15}`);
  headline `B_sys` is 2.4 MHz / 8.8 MHz (§4.1).
- SCA solver is **frozen**. Characterization vs `J`, `I`, `λ`, `T_k`,
  CPU, and Random/K-means/PSO lives in `docs/EXPERIMENTS.md`.
- Fig. 11 heterogeneous-λ AoDT check: `python -m uavdt fig11`.
- TD3 (Algorithm 2) is out of scope.
