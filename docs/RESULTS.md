# Experimental Results

**Khalaf et al. (IEEE TNSM, 2026) — `uavdt` reproduction**


|                            |                                                                                                    |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| **Scope**                  | Our experimental outcomes on the fresh `uavdt` simulator                                           |
| **Not in scope**           | Numeric comparison of Mbps figures to the paper’s §VII plots                                       |
| **Last full regeneration** | 2026-09-08 (full pipeline: pytest 125/125, primary + cap15 + 500 m campaigns, SCA-joint + tk08 follow-ups, fig11, spot-validate, analysis/plots) |
| **Primary campaign**       | `results/campaign_8.8mhz_cap25_si12k.json` (100 m, 25% per-link cap)           |
| **Tighter-cap sensitivity**| `results/campaign_8.8mhz_cap15_n20.json` (100 m, 15% per-link cap)             |
| **Paper-field test**       | `results/campaign_8.8mhz_cap25_si12k_500m.json` (500 m, 25% cap)               |
| **Related docs**           | `[EXPERIMENTS.md](EXPERIMENTS.md)` · `[REPRODUCTION.md](REPRODUCTION.md)` · `[param.md](param.md)` |


---

## Table of contents

1. [Eq. (6) + constraint (27): the 20 kHz ceiling](#0-eq-6--constraint-27-the-20-khz-ceiling)
2. [Audit summary](#audit-summary-paper-vii-vs-this-implementation)
3. [Campaign inventory](#campaign-inventory)
4. [Default scenario](#default-scenario-headline-point)
5. [Bandwidth sweep](#1-bandwidth-sweep-qualitative)
6. [Sweep axes (8.8 MHz, 15% sensitivity)](#2-sweep-axes-at-88-mhz-15-cap)
7. [Paired statistics](#3-paired-statistics-sca-vs-baselines-same-seed)
8. [Solver validation](#4-solver-validation)
9. [Sensibility checklist](#5-sensibility-checklist)
10. [Known limitations](#6-known-limitations-this-milestone)
11. [Paper field 500 m](#26-paper-field-500--500-m-88-mhz-25-cap)
12. [Primary campaign 25%](#27-primary-campaign-88-mhz-25-cap)
13. [SCA-joint probe](#28-sca-joint-methodology-probe)
14. [Suggested writeup sentences](#7-suggested-writeup-sentences-copy-ready)
15. [AoDT extras & Fig. 11](#8-aodt-extras-eqs-1017-fcfs--fcfs-p--lcfs-s-fig-11)
16. [How to regenerate](#9-how-to-regenerate)
17. [S_i / L correction](#10-s_i--l-correction--settled-defaults-vs-old-placeholders)

---

## At a glance

> **Lead finding.** Eq. (6) plus constraint (27) give, for any powers, path losses, or noise figures,
>
>
> \sum_{i,j} B_{ij}\log_2(1+\mathrm{SNR}*{ij})
> \le B*{\mathrm{sys}}\log_2(1+\mathrm{SNR}_{\max}).
>
>
> If Table II’s `B_sys = 20{,}000` Hz is the sum cap in (27), then even `SNR_max = 10^{15}` (essentially noiseless, `\log_2(1+\mathrm{SNR})\approx 49.83`) caps the whole system at **0.997 Mbps**, not 7–14 Mbps. That ceiling does not depend on `a`, `b`, `\eta_{\mathrm{LoS/NLoS}}`, `\sigma`, field size, or this simulator. See [§0](#0-eq-6--constraint-27-the-20-khz-ceiling).

### Experimental setup


| Parameter       | Value                                                                                                                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Area**        | **100 × 100 m** (headline); **500 × 500 m** field test at 25% cap ([§2.6](#26-paper-field-500--500-m-88-mhz-25-cap)); 20 kHz check at both fields ([§0.3](#03-control-paper-500--500-m-field-still-20-khz)) |
| **Seeds**       | 20 consecutive seeds per sweep point (`seed_start = 1`)                                                                                                                                                   |
| **Methods**     | SCA, random, k-means, PSO *(PSO is external, not in the paper)*                                                                                                                                           |
| **Score**       | Python `evaluate()` on every method; placement baselines re-scored with the same frozen-geometry bandwidth LP as SCA’s B step                                                                             |
| **SCA backend** | CVXPY for campaigns; MATLAB CVX+MOSEK spot-validated                                                                                                                                                      |


### Headline result @ J = 3 (8.8 MHz, 25% primary cap)


| Field / role                              | Cap     | SCA       | Random | K-means | PSO   | Feasible | Spread    |
| ----------------------------------------- | ------- | --------- | ------ | ------- | ----- | -------- | --------- |
| **100 × 100 m** (primary)                 | **25%** | **8.946** | 8.928  | 8.901   | 8.905 | 100%     | **0.046** |
| 100 × 100 m (tighter-cap sensitivity)     | 15%     | 8.895     | 8.765  | 8.838   | 8.854 | 100%     | 0.129     |
| 500 × 500 m (field test, same 25% cap)    | 25%     | 8.300     | 7.947  | 7.474   | 8.122 | 100%     | 0.826     |


**25% is the primary cap:** it is the leftover-dump stress test adopted independently of the SCA-vs-random p-value. At default J = 3, SCA vs random is **not** significant (p=0.123). A **15%** cap remains as a tighter-cap sensitivity (search-selected; SCA vs random p&lt;0.001 at J=3, but the gap is not practically large at J=4–5). See [§2.7](#27-primary-campaign-88-mhz-25-cap) and [§2](#2-sweep-axes-at-88-mhz-15-cap).

The 500 m campaign uses the **same 25% primary cap**. See [§2.6](#26-paper-field-500--500-m-88-mhz-25-cap).

### Last regeneration

**2026-09-08 (full pipeline)** — Re-ran the complete regeneration ledger ([§9](#9-how-to-regenerate)) on current code:

- `pytest` **125/125** (includes `test_sca_joint.py` and other SCA-joint coverage)
- `check_bsys_20khz.py` — 0% feasible at 100 m and 500 m (unchanged)
- Primary `campaign_8.8mhz_cap25_si12k.json` + init-repair + paired winrate/losses
- Cap15 `campaign_8.8mhz_cap15_n20.json` + fig11 + paired winrate
- 500 m `campaign_8.8mhz_cap25_si12k_500m.json` + figures
- `fig11_8.8mhz_cap25_si12k.json` — **50/50** sensibility checks pass
- `spot-validate` — CVXPY vs MATLAB `agreement: ok` (no cap + 25% cap)
- SCA-joint probe + tk08 follow-ups (cohesive **20/20**, construction **40/40**, tradeoff gap mean **0.586** Mbps)
- `analyze_campaigns.py` + `plot_paper_figures.py` (primary, cap15, 500 m)
- Bandwidth preset sweep (`run_all_bandwidth_campaigns.ps1`) — Sep 7 artifacts retained; numbers match fresh primary run

One-command replay: `scripts/run_full_regeneration.ps1` (adds bandwidth sweep if uncommented).

**2026-09-08 (earlier)** — T_k=0.8 s audit trail closed: hand construction → best-SE joint null → full-sweep null → process-cohesive candidate **20/20** → construction **40/40** → sync tradeoff gap **0.59 Mbps** mean (median 0.57, 40 geometries). See [§2.8](#28-sca-joint-methodology-probe).

**2026-09-07** — restore 25% as primary; keep 15% as tighter-cap sensitivity; 500 m field test. Mechanism diagnostics: `cap_binding_diagnostic.json`, `se_spread_by_j.json`.

**2026-09-05** — first 100 m 25% campaign and analysis:

- `pytest` **108/108** *(superseded by 125/125 after SCA-joint tests)*
- Primary campaign `campaign_8.8mhz_cap25_si12k.json` (all axes, 20 seeds, CVXPY)
- `fig11_8.8mhz_cap25_si12k.json`; spot-validate (8.8 MHz no cap + cap 25%)
- Full bandwidth preset sweep: `scripts/run_all_bandwidth_campaigns.ps1`

---

## 0. Eq. (6) + constraint (27): the 20 kHz ceiling

### 0.1 Model-free bound *(lead with this)*

Constraint (27) is `\sum_{i,j} B_{ij} \le B_{\mathrm{sys}}`. Eq. (6) is `r_{ij} = B_{ij}\log_2(1+\mathrm{SNR}_{ij})`. The objective (20) is the sum of associated rates, which cannot exceed the sum of all `r_{ij}`. Therefore

```text
R_sum  ≤  Σ B_ij · log2(1+SNR_ij)  ≤  B_sys · log2(1+SNR_max)
```

for **any** choice of `p_i`, path loss, and noise. With `B_sys = 20{,}000` Hz:


| SNR_max                    | log₂(1+SNR) | Ceiling        |
| -------------------------- | ----------- | -------------- |
| 1                          | 1.00        | 0.020 Mbps     |
| 10³                        | 9.97        | 0.199 Mbps     |
| 10⁶                        | 19.93       | 0.399 Mbps     |
| 10¹⁵ *(absurdly generous)* | 49.83       | **0.997 Mbps** |


Figs. 6–10 sit at **7–14 Mbps** — seven to fourteen times above even the noiseless fantasy.


| Helper           | Command                                    |
| ---------------- | ------------------------------------------ |
| Ceiling function | `uavdt.channel.sum_rate_ceiling_bit_per_s` |
| Re-run check     | `python scripts/check_bsys_20khz.py`       |


### 0.2 Table II label vs constraint (27) — closed, as a fork

Table II’s row text is literally **“Minimum bandwidth allocation, `B_sys`, 20{,}000 Hz”**, in parallel with the previous row (“Minimum data rate, `R_min`”). Constraint (27) and the paragraph that explains it are a **ceiling on total uplink bandwidth**: *“the total uplink bandwidth of the system does not exceed the available system bandwidth of value `B_sys`.”*

Problem (P) contains **no** constraint `B_{ij} \ge B_{\mathrm{sys}}` or `\sum B_{ij} \ge B_{\mathrm{sys}}`. The only occurrence of the symbol `B_sys` in the program is the sum cap (27). The paper never states a second bandwidth number.


| Reading                                              | What 20 kHz is                                       | What Figs. 6–10 do to it                                                                                                                                                                                                 |
| ---------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **A — mathematics of (27) governs** *(adopted here)* | Sum cap `\sum B_{ij} \le 20` kHz                     | **Ruled out.** Y-axis is 7–14 Mbps; §0.1 caps any SNR at 0.997 Mbps. Fig. 7 also *grows* with `I` (SCA “exceeding approximately 14 Mbps for 32 devices”), which a 20 kHz shared pool cannot do.                          |
| **B — table English governs**                        | A per-link or per-device **floor**, not the (27) cap | **Not ruled out by shape.** Linear-in-`I` growth is what a per-link allocation looks like. The (27) cap is then **undisclosed**. On the *written* radio the stated 20 kHz floor is tens of times too small (next table). |


**Why we adopt Reading A:**

1. Table II is the parameter table for the symbols in Problem (P).
2. Those symbols appear in (27) only as a sum ceiling.
3. The explanatory sentence of (27) says “available system bandwidth.”
4. “Minimum” on that row is the same adjective as `R_min`, which **is** a floor (constraint 25) — a table-editing collision, not a second constraint.

Reading B is named so a critic cannot say the table English was ignored.

#### Reading B magnitude *(not a rounding error)*

Invert Eq. (6) under an equal per-link split, `B_i = R / (I \log_2(1+\mathrm{SNR}))`, using the control’s observed `\mathrm{SNR}_{\max}\approx 1.03` (best-case: the *smallest* `B_i` that can hit the published Mbps). Helper: `uavdt.channel.per_link_hz_for_target_rate`.


| Published anchor (paper text)       | I   | B_i needed  | vs stated 20 kHz | Implied ΣB      |
| ----------------------------------- | --- | ----------- | ---------------- | --------------- |
| Fig. 6 SCA ~8.8 Mbps (`J = 5`)      | 10  | **862 kHz** | **43×**          | 8.62 MHz (431×) |
| Figs. 6–10 band, 7 Mbps at `I = 10` | 10  | 685 kHz     | 34×              | 6.85 MHz (343×) |
| Fig. 7 SCA ~14 Mbps                 | 32  | 428 kHz     | 21×              | 13.7 MHz (685×) |


At a fantasy `\mathrm{SNR}=10^{15}` the stated 20 kHz *floor* would suffice (`B_i` drops to 9–18 kHz). That is why Reading B is not a physical impossibility in the same parameter-free sense as Reading A. It *is* an unstated parameter **21–43×** the table row on the radio Table II actually writes (`\mathrm{SNR}\approx 1`), or a hidden (27) cap of **~7–14 MHz** (hundreds of times 20 kHz as a pool). Neither is a units typo of 20{,}000 Hz.

### 0.2 Sensitivity: radians vs degrees in Eq. (4)

Table II gives `a = 9.61`, `b = 0.16` in the usual Al-Hourani **degree** convention, but Eq. (4) writes \(\theta=\arcsin(H/d_{ij})\) with no conversion. This repo defaults to **radians** (`los_angle_unit="rad"`); `deg` is an opt-in alternate reading.

**Sensitivity check (ran 2026-09-08).** Same zenith link and default campaign point (J = 3, I = 10, seed 1, 8.8 MHz, 25% cap):

| `los_angle_unit` | Zenith SNR | Default-point sum rate |
| --- | --- | --- |
| **rad** (default) | **1.03** | **8.91 Mbps** |
| deg | 90.5 | 57.23 Mbps |

Only the radian reading is consistent with §0.1 (\(\mathrm{SNR}\approx 1\) at zenith) and the feasible-axis Mbps tables. The degree reading would make every headline rate meaningless. **Adopted reading: radians.** Do not compare to papers that silently apply a degree conversion without stating it.

### 0.3 Control: paper 500 × 500 m field, still 20 kHz

A critic can object that the infeasible 20 kHz result was obtained on a **100 × 100 m** field, so two things were changed at once.

- The bound in §0.1 does **not** depend on area.
- The simulator check does not either, in the direction that would help the paper: at 500 × 500 m the zenith SNR is the same (`H = 100` m) and typical links are worse.

**Setup:** default point `I=10`, `J=3`, 20 seeds, random and k-means with the same frozen-`q` bandwidth LP as the campaign (`results/check_bsys_20khz.json`).

#### Mechanism of the realized rate

At 20 kHz the QoS bandwidth LP is infeasible (`R_{\min}=10` kbps on ten links needs ~102 kHz). Evaluate then falls back to **equal-share** `B_{\mathrm{sys}}/I = 2` kHz per associated link:

```text
R_sum          = B_sys · mean_associated_SE     (what we report)
best-link dump = B_sys · max_SE                 (geometry ceiling)
```

Max SNR is set by a UAV over an IoT at `H = 100` m, so `max_SE` (and the dump ceiling) is **area-independent**. Mean SE is not: a 500 m field has longer associated links.


| Field               | Max SNR | Mean assoc. SE | Best-link dump | Equal-share pred. | Random / k-means   | Feasible |
| ------------------- | ------- | -------------- | -------------- | ----------------- | ------------------ | -------- |
| 100 × 100 m         | 1.03    | 0.973          | 0.020 Mbps     | 0.019 Mbps        | 0.019 / 0.020 Mbps | **0%**   |
| 500 × 500 m (paper) | 1.03    | 0.635          | 0.020 Mbps     | 0.013 Mbps        | 0.012 / 0.013 Mbps | **0%**   |


The 500 m drop (0.020 → 0.013 Mbps) is equal-share averaging over worse links, **not** a different SNR regime for the §0.1 bound. Dumping the whole 20 kHz pool on the best link would still be ~0.020 Mbps on both fields. **Area is not a confounder** for the 0%-feasible conclusion.

---

## Audit summary (paper §VII vs this implementation)

> The 20 kHz finding does not wait on the simulator: **Eq. (6) + (27) already cap Table II `B_sys` at 0.997 Mbps even at `SNR=10^{15}`** (§0). We implemented the Khalaf–Itani–Sharafeddine UAV-aided digital-twin IoT model (Eqs. (1)–(6), (11), (13), (16)–(17), Problem (P)) in a **100 × 100 m** field with faithful Table II radio/compute parameters except bandwidth and two external task quantities (`S_i`, `L`), and repeated the 20 kHz check at **500 × 500 m**.
>
> Under **Table II `B_sys = 20 kHz` as the (27) cap**, every method fails QoS and AoDT: feasible fraction **0%**, sum rate **~0.02 Mbps** at 100 m and **~0.012–0.013 Mbps** at 500 m — matching the written channel’s `SNR\approx 1` ceiling, not a solver artifact. We therefore **do not treat the paper’s Fig. 6–10 Mbps curves as a reproduction target**.
>
> For **feasible** bandwidths (2.4 MHz and 8.8 MHz), behaviour is internally coherent:
>
> - **No per-link cap:** all methods saturate near **~8.98 Mbps** (leftover spectrum piles onto the best link).
> - **25% per-link cap** *(external parameter, not Problem (P)):* methods separate; **SCA beats k-means and PSO** on paired seeds with FDR q < 0.05 on **25/25** unique sweep points for both.
> - **vs random:** edge is **regime-limited** (FDR **15/25**; large at J = 1–2 and high I, not significant at default J = 3).
>
> The credible finding is **model-consistent placement comparison under feasible spectrum**, not numeric agreement with §VII.

### Second audit finding: Eq. (17) vs Fig. 11 narrative


| Aspect                 | Status                                                                                                                            |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Problem (P) score**  | Eq. (17): `max_i D_i + (1/λ_min)(1 + Σλ/μ)`                                                                                       |
| **Paper Fig. 11 text** | Heterogeneous within-group λ should sit **between** uniform fast and uniform slow                                                 |
| **Event simulator**    | fast < heterogeneous < slow ✓                                                                                                     |
| **Eq. (17)**           | Heterogeneous scores **worse than uniform-slow** (`Σλ` rises while `λ_min` stays at slow end, 0.8/s)                              |
| **Implementation**     | **Resolved** — coded as published (`aodt.py`, `evaluate()`); unit tests + Fig. 11 sensibility (50/50) pass; **not a code defect** |
| **Paper intent**       | **Open** — whether (17) is what the authors physically intended given Fig. 11 prose                                               |


SCA optimizes (17) as written, so the objective does **not** behave the way the narrative figure describes. The hetero−slow gap **grew** under settled `S_i`/`L` (see [§8.1](#81-fig-11-i--10-k-means-88-mhz-25-cap-20-seeds)).

### Third audit finding: Eq. (4) angle unit

| Aspect | Status |
| --- | --- |
| **Ambiguity** | Table II `a`, `b` are degree-based Al-Hourani constants; Eq. (4) writes \(\arcsin(H/d)\) with no conversion |
| **Sensitivity** | **Checked** — rad: zenith SNR **1.03**, J = 3 seed 1 **8.91 Mbps**; deg: SNR **90.5**, **57.23 Mbps** |
| **Adopted reading** | **Radians** (`los_angle_unit="rad"`, default). Degree reading is inconsistent with §0 and all headline tables |
| **Detail** | [§0.2](#02-sensitivity-radians-vs-degrees-in-eq-4); `docs/REPRODUCTION.md` §3 |

---

## Campaign inventory

### Primary & diagnostic campaigns


| File                                            | B_sys       | Per-link cap | n_runs | Role                                                              |
| ----------------------------------------------- | ----------- | ------------ | ------ | ----------------------------------------------------------------- |
| `results/check_bsys_20khz.json`                 | 20 kHz      | none         | 20     | Model-free ceiling + 100 m vs **500 m** control                   |
| `results/campaign_20khz.json`                   | 20 kHz      | none         | 20     | Table II diagnostic, 100 m, all axes (infeasible)                 |
| `results/campaign_20khz_cap25.json`             | 20 kHz      | 25%          | 20     | Same, capped                                                      |
| `results/campaign_2.4mhz.json`                  | 2.4 MHz     | none         | 20     | Mid-bandwidth, saturated                                          |
| `results/campaign_2.4mhz_cap25.json`            | 2.4 MHz     | 25%          | 20     | Mid-bandwidth, differentiated                                     |
| `results/campaign_8.8mhz_n20.json`              | 8.8 MHz     | none         | 20     | Saturated control (2026-09-05 refresh)                            |
| `results/campaign_8.8mhz_cap25_si12k.json`  | **8.8 MHz** | **25%**      | **20** | **Primary campaign** (100 × 100 m, leftover-dump stress test)     |
| `results/campaign_8.8mhz_cap15_n20.json`        | 8.8 MHz     | 15%          | 20     | Tighter-cap **sensitivity** (100 m; search-selected)              |
| `results/campaign_8.8mhz_cap25_si12k_500m.json` | 8.8 MHz     | 25%          | 20     | **Paper-field test** (500 × 500 m, same 25% primary cap)          |
| `results/campaign_8.8mhz_cap25_n20.json`        | 8.8 MHz     | 25%          | 20     | Same 25% config; output of `run_all_bandwidth_campaigns.ps1`      |


### Superseded


| File                                   | Notes                             |
| -------------------------------------- | --------------------------------- |
| `results/campaign_20260904_cap25.json` | Old `S_i`/`L` placeholders        |
| `results/compare_si_l_defaults.json`   | Old vs settled side-by-side @ J=3 |


### Supporting artifacts


| File                                                       | Contents                                            |
| ---------------------------------------------------------- | --------------------------------------------------- |
| `results/campaign_8.8mhz_cap25_si12k_paired.json` / `.csv` | Paired stats (25% primary)                          |
| `results/campaign_8.8mhz_cap25_si12k_losses.json`          | Loss-seed forensics (J = 3, 25% primary)            |
| `results/campaign_8.8mhz_cap15_n20_paired.json` / `.csv`   | Paired SCA vs baselines (15% sensitivity)           |
| `results/campaign_8.8mhz_cap15_n20_losses.json`            | Loss-seed forensics (J = 3, 15% sensitivity)        |
| `results/compare_cap15_vs_cap25.json`                      | Side-by-side 15% vs 25%                             |
| `results/cap_binding_diagnostic.json`                      | Leftover-dump / cap-binding audit (J=1–5)            |
| `results/bw_cap_by_J_grid.json`                            | 30-cell cap×J search (6 caps × J=1–5; single FDR)    |
| `results/bw_boundary_refine_j3.json`                       | J=3 bisection 23.5–24.5% + 33-test FDR + variance audit |
| `results/se_spread_by_j.json`                              | SE max/mean/min vs J (why the gap fades)            |
| `results/campaign_8.8mhz_cap25_si12k_scajoint.json`        | SCA-joint probe (full axes; method=`sca_joint` only) |
| `results/tk08_scajoint_feasibility.json`                   | T_k=0.8 s frozen SCA vs best-SE SCA-joint (20 seeds) |
| `results/tk08_scajoint_cohesive.json`                      | T_k=0.8 s SCA-joint + process-cohesive candidate (20 seeds) |
| `results/tk08_cohesive_construction.json`                  | T_k=0.8 s centroid-cohesive construction (seeds 1–40) |
| `results/tk08_sync_tradeoff_gap.json`                        | 40-geometry rate-for-synchronization gap (fixed k-means q) |
| `results/sca_joint_vs_frozen.json`                         | Sweep comparison vs frozen SCA (Δ>0.05 Mbps bar)     |
| `results/sca_joint_default_runtime.json`                   | Wall-clock / iterations at default J=3               |
| `results/spot_validate_8.8mhz.json`                        | CVXPY vs MATLAB MOSEK (`agreement: ok`)             |
| `results/fig11_8.8mhz_cap25_si12k.json` / `.csv`           | Fig. 11 λ patterns (primary; equal-share)           |
| `results/fig11_8.8mhz_cap15.json` / `.csv`                 | Fig. 11 (15% sensitivity; numerically identical)    |
| `results/fig11_8.8mhz_cap25.json` / `.csv`                 | Fig. 11 (old `S_i`/`L` placeholders)                |
| `results/figures/`                                         | Primary plots (25%); `figures/cap15/` holds 15%     |


**Analysis scripts:** `scripts/paired_winrate.py`, `scripts/analyze_sca_vs_random_losses.py`, `scripts/analyze_campaigns.py`, `scripts/run_sca_joint_campaign.py`, `scripts/run_tk08_followup.py`, `scripts/bw_cap_by_j_grid.py`, `scripts/bw_boundary_refine_j3.py`, `scripts/cap_binding_diagnostic.py`

---

## Default scenario (headline point)

Unless noted:


| Parameter      | Value                        |
| -------------- | ---------------------------- |
| IoT devices    | **I = 10**                   |
| UAVs           | **J = 3**                    |
| Arrival rate   | **λ = 2/s**                  |
| AoDT threshold | **T_k = 2.8 s**              |
| UAV CPU        | **f_j = 2×10⁸ cycles/s**     |
| Process groups | **K = 2**, **5 IoTs each**   |
| Task size      | **S_i = 12,000 bytes**       |
| Task cycles    | **L = 3.75×10⁶ cycles/task** |
| Per-link cap   | **25%** of `B_sys` (2.20 MHz/link) primary leftover-dump stress test; **15%** is a tighter-cap sensitivity |


---

## 1. Bandwidth sweep (qualitative)

### 1.1 20 kHz (Table II value)

The argument to lead with is §0.1, not this campaign table. The 100 × 100 m sweep confirms the solver agrees with the one-line bound.


| Observation                                          | Value                 |
| ---------------------------------------------------- | --------------------- |
| Model-free ceiling (`SNR_max=10^{15}`)               | **0.997 Mbps**        |
| Written-channel ceiling (observed `SNR\approx 1.03`) | **0.020 Mbps**        |
| Feasible fraction (all methods, all points, 100 m)   | **0%**                |
| Typical sum rate (100 m)                             | **~0.019–0.020 Mbps** |
| Typical sum rate (500 m control)                     | **~0.012–0.013 Mbps** |
| Cap vs no-cap                                        | **No difference**     |


**Interpretation:** If 20 kHz is the (27) cap, total spectrum is the bottleneck and 7–14 Mbps is impossible without trusting the simulator. QoS (`R_min = 10` kbps per active link) also fails: meeting it on ten associated links needs on the order of **~102 kHz** at 100 m (`∑_i R_min / SE_{ij}`), and more at 500 m where SE is worse. AoDT cannot be met either. `S_i` / `L` do not enter the §0.1 bound.

**Settled `S_i` / `L` defaults:** `S_i = 12{,}000` bytes (`96{,}000` bit), `L = 3.75\times10^6` cycles/task (`\mu \approx 53.3` /s). Not in Table II. Prior placeholder runs used `10{,}000` bit and `L=10^6`; see [§10](#10-s_i--l-correction--settled-defaults-vs-old-placeholders). Primary campaign `campaign_8.8mhz_cap25_si12k.json` uses the settled values.

- `S_i` only enters upload delay (Eq. 11) and thus the AoDT bandwidth floor.
- `L` only scales `μ_j`; it does not enter the radio model.

### 1.2 2.4 MHz


| Config  | SCA @ J=3  | All methods feasible? |
| ------- | ---------- | --------------------- |
| No cap  | 2.434 Mbps | Yes (100%)            |
| Cap 25% | 2.428 Mbps | Yes (100%)            |


At J = 3 with cap 25% (`campaign_2.4mhz_cap25.json`, 2026-09-05): SCA 2.428, PSO 2.423, k-means 2.420, random 2.414 Mbps — gaps **< 0.01 Mbps**. Without cap, PSO/k-means are within **~0.005 Mbps** of SCA. Ordering still favours SCA at J = 3 but margins are tighter than the 8.8 MHz 25% primary campaign.

### 1.3 8.8 MHz, no cap


| Method @ J=3 | Mean Mbps | Feasible |
| ------------ | --------- | -------- |
| SCA          | 8.971     | 100%     |
| Random       | 8.954     | 100%     |
| K-means      | 8.947     | 100%     |
| PSO          | 8.943     | 100%     |


Spread **< 0.03 Mbps** (`campaign_8.8mhz_n20.json`, 2026-09-05). **λ**, **CPU**, and most **AoDT** points (when feasible) give **identical** per-seed rates — the objective is communication-limited only.

> **Not useful for ranking methods.**

### 1.4 8.8 MHz, 25% cap *(primary)*


| Method @ J=3 | Mean Mbps | Feasible |
| ------------ | --------- | -------- |
| **SCA**      | **8.946** | 100%     |
| Random       | 8.928     | 100%     |
| PSO          | 8.905     | 100%     |
| K-means      | 8.901     | 100%     |


Spread **0.046 Mbps**. SCA vs random is **+0.019 Mbps, 15/20, p=0.123** (not significant). This is still the primary leftover-dump stress test: the cap was chosen independently of that p-value. See [§2.7](#27-primary-campaign-88-mhz-25-cap).

### 1.5 8.8 MHz, 15% cap *(tighter-cap sensitivity)*


| Method @ J=3 | Mean Mbps | Feasible |
| ------------ | --------- | -------- |
| **SCA**      | **8.895** | 100%     |
| PSO          | 8.854     | 100%     |
| K-means      | 8.838     | 100%     |
| Random       | 8.765     | 100%     |


Spread **0.129 Mbps**. SCA vs random is **+0.129 Mbps, 20/20, p&lt;0.001** (Bonferroni p_adj &lt; 0.001). Keep this campaign as a sensitivity: 15% was search-selected and the J-sweep Δ > 0.05 Mbps bar fails at J = 4–5. Full axis tables are in [§2](#2-sweep-axes-at-88-mhz-15-cap).

---

## 2. Sweep axes at 8.8 MHz, 15% cap

**Source:** `campaign_8.8mhz_cap15_n20.json`

These tables are a tighter-cap **sensitivity**, not the primary ranking campaign. Primary 25% means and paired stats are in [§2.7](#27-primary-campaign-88-mhz-25-cap) and [§3.1](#31-default-point--j--3-i--10-25-primary).

Campaign Mbps means are `mean(all 20 seeds)` — `campaign._summarize` does not drop infeasible seeds. After the (24) init repair, **every row in §2.1–2.3 and §2.5 is 100% feasible**. The only exception is **§2.4 T_k = 0.8 s**.

### 2.1 Fig. 6 analogue — UAV count J (I = 10)


| J   | SCA   | Random | K-means | PSO   |
| --- | ----- | ------ | ------- | ----- |
| 1   | 8.488 | 8.100  | 8.382   | 8.411 |
| 2   | 8.776 | 8.564  | 8.687   | 8.758 |
| 3   | 8.895 | 8.765  | 8.838   | 8.854 |
| 4   | 8.939 | 8.892  | 8.901   | 8.910 |
| 5   | 8.959 | 8.943  | 8.937   | 8.942 |


**Analysis**

- **J = 1–3:** SCA gains the most when UAVs are scarce and the 15% cap binds — spatial placement and joint B allocation matter. Spread at J = 1 is **0.388 Mbps**. J = 1 and J = 2 are **100% feasible** for every method.
- **J = 4–5:** Rates converge; SCA still leads but the random gap shrinks to ~0.016–0.047 Mbps.
- Vs PSO at **J = 2** the mean edge is only +0.018 Mbps and the paired FDR test is **not** significant (q = 0.123). That is a tight-PSO cell, not a ranking reversal (SCA still has the higher mean).
- Monotonic **non-decreasing** sum rate with J for all methods — sensible.

### 2.2 Fig. 7 analogue — IoT count I (J = 3)


| I   | SCA   | Random | K-means | PSO   | SCA feasible |
| --- | ----- | ------ | ------- | ----- | ------------ |
| 10  | 8.895 | 8.765  | 8.838   | 8.854 | 100%         |
| 16  | 8.914 | 8.843  | 8.858   | 8.866 | 100%         |
| 20  | 8.913 | 8.855  | 8.870   | 8.889 | 100%         |
| 24  | 8.910 | 8.849  | 8.872   | 8.881 | 100%         |
| 28  | 8.907 | 8.833  | 8.880   | 8.884 | 100%         |
| 32  | 8.895 | 8.819  | 8.870   | 8.878 | 100%         |


**Analysis**

- SCA stays above k-means/PSO/random at every I. The random gap is **wider** than at 25% (~0.07–0.13 Mbps vs ~0.02–0.07).
- SCA’s mean is not monotone in I (a small bump at I = 16, then a slow decline). That is expected: more IoTs add both extra links and tighter AoDT floors.
- At **I = 28–32**, putting both process groups on one UAV would violate (24) (56/s and 64/s vs μ ≈ 53.3 /s), but a **split** assignment is feasible (28/s and 32/s each). After repairing init when (24) fails, **all 20 seeds are feasible** at both ticks for **every method**.
- SCA wins paired FDR tests vs random and vs k-means on all unique I ticks.

### 2.3 Fig. 8 analogue — Arrival rate λ (I = 10, J = 3)


| λ (/s) | SCA   | Random | K-means | PSO   |
| ------ | ----- | ------ | ------- | ----- |
| 1.0    | 8.891 | 8.761  | 8.836   | 8.852 |
| 1.5    | 8.893 | 8.764  | 8.837   | 8.853 |
| 2.0    | 8.895 | 8.765  | 8.838   | 8.854 |
| 2.5    | 8.895 | 8.766  | 8.838   | 8.854 |
| 3.0    | 8.896 | 8.766  | 8.838   | 8.854 |
| 3.5    | 8.896 | 8.767  | 8.838   | 8.854 |


**Analysis:** With settled `L`, AoDT binds at `T_k = 2.8 s` and the comm score **varies slightly** with λ. Differences are **< 0.01 Mbps** on SCA. **Do not pool λ rows with J = 3 in meta-analyses** (they duplicate the default at λ = 2.0 only).

### 2.4 Fig. 9 analogue — AoDT threshold T_k (I = 10, J = 3)


| T_k (s)   | SCA Mbps      | SCA feasible | Notes                                               |
| --------- | ------------- | ------------ | --------------------------------------------------- |
| **0.8**   | 8.842         | **0%**       | Mean of 20 **infeasible** scores; not a model limit |
| 1.2       | 8.852         | 100%         |                                                     |
| 1.6       | 8.878         | 100%         |                                                     |
| 2.0       | 8.887         | 100%         |                                                     |
| 2.4       | 8.892         | 100%         |                                                     |
| 2.8 – 3.0 | 8.895 – 8.896 | 100%         | Plateau at default                                  |


**Analysis**

- T_k = 0.8 s is **0% feasible** for SCA, k-means, and random under frozen nearest-association (PSO 2/20). **Do not publish that 0% as a Problem (P) limit.**
- Eq. (17) slack: T_k - Q - T_{\mathrm{u2u}} on forwarded links. With \lambda=2/s, |N_k|=5, \mu\approx 53.3/s, Q\approx 0.594 s → Q+T_{\mathrm{u2u}}\approx 0.894 s **> 0.8 s** even at infinite rate.
- The same 20 campaign seeds, and 20 held-out geometries (seeds 21–40), admit a legal assignment at the **same k-means \(q\)**: put all of \(N_k\) on one UAV (process-cohesive \(a_{ij}\)). That construction is feasible **40/40** (`tk08_cohesive_construction.json`). Nearest-UAV stays **0/40** (bandwidth LP infeasible; upload slack \(T_k-Q-T_{\mathrm{u2u}}=-0.094\) s). J = 1 is also 20/20 (no other UAV to forward to). PSO is 2/20 by parking both groups on one UAV.
- Letting sequential SCA rematch \(a_{ij}\) to the current best-SE UAV does **not** find that map (§2.8): frozen SCA and best-SE SCA-joint are both **0/20** at T_k = 0.8 s. Best-SE coincides with nearest at the infeasible init, so that discrete step is a no-op.
- Adding process-cohesive grouping as a **second rematch candidate** does find it: SCA-joint is then **20/20** feasible (§2.8). That is a probe flag, not the headline solver. The radio cost of meeting \(T_k=0.8\) s via grouping is **~0.6 Mbps** median at fixed \(q\) across **40/40** geometries (range 0.29–0.98; §I accuracy–synchronization tradeoff).
- For T_k ≥ 1.2 s the comm objective rises toward the default plateau.

### 2.5 Fig. 10 analogue — UAV CPU f_j (I = 10, J = 3)


| f_j (×10⁸ c/s) | SCA Mbps      | SCA feasible |
| -------------- | ------------- | ------------ |
| 0.5            | 8.892         | 100%         |
| 1.0            | 8.894         | 100%         |
| 1.5 – 2.5      | 8.894 – 8.895 | 100%         |


**Analysis:** At f_j = 0.5×10⁸, μ ≈ 13.3 /s; splitting groups (10/s each) is feasible. After init repair, **20/20 seeds are feasible** and the comm score sits on the default plateau. **Do not pool CPU rows with J = 3 in meta-analyses.**

### 2.6 Paper field 500 × 500 m (8.8 MHz, 25% cap)

**Source:** `campaign_8.8mhz_cap25_si12k_500m.json` (2026-09-07, 20 seeds). Field-size test at the **same 25% primary cap**. A 500 m campaign at 15% has not been run.

Same 25% cap as the 100 m primary campaign, but IoTs and UAVs are placed in the paper’s **500 × 500 m** field. Zenith max SNR is unchanged (`H = 100` m); mean link quality is worse, so sum rates drop ~0.6–0.7 Mbps at J = 3 relative to 100 m **at the same 25% cap**.

#### Default point — J = 3, I = 10 (both columns 25% cap)


| Method  | 100 m (25% primary) | 500 m (25%) | Δ (500 − 100) |
| ------- | ---------------- | ----------- | ------------- |
| **SCA** | 8.946            | **8.300**   | −0.646        |
| PSO     | 8.905            | 8.122       | −0.783        |
| Random  | 8.928            | 7.947       | −0.981        |
| K-means | 8.901            | 7.474       | −1.427        |


All methods **100% feasible** at J = 3 on both fields.

#### Fig. 6 analogue — UAV count J (I = 10)


| J   | SCA (500 m) | Random | K-means | PSO   |
| --- | ----------- | ------ | ------- | ----- |
| 1   | 6.030       | 4.980  | 4.580   | 6.020 |
| 2   | 7.490       | 6.760  | 6.170   | 7.520 |
| 3   | 8.300       | 7.947  | 7.474   | 8.122 |
| 4   | 8.580       | 8.480  | 8.110   | 8.410 |
| 5   | 8.760       | 8.670  | 8.530   | 8.610 |


**Analysis**

- SCA leads at every J; the gap vs k-means is **largest at J = 3** (~0.83 Mbps) where placement and bandwidth allocation matter most on a large field.
- At J = 1, PSO nearly matches SCA (~6.02 vs 6.03 Mbps); k-means lags (~4.58 Mbps).
- I = 32: random drops to **95%** feasible (one seed fails QoS/AoDT); SCA/k-means/PSO stay 100%.
- T_k = 0.8 s remains **0% feasible** for all methods under nearest-a (same discrete-init artefact as §2.4 / §2.8; not a Problem (P) limit).
- Method ranking is unchanged: **SCA > PSO ≈ random > k-means** at 500 m, with larger separations than at 100 m.

Plot: `python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap25_si12k_500m.json --out-dir results/figures/500m`

### 2.7 Primary campaign: 8.8 MHz, 25% cap

**Source:** `campaign_8.8mhz_cap25_si12k.json` (100 × 100 m).

25% is the **primary leftover-dump stress test**, chosen independently of the SCA-vs-random p-value. Without a per-link cap the frozen-q LP puts ~95% of `B_sys` on the best-SE link at every J; a 25% cap pins max share at 0.25 (~4 links to exhaust the pool). A 15% cap pins it at 0.15 (~7 links) and remains a tighter-cap **sensitivity** (search-selected from a cap×J grid). The J fade of the rate gap is **SE spread**, not dump fade: `SE_max` stays ≈ 1.021 (zenith at H = 100 m) while `SE_min` / `SE_mean` rise with J. See `results/cap_binding_diagnostic.json` and `results/se_spread_by_j.json`.

At default J = 3, SCA vs random is **not** significant (p = 0.123). That does not demote 25% to a control.

#### Default point — J = 3, I = 10


| Method  | 25% (primary) | 15% (sensitivity) | Δ (15 − 25) |
| ------- | ------------- | ----------------- | ----------- |
| **SCA** | **8.946**     | 8.895             | −0.052      |
| PSO     | 8.905         | 8.854             | −0.051      |
| K-means | 8.901         | 8.838             | −0.063      |
| Random  | 8.928         | 8.765             | −0.162      |


Random drops the most when the cap tightens; SCA’s ranking edge **widens** at 15%. Paired vs-random at 25%: +0.019 ± 0.042 Mbps, 15/20, Wilcoxon p = 0.123. Full 25% axis tables remain in the primary campaign JSON; do not mix them with the 15% tables in §2.1–2.5.

#### Cap × J search and boundary refinement

**Question:** is 15% “significant vs random” a general property of any cap, or a search-selected cell in a larger landscape?

**What ran.** Exploratory **30-cell** grid: caps **15%, 18%, 20%, 23%, 25%, 30%** × **J = 1–5**, 20 seeds each, SCA vs random at 8.8 MHz (`bw_cap_by_J_grid.json`). One **BH-FDR** correction across all 30 Wilcoxon p-values. Practical-effect audit: **Δ > 0.05 Mbps** (same bar as elsewhere). Mechanism audit: frozen-geometry leftover-dump binding (`cap_binding_diagnostic.json`). J = 3 **bisection** at **23.5%, 24%, 24.5%** with the 30 grid points pooled into a **33-test FDR** (`bw_boundary_refine_j3.json`). Loose-cap **40%/50%** J = 3 rows in the boundary file are variance diagnostics only (excluded from the 33-test FDR).

**Landscape (not two endpoints).** Significance vs random is **U-shaped in cap%**, not monotonic:

| Cap | J = 3 Δ (Mbps) | J = 3 FDR q | FDR-sig at **all** J = 1–5? | J = 3 clears Δ > 0.05? |
| --- | --- | --- | --- | --- |
| 15% | +0.129 | 8.2×10⁻⁶ | **Yes** | **Yes** |
| 18% | +0.085 | 8.2×10⁻⁶ | No | Yes |
| 20% | +0.060 | 1.1×10⁻⁵ | No | Yes |
| 23% | +0.035 | 8.8×10⁻⁴ | No | No |
| **25%** (primary) | +0.019 | 0.137 | No | No |
| 30% | +0.011 | 0.032 | **Yes** | No |

Only **15%** and **30%** are FDR-significant at every J simultaneously — but for opposite reasons. At **15%** the cap forces leftover spectrum across ~6 associated links (mean `n_at_cap` = 6 at J = 3), so placement and B allocation matter; Δ is large at low J and shrinks at J = 4–5 (0.047 / 0.016 Mbps — below the practical bar). At **30%** the raw gap stays **< 0.02 Mbps** at J = 3 while `std(Δ)` falls (0.042 → 0.027 Mbps); standardized effect `Δ/std` stays ~0.4. Later p &lt; 0.05 values in the 25–50% band are **variance shrinkage on an already-tiny gap**, not effect-size recovery (`bw_boundary_refine_j3.json` paragraph).

**J = 3 crossover (precise).** After 33-test FDR, significance holds through **23.5%** (q = 0.0052, Δ = +0.029 Mbps) and turns off at **24.0%** (q = 0.101, Δ = +0.022 Mbps). That brackets the switch between “tight-cap regime” and “primary 25% leftover-dump stress test.”

**Practical-effect verdict.** **No cap** survives “FDR-significant **and** Δ > 0.05 Mbps at all J = 1–5.” 15% was retained as a **tighter-cap sensitivity** (search-selected), not as an independently chosen primary. **25%** stays primary because it was adopted as the leftover-dump stress test **before** the grid search, and because vs-random at default J = 3 is honestly n.s. (p = 0.123) — not because the grid “picked” 25%.

**Why this matters.** Without this subsection, the doc reads like “we tried 15% and 25%.” The actual finding is a **characterized cap landscape**: tight caps concentrate the LP onto more links (real effect); loose caps let the LP dump on one link (tiny Δ, variance-driven significance); the primary 25% point sits in the middle where effect size is negligible even when k-means/PSO comparisons remain significant.

### 2.8 SCA-joint methodology probe

**Primary question (pre-registered):** is freezing \(a_{ij}\) / \(b_{ij}\) after nearest-UAV init a limitation of Problem (P), or only of this sequential SCA solver?

**What ran.** A separate solver (`uavdt.sca_joint`, campaign tag `method="sca_joint"`) keeps Algorithm 1’s joint Taylor-LP on \((q,B)\), then at each outer iteration proposes discrete re-associations and `cpu_stable_processing` for (23)/(24). The discrete update is accepted only if `evaluate()` becomes newly feasible or the true sum rate improves (same gate as the \(q,B\) step). Frozen SCA was not modified. Headline files tagged `sca` were not overwritten. Default rematch is best-SE only; later steps add an optional process-cohesive candidate and a 40-seed construction sweep.

**Sources:** `results/tk08_scajoint_feasibility.json`, `results/tk08_scajoint_cohesive.json`, `results/tk08_cohesive_construction.json`, `results/tk08_sync_tradeoff_gap.json`, `results/campaign_8.8mhz_cap25_si12k_scajoint.json`, `results/sca_joint_vs_frozen.json`, `results/sca_joint_default_runtime.json`. Frozen numbers are from `campaign_8.8mhz_cap25_si12k.json`. Practical-effect bar: **Δ > 0.05 Mbps**. SCA-joint vs random/k-means/PSO is **not** a replacement headline comparison.

#### Investigation sequence (T_k = 0.8 s)

The Fig. 9 row at \(T_k=0.8\) s looked like a model limit (0% feasible for SCA, k-means, random). The work below is the audit trail — each step either falsified a hypothesis or narrowed it.

| Step | Suspicion / test | Outcome | Artifact |
| --- | --- | --- | --- |
| **1** | Campaign 0% means Problem (P) is infeasible at \(T_k=0.8\) s | **Falsified.** Hand construction: put all of \(N_k\) on one UAV at the same k-means \(q\) → feasible on campaign seeds | `tk08_scajoint_feasibility.json` (side field) |
| **2** | Unfreezing \(a_{ij}\) with best-SE rematch will find that map | **Null.** SCA-joint (best-SE only) **0/20**; best-SE = nearest at init, so rematch is a no-op; UAVs never move | `tk08_scajoint_feasibility.json` |
| **3** | Maybe association matters on the full axis sweep | **Null for Mbps.** Best-SE SCA-joint ≥ frozen SCA everywhere; max Δ **+0.006 Mbps**; **0/29** feasibility differences on feasible points | `campaign_8.8mhz_cap25_si12k_scajoint.json`, `sca_joint_vs_frozen.json` |
| **4** | Refine: blocker is “greedy never proposes grouping,” not LP / evaluate bug | **Targeted test.** One extra rematch candidate: process-cohesive \(a_{ij}\) beside best-SE | (design) |
| **5** | Process-cohesive candidate inside SCA-joint | **Confirmed.** **20/20** feasible; every seed accepts `process_cohesive`; LP and `evaluate()` were not the blocker | `tk08_scajoint_cohesive.json` |
| **6** | Single-seed construction might be a geometry accident | **Generalized.** Centroid-cohesive construction feasible **40/40** (seeds 1–20 + held-out 21–40); nearest stays **0/40** | `tk08_cohesive_construction.json` |
| **7** | Does the throughput cost of grouping also generalize? | **Yes, with spread explained by SE loss.** At fixed k-means \(q\), best-SE @ \(T_k=1.2\) s minus cohesive @ \(T_k=0.8\) s: mean **0.59 Mbps**, median **0.57**, range **0.29–0.98** (40/40). Spread tracks \(\sum \Delta\mathrm{SE}\) (\(r\approx 0.92\)), not count of IoTs off best link (\(r\approx 0.14\)) | `tk08_sync_tradeoff_gap.json` |

**Mechanistic thread.** Splitting \(N_k\) across UAVs forces \(T_{\mathrm{u2u}}\) on forwarded links; with settled \(\lambda\) and \(|N_k|=5\), \(Q+T_{\mathrm{u2u}}\approx 0.894>0.8\) even at infinite rate — so nearest-UAV init cannot pass AoDT. Process-cohesive assignment removes forwarding (\(T_k-Q=+0.206\) s slack on every geometry). Best-SE rematch never proposes that map because per-IoT SE ranking coincides with nearest at the infeasible init. Adding the cohesive candidate is the direct test of that story; it passes.

#### Summary at T_k = 0.8 s (I = 10, J = 3)

| Method | Feasible | Mean Mbps | Process-cohesive \(a_{ij}\) |
| --- | --- | --- | --- |
| Frozen SCA | **0/20** | 8.909 (infeasible diagnostic) | 0/20 |
| SCA-joint (best-SE only) | **0/20** | 8.909 (infeasible diagnostic) | 0/20 |
| SCA-joint + process-cohesive candidate | **20/20** | **8.393** (feasible) | 20/20 |
| Hand construction at k-means \(q\) | **40/40** (20 campaign + 20 held-out) | 8.275 at frozen \(q\) | 40/40 |

The two Mbps columns are **not** a head-to-head ranking: 8.909 is the mean of infeasible nearest-UAV scores (AoDT fails; sum rate is still computed). 8.393 is the mean of **feasible** scores after accepting process-cohesive grouping and running the \((q,B)\) loop.

#### Why the feasible cohesive rate is lower — and what it means for §I

Readers will notice 8.393 < 8.909 and wonder whether feasibility was bought cheaply. Two separate points:

1. **Do not compare those two means directly** — one averages failing runs, the other feasible converged solutions.
2. **There is a real, quantified radio-efficiency cost to grouping** — and it is a direct instance of the accuracy–synchronization tradeoff the paper motivates in **§I** (higher communication throughput vs. meeting a tight process AoDT deadline).

**Definition (fixed k-means \(q\), association only).** For each seed, compare two feasible frozen-\(q\) points on the same IoT placement: **best-SE** \(a_{ij}\) at \(T_k=1.2\) s (AoDT feasible under per-IoT radio gain) vs **process-cohesive** \(a_{ij}\) at \(T_k=0.8\) s (AoDT feasible under no-forwarding synchronization). The gap is the throughput “price” of meeting the tighter deadline via grouping rather than splitting \(N_k\).

**40-geometry check** (`tk08_sync_tradeoff_gap.json`; seeds 1–20 campaign + 21–40 held-out):

| Statistic | Gap (Mbps) |
| --- | --- |
| Mean | **0.59** |
| Median (p50) | **0.57** |
| Std | 0.16 |
| Range | 0.29 – 0.98 |
| p10 – p90 | 0.38 – 0.81 |
| Campaign mean (seeds 1–20) | 0.60 |
| Held-out mean (seeds 21–40) | 0.57 |
| In [0.5, 0.7] Mbps | 18 / 40 (45%) |

The magnitude **generalizes** with the feasibility result: both panels agree, and the median stays near **~0.6 Mbps**. It is **not** a single fixed constant. Quote **~0.6 Mbps (median ~0.57, p10–p90 ~0.38–0.81)** rather than treating 0.60 as exact.

**What drives the spread (checked).** Cohesive grouping moves **3–8 IoTs per seed** off their best-SE UAV (mean **4.9**). That count alone does **not** predict the gap (\(r\approx 0.14\)). The gap tracks **how much spectral efficiency is sacrificed**: \(\sum_i(\mathrm{SE}_{ij}^{\mathrm{coh}}-\mathrm{SE}_{ij}^{\mathrm{best}})\) correlates with gap at **\(r\approx 0.92\)**. Two seeds make the distinction clear:

- **Seed 16** (max gap **0.98 Mbps**): only **6/10** IoTs leave best-SE, but several land on much worse links (\(\Delta\mathrm{SE}\approx -0.27\) to \(-0.35\) per IoT; total \(\Delta\sum\mathrm{SE}=-1.43\)).
- **Seed 23** (only **0.47 Mbps** despite **8/10** off best): every moved IoT’s alternative link is nearly as good (\(\Delta\mathrm{SE}\) mostly \(-0.03\) to \(-0.15\)).

So the tail is not an outlier artefact or a second hidden mechanism — it is the same radio-geometry cost, with magnitude set by **how bad the cohesive links are**, not simply how many devices move. Nothing else is needed to explain the 0.29–0.98 Mbps range.

**Paper framing.** This is not a solver-debugging footnote. It is an **experimental confirmation inside our reproduction** of the paper’s central conceptual claim: you can gain synchronization (satisfy Eq. (17) at \(T_k=0.8\) s by keeping each \(N_k\) on one UAV) only by giving up roughly half a megabit per second of sum rate at the same deployment geometry. Nearest/best-SE association maximizes the communication objective but fails the deadline; process-cohesive association pays the radio tax and passes. That is the rate-vs-AoDT tension stated in §I, observed quantitatively at the Fig. 9 edge case.

The SCA-joint solver mean (8.393 Mbps) is slightly **above** the frozen-\(q\) cohesive construction mean (8.275 Mbps) because UAVs move after grouping; association stays process-cohesive.

#### Full sweep vs frozen SCA (best-SE rematch; 8.8 MHz, 25% cap, 20 seeds)

Step 3 above: pre-registered SCA-joint ≥ frozen SCA everywhere. **Holds** under the 0.05 Mbps bar (0/29 points with joint lower; 0/29 practical rate gaps; 0/29 feasibility differences on feasible-axis points). Largest Δ is **+0.006 Mbps** at default J = 3. Rematch *does* fire after UAVs move (463 accepted discrete updates over 580 seeds; association changed on 198/580), but the rate effect stays well below the bar.

| Sweep point | Frozen SCA Mbps / feas. | SCA-joint Mbps / feas. | Δ Mbps | Verdict |
| --- | --- | --- | --- | --- |
| J = 1 | 8.758 / 100% | 8.758 / 100% | +0.000 | negligible |
| J = 2 | 8.886 / 100% | 8.889 / 100% | +0.003 | negligible |
| J = 3 | 8.946 / 100% | 8.952 / 100% | +0.006 | negligible |
| J = 4 | 8.971 / 100% | 8.972 / 100% | +0.001 | negligible |
| J = 5 | 8.977 / 100% | 8.978 / 100% | +0.000 | negligible |
| I = 10 | 8.946 / 100% | 8.952 / 100% | +0.006 | negligible |
| I = 16 | 8.942 / 100% | 8.943 / 100% | +0.001 | negligible |
| I = 20 | 8.934 / 100% | 8.936 / 100% | +0.002 | negligible |
| I = 24 | 8.925 / 100% | 8.928 / 100% | +0.003 | negligible |
| I = 28 | 8.919 / 100% | 8.919 / 100% | +0.000 | negligible |
| I = 32 | 8.906 / 100% | 8.907 / 100% | +0.001 | negligible |
| λ = 1.0 /s | 8.944 / 100% | 8.947 / 100% | +0.003 | negligible |
| λ = 1.5 /s | 8.947 / 100% | 8.951 / 100% | +0.004 | negligible |
| λ = 2.0 /s | 8.946 / 100% | 8.952 / 100% | +0.006 | negligible |
| λ = 2.5 /s | 8.947 / 100% | 8.952 / 100% | +0.005 | negligible |
| λ = 3.0 /s | 8.948 / 100% | 8.952 / 100% | +0.005 | negligible |
| λ = 3.5 /s | 8.948 / 100% | 8.953 / 100% | +0.005 | negligible |
| T_k = 0.8 s | 8.909 / **0%** | 8.909 / **0%** | +0.000 | negligible |
| T_k = 1.2 s | 8.889 / 100% | 8.894 / 100% | +0.006 | negligible |
| T_k = 1.6 s | 8.925 / 100% | 8.930 / 100% | +0.005 | negligible |
| T_k = 2.0 s | 8.938 / 100% | 8.942 / 100% | +0.003 | negligible |
| T_k = 2.4 s | 8.945 / 100% | 8.948 / 100% | +0.003 | negligible |
| T_k = 2.8 s | 8.946 / 100% | 8.952 / 100% | +0.006 | negligible |
| T_k = 3.0 s | 8.948 / 100% | 8.953 / 100% | +0.005 | negligible |
| f_j = 0.5×10⁸ | 8.946 / 100% | 8.949 / 100% | +0.002 | negligible |
| f_j = 1.0×10⁸ | 8.948 / 100% | 8.952 / 100% | +0.004 | negligible |
| f_j = 1.5×10⁸ | 8.946 / 100% | 8.952 / 100% | +0.006 | negligible |
| f_j = 2.0×10⁸ | 8.946 / 100% | 8.952 / 100% | +0.006 | negligible |
| f_j = 2.5×10⁸ | 8.947 / 100% | 8.953 / 100% | +0.006 | negligible |

Duplicate default-scenario rows (J = 3 / I = 10 / λ = 2 / T_k = 2.8 / f_j = 2×10⁸) are independent re-solves of the same 20 seeds; they agree. The T_k = 0.8 s row is the **best-SE** SCA-joint campaign (0/20). The cohesive-candidate follow-up (20/20) is a separate file and is not mixed into this table.

**Verdict on the story.** The sequence above closes the T_k = 0.8 s false alarm: Problem (P) is feasible (40/40 construction); sequential SCA’s 0% is a nearest/best-SE proposal failure, confirmed when one cohesive candidate recovers 20/20. The throughput cost of synchronization generalizes too: **~0.6 Mbps** median at fixed \(q\) across 40 geometries (§I tradeoff, not a solver artifact). On every other sweep point, best-SE rematch does not move the headline (max Δ **+0.006 Mbps**). Frozen nearest-UAV association was costing the *model* claim at T_k = 0.8 s, not the Mbps tables. The cohesive rematch flag is a hypothesis test, not a headline solver. TD3’s binaries-frozen decision is independent of this probe.

#### Wall-clock / iterations at the default point (TD3 write-up)

Paper Fig. 5 flags SCA’s iteration/scalability cost as the gap TD3 is meant to close. Adding a discrete re-match was expected to make that worse.

| | Frozen SCA | SCA-joint | Ratio (joint / frozen) |
| --- | --- | --- | --- |
| Mean wall-clock (20 seeds) | 0.891 s | 0.888 s | **1.00×** |
| Mean outer iterations | 27.7 | 27.9 | 1.01× |
| Mean Mbps | 8.946 | 8.952 | — |
| Feasible | 20/20 | 20/20 | — |

The extra work is an \(O(IJ)\) SE argmax plus an occasional frozen-\(q\) bandwidth LP when the ranking actually changes. At I = 10, J = 3 that is lost in the noise of the joint Taylor LP (~28 iterations, ~0.9 s). SCA’s scalability weakness for a later TD3 comparison remains the **convexified (q, B) loop**, not this greedy discrete search. The process-cohesive candidate is one extra frozen-\(q\) LP per rematch attempt; at T_k = 0.8 s that probe averaged **1.23 s / 24.7 iters** because it actually entered the \((q,B)\) loop after grouping. That is still the convex loop, not a combinatorial association search.

---

## 3. Paired statistics (SCA vs baselines, same seed)

**Primary source:** `campaign_8.8mhz_cap25_si12k_paired.json`  
**Sensitivity source:** `campaign_8.8mhz_cap15_n20_paired.json`  
**Delta:** SCA − baseline (Mbps) · **Test:** Wilcoxon signed-rank (two-sided) · **Std:** sample std of 20 paired deltas, not SEM

### 3.1 Default point — J = 3, I = 10 (25% primary)


| Baseline | Mean Δ | Std Δ | Wins  | Wilcoxon p | Bonferroni p_adj | Verdict         |
| -------- | ------ | ----- | ----- | ---------- | ---------------- | --------------- |
| Random   | +0.019 | 0.042 | 15/20 | 0.123      | 0.369            | Not significant |
| K-means  | +0.046 | 0.019 | 20/20 | &lt;0.001  | &lt;0.001        | **Significant** |
| PSO      | +0.042 | 0.026 | 19/20 | &lt;0.001  | &lt;0.001        | **Significant** |


FDR vs random at 25%: **15 / 25** unique points — **not** at default J = 3. Five losing seeds vs random (7, 11, 13, 18, 19): random drew a better capped-LP geometry; SCA initializes from k-means and does not search the random basin. That collapse is expected under a leftover-dump stress test that does not pin leftover onto as many links as 15%. It does **not** demote 25% from primary.

### 3.2 Same tests at 15% (tighter-cap sensitivity)


| Baseline | Mean Δ | Std Δ | Wins  | Wilcoxon p | Bonferroni p_adj | Verdict         |
| -------- | ------ | ----- | ----- | ---------- | ---------------- | --------------- |
| Random   | +0.129 | 0.187 | 20/20 | &lt;0.001  | &lt;0.001        | **Significant** |
| K-means  | +0.057 | 0.034 | 20/20 | &lt;0.001  | &lt;0.001        | **Significant** |
| PSO      | +0.041 | 0.025 | 19/20 | &lt;0.001  | &lt;0.001        | **Significant** |


**Wilcoxon vs paired t (vs random):** Wilcoxon p &lt; 0.001, paired t p = 0.006 on the same 20 deltas. All 20 signs are positive. Right-skewed (skew ≈ +3.13); largest win is seed 3 (+0.872 Mbps); smallest is seed 4 (+0.001 Mbps). **Zero losses vs random.**

### 3.3 Regime map — BH-FDR over 25 unique sweep points *(15% sensitivity)*

Repeated default-scenario copies (λ = 2.0 and f_j = 2.0 duplicate of J = 3; I = 10 duplicate of J = 3) excluded.


| Baseline    | FDR q < 0.05 | Where significant                                                                 |
| ----------- | ------------ | --------------------------------------------------------------------------------- |
| **Random**  | **25 / 25**  | All unique J, I, λ, T_k, f_j points                                               |
| **K-means** | **25 / 25**  | All unique points                                                                 |
| **PSO**     | **23 / 25**  | All unique points **except** J = 2 (q = 0.123) and T_k = 1.2 s (q = 0.055)        |


> **Sensitivity pattern:** Under the 15% cap, SCA’s advantage vs **random** and **k-means** holds on the full unique grid, **including default J = 3**. Vs **PSO** the edge is significant at the default point but not at two tight cells (J = 2; T_k = 1.2 s). Do not treat this as the primary ranking campaign: 15% was search-selected, and the practical Δ > 0.05 Mbps bar fails at J = 4–5.

*T_k = 0.8 under frozen nearest-a is all-infeasible for SCA/k-means/random (PSO 2/20); the paired gap there is among infeasible nearest-a scores, not a feasible QoS win.*

### 3.4 Why k-means and PSO lose (structural, not noise)

At J = 3, mean over 20 seeds (`campaign_8.8mhz_cap15_n20_losses.json` → `proxy_mismatch_j3`; 15% is the more informative cap because it binds harder):


| Method  | Equal-share (Mbps) | Capped LP (Mbps) | LP − equal |
| ------- | ------------------ | ---------------- | ---------- |
| Random  | 8.432              | 8.765            | **+0.334** |
| K-means | 8.692              | 8.838            | +0.146     |
| PSO     | 8.742              | 8.854            | +0.112     |


- PSO’s **inner** fitness uses equal-share bandwidth; k-means minimizes spatial spread.
- The **campaign score** is the capped LP.
- Equal-share is `B_sys/I = 0.88` MHz/link, which is **below** both the 15% and 25% caps, so the equal-share column is the same at both caps. The LP gain is smaller at 15% because the cap binds.
- SCA optimizes under the LP objective (via SCA steps), so it consistently beats proxies that do not.

---

## 4. Solver validation

### Spot validate (`spot_validate_8.8mhz.json`, seed 1, max_iterations=30)


| Config           | CVXPY       | MATLAB      | abs_obj_diff (bit/s) | se_max_abs_diff |
| ---------------- | ----------- | ----------- | -------------------- | --------------- |
| 8.8 MHz, no cap  | 8.9677 Mbps | 8.9677 Mbps | **0**                | ~1e-15          |
| 8.8 MHz, cap 25% | 8.9600 Mbps | 8.9600 Mbps | **0.63**             | ~1e-15          |


Both configs: `agreement: ok`. Cap-25 seed 1 matches the campaign CVXPY rate exactly (8.960041 Mbps).

### Default point — 20 paired seeds (J = 3, 25% primary)

**Source:** `scripts/compare_sca_cvxpy_matlab.py` → `results/sca_cvxpy_vs_matlab_j3.json`


| Quantity                | Value                           |
| ----------------------- | ------------------------------- |
| Mean CVXPY              | **8.946 Mbps**                  |
| Mean MATLAB             | **8.949 Mbps**                  |
| Mean Δ (MATLAB − CVXPY) | **+0.003 Mbps**                 |
| Max |Δ|                 | **0.026 Mbps** (seeds 2, 7)     |
| Median |Δ|              | **~0** (17/20 within 1e-5 Mbps) |
| Wilcoxon p (20 paired)  | **0.91** (not significant)      |
| Feasible both           | **20/20**                       |


Differences are **path-dependent** (MATLAB sometimes takes more accepted position steps), not a channel mismatch. Campaign tables stay on CVXPY. This 20-seed MATLAB pair was run at the **25% primary cap**; seed-1 spot-validate covers no-cap and 25% (`agreement: ok`).

---

## 5. Sensibility checklist


| Check                               | Result                                                          |
| ----------------------------------- | --------------------------------------------------------------- |
| Unit tests                          | Pass                                                            |
| 20 kHz infeasible                   | Yes — 0% at 100 m and 500 m; model-free cap 0.997 Mbps (§0)     |
| Rates ≤ bandwidth ceiling           | Yes — ~8.97 Mbps (no cap), ~8.90 (cap15), ~8.95 (cap25), ~2.43 at 2.4 MHz |
| SCA best on feasible points (mean)  | Yes                                                             |
| λ / CPU vary slightly when feasible | Yes — < 0.03 Mbps                                               |
| T_k = 0.8 infeasible                | **No** as a model limit — process-cohesive \(a_{ij}\) feasible **40/40** at k-means \(q\) (seeds 1–40); frozen SCA and best-SE SCA-joint **0/20**; SCA-joint + cohesive candidate **20/20** (§2.4, §2.8, §6) |
| More UAVs help under cap (J = 1→3)  | Yes                                                             |
| Method collapse without cap         | Yes — expected saturation                                       |
| Paired stats + FDR documented       | Yes                                                             |
| Eq. (17) implementation             | **Resolved** — matches published formula (§8.1)                 |
| Eq. (17) vs Fig. 11 narrative       | **Open (paper intent)** — not a code defect (§8.1)              |


> **Verdict:** Results are **sensible and sufficient to move on** to writing and figures. They are **not** a numeric reproduction of the paper’s §VII curves — that gap is **expected and documented**, not a failure of this repo.

---

## 6. Known limitations (this milestone)

Three published-looking “infeasibility” claims were artifacts. **All three are closed** with campaign-scale evidence. Frozen nearest-UAV \(a_{ij}\) is not an open model question.

### False alarms *(resolved — do not publish as model limits)*


| #   | Claim that looked like a model limit                     | What it actually was                                       | Status |
| --- | -------------------------------------------------------- | ---------------------------------------------------------- | ------ |
| 1   | Settled S_i/L vs placeholders: AoDT not binding, μ=200/s | External task size/cycles were placeholders                | **Resolved** — S_i=12,000 bytes, L=3.75×10⁶, μ≈53.3/s; AoDT binds at T_k=2.8 s |
| 2   | I = 28–32 and f_j=0.5×10⁸ “infeasible” (55/60/65%)       | Majority-of-association b_ij put both processes on one UAV | **Resolved** — rematch process→UAV when (24) fails; 20/20 feasible |
| 3   | T_k=0.8 s “too tight for the model” (0%)                 | Frozen nearest \(a_{ij}\) splits \(N_k\); forwarded links have \(Q+T_{\mathrm{u2u}}\approx 0.894>0.8\) even at infinite rate | **Resolved as a model claim.** Centroid-cohesive \(a_{ij}\) at k-means \(q\): **40/40** feasible (campaign 1–20 + held-out 21–40; `tk08_cohesive_construction.json`). J = 1: 20/20. Frozen SCA and best-SE SCA-joint: **0/20**. SCA-joint + process-cohesive rematch candidate: **20/20** (`tk08_scajoint_cohesive.json`). The campaign 0% is the discrete init, not Problem (P). |


### Sequential SCA and association *(measured)*

Algorithm 1 as implemented freezes \(a_{ij}\) and \(b_{ij}\) after nearest-UAV / CPU-stable init. That used to sit here as a soft “scoped to this solver” caveat backed by one constructed point. The SCA-joint probe (§2.8) measured it on the primary 25% grid (20 seeds, all axes).

| Test | Result |
| --- | --- |
| Is Problem (P) feasible at \(T_k=0.8\) s? | **Yes.** Process-cohesive \(a_{ij}\) at k-means \(q\): **40/40** (seeds 1–20 and held-out 21–40). J = 1: 20/20. Slack sign (\(T_k-Q=+0.206\) s vs \(T_k-Q-T_{\mathrm{u2u}}=-0.094\) s) is geometry-independent; the bandwidth LP still cleared all 40 geometries. |
| Frozen SCA feasible at \(T_k=0.8\) s? | **0/20** |
| SCA-joint (best-SE rematch) feasible at \(T_k=0.8\) s? | **0/20** (0 rematch accepts). Best-SE coincides with nearest at that init, so the discrete step is a no-op. |
| SCA-joint + process-cohesive candidate feasible at \(T_k=0.8\) s? | **20/20**. Accepts `process_cohesive` (never best-SE); 0 forwarding at termination. Confirms the blocker was “greedy never proposes grouping,” not an LP/evaluate bug. |
| Radio cost of grouping at fixed \(q\) | **~0.6 Mbps** median (0.57), mean 0.59, p10–p90 **0.38–0.81**, range **0.29–0.98** over **40/40** geometries. Spread tracks \(\sum\Delta\mathrm{SE}\) (\(r\approx 0.92\)), not IoT count off best link (\(r\approx 0.14\); see seed 16 vs 23 in §2.8) |
| Do feasible-axis Mbps move if association is unfrozen (best-SE)? | **No, not practically.** 29 sweep points; max Δ **+0.006 Mbps** (default J = 3); **0/29** above the 0.05 Mbps bar; **0/29** feasibility differences. |
| Runtime cost of best-SE rematch (default J = 3) | Frozen 0.891 s / 27.7 iters vs joint 0.888 s / 27.9 iters (**1.00×**). |

**Model fact:** Problem (P) is feasible at \(T_k=0.8\) s on this scenario, including held-out placements. Sequential SCA’s 0% is a nearest-UAV / best-SE coincidence — not a remaining scope note.

**Headline tables:** unfreezing association with best-SE rematch does not change frozen-SCA Mbps. The cohesive-candidate flag is a hypothesis test, not a replacement solver. SCA-joint vs random/k-means/PSO is not a replacement comparison. TD3’s binaries-frozen decision is independent of this probe.

### Other *(by design / out of scope / audit notes)*


| Item                          | Status                                                    |
| ----------------------------- | --------------------------------------------------------- |
| Area 100 × 100 m (headline)   | **By design.** 20 kHz also checked at 500 × 500 m (§0.3)  |
| TD3 (Algorithm 2)             | Not implemented                                           |
| Paper Mbps targets            | Explicitly not pursued                                    |
| 20 kHz as (27) cap            | **Real** model infeasibility — not a solver artifact (§0) |
| 15% per-link cap (sensitivity) | Search-selected from cap×J grid; FDR-sig at all J but practical Δ fails at J = 4–5 (§2.7) |
| 25% per-link cap (primary)     | Leftover-dump stress test; vs-random n.s. at J = 3; sits in the 23.5–24% FDR crossover (§2.7) |
| PSO                           | External baseline; equal-share inner fitness              |
| Eq. (17) vs Fig. 11 narrative | **Open (paper intent)** — not a code bug (§8.1)           |
| Eq. (4) angle unit            | **Checked.** Default radians; deg reading gives SNR ≈ 90 at zenith and ~57 Mbps at J = 3 (§0.2). Radians adopted. |
| Idle compute UAVs under majority \(b_{ij}\) | **Checked, default scenario.** Constraint (23) allows one processing UAV per process; with K = 2, J = 3, majority vote often puts both processes on one UAV. **20/20** default seeds have ≥1 UAV with **zero** processing load (mean **1.35** idle compute UAVs); all three UAVs still carry associated uplink traffic. Legal under (23)+(24); `cpu_stable_processing` unchanged when majority is already stable. Does not move headline Mbps tables. |
| SCA `stop_reason` when `accepted_steps = 0` | **Documented nit.** If the iteration cap is hit with zero accepted moves and `step_size` still above `min_step_size`, Python reports `MAX_ITERATIONS` rather than `STEP_SIZE_LIMIT` (`classify_stop_reason` explicitly exempts that case). Does not affect scoring; uncapped runs that never accept a step may show `MAX_ITERATIONS` in diagnostics. |


---

## 7. Suggested writeup sentences (copy-ready)

### Audit (short)

> Eq. (6) and constraint (27) imply R_{\mathrm{sum}}\le B_{\mathrm{sys}}\log_2(1+\mathrm{SNR}*{\max}). With B*{\mathrm{sys}}=20\mathrm{kHz} this is at most 0.997 Mbps even at \mathrm{SNR}=10^{15}, so Figs. 6–10 (7–14 Mbps, and Fig. 7 increasing with I) rule out reading Table II’s 20 kHz as the (27) sum cap. If the table’s “Minimum bandwidth allocation” is instead a per-link floor, matching Fig. 6’s 8.8 Mbps at I=10 on the written channel (\mathrm{SNR}\approx 1.03) needs ~862 kHz per link — **43×** the stated 20 kHz — or an undisclosed (27) cap of ~8.6 MHz. Under the cap reading, 20 kHz is 0% feasible at both 100 × 100 m and 500 × 500 m. At feasible bandwidths the primary leftover-dump stress test is a **25%** per-link cap (not part of Problem (P)). At default J = 3, SCA vs random is not significant (Wilcoxon p = 0.123); SCA vs k-means and vs PSO remain significant. A **15%** cap is retained as a tighter-cap sensitivity (search-selected; SCA vs random Bonferroni p_adj &lt; 0.001 at J = 3), not as an independently chosen ranking config.

### Headline result (cap 25% primary, J = 3)

> SCA achieves 8.946 Mbps mean sum rate vs 8.928 (random), 8.901 (k-means), and 8.905 (PSO). Paired vs random is +0.019 ± 0.042 Mbps, 15/20, Wilcoxon p = 0.123. Vs k-means and vs PSO the paired tests remain significant (20/20 and 19/20, both p &lt; 0.001).

### T_k = 0.8 s is not a Problem (P) limit

> At \(T_k=0.8\) s the sequential SCA campaign is 0/20 feasible under nearest-UAV association, because splitting \(N_k\) forces a \(T_{\mathrm{u2u}}\) hop and \(Q+T_{\mathrm{u2u}}\approx 0.894>0.8\). Hand construction and a 40-geometry check showed Problem (P) is still feasible (process-cohesive \(a_{ij}\) at k-means \(q\), 40/40). Best-SE SCA-joint did not find that map (0/20); adding a process-cohesive rematch candidate did (20/20). At fixed \(q\), grouping trades **~0.6 Mbps** median sum rate (p10–p90 ~0.38–0.81 across 40 seeds) for meeting the tight deadline — a direct, quantified instance of the accuracy–synchronization tradeoff the paper motivates in §I. Do not publish the 0% column as a model limit; do publish the frozen-SCA Mbps tables as the headline.

### Accuracy–synchronization tradeoff (§I, quantified)

> In our reproduction, keeping each process group on one UAV (no \(T_{\mathrm{u2u}}\)) makes \(T_k=0.8\) s feasible where nearest-UAV association fails, at a median cost of **~0.57 Mbps** sum rate at the same k-means geometry (mean 0.59, 40 seeds). This is an experimental confirmation of the rate-vs-AoDT tension stated in the paper’s introduction — not a solver footnote.

### Tighter-cap sensitivity (cap 15%, J = 3)

> At 15% the same comparison is SCA 8.895 vs random 8.765 Mbps (+0.129 ± 0.187, 20/20, Wilcoxon p &lt; 0.001). Report 15% as a search-selected sensitivity, not as if it were chosen like the 100 × 100 m field or the settled S_i/L defaults.

### Comment to authors / editor (Table II B_sys)

> Eq. (6) and constraint (27) imply that the sum rate cannot exceed B_{\mathrm{sys}}\log_2(1+\mathrm{SNR}*{\max}). Table II lists B*{\mathrm{sys}}=20{,}000 Hz. Even at \mathrm{SNR}=10^{15} that ceiling is 0.997 Mbps, while Figs. 6–10 report 7–14 Mbps. Those axes rule out using 20 kHz as the total uplink cap in (27). Could the authors confirm whether Table II’s 20 kHz is the (27) sum cap, a per-link floor, or a typographical error?

### Eq. (17) vs Fig. 11 narrative

> The paper states heterogeneous within-group arrival rates should lie between uniform fast and uniform slow. Our queue simulator shows fast < heterogeneous < slow, but Eq. (17) — the score inside Problem (P) — ranks heterogeneous **above** uniform-slow because the formula is slowest-λ-limited yet penalizes total load via Σλ. **Implementation is resolved** (formula behaviour, not a code defect). **Paper intent remains open.**

---

## 8. AoDT extras (Eqs. (10)–(17), FCFS / FCFS-P / LCFS-S, Fig. 11)

Problem (P) still scores **Eq. (17)**. This pass adds the rest of the paper’s AoDT block except TD3:

- Instantaneous age (10)
- Neglected download (12) as `Z = 0`
- LCFS-S closed forms (14)–(15)
- Kaul M/M/1 **FCFS** AoI
- Event-driven **FCFS**, **FCFS-P**, **LCFS-S**

**Single-source unit tests (ρ = 0.6):** LCFS-S matches Eq. (14); FCFS matches the Kaul formula; LCFS-S age < FCFS-P ≤ FCFS.

**Multi-source process age is not Eq. (17).** Eq. (17) is `max D_i + (1/λ_min)(1 + Σλ/μ)`. The simulator’s process age is the time-average of `max_i ζ_i(t)` (Eq. (10) over the group).

### 8.1 Fig. 11 (I = 10, k-means, 8.8 MHz, 25% primary, 20 seeds)

**Primary file:** `results/fig11_8.8mhz_cap25_si12k.json` (settled `S_i`/`L`). Equal-share `B_sys/I = 0.88` MHz/link is below both the 15% and 25% caps, so Fig. 11 numbers match `fig11_8.8mhz_cap15.json`.

#### Bandwidth convention


| Context               | Bandwidth model       | J = 3 sum rate | AoDT                 |
| --------------------- | --------------------- | -------------- | -------------------- |
| Fig. 11 `evaluate()`  | Equal-share           | 8.69 Mbps      | Eq. (17) ≈ 1.0–1.9 s |
| Campaign k-means (LP) | Frozen-q bandwidth LP | 8.84 Mbps      | Binds at T_k = 2.8 s |


λ does not enter Eq. (20); rates are identical across Fig. 11 patterns at fixed J.

#### Fig. 11 sensibility gate (50 / 50)

Automated checks in `sensibility_checks()` (`src/uavdt/experiments/fig11.py`), 10 per UAV count J ∈ {1,…,5}. Passing means the AoDT implementation and λ patterns behave coherently; it does **not** certify agreement with unpublished paper curves.

#### Formula vs narrative *(second audit finding)*


| Evaluator                        | J = 3 ordering                                      | Matches paper narrative? |
| -------------------------------- | --------------------------------------------------- | ------------------------ |
| **FCFS simulator** (process-max) | fast (1.46 s) < hetero (2.16 s) < slow (3.13 s)     | **Yes**                  |
| **Eq. (17)** (Problem P score)   | fast (1.00 s) < slow (1.74 s) **< hetero (1.89 s)** | **No**                   |


**Mechanism:** Eq. (17) uses `λ_min` (0.8/s) but the queue term `(1 + Σλ/μ)/λ_min` grows with total load. Heterogeneous mixes fast and slow sources, so Σλ exceeds uniform-slow even though λ_min is unchanged.


| S_i/L            | Eq. (17) fast | Eq. (17) slow | Eq. (17) hetero | hetero − slow |
| ---------------- | ------------- | ------------- | --------------- | ------------- |
| Old placeholders | 0.83 s        | 1.57 s        | 1.62 s          | +0.05 s       |
| Settled defaults | 1.00 s        | 1.74 s        | 1.89 s          | **+0.14 s**   |


#### Status split


| Question                                          | Status       | Notes                                                    |
| ------------------------------------------------- | ------------ | -------------------------------------------------------- |
| Is Eq. (17) implemented correctly?                | **Resolved** | `average_aodt_s()` matches published formula; tests pass |
| Does Eq. (17) match Fig. 11 narrative?            | **No**       | Formula ranks hetero worse than uniform-slow             |
| Did the authors intend (17) given that narrative? | **Open**     | Outside this repo                                        |


#### Default J = 3 results (mean over 20 seeds)


| Pattern                     | Eq. (17) max (s) | Eq. (15) max (s) | Sim source FCFS (s) | Sim process-max FCFS (s) | Sum rate (Mbps) |
| --------------------------- | ---------------- | ---------------- | ------------------- | ------------------------ | --------------- |
| Uniform fast (λ₁=2, λ₂=3)   | 0.998            | 0.923            | 0.694               | 1.455                    | 8.692           |
| Heterogeneous (0.8…3)       | 1.886            | 1.682            | 0.932               | 2.155                    | 8.692           |
| Uniform slow (λ₁=0.8, λ₂=1) | 1.743            | 1.668            | 1.403               | 3.131                    | 8.692           |


Fig. 11 sensibility gate: **50/50** pass (2026-09-05).

**What aligns with the paper:** simulator ordering fast < heterogeneous < slow at every J = 1…5; sum rate identical across λ patterns at fixed J.

**What does not:** Eq. (17) ranks heterogeneous above uniform-slow, contrary to the paper’s stated story.

```bash
python -m uavdt aodt-compare --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --placement kmeans
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --out results/fig11_8.8mhz_cap25_si12k.json
```

---

## 9. How to regenerate

```powershell
$env:PYTHONPATH="src"

python -m pytest
python scripts/check_bsys_20khz.py
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --solver cvxpy --out results/campaign_8.8mhz_cap25_si12k.json
python scripts/rerun_init_repair_points.py
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/analyze_sca_vs_random_losses.py results/campaign_8.8mhz_cap25_si12k.json
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --out results/fig11_8.8mhz_cap25_si12k.json
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-iterations 30
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --max-iterations 30
python scripts/analyze_campaigns.py
python scripts/plot_paper_figures.py

# Tighter-cap sensitivity (100 m, 15% cap) — optional
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.15 --n-runs 20 --solver cvxpy --out results/campaign_8.8mhz_cap15_n20.json
python scripts/paired_winrate.py results/campaign_8.8mhz_cap15_n20.json
python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap15_n20.json --fig11 results/fig11_8.8mhz_cap15.json --out-dir results/figures/cap15

# Paper field (500 × 500 m) — field-size test, same 25% primary cap
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --solver cvxpy --area-m 500 --out results/campaign_8.8mhz_cap25_si12k_500m.json
python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap25_si12k_500m.json --out-dir results/figures/500m

# SCA-joint methodology probe (does not overwrite frozen-SCA campaign JSON)
python scripts/run_sca_joint_campaign.py
python scripts/run_tk08_followup.py
```

Full bandwidth preset sweep (long): `scripts/run_all_bandwidth_campaigns.ps1`

One-command replay of §9 (except bandwidth sweep): `scripts/run_full_regeneration.ps1`

### Run timings (representative)


| Step                             | Duration |
| -------------------------------- | -------- |
| `pytest` (125 tests)             | ~4 s     |
| Primary 25% campaign (100 m)     | ~9 min   |
| Cap15 campaign (100 m)           | ~11 min  |
| 500 m field campaign             | ~11 min  |
| SCA-joint full-axis probe        | ~10 min  |
| tk08 follow-ups (3 steps)        | ~1 min   |
| Init-repair                      | ~1.5 min |
| Analysis + fig11 + plots         | < 3 min  |
| Full bandwidth sweep (7 configs) | ~36 min  |


**Logs:** `results/campaign_8.8mhz_cap25_si12k_run.log`, `results/rerun_init_repair.log`, `results/check_bsys_latest.log`, `results/spot_validate_nocap.log`, `results/spot_validate_cap25.log`, `results/run_all_bandwidth_campaigns.log`

---

## 10. S_i / L correction — settled defaults vs old placeholders

### Problem

`config.py` previously used `S_i = 10{,}000` **bit** (~1,250 bytes) and `L = 10^6` cycles (`μ = 200` /s). Settled choices: **12,000 bytes** and **3.75×10⁶** cycles (`μ ≈ 53.3` /s).

### Fix

`EXTERNAL_TASK_SIZE_BYTES = 12_000`, `EXTERNAL_TASK_SIZE_BITS = 96_000`, `EXTERNAL_TASK_CYCLES = 3.75e6` in `src/uavdt/config.py`.

### Side-by-side @ J=3

`scripts/compare_si_l_defaults.py` → `results/compare_si_l_defaults.json`


| Quantity                  | Old placeholders | Settled defaults | Better?                                    |
| ------------------------- | ---------------- | ---------------- | ------------------------------------------ |
| S_i                       | 1,250 bytes      | **12,000 bytes** | Matches experimental choice                |
| L                         | 1×10⁶ cycles     | **3.75×10⁶**     | Matches experimental choice                |
| μ                         | 200 /s           | **53.3 /s**      | Heavier queue; AoDT more meaningful        |
| Mean max AoDT @ J=3 (SCA) | ~1.83 s          | **~2.80 s**      | **Yes** — sits at T_k (binding)            |
| SCA mean Mbps             | 8.964            | 8.946            | Slightly lower (more B to AoDT floors)     |
| Random mean Mbps          | 8.950            | 8.928            | Same pattern                               |
| SCA − random (Mbps)       | +0.014           | **+0.018**       | Slightly wider edge                        |
| SCA − k-means (Mbps)      | +0.055           | +0.046           | Slightly narrower                          |
| Feasible @ 8.8 MHz cap25  | 100%             | 100%             | Unchanged                                  |
| 20 kHz feasible (5 seeds) | 0/5              | 0/5              | Unchanged                                  |
| Fig. 11 sensibility (J=3) | 10/10            | 10/10            | Eq. (17) hetero−slow gap +0.05→**+0.14 s** |


> **Verdict:** Settled S_i/L are **better for model fidelity** — Eq. (17) now **binds at** T_k = 2.8 s. Sum rates drop ~0.02 Mbps; **method ranking is unchanged**. This does **not** close the gap to paper §VII Mbps.

```bash
python scripts/compare_si_l_defaults.py
python -m uavdt campaign --axis uavs --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --out results/campaign_8.8mhz_cap25_si12k.json
```

