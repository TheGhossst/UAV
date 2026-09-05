# Current results — Khalaf et al. (IEEE TNSM, 2026) reproduction

This document records **our** experimental outcomes on the fresh `uavdt` simulator.
It does **not** compare Mbps figures to the paper’s §VII plots. Table II’s
`B_sys = 20 kHz` is infeasible under the stated model; the paper’s 7–14 Mbps
curves are treated as **not credible** outputs of that parameter set (see audit
below).

**Area:** **100 × 100 m** (guide requirement; paper text uses 500 × 500 m).
**Seeds:** 20 consecutive seeds per sweep point (`seed_start = 1`).
**Methods:** SCA, random, k-means, PSO (PSO is external, not in the paper).
**Score:** Python `evaluate()` on every method. Placement baselines are
re-scored with the same frozen-geometry bandwidth LP as SCA’s B step.
**SCA backend:** CVXPY for campaigns; MATLAB CVX+MOSEK spot-validated.

Ledger for how to re-run: `docs/EXPERIMENTS.md`. Parameter ledger:
`docs/REPRODUCTION.md`, `docs/param.md`.

---

## Audit paragraph (paper §VII vs this implementation)

We implemented the Khalaf–Itani–Sharafeddine UAV-aided digital-twin IoT model
(Eqs. (1)–(6), (11), (13), (16)–(17), Problem (P)) in a **100 × 100 m** field
with faithful Table II radio/compute parameters except bandwidth and two
external task quantities (`S_i`, `L`) needed to evaluate upload delay and
service rate. Under **Table II `B_sys = 20 kHz`**, every method fails QoS and
AoDT on every sweep point: feasible fraction **0%**, sum rate stuck at
**~0.02 Mbps** (the hard bandwidth ceiling). That outcome is expected: the
model’s minimum-rate and AoDT floors require far more spectrum than 20 kHz can
provide, and Shannon-style bounds forbid multi-Mbps throughput at that
bandwidth. We therefore **do not treat the paper’s Fig. 6–10 Mbps curves as a
reproduction target**; they are inconsistent with the paper’s own equations
and Table II when evaluated faithfully. For **feasible** bandwidths (2.4 MHz and
8.8 MHz), behaviour is internally coherent: at 8.8 MHz **without** a per-link
cap, all methods saturate near **~8.98 Mbps** (leftover spectrum piles onto the
best link); with a **25% per-link cap** (external parameter, not Problem (P)),
methods separate and **SCA beats k-means and PSO** on paired seeds with FDR
q < 0.05 on **25/25** and **24/25** unique sweep points respectively; the edge
over **random** is **regime-limited** (large at J = 1–2 and high I, not
significant at the default J = 3 point). The
credible finding is **model-consistent placement comparison under feasible
spectrum**, not numeric agreement with §VII.

**Second audit finding (Eq. (17) vs Fig. 11 narrative):** Problem (P) scores
process AoDT with Eq. (17), `max_i D_i + (1/λ_min)(1 + Σλ/μ)`. The paper’s
Fig. 11 text says heterogeneous within-group λ should sit **between** uniform
fast and uniform slow. Under faithful evaluation, the **event simulator** shows
that ordering (fast &lt; heterogeneous &lt; slow), but **Eq. (17) does not**:
heterogeneous scores **worse than uniform-slow** because the `Σλ` term rises
while `λ_min` stays at the slow end (0.8/s). **Implementation: resolved** —
Eq. (17) is coded as published (`aodt.py`, `evaluate()`), confirmed by unit
tests and Fig. 11 sensibility (50/50); this is a property of the written
formula, **not a code defect** — do not hunt for an implementation bug here.
**Paper intent: open** — whether (17) is what the authors physically intended
given Fig. 11’s fast &lt; hetero &lt; slow prose. SCA optimizes (17) as written,
so the objective does **not** behave the way the narrative figure describes.
The hetero−slow gap **grew** under settled `S_i`/`L` (see §8.1).

---

## Campaign inventory

| File | `B_sys` | Per-link cap | `n_runs` | Role |
| --- | --- | --- | --- | --- |
| `results/campaign_20khz.json` | 20 kHz | none | 20 | Table II diagnostic (infeasible) |
| `results/campaign_20khz_cap25.json` | 20 kHz | 25% | 20 | Same, capped |
| `results/campaign_2.4mhz.json` | 2.4 MHz | none | 20 | Mid-bandwidth, saturated |
| `results/campaign_2.4mhz_cap25.json` | 2.4 MHz | 25% | 20 | Mid-bandwidth, differentiated |
| `results/campaign_8.8mhz_n20.json` | 8.8 MHz | none | 20 | Headline, saturated |
| `results/campaign_8.8mhz_cap25_si12k.json` | 8.8 MHz | 25% | 20 | **Primary comparison config** (settled `S_i`/`L`) |
| `results/campaign_20260904_cap25.json` | 8.8 MHz | 25% | 20 | Superseded — old `S_i`/`L` placeholders |
| `results/compare_si_l_defaults.json` | 8.8 MHz | 25% | 20 | Old vs settled side-by-side @ J=3 |

Supporting artifacts:

- `results/campaign_8.8mhz_cap25_si12k_paired.json` / `.csv` — paired SCA vs baselines
- `results/campaign_8.8mhz_cap25_si12k_losses.json` — loss-seed forensics (J = 3)
- `results/campaign_20260904_cap25_paired.json` / `.csv` — paired stats (old `S_i`/`L`)
- `results/spot_validate_8.8mhz.json` — CVXPY vs MATLAB MOSEK (agreement: ok)
- `results/fig11_8.8mhz_cap25_si12k.json` / `.csv` — Fig. 11 λ patterns, settled `S_i`/`L`
- `results/fig11_8.8mhz_cap25.json` / `.csv` — Fig. 11 (old `S_i`/`L` placeholders)

Analysis scripts: `scripts/paired_winrate.py`, `scripts/analyze_sca_vs_random_losses.py`,
`scripts/analyze_campaigns.py`.

---

## Default scenario (headline point)

Unless noted, this is **I = 10**, **J = 3**, **λ = 2/s**, **T_k = 2.8 s**,
**f_j = 2×10⁸ cycles/s**, **K = 2** process groups with **5 IoTs each**,
**S_i = 12,000 bytes**, **L = 3.75×10⁶** cycles/task.

---

## 1. Bandwidth sweep (qualitative)

### 1.1 20 kHz (Table II value)

| Observation | Value |
| --- | --- |
| Feasible fraction (all methods, all points) | **0%** |
| Typical sum rate | **~0.019–0.020 Mbps** |
| Cap vs no-cap | **No difference** |

**Interpretation:** Total spectrum is the bottleneck. QoS (`R_min = 10 kbps`
per active link) and AoDT cannot be met simultaneously. This is the strongest
single check that §VII Mbps plots are not produced by this model + Table II.

**`S_i` / `L` defaults (settled):** `S_i = 12{,}000` bytes (`96{,}000` bit),
`L = 3.75\times10^6` cycles/task (`\mu \approx 53.3` /s). Not in Table II.
Prior placeholder runs used `10{,}000` bit and `L=10^6`; see §10. The primary
campaign `campaign_8.8mhz_cap25_si12k.json` uses the settled values.

1. **Shannon ceiling (Eq. 6):** at 20 kHz the whole system is bounded at
   **~0.13 Mbps**, independent of `S_i` and `L`. That alone rules out 7–14 Mbps.
2. **QoS floor (Eq. 25):** meeting `R_min` on ten associated links needs on the
   order of **~102 kHz** total (`∑_i R_min / SE_ij` at a typical geometry) —
   also independent of `S_i`/`L`.
3. **Empirical:** `campaign_20khz*.json` — 0% feasible at the defaults above.

`S_i` only enters upload delay (Eq. 11) and thus the AoDT bandwidth floor; making
`S_i` smaller could ease AoDT but would not relax the QoS or Shannon limits.
`L` only scales `μ_j`; it does not enter the radio model. A formal grid over
`S_i` and `L` is still **open** if we want to quote a feasible `(S_i, L)` region
at 20 kHz, but none is expected while (1)–(2) hold.

### 1.2 2.4 MHz

| Config | SCA @ J=3 (Mbps) | All methods feasible? |
| --- | --- | --- |
| No cap | 2.447 | Yes (100%) |
| Cap 25% | 2.442 | Yes (100%) |

Methods differ by **&lt; 0.01 Mbps** without cap (bandwidth-saturated). With cap,
SCA leads k-means/PSO by **~0.05 Mbps** at J = 3; ordering matches 8.8 MHz cap25
but at lower absolute rates (~2.45 Mbps ceiling).

### 1.3 8.8 MHz, no cap

| Method @ J=3 | Mean Mbps | Feasible |
| --- | --- | --- |
| SCA | 8.984 | 100% |
| Random | 8.980 | 100% |
| K-means | 8.958 | 100% |
| PSO | 8.952 | 100% |

Spread **&lt; 0.04 Mbps**. **λ**, **CPU**, and most **AoDT** points (when feasible)
give **identical** per-seed rates — the objective is communication-limited only.
**Not useful for ranking methods.**

### 1.4 8.8 MHz, 25% cap (primary)

| Method @ J=3 | Mean Mbps | Feasible |
| --- | --- | --- |
| **SCA** | **8.946** | 100% |
| Random | 8.928 | 100% |
| K-means | 8.901 | 100% |
| PSO | 8.905 | 100% |

Method separation is clear. This is the configuration used for defensible
placement comparisons below.

---

## 2. Sweep axes at 8.8 MHz, 25% cap (`campaign_8.8mhz_cap25_si12k.json`)

### 2.1 Fig. 6 analogue — UAV count J (I = 10)

| J | SCA | Random | K-means | PSO |
| --- | --- | --- | --- | --- |
| 1 | 8.758 | 8.500 | 8.565 | 8.597 |
| 2 | 8.886 | 8.798 | 8.790 | 8.848 |
| 3 | 8.946 | 8.928 | 8.901 | 8.905 |
| 4 | 8.971 | 8.966 | 8.939 | 8.942 |
| 5 | 8.977 | 8.975 | 8.964 | 8.963 |

**Analysis:**

- **J = 1–3:** SCA gains the most when UAVs are scarce and the 25% cap binds —
  spatial placement and joint B allocation matter.
- **J = 4–5:** Rates converge; random nearly matches SCA (~0.001–0.006 Mbps gap).
  Extra UAVs add spatial reuse slack; placement optimization buys little.
- Monotonic **non-decreasing** sum rate with J for all methods — sensible.

### 2.2 Fig. 7 analogue — IoT count I (J = 3)

| I | SCA | Random | K-means | PSO | SCA feasible |
| --- | --- | --- | --- | --- | --- |
| 10 | 8.946 | 8.928 | 8.901 | 8.905 | 100% |
| 16 | 8.942 | 8.911 | 8.893 | 8.899 | 100% |
| 20 | 8.934 | 8.898 | 8.898 | 8.912 | 100% |
| 24 | 8.925 | 8.881 | 8.891 | 8.899 | 100% |
| 28 | 8.908 | 8.861 | 8.895 | 8.900 | **55%** |
| 32 | 8.898 | 8.840 | 8.883 | 8.889 | **60%** |

**Analysis:** SCA stays **~0.01–0.06 Mbps** above k-means/PSO as I grows; random
closes part of the gap. At **I = 28–32** the heavier task load (`μ ≈ 53.3` /s)
causes partial infeasibility (SCA init can fail on ~9/20 seeds at I = 28).
SCA still wins paired tests on I = 16–32 vs k-means/PSO and vs random on
I = 16–32 (FDR q &lt; 0.05).

### 2.3 Fig. 8 analogue — Arrival rate λ (I = 10, J = 3)

| λ (/s) | SCA | Random | K-means | PSO |
| --- | --- | --- | --- | --- |
| 1.0 | 8.944 | 8.918 | 8.898 | 8.902 |
| 1.5 | 8.947 | 8.925 | 8.900 | 8.904 |
| 2.0 | 8.946 | 8.928 | 8.901 | 8.905 |
| 2.5 | 8.947 | 8.929 | 8.901 | 8.905 |
| 3.0 | 8.948 | 8.930 | 8.902 | 8.905 |
| 3.5 | 8.948 | 8.930 | 8.902 | 8.905 |

**Analysis:** With settled `L`, AoDT binds at `T_k = 2.8 s` and the comm score
**varies slightly** with λ (placement affects AoDT bandwidth floors). Differences
are **&lt; 0.03 Mbps** — still small, but no longer bit-identical across λ.
**Do not pool λ rows with J = 3 in meta-analyses** (they are repeated-default
copies at λ = 2.0 only).

### 2.4 Fig. 9 analogue — AoDT threshold T_k (I = 10, J = 3)

| T_k (s) | SCA Mbps | SCA feasible | Notes |
| --- | --- | --- | --- |
| **0.8** | 8.909 | **0%** | Too tight for all methods |
| 1.2 | 8.889 | 100% | |
| 1.6 | 8.925 | 100% | |
| 2.0 | 8.938 | 100% | |
| 2.4 | 8.945 | 100% | |
| 2.8 – 3.0 | 8.946 – 8.948 | 100% | Plateau at default |

**Analysis:** T_k = 0.8 s is correctly infeasible. For T_k ≥ 1.2 s the comm
objective rises toward the default plateau as the AoDT floor relaxes. At
T_k = 0.8, reported Mbps are **not** feasible QoS/AoDT solutions — treat as
diagnostic only.

### 2.5 Fig. 10 analogue — UAV CPU f_j (I = 10, J = 3)

| f_j (×10⁸ c/s) | SCA Mbps | SCA feasible |
| --- | --- | --- |
| **0.5** | 8.928 | **65%** |
| 1.0 | 8.948 | 100% |
| 1.5 – 2.5 | 8.946 – 8.947 | 100% |

**Analysis:** At **f_j = 0.5×10⁸** the lower service rate (`μ ≈ 53.3` /s with
settled `L`) makes 35% of seeds infeasible. Above 1.0×10⁸ the comm score
varies only slightly (&lt; 0.02 Mbps) when feasible. **Do not pool CPU rows
with J = 3 in meta-analyses** (λ = 2.0 / f_j = 2.0 duplicate the default).

---

## 3. Paired statistics (SCA vs baselines, same seed)

Source: `campaign_8.8mhz_cap25_si12k_paired.json`. Delta = SCA − baseline (Mbps).
Primary test: **Wilcoxon signed-rank** (two-sided). `+/-` in quotes is **sample
std of 20 paired deltas**, not SEM.

### 3.1 Default point — J = 3, I = 10

| Baseline | Mean Δ | Std Δ | Wins | Wilcoxon p | Bonferroni p_adj (m=3) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Random | +0.019 | 0.042 | 15/20 | 0.123 | **0.369** | Not significant |
| K-means | +0.046 | 0.019 | 20/20 | &lt;0.001 | &lt;0.001 | **Significant** |
| PSO | +0.042 | 0.026 | 19/20 | &lt;0.001 | &lt;0.001 | **Significant** |

**Wilcoxon vs paired t at J = 3 vs random:** Wilcoxon p = 0.123, paired t
p = 0.062 on the **same** 20 deltas. This is a **shape** issue: 15/20 positive
signs, but mean (0.019 Mbps) sits inside a wide spread (std 0.042). Distribution
is right-skewed (skew ≈ +1.15); largest outlier is a **win** (seed 3, +0.130 Mbps),
not a catastrophic SCA failure.

**Five losing seeds vs random (7, 11, 13, 18, 19):** Non-adjacent seeds; SCA
**CONVERGED** on four and hit **MAX_ITERATIONS** on seed 7; SCA still beat
k-means on each. Random drew a better capped-LP geometry; SCA initializes from
k-means and does not search the random basin.

### 3.2 Regime map — BH-FDR over 25 unique sweep points

Repeated default-scenario copies (λ = 2.0 and f_j = 2.0 duplicate of J = 3;
I = 10 duplicate of J = 3) excluded.

| Baseline | Points with FDR q &lt; 0.05 | UAV sweep |
| --- | --- | --- |
| **K-means** | **25 / 25** | J = 1…5 all significant |
| **PSO** | **24 / 25** | All J and I except I = 28 (n.s.) |
| **Random** | **14 / 25** | Significant at J = 1–2, 4, I = 16–32, λ = 1, T_k = 0.8–2.4; **not** at J = 3, 5, λ ≥ 1.5, T_k = 3.0, f_j = 0.5×10⁸ |

\*T_k = 0.8 point is all-infeasible; significant paired gap there reflects
infeasible geometry, not a feasible QoS win.

**Publishable pattern:** SCA’s advantage is **strong vs coverage-based baselines**
(k-means, PSO) under bandwidth stress; vs **random** it appears when UAV count is
low or IoT count is high — not at the default J = 3 point (Wilcoxon p = 0.123).

### 3.3 Why k-means and PSO lose (structural, not noise)

At J = 3, mean over 20 seeds:

| Method | Equal-share score (Mbps) | Capped LP score (Mbps) | LP − equal (Mbps) |
| --- | --- | --- | --- |
| Random | 8.432 | 8.928 | **+0.496** |
| K-means | 8.692 | 8.901 | +0.209 |
| PSO | 8.742 | 8.905 | +0.163 |

PSO’s **inner** fitness uses equal-share bandwidth; k-means minimizes spatial
spread. The **campaign score** is the capped LP. Random placements look poor
under equal share but gain the most when the LP can concentrate 25% of B_sys on
the best links. SCA optimizes under the LP objective (via SCA steps), so it
consistently beats proxies that do not — even when random occasionally lucks into
a good geometry.

---

## 4. Solver validation

`spot_validate_8.8mhz.json` (seed 1):

| Config | CVXPY Mbps | MATLAB Mbps | `se_max_abs_diff` |
| --- | --- | --- | --- |
| 8.8 MHz, no cap | 8.9831 | 8.9831 | ~1e-15 |
| 8.8 MHz, cap 25% | 8.9797 | 8.9795 | ~1e-15 |

Campaign SCA uses CVXPY; MATLAB path is validated for the frozen SCA core.

---

## 5. Sensibility checklist

| Check | Result |
| --- | --- |
| Unit tests (90) | Pass |
| 20 kHz infeasible | Yes — 0% feasible, ~0.02 Mbps (independent of `S_i`/`L`; see §1.1) |
| Rates ≤ bandwidth ceiling | Yes — ~8.98 Mbps at 8.8 MHz, ~2.45 at 2.4 MHz |
| SCA best on feasible points (mean) | Yes |
| λ / CPU vary slightly when feasible | Yes — &lt; 0.03 Mbps (AoDT binds at T_k) |
| T_k = 0.8 infeasible | Yes |
| More UAVs help under cap (J = 1→3) | Yes |
| Method collapse without cap | Yes — expected saturation |
| Paired stats + FDR documented | Yes |
| Eq. (17) implementation | **Resolved** — matches published formula (§8.1) |
| Eq. (17) vs Fig. 11 narrative | **Open (paper intent)** — documented; not a code defect (§8.1) |

**Verdict:** Results are **sensible and sufficient to move on** to writing and
figures. They are **not** a numeric reproduction of the paper’s §VII curves, and
that gap is **expected and documented**, not a failure of this repo.

---

## 6. Known limitations (this milestone)

| Item | Status |
| --- | --- |
| Area 100 × 100 m | **By design** (guide); not changed to 500 × 500 m |
| TD3 (Algorithm 2) | Not implemented |
| Paper Mbps targets | Explicitly not pursued |
| Eq. (17) implementation | **Resolved** — coded as published; unit tests + Fig. 11 sensibility |
| Eq. (17) vs Fig. 11 narrative | **Open (paper intent)** — formula ranks hetero worse than slow; not a code bug (§8.1) |
| `S_i`, `L` | **Settled:** 12,000 bytes, `L=3.75×10⁶` (see §10) |
| 25% per-link cap | External parameter (not Problem (P)) |
| SCA init | K-means + frozen association/processing |
| PSO | External baseline; equal-share inner fitness |

---

## 7. Suggested writeup sentences (copy-ready)

**Audit (short):**  
*Under faithful implementation of the Khalaf et al. model with Table II parameters
in a 100 × 100 m field, B_sys = 20 kHz yields 0% feasible deployments and
~0.02 Mbps sum rate; the paper’s multi-Mbps §VII curves are incompatible with
this parameter set. At feasible bandwidths, SCA outperforms k-means and PSO under
a 25% per-link cap (paired Wilcoxon, FDR q &lt; 0.05 on 25/25 and 24/25 unique
points respectively); the advantage over random is limited to low UAV counts and
high IoT density and is **not** significant at the default J = 3 configuration
(Wilcoxon p = 0.123, Bonferroni-adjusted p = 0.369).*

**Headline result (cap 25%, J = 3):**  
*SCA achieves 8.946 Mbps mean sum rate vs 8.901 (k-means) and 8.905 (PSO),
winning 20/20 and 19/20 paired seeds respectively (p &lt; 0.001); vs random
(8.928 Mbps) the gain is 0.019 ± 0.042 Mbps with 15/20 wins (Wilcoxon p = 0.123,
Bonferroni-adjusted p = 0.369).*

**Eq. (17) vs Fig. 11 narrative:**  
*The paper states heterogeneous within-group arrival rates should lie between
uniform fast and uniform slow. Our queue simulator shows fast &lt; heterogeneous
&lt; slow, but Eq. (17) — the score inside Problem (P) — ranks heterogeneous
**above** uniform-slow (worse AoDT) because the formula is slowest-λ-limited yet
penalizes total load via `Σλ`. At J = 3 the hetero−slow excess is +0.14 s under
settled parameters (+0.05 s under old placeholders). **Implementation is resolved:**
Eq. (17) is coded as published and verified (unit tests, Fig. 11 sensibility
50/50) — this is formula behaviour, not a code defect. **Paper intent remains
open:** whether the authors meant Problem (P) to use this expression given their
Fig. 11 prose. Either way, SCA optimizes a score whose λ-heterogeneity ordering
differs from the narrative figure.*

---

## 8. AoDT extras (Eqs. (10)–(17), FCFS / FCFS-P / LCFS-S, Fig. 11)

Problem (P) still scores **Eq. (17)**. This pass adds the rest of the paper’s
AoDT block except TD3: instantaneous age (10), neglected download (12) as
`Z = 0`, LCFS-S closed forms (14)–(15), Kaul M/M/1 **FCFS** AoI, and
event-driven **FCFS**, **FCFS-P** (same-source waiting-room replacement),
and **LCFS-S** (paper: preemption in service).

**Single-source unit tests (ρ = 0.6):** LCFS-S matches Eq. (14); FCFS matches
the Kaul formula; LCFS-S age < FCFS-P ≤ FCFS. A constant upload delay and a
non-zero `Z` add to the simulated age as expected.

**Multi-source process age is not Eq. (17).** Eq. (17) is
`max D_i + (1/λ_min)(1 + Σλ/μ)`. The simulator’s process age is the
time-average of `max_i ζ_i(t)` (Eq. (10) over the group). With five IoTs that
max is larger than (17), which is the right inequality. At default load
(μ = 200 /s, Σλ ≈ 10 /s) FCFS-P ≈ FCFS (almost no waiting-room replacement).
LCFS-S **drops** stale packets (as the paper describes) but can **raise**
`max_i ζ_i` because preemption is unfair to the lagging source. That does
not contradict the single-source LCFS-S advantage.

### 8.1 Fig. 11 (I = 10, k-means, 8.8 MHz, 25% cap, 20 seeds)

Same IoT geometry across λ patterns for each seed. Primary file:
`results/fig11_8.8mhz_cap25_si12k.json` (settled `S_i`/`L`). Older run with
placeholder parameters: `results/fig11_8.8mhz_cap25.json`.

**Fig. 11 sensibility gate (50 / 50):** automated checks in
`sensibility_checks()` (`src/uavdt/experiments/fig11.py`), **10 per UAV count**
`J ∈ {1,…,5}`. They are **not** unit tests and **not** a Mbps target. Each `J`
asserts: (i) Eq. (17) max age — fast λ &lt; slow and fast &lt; heterogeneous;
heterogeneous closer to slow than to fast (slowest-λ-limited formula); (ii) FCFS
simulator — mean per-source age and process-max age ordered fast &lt; hetero &lt;
slow; (iii) closed forms — Eq. (17) ≥ Eq. (15), FCFS closed &gt; Eq. (15), sim
process-max &gt; Eq. (17) − 0.05 s; (iv) LCFS-S / FCFS-P / FCFS simulators
return finite mean-source ages at fast λ. Passing means the AoDT implementation
and Fig. 11 λ patterns behave coherently; it does **not** certify agreement with
the paper’s unpublished Fig. 11 curves — and check (i) deliberately encodes what
Eq. (17) **does**, not what the paper **says** heterogeneous should do.

#### Formula vs narrative (second audit finding)

Paper text (Fig. 11 caption): heterogeneous λ performance should lie **under**
uniform fast and **above** uniform slow — i.e. fast &lt; heterogeneous &lt; slow
for process age. Two evaluators disagree:

| Evaluator | J = 3 ordering (settled `S_i`/`L`) | Matches paper narrative? |
| --- | --- | --- |
| **FCFS simulator** (process-max of Eq. (10)) | fast (1.46 s) &lt; hetero (2.16 s) &lt; slow (3.13 s) | **Yes** |
| **Eq. (17)** (Problem (P) score) | fast (1.00 s) &lt; slow (1.74 s) **&lt; hetero (1.89 s)** | **No** — hetero **worse than slow** |

Mechanism: Eq. (17) uses `λ_min` (0.8/s for both slow and heterogeneous groups)
so the slowest source sets the baseline, but the queue term `(1 + Σλ/μ)/λ_min`
**also** grows with total load. Heterogeneous mixes fast and slow sources, so
`Σλ` exceeds uniform-slow even though `λ_min` is unchanged — heterogeneous
scores **higher** (worse) AoDT than uniform-slow. The simulator’s
fast &lt; hetero &lt; slow ordering comes from time-averaging `max_i ζ_i(t)`, which
is **not** what Eq. (17) computes.

The hetero − slow excess on Eq. (17) **widened** when `S_i`/`L` were corrected
(larger upload delays in `D_i`, heavier queues at `μ ≈ 53.3` /s):

| `S_i`/`L` | Eq. (17) fast | Eq. (17) slow | Eq. (17) hetero | hetero − slow |
| --- | ---: | ---: | ---: | ---: |
| Old placeholders | 0.83 s | 1.57 s | 1.62 s | **+0.05 s** |
| Settled defaults | 1.00 s | 1.74 s | 1.89 s | **+0.14 s** |

Simulator process-max ordering is unchanged (fast &lt; hetero &lt; slow at every
J).

**Status split (read this before debugging):**

| Question | Status | Notes |
| --- | --- | --- |
| Is Eq. (17) implemented correctly? | **Resolved** | `average_aodt_s()` matches the published formula; `test_fig11_eq17_slowest_limited_on_one_seed` and Fig. 11 sensibility (50/50) pass. **No code bug to find.** |
| Does Eq. (17) match the paper’s Fig. 11 narrative? | **No** | Formula ranks hetero **worse than** uniform-slow (see table above). |
| Did the authors intend (17) given that narrative? | **Open** | Outside this repo. SCA optimizes (17) as written; the mismatch is between paper text and paper equation, not between our code and Eq. (17). |

Default J = 3, settled `S_i`/`L` (mean over 20 seeds):

| Pattern | Eq. (17) max (s) | Eq. (15) max (s) | Sim mean source FCFS (s) | Sim process-max FCFS (s) | Sum rate (Mbps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform fast (λ₁=2, λ₂=3) | 0.998 | 0.923 | 0.694 | 1.455 | 8.692 |
| Heterogeneous (0.8…3 in each group) | 1.886 | 1.682 | 0.932 | 2.155 | 8.692 |
| Uniform slow (λ₁=0.8, λ₂=1) | 1.743 | 1.668 | 1.403 | 3.131 | 8.692 |

**What still aligns with the paper:**

- **Simulator** mean-source and process-max: fast &lt; heterogeneous &lt; slow at
  every J = 1…5 — matches Fig. 11 narrative.
- Sum rate is **identical** across λ patterns at fixed J (8.69 Mbps at J = 3).
  λ does not enter Eq. (20); with settled `L`, Eq. (17) max ≈ 1.9 s sits below
  `T_k = 2.8 s` on this check (campaign default binds at 2.8 s via placement).
  Feasible fraction 100%. Rate still rises with J (8.09 → 8.86 Mbps from J = 1
  to 5) via geometry.

**What does not:** Eq. (17) ranks heterogeneous **above** uniform-slow, contrary
to the paper’s stated “between fast and slow” story. That gap is **formula
behaviour** (implementation resolved — see status table above), and it **grew**
under corrected parameters.

```text
python -m uavdt aodt-compare --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --placement kmeans
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --out results/fig11_8.8mhz_cap25_si12k.json
```

---

## 9. How to regenerate

```text
$env:PYTHONPATH="src"

python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --solver cvxpy --out results/campaign_8.8mhz_cap25_si12k.json
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/analyze_sca_vs_random_losses.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/analyze_campaigns.py
```

Full campaign re-run (long): `scripts/run_all_bandwidth_campaigns.ps1`.

---

## 10. `S_i` / `L` correction — settled defaults vs old placeholders

**Problem:** `config.py` previously used `S_i = 10{,}000` **bit** (~1,250 bytes) and
`L = 10^6` cycles (`\mu = 200` /s). Our settled experimental choices are
**12,000 bytes** and **3.75×10⁶** cycles (`\mu \approx 53.3` /s). Files
`campaign_20260904_cap25*.json` used the **old** placeholders; the primary
campaign is `campaign_8.8mhz_cap25_si12k.json`.

**Fix:** `EXTERNAL_TASK_SIZE_BYTES = 12_000`, `EXTERNAL_TASK_SIZE_BITS = 96_000`,
`EXTERNAL_TASK_CYCLES = 3.75e6` in `src/uavdt/config.py`.

**Full campaign:** `campaign_8.8mhz_cap25_si12k.json` — all axes, 20 seeds,
settled defaults. Paired stats in `campaign_8.8mhz_cap25_si12k_paired.json`.

**Side-by-side @ J=3:** `scripts/compare_si_l_defaults.py` — I=10, J=3, 8.8 MHz,
25% cap, 20 seeds. Output: `results/compare_si_l_defaults.json`.

| Quantity | Old placeholders | Settled defaults | Better? |
| --- | ---: | ---: | --- |
| `S_i` | 1,250 bytes (10k bit) | **12,000 bytes** | Matches our experimental choice |
| `L` | 1×10⁶ cycles | **3.75×10⁶** | Matches our experimental choice |
| `μ` | 200 /s | **53.3 /s** | Heavier queue; AoDT more meaningful |
| Mean max AoDT @ J=3 (SCA) | ~1.83 s | **~2.80 s** | **Yes** — sits at `T_k` (binding) |
| SCA mean Mbps | 8.964 | 8.946 | Slightly lower (more B to AoDT floors) |
| Random mean Mbps | 8.950 | 8.928 | Same pattern |
| SCA − random (Mbps) | +0.014 | **+0.018** | Slightly wider edge |
| SCA − k-means (Mbps) | +0.055 | +0.046 | Slightly narrower |
| Feasible @ 8.8 MHz cap25 | 100% | 100% | Unchanged |
| 20 kHz feasible (5 seeds) | 0/5 | 0/5 | Unchanged (QoS/Shannon, not `S_i`/`L`) |
| Fig. 11 sensibility (J=3) | 10/10 | 10/10 | Coherent; Eq. (17) hetero−slow gap +0.05→**+0.14 s** |

**Verdict:** Settled `S_i`/`L` are **better for model fidelity** — upload delay and
queue load are in the range we originally chose; Eq. (17) now **binds at**
`T_k = 2.8` s instead of sitting ~1 s below it. Sum rates drop ~0.02 Mbps
because larger tasks pull bandwidth into AoDT floors; **method ranking is
unchanged** (SCA still leads; edge over random slightly larger). This does **not**
close the gap to paper §VII Mbps (still not a target). Re-run headline campaigns
with the new defaults before citing Mbps numbers in a writeup.

```text
python scripts/compare_si_l_defaults.py
python -m uavdt campaign --axis uavs --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --out results/campaign_8.8mhz_cap25_si12k.json
```
