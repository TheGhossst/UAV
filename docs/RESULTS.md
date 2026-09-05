# Current results — Khalaf et al. (IEEE TNSM, 2026) reproduction

This document records **our** experimental outcomes on the fresh `uavdt` simulator.
It does **not** compare Mbps figures to the paper’s §VII plots.

**Lead finding:** Eq. (6) plus constraint (27) give, for any powers, path
losses, or noise figures,

\[
\sum_{i,j} B_{ij}\log_2(1+\mathrm{SNR}_{ij})
\le B_{\mathrm{sys}}\log_2(1+\mathrm{SNR}_{\max}).
\]

If Table II’s `B_sys = 20{,}000` Hz is the sum cap in (27), then even
`SNR_max = 10^{15}` (essentially noiseless, `\log_2(1+\mathrm{SNR})\approx 49.83`)
caps the whole system at **0.997 Mbps**, not 7–14 Mbps. That one-line ceiling
does not use `a`, `b`, `\eta_{\mathrm{LoS/NLoS}}`, `\sigma`, field size, or this
simulator. See §0.

**Area:** headline campaigns use **100 × 100 m**; the 20 kHz check was also
run at the paper’s **500 × 500 m** field (same 0% feasible conclusion; §0.3).
**Seeds:** 20 consecutive seeds per sweep point (`seed_start = 1`).
**Methods:** SCA, random, k-means, PSO (PSO is external, not in the paper).
**Score:** Python `evaluate()` on every method. Placement baselines are
re-scored with the same frozen-geometry bandwidth LP as SCA’s B step.
**SCA backend:** CVXPY for campaigns; MATLAB CVX+MOSEK spot-validated.

Ledger for how to re-run: `docs/EXPERIMENTS.md`. Parameter ledger:
`docs/REPRODUCTION.md`, `docs/param.md`.

**Last full regeneration:** 2026-09-05 — `pytest` 108/108;
`scripts/check_bsys_20khz.py`; primary campaign
`campaign_8.8mhz_cap25_si12k.json` (all axes, 20 seeds, CVXPY);
`scripts/rerun_init_repair_points.py`; paired/losses analysis;
`fig11_8.8mhz_cap25_si12k.json`; `spot-validate` (8.8 MHz no cap + cap 25%).
Full bandwidth preset sweep: `scripts/run_all_bandwidth_campaigns.ps1`
(completed 2026-09-05 — refreshes `campaign_20khz*.json`, `campaign_2.4mhz*.json`,
`campaign_8.8mhz_n20.json`, `campaign_8.8mhz_cap25_n20.json`).

---

## 0. Eq. (6) + constraint (27): the 20 kHz ceiling

### 0.1 Model-free bound (lead with this)

Constraint (27) is `\sum_{i,j} B_{ij} \le B_{\mathrm{sys}}`. Eq. (6) is
`r_{ij} = B_{ij}\log_2(1+\mathrm{SNR}_{ij})`. The objective (20) is the sum of
associated rates, which cannot exceed the sum of all `r_{ij}`. Therefore

```text
R_sum  ≤  Σ B_ij · log2(1+SNR_ij)  ≤  B_sys · log2(1+SNR_max)
```

for **any** choice of `p_i`, path loss, and noise. With `B_sys = 20{,}000` Hz:

| `SNR_max` | `\log_2(1+\mathrm{SNR})` | Ceiling |
| ---: | ---: | ---: |
| 1 | 1.00 | 0.020 Mbps |
| `10^3` | 9.97 | 0.199 Mbps |
| `10^6` | 19.93 | 0.399 Mbps |
| `10^{15}` (absurdly generous) | 49.83 | **0.997 Mbps** |

Figs. 6–10 sit at **7–14 Mbps**, seven to fourteen times above even the
noiseless fantasy. Helper: `uavdt.channel.sum_rate_ceiling_bit_per_s`.
Re-run: `python scripts/check_bsys_20khz.py`.

### 0.2 Table II label vs constraint (27) — closed, as a fork

Table II’s row text is literally **“Minimum bandwidth allocation,
`B_sys`, 20{,}000 Hz”**, in parallel with the previous row (“Minimum data
rate, `R_min`”). Constraint (27) and the paragraph that explains it are a
**ceiling on total uplink bandwidth**: “the total uplink bandwidth of the
system does not exceed the available system bandwidth of value `B_sys`.”
Problem (P) contains **no** constraint `B_{ij} \ge B_{\mathrm{sys}}` or
`\sum B_{ij} \ge B_{\mathrm{sys}}`. The only occurrence of the symbol
`B_sys` in the program is the sum cap (27). The paper never states a second
bandwidth number.

What the paper’s own Fig. 6–10 axes do to each reading:

| Reading | What 20 kHz is | What Figs. 6–10 do to it |
| --- | --- | --- |
| **A — mathematics of (27) governs** (adopted here) | Sum cap `\sum B_{ij} \le 20` kHz | **Ruled out.** Y-axis is 7–14 Mbps; §0.1 caps any SNR at 0.997 Mbps. Fig. 7 also *grows* with `I` (SCA “exceeding approximately 14 Mbps for 32 devices”), which a 20 kHz shared pool cannot do. |
| **B — table English governs** | A per-link or per-device **floor**, not the (27) cap | **Not ruled out by shape.** Linear-in-`I` growth is what a per-link allocation looks like. The (27) cap is then **undisclosed**. On the *written* radio the stated 20 kHz floor is tens of times too small (next paragraph). |

We adopt **Reading A** because (i) Table II is the parameter table for the
symbols in Problem (P), (ii) those symbols appear in (27) only as a sum
ceiling, (iii) the explanatory sentence of (27) says “available system
bandwidth,” and (iv) “Minimum” on that row is the same adjective as `R_min`,
which **is** a floor (constraint 25) — a table-editing collision, not a second
constraint. Reading B is named so a critic cannot say the table English was
ignored.

**Reading B magnitude (not a rounding error).** Invert Eq. (6) under an equal
per-link split, `B_i = R / (I \log_2(1+\mathrm{SNR}))`, using the control’s
observed `\mathrm{SNR}_{\max}\approx 1.03` (best-case: the *smallest* `B_i`
that can hit the published Mbps). Helper:
`uavdt.channel.per_link_hz_for_target_rate`.

| Published anchor (paper text) | `I` | `B_i` needed | vs stated 20 kHz | Implied `\sum B` |
| --- | ---: | ---: | ---: | ---: |
| Fig. 6 SCA ~8.8 Mbps (`J = 5`) | 10 | **862 kHz** | **43×** | 8.62 MHz (431×) |
| Figs. 6–10 band, 7 Mbps at `I = 10` | 10 | 685 kHz | 34× | 6.85 MHz (343×) |
| Fig. 7 SCA ~14 Mbps | 32 | 428 kHz | 21× | 13.7 MHz (685×) |

At a fantasy `\mathrm{SNR}=10^{15}` the stated 20 kHz *floor* would suffice
(`B_i` drops to 9–18 kHz). That is why Reading B is not a physical
impossibility in the same parameter-free sense as Reading A. It *is* an
unstated parameter **21–43×** the table row on the radio Table II actually
writes (`\mathrm{SNR}\approx 1`), or a hidden (27) cap of **~7–14 MHz**
(hundreds of times 20 kHz as a pool). Neither is a units typo of 20{,}000 Hz.

### 0.3 Control: paper 500 × 500 m field, still 20 kHz

A critic can object that the infeasible 20 kHz result was obtained on a
**100 × 100 m** field, so two things were changed at once. The bound in §0.1
does not depend on area. The simulator check does not either, in the
direction that would help the paper: at 500 × 500 m the zenith SNR is the
same (`H = 100` m) and typical links are worse.

Default point `I=10`, `J=3`, 20 seeds, random and k-means with the same
frozen-`q` bandwidth LP as the campaign (`results/check_bsys_20khz.json`).

**Mechanism of the realized rate.** At 20 kHz the QoS bandwidth LP is
infeasible (`R_{\min}=10` kbps on ten links needs ~102 kHz). Evaluate then
falls back to **equal-share** `B_{\mathrm{sys}}/I = 2` kHz per associated
link, so

```text
R_sum          = B_sys · mean_associated_SE     (what we report)
best-link dump = B_sys · max_SE                 (geometry ceiling)
```

Max SNR is set by a UAV over an IoT at `H = 100` m, so `max_SE` (and the
dump ceiling) is **area-independent**. Mean SE is not: a 500 m field has
longer associated links.

| Field | Max SNR | Mean associated SE | Best-link dump | Equal-share pred. | Random / k-means realized | Feasible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 × 100 m | 1.03 | 0.973 | 0.020 Mbps | 0.019 Mbps | 0.019 / 0.020 Mbps | **0%** |
| 500 × 500 m (paper) | 1.03 | 0.635 | 0.020 Mbps | 0.013 Mbps | 0.012 / 0.013 Mbps | **0%** |

The 500 m drop (0.020 → 0.013 Mbps) is equal-share averaging over worse
links, **not** a different SNR regime for the §0.1 bound. Dumping the whole
20 kHz pool on the best link would still be ~0.020 Mbps on both fields.
Area is not a confounder for the 0%-feasible conclusion.

---

## Audit paragraph (paper §VII vs this implementation)

The 20 kHz finding does not wait on the simulator: **Eq. (6) + (27) already
cap Table II `B_sys` at 0.997 Mbps even at `SNR=10^{15}`** (§0). We then
implemented the Khalaf–Itani–Sharafeddine UAV-aided digital-twin IoT model
(Eqs. (1)–(6), (11), (13), (16)–(17), Problem (P)) in a **100 × 100 m** field
with faithful Table II radio/compute parameters except bandwidth and two
external task quantities (`S_i`, `L`) needed to evaluate upload delay and
service rate, and repeated the 20 kHz check at **500 × 500 m**. Under
**Table II `B_sys = 20 kHz` as the (27) cap**, every method fails QoS and
AoDT: feasible fraction **0%**, sum rate **~0.02 Mbps** at 100 m and
**~0.012–0.013 Mbps** at 500 m — matching the written channel’s
`SNR\approx 1` ceiling, not a solver artifact. We therefore **do not treat
the paper’s Fig. 6–10 Mbps curves as a reproduction target**. For **feasible**
bandwidths (2.4 MHz and
8.8 MHz), behaviour is internally coherent: at 8.8 MHz **without** a per-link
cap, all methods saturate near **~8.98 Mbps** (leftover spectrum piles onto the
best link); with a **25% per-link cap** (external parameter, not Problem (P)),
methods separate and **SCA beats k-means and PSO** on paired seeds with FDR
q < 0.05 on **25/25** unique sweep points for both; the edge
over **random** is **regime-limited** (FDR **15/25**; large at J = 1–2 and high I, not
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
| `results/check_bsys_20khz.json` | 20 kHz | none | 20 | Model-free ceiling + 100 m vs **500 m** control |
| `results/campaign_20khz.json` | 20 kHz | none | 20 | Table II diagnostic, 100 m, all axes (infeasible) |
| `results/campaign_20khz_cap25.json` | 20 kHz | 25% | 20 | Same, capped |
| `results/campaign_2.4mhz.json` | 2.4 MHz | none | 20 | Mid-bandwidth, saturated |
| `results/campaign_2.4mhz_cap25.json` | 2.4 MHz | 25% | 20 | Mid-bandwidth, differentiated |
| `results/campaign_8.8mhz_n20.json` | 8.8 MHz | none | 20 | Headline, saturated (2026-09-05 refresh) |
| `results/campaign_8.8mhz_cap25_si12k.json` | 8.8 MHz | 25% | 20 | **Primary comparison config** (settled `S_i`/`L`) |
| `results/campaign_8.8mhz_cap25_n20.json` | 8.8 MHz | 25% | 20 | Same config as si12k; output of `run_all_bandwidth_campaigns.ps1` |
| `results/campaign_20260904_cap25.json` | 8.8 MHz | 25% | 20 | Superseded — old `S_i`/`L` placeholders |
| `results/compare_si_l_defaults.json` | 8.8 MHz | 25% | 20 | Old vs settled side-by-side @ J=3 |

Supporting artifacts:

- `results/check_bsys_20khz.json` — Eq. (6)+(27) ceilings; 20 kHz at 100 m and 500 m
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

The argument to lead with is §0.1, not this campaign table. The 100 × 100 m
sweep is a confirmation that the solver agrees with the one-line bound.

| Observation | Value |
| --- | --- |
| Model-free ceiling (`SNR_max=10^{15}`) | **0.997 Mbps** |
| Written-channel ceiling (observed `SNR\approx 1.03`) | **0.020 Mbps** |
| Feasible fraction (all methods, all points, 100 m) | **0%** |
| Typical sum rate (100 m) | **~0.019–0.020 Mbps** |
| Typical sum rate (500 m control) | **~0.012–0.013 Mbps** |
| Cap vs no-cap | **No difference** |

**Interpretation:** If 20 kHz is the (27) cap, total spectrum is the bottleneck
and 7–14 Mbps is impossible without trusting the simulator. QoS
(`R_min = 10` kbps per active link) also fails: meeting it on ten associated
links needs on the order of **~102 kHz** at 100 m (`∑_i R_min / SE_{ij}`),
and more at 500 m where SE is worse. AoDT cannot be met either. `S_i` / `L`
do not enter the §0.1 bound.

**`S_i` / `L` defaults (settled):** `S_i = 12{,}000` bytes (`96{,}000` bit),
`L = 3.75\times10^6` cycles/task (`\mu \approx 53.3` /s). Not in Table II.
Prior placeholder runs used `10{,}000` bit and `L=10^6`; see §10. The primary
campaign `campaign_8.8mhz_cap25_si12k.json` uses the settled values.

`S_i` only enters upload delay (Eq. 11) and thus the AoDT bandwidth floor; making
`S_i` smaller could ease AoDT but would not relax the QoS floor or the §0.1
Shannon cap. `L` only scales `μ_j`; it does not enter the radio model.

### 1.2 2.4 MHz

| Config | SCA @ J=3 (Mbps) | All methods feasible? |
| --- | --- | --- |
| No cap | 2.434 | Yes (100%) |
| Cap 25% | 2.428 | Yes (100%) |

At J = 3 with cap 25% (fresh `campaign_2.4mhz_cap25.json`, 2026-09-05): SCA
2.428, PSO 2.423, k-means 2.420, random 2.414 Mbps — gaps **&lt; 0.01 Mbps**.
Without cap, PSO/k-means are within **~0.005 Mbps** of SCA. Ordering still favours
SCA at J = 3 but margins are tighter than the 8.8 MHz cap25 campaign.

### 1.3 8.8 MHz, no cap

| Method @ J=3 | Mean Mbps | Feasible |
| --- | --- | --- |
| SCA | 8.971 | 100% |
| Random | 8.954 | 100% |
| K-means | 8.947 | 100% |
| PSO | 8.943 | 100% |

Spread **&lt; 0.03 Mbps** (`campaign_8.8mhz_n20.json`, 2026-09-05). **λ**, **CPU**,
and most **AoDT** points (when feasible) give **identical** per-seed rates — the
objective is communication-limited only. **Not useful for ranking methods.**

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

Campaign Mbps means are `mean(all 20 seeds)` — `campaign._summarize` does not
drop infeasible seeds. After the (24) init repair, **every row in §2.1–2.3
and §2.5 is 100% feasible**, so those tables are feasible-set means. The
only exception in this section is **§2.4 T_k = 0.8 s** (below).

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
  spatial placement and joint B allocation matter. **J = 1 and J = 2 are 100%
  feasible** for every method (AoDT slack ≈ 2.21 s and 1.91 s). The cap
  lowers Mbps; it does not make the point infeasible.
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
| 28 | 8.919 | 8.858 | 8.894 | 8.899 | 100% |
| 32 | 8.906 | 8.838 | 8.883 | 8.891 | 100% |

**Analysis:** SCA stays **~0.02–0.06 Mbps** above k-means/PSO as I grows; random
closes part of the gap at low I and falls further behind at high I. At
**I = 28–32**, putting both process groups on one UAV would violate (24)
(56/s and 64/s vs μ ≈ 53.3 /s), but a **split** assignment is feasible
(28/s and 32/s each). Older 55%/60% “feasible” figures counted
majority-of-association collisions at init — SCA freezes `b_ij` after that
step, so a colliding seed stayed infeasible. After repairing init when (24)
fails, **all 20 seeds are feasible** at both ticks for **every method**
(random 10%→100% / 25%→100%; k-means 55%→100% / 60%→100%; PSO was already
100% because its inner fitness penalizes unstable queues). The same
`cpu_stable_processing` helper scores SCA init and the placement
baselines; this was not an SCA-only repair. SCA still wins paired
tests on I = 16–32 vs k-means, PSO, and random (FDR q &lt; 0.05), including
vs PSO at I = 28 (previously n.s. because infeasible seeds mixed the
comparison).

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
| **0.8** | 8.909 | **0%** | Mean of 20 **infeasible** scores; not a model limit (see analysis) |
| 1.2 | 8.889 | 100% | |
| 1.6 | 8.925 | 100% | |
| 2.0 | 8.938 | 100% | |
| 2.4 | 8.945 | 100% | |
| 2.8 – 3.0 | 8.946 – 8.948 | 100% | Plateau at default |

**Analysis:** T_k = 0.8 s is **0% feasible** for SCA, k-means, and random
under this implementation’s frozen nearest-association. That 0% **does not
survive a different discrete init** — do not publish it as a Problem (P)
limit. Eq. (17) slack is \(T_k - Q - T_{\mathrm{u2u}}\) on forwarded links.
With \(\lambda=2\)/s, \(|N_k|=5\), \(\mu\approx 53.3\)/s, \(Q\approx 0.594\) s,
so \(Q+T_{\mathrm{u2u}}\approx 0.894\) s \(> 0.8\) s even at infinite rate.
Nearest-a splits both process groups on **all 20** k-means seeds (IoTs are
placed uniformly; \(N_k\) is an index slice, not a spatial cluster), so
some IoTs always forward. The (24) init repair does **not** fire (I = 10,
majority is CPU-stable). Checks that *do* find a legal point: **J = 1**
(cannot forward) is 20/20 feasible at T_k = 0.8; the **same k-means UAV
positions** with process-cohesive \(a_{ij}\) (all of \(N_k\) on one UAV)
are 20/20 feasible; PSO is 2/20 feasible (seeds 8, 20) by putting **both
process groups on one UAV** (`assoc_uavs = {1}`, `fwd = 0`, slack +0.206 s
— the J = 1 geometry). Sharing that UAV is legal at I = 10 (20/s < μ).
The other 18 PSO seeds still split \(N_k\) across three UAVs and stay
infeasible. PSO’s published mixed mean at this tick (8.61 Mbps all-seed vs
8.38 feasible-only) is **not** in the table above. For T_k ≥ 1.2 s the comm
objective rises toward the default plateau as the AoDT floor relaxes. At
T_k = 0.8, the SCA 8.909 Mbps figure is the mean of 20 infeasible
nearest-a scores — diagnostic only, not “the model cannot meet 0.8 s.”

### 2.5 Fig. 10 analogue — UAV CPU f_j (I = 10, J = 3)

| f_j (×10⁸ c/s) | SCA Mbps | SCA feasible |
| --- | --- | --- |
| 0.5 | 8.946 | 100% |
| 1.0 | 8.948 | 100% |
| 1.5 – 2.5 | 8.946 – 8.947 | 100% |

**Analysis:** At **f_j = 0.5×10⁸**, μ ≈ **13.3** /s. Both groups on one UAV
(20/s) violates (24); splitting (10/s each) is feasible. The old 65% figure
was the same init-collision artifact as I = 28–32, not model infeasibility.
After the init repair, **20/20 seeds are feasible** and the comm score sits
on the default plateau (CPU does not enter Eq. (20) once queues are stable).
Above 1.0×10⁸ the score varies only slightly (&lt; 0.02 Mbps). **Do not pool CPU
rows with J = 3 in meta-analyses** (λ = 2.0 / f_j = 2.0 duplicate the default).

---

## 3. Paired statistics (SCA vs baselines, same seed)

Source: `campaign_8.8mhz_cap25_si12k_paired.json`. Delta = SCA − baseline (Mbps).
Primary test: **Wilcoxon signed-rank** (two-sided). `+/-` in quotes is **sample
std of 20 paired deltas**, not SEM.

### 3.1 Default point — J = 3, I = 10

| Baseline | Mean Δ | Std Δ | Wins | Wilcoxon p | Bonferroni p_adj (m=3) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Random | +0.019 | 0.042 | 15/20 | 0.123 | **0.369** | Not significant |
| K-means | +0.046 | 0.019 | 20/20 | &lt;0.001 | **&lt;0.001** | **Significant** |
| PSO | +0.042 | 0.026 | 19/20 | &lt;0.001 | **&lt;0.001** | **Significant** |

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
| **PSO** | **25 / 25** | All unique J, I, λ, T_k, f_j points |
| **Random** | **15 / 25** | Significant at J = 1–2, 4, I = 16–32, λ = 1, T_k = 0.8–2.4, f_j = 0.5×10⁸; **not** at J = 3, 5, λ ≥ 1.5, T_k = 3.0, f_j ≥ 1.0×10⁸ |

The I = 28, I = 32, and \(f_j = 0.5\times 10^8\) FDR calls from before the
(24) init repair mixed jointly feasible seeds with seeds whose majority
vote violated (24). Wilcoxon still used n = 20 Mbps values, but those SCA
scores were not 20 feasible geometries — infeasible seeds were scored
anyway (fallback \(B\), failed (24)). That is a **confounded sample**, not
a random 20. The calls in this table are 20/20 jointly feasible and
**supersede** the earlier ones; they are not a same-sample update.

\*T_k = 0.8 under frozen nearest-a is all-infeasible for SCA/k-means/random
(PSO 2/20); the paired gap there is among infeasible nearest-a scores, not
a feasible QoS win. A legal no-forwarding assignment exists (§2.4).

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

(Source: `campaign_8.8mhz_cap25_si12k_losses.json` → `proxy_mismatch_j3`, 2026-09-05.)

PSO’s **inner** fitness uses equal-share bandwidth; k-means minimizes spatial
spread. The **campaign score** is the capped LP. Random placements look poor
under equal share but gain the most when the LP can concentrate 25% of B_sys on
the best links. SCA optimizes under the LP objective (via SCA steps), so it
consistently beats proxies that do not — even when random occasionally lucks into
a good geometry.

---

## 4. Solver validation

`spot_validate_8.8mhz.json` (seed 1, `max_iterations=30`):

| Config | CVXPY Mbps | MATLAB Mbps | `abs_obj_diff` (bit/s) | `se_max_abs_diff` |
| --- | --- | --- | --- | --- |
| 8.8 MHz, no cap | 8.9677 | 8.9677 | **0** | ~1e-15 |
| 8.8 MHz, cap 25% | 8.9600 | 8.9600 | **0.63** | ~1e-15 |

Both configs: `agreement: ok`. Cap-25 seed 1 matches the campaign CVXPY rate
exactly (8.960041 Mbps).

**Headline point — 20 paired seeds (J = 3, cap 25%, 2026-09-05):**  
`scripts/compare_sca_cvxpy_matlab.py` → `results/sca_cvxpy_vs_matlab_j3.json`.
CVXPY rates from `campaign_8.8mhz_cap25_si12k.json`; MATLAB re-run per seed.

| Quantity | Value |
| --- | --- |
| Mean CVXPY | **8.946 Mbps** |
| Mean MATLAB | **8.949 Mbps** |
| Mean Δ (MATLAB − CVXPY) | **+0.003 Mbps** |
| Max \|Δ\| | **0.026 Mbps** (seeds 2, 7) |
| Median \|Δ\| | **~0** (17/20 seeds within 1e-5 Mbps) |
| Wilcoxon p (20 paired) | **0.91** (not significant) |
| Feasible both | **20/20** |

Differences are **path-dependent** (MATLAB sometimes takes more accepted position
steps and lands on a slightly better geometry), not a channel or `se_max_abs_diff`
mismatch (`~1e-15` on every seed). Campaign tables stay on CVXPY; MATLAB is
validated as equivalent for headline reporting.

Campaign SCA uses CVXPY; MATLAB CVX+MOSEK is spot-validated on seed 1 and
cross-checked on all 20 headline seeds.

---

## 5. Sensibility checklist

| Check | Result |
| --- | --- |
| Unit tests | Pass |
| 20 kHz infeasible | Yes — 0% at 100 m and 500 m; model-free cap 0.997 Mbps (§0) |
| Rates ≤ bandwidth ceiling | Yes — ~8.97 Mbps at 8.8 MHz (no cap), ~8.95 (cap25), ~2.43 at 2.4 MHz |
| SCA best on feasible points (mean) | Yes |
| λ / CPU vary slightly when feasible | Yes — &lt; 0.03 Mbps (AoDT binds at T_k) |
| T_k = 0.8 infeasible | **No** as a model limit — 0% under frozen nearest-a (\(T_{\mathrm{u2u}}\)); 20/20 with no-forwarding \(a_{ij}\) (§2.4) |
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

Three published-looking “infeasibility” or “the model can’t do that”
claims were artifacts. One remaining restriction is real but scoped to
**this sequential solver**, not Problem (P).

### False alarms (resolved — do not publish as model limits)

| # | Claim that looked like a model limit | What it actually was | Status |
| --- | --- | --- | --- |
| 1 | Settled \(S_i\)/\(L\) vs placeholders: AoDT not binding, \(\mu=200\)/s | External task size/cycles were placeholders, not Table II | **Resolved** — \(S_i=12{,}000\) bytes, \(L=3.75\times 10^6\), \(\mu\approx 53.3\)/s; AoDT binds at \(T_k=2.8\) s (§10) |
| 2 | I = 28–32 and \(f_j=0.5\times 10^8\) “infeasible” (55/60/65%) | Majority-of-association \(b_{ij}\) put both processes on one UAV; (24) failed at init and \(b\) stayed frozen | **Resolved** — rematch process→UAV when (24) fails; 20/20 feasible; old FDR calls superseded (§2.2, §2.5, §3.2) |
| 3 | \(T_k=0.8\) s “too tight for the model” (SCA/k-means/random 0%) | Frozen nearest \(a_{ij}\) splits \(N_k\); \(Q+T_{\mathrm{u2u}}\approx 0.894>0.8\) even at infinite rate | **Resolved as a model claim** — J = 1 and process-cohesive \(a_{ij}\) are 20/20 feasible; PSO 2/20 by parking both groups on one UAV (§2.4) |

### Real, scoped limitation (this solver)

**Frozen discrete \(a_{ij}\) after nearest-UAV init.** Sequential SCA never
rematches association. A legal no-forwarding map exists at the same k-means
\(q\) (and at J = 1); this code does not search it, so SCA/k-means/random
stay 0% at \(T_k=0.8\) s. \(b_{ij}\) is also frozen after init, but (24)
collisions are now repaired once. Scope: this Algorithm 1 stand-in, not
Problem (P). Changing default \(a_{ij}\) would move the campaign.

### Other (by design / out of scope)

| Item | Status |
| --- | --- |
| Area 100 × 100 m (headline) | **By design.** 20 kHz also checked at paper 500 × 500 m (§0.3) |
| TD3 (Algorithm 2) | Not implemented |
| Paper Mbps targets | Explicitly not pursued |
| 20 kHz as (27) cap | **Real** model infeasibility (ceiling 0.997 Mbps) — not a solver artifact (§0) |
| 25% per-link cap | External parameter (not Problem (P)) |
| PSO | External baseline; equal-share inner fitness |
| Eq. (17) vs Fig. 11 narrative | **Open (paper intent)** — formula ranks hetero worse than slow; not a code bug (§8.1) |

---

## 7. Suggested writeup sentences (copy-ready)

**Audit (short):**  
*Eq. (6) and constraint (27) imply \(R_{\mathrm{sum}}\le B_{\mathrm{sys}}\log_2(1+\mathrm{SNR}_{\max})\). With \(B_{\mathrm{sys}}=20\,\mathrm{kHz}\) this is at most 0.997 Mbps even at \(\mathrm{SNR}=10^{15}\), so Figs. 6–10 (7–14 Mbps, and Fig. 7 increasing with \(I\)) rule out reading Table II’s 20 kHz as the (27) sum cap. If the table’s “Minimum bandwidth allocation” is instead a per-link floor, matching Fig. 6’s 8.8 Mbps at \(I=10\) on the written channel (\(\mathrm{SNR}\approx 1.03\)) needs ~862 kHz per link — **43×** the stated 20 kHz — or an undisclosed (27) cap of ~8.6 MHz. Under the cap reading, 20 kHz is 0% feasible at both 100 × 100 m and 500 × 500 m. At feasible bandwidths, SCA outperforms k-means and PSO under
a 25% per-link cap (paired Wilcoxon, FDR q &lt; 0.05 on 25/25 unique
points vs k-means and vs PSO); the advantage over random is limited to low UAV counts and
high IoT density and is **not** significant at the default J = 3 configuration
(Wilcoxon p = 0.123, Bonferroni-adjusted p = 0.369).*

**Headline result (cap 25%, J = 3):**  
*SCA achieves 8.946 Mbps mean sum rate vs 8.901 (k-means) and 8.905 (PSO),
winning 20/20 and 19/20 paired seeds respectively (p &lt; 0.001); vs random
(8.928 Mbps) the gain is 0.019 ± 0.042 Mbps with 15/20 wins (Wilcoxon p = 0.123,
Bonferroni-adjusted p = 0.369).*

**Comment to authors / editor (Table II \(B_{\mathrm{sys}}\)):**  
*Eq. (6) and constraint (27) imply that the sum rate cannot exceed \(B_{\mathrm{sys}}\log_2(1+\mathrm{SNR}_{\max})\). Table II lists \(B_{\mathrm{sys}}=20{,}000\) Hz. Even at \(\mathrm{SNR}=10^{15}\) that ceiling is 0.997 Mbps, while Figs. 6–10 report 7–14 Mbps (Fig. 6: SCA ~8.8 Mbps at five UAVs; Fig. 7: SCA ~14 Mbps at 32 IoTs). Those axes rule out using 20 kHz as the total uplink cap in (27). Table II’s row text is “Minimum bandwidth allocation.” If 20 kHz was intended as a per-link floor rather than the (27) cap, the manuscript does not state the value (27) actually used; matching the published 8.8 Mbps at \(I=10\) on the Table II radio (\(\mathrm{SNR}\approx 1\)) would require on the order of 860 kHz per associated link (about 40× the tabulated 20 kHz), i.e. a system pool of several MHz. Could the authors confirm whether Table II’s 20 kHz is the (27) sum cap, a per-link floor, or a typographical error (for example MHz written as Hz)?*

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

**Bandwidth convention (not a code inconsistency):** Fig. 11 calls
`evaluate()` with no allocation, so `B_ij` is **equal-share** among associated
links. Campaigns re-score the same k-means geometry with the frozen-`q`
**bandwidth LP** (same LP as SCA’s B step). Equal share gives every link a
thick pipe, so upload delays are small (Eq. (17) ≈ 1.0–1.9 s here) and the
sum rate is 8.69 Mbps at J = 3. The LP starves weak links down to the AoDT
floor, so campaign k-means at the same J = 3 point is 8.90 Mbps with AoDT
binding at `T_k = 2.8` s. λ does not enter Eq. (20); rates are identical
across Fig. 11 patterns at fixed J because the radio geometry is.

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

(Fresh `fig11_8.8mhz_cap25_si12k.json`, J = 3 means: Eq. (17) fast 0.998 s, slow
1.743 s, hetero 1.886 s → hetero − slow **+0.14 s**.)

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

(Fig. 11 sensibility gate: **50/50** pass on 2026-09-05 re-run.)

**What still aligns with the paper:**

- **Simulator** mean-source and process-max: fast &lt; heterogeneous &lt; slow at
  every J = 1…5 — matches Fig. 11 narrative.
- Sum rate is **identical** across λ patterns at fixed J (8.69 Mbps at J = 3).
  λ does not enter Eq. (20). That 8.69 Mbps is **equal-share** bandwidth (Fig. 11
  `evaluate()` default). Campaign k-means at the same J = 3 uses the **LP** and
  reports 8.90 Mbps with AoDT binding at 2.8 s — see the convention note at the
  start of §8.1. Feasible fraction 100%. Rate still rises with J (8.09 → 8.86 Mbps
  from J = 1 to 5) via geometry.

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

python -m pytest
python scripts/check_bsys_20khz.py
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --solver cvxpy --out results/campaign_8.8mhz_cap25_si12k.json
python scripts/rerun_init_repair_points.py
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/analyze_sca_vs_random_losses.py results/campaign_8.8mhz_cap25_si12k.json
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --out results/fig11_8.8mhz_cap25_si12k.json
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-iterations 30
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --max-iterations 30
python scripts/compare_sca_cvxpy_matlab.py
python scripts/analyze_campaigns.py
```

Full bandwidth preset sweep (long): `scripts/run_all_bandwidth_campaigns.ps1`.

**2026-09-05 run log:** primary campaign ~9 min; init-repair ~1.5 min; analysis
scripts &lt; 2 min; fig11 ~2 min; spot-validate uses MATLAB when available.
Full bandwidth sweep (`run_all_bandwidth_campaigns.ps1`) ~36 min for six
presets. Artifacts: `results/campaign_8.8mhz_cap25_si12k_run.log`,
`results/run_all_bandwidth_campaigns.log`, `results/rerun_init_repair.log`,
`results/check_bsys_20khz.json`.

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
