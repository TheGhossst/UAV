# Observations for the paper

Snapshot: 2026-10-11. Collected from the code in `src/uavdt/` and the result artifacts listed under each item. Each observation has an ID (O1, O2, ...) so it can be referenced from the draft.

Status tags:

- **[MEASURED]**: the artifact is on disk and the number was recomputed for this file.
- **[REPORTED]**: the run was done and its numbers were recorded, but the JSON is no longer under `results/` (gitignored). Regenerate it with the listed command before citing.
- **[LEGACY 8.8 MHz]**: measured on the older 8.8 MHz protocol and not rerun at 10 MHz. Say so in the paper or rerun it.
- **[PENDING]**: still running or not started. The placeholder is `TBD`.

Conventions, unless a row says otherwise:

- Seeds 1–100 from the frozen scenario banks `data/scenario_bank/n100_i10_j3_{100m,500m}.json`.
- Defaults: I = 10, J = 3, H = 100 m, λ = 2/s, T_k = 2.8 s, f_j = 2×10⁸, B_sys = 10 MHz.
- All methods are scored by the same `evaluate()` and the same frozen-q bandwidth LP.
- A "practical" win is a paired gain above **0.05 Mbps**. A tie is |Δ| ≤ 1 bit/s.
- Paired tests are Wilcoxon signed-rank on the same seed.
- The win/loss counts below are W/L/T (wins/losses/ties).

---

## 1. Setup facts the paper must state

These are deliberate departures from Khalaf et al., or quantities the paper does not give. A reviewer will look for each of them.

| ID | Item | Value used here | Paper | Source |
| --- | --- | --- | --- | --- |
| O1 | System bandwidth `B_sys` | **10 MHz** (headline); 8.8 MHz in the legacy runs | Table II lists 20 kHz | `src/uavdt/config.py` (`PRIMARY_B_SYS_HZ`) |
| O2 | Per-link cap `max_bw_share` | **25%** (2.5 MHz/link) headline; sweep over none, 10, 12, 15, 18, 20, 22, 25% | Not in Problem (P) | `config.py`, `sca/cvx_problem.py` |
| O3 | Field | 100 × 100 m **and** 500 × 500 m | 500 × 500 m | `results/run_10mhz_cap25/` |
| O4 | Task size `S_i`, CPU cycles `L` | S = 12,000 bytes, L = 3.75×10⁶ cycles | Not specified | `docs/param.md` |
| O5 | Angle unit in Eq. (4) | Radians, as Eq. (4) is written | Ambiguous | `channel.py` (`los_angle_unit="rad"`) |
| O6 | Noise | Fixed σ² = 10⁻⁴ W, so SNR = p·g/σ² does not depend on B | Eq. (6) as written | `channel.py::snr` |
| O7 | Association and processing | Fixed after nearest-UAV initialization (`dynamic_assignment=False`) | Algorithm 1 is ambiguous | `sca/settings.py` |

**O1 to O3 are why the absolute Mbps numbers here cannot be compared with Figs. 6–10 of the original paper.** State this once, early, and present the work as a model-consistent comparison of methods, not a numeric replication.

---

## 2. Audit findings on the original paper

**O8. Table II bandwidth cannot produce the paper's figures. [MEASURED, model-free]**
Eq. (6) plus constraint (27) give R_sum ≤ B_sys·log₂(1 + SNR_max). With B_sys = 20 kHz the ceiling is **0.997 Mbps even at SNR = 10¹⁵**, while Figs. 6–10 report 7–14 Mbps. Under the 20 kHz reading every method is 0% feasible, at about 0.02 Mbps on the 100 m field and 0.012–0.013 Mbps on the 500 m field. Matching Fig. 6's 8.8 Mbps would need about 862 kHz per link, 43× the stated value. Source: `docs/results_old.md` §0, §7 (copy-ready text there).

**O9. With radians, the LoS term in Eq. (4) is essentially constant. [MEASURED]**
With the Table II constants a = 9.61 and b = 0.16 and the angle in radians, P_LoS ≈ 0.023–0.028 at every elevation. The channel is effectively free-space path loss plus a constant (about 20.4 dB), so it depends on distance only. The zenith SNR is 1.03.
If degrees are used instead, the zenith SNR is 90.5 and the default-point rate is 57.23 Mbps instead of 8.91 Mbps (8.8 MHz). That is inconsistent with every Mbps axis in the paper.
Source: `results_old.md` §0.2.

**O10. Eq. (17) ranks heterogeneous arrival rates below uniform-slow, which contradicts the Fig. 11 narrative. [LEGACY 8.8 MHz]**
The event simulator gives fast < heterogeneous < slow, which matches the paper's text. Eq. (17) does not: because it is limited by the slowest λ and penalizes Σλ, it scores the heterogeneous case as worse than uniform-slow. The code matches the published formula, so this is a question about what the authors intended, not a code defect. Source: `results_old.md` "Second audit finding", §8.1.

**O11. The paper's own DRL method (TD3, Algorithm 2) loses to SCA. [LEGACY 8.8 MHz]**

| Cell | TD3 | SCA | TD3 − SCA | Seeds |
| --- | --- | --- | --- | --- |
| 100 m, J = 3 | 8.917 | 8.946 | **−0.030** | 2/20 seeds favor TD3, p < 0.001 |
| 500 m, n100 | 7.510 | 8.228 | **−0.718** | 0/100 seeds favor TD3 |

- Across the full sweep, TD3's mean is above SCA's on **0/29** points.
- TD3 costs about 97 s per seed, against about 0.9 s for SCA.
- TD3 was not rerun at 10 MHz. The sign is unlikely to flip, but either say "8.8 MHz" or rerun it.

Source: `results_old.md` §2.10.

---

## 3. Structure of the problem

These explain why the numbers look the way they do. Put them in a "problem structure" or "discussion" subsection.

**O12. At frozen q, the bandwidth subproblem is a linear program with a closed-form solution. [MEASURED, analytic]**
Because SNR does not depend on B (O6), rate = B·SE is linear in B. With per-link floors, the per-link cap, and Σ B ≤ B_sys, the optimum is a bounded fractional knapsack:

1. Give every link its floor, max(R_min/SE, S/(SE·slack)).
2. Fill the remaining bandwidth greedily by SE, each link up to the cap.

The leftover bandwidth therefore piles onto the ⌈1/c⌉ highest-SE links (4 links at c = 25%). In the headline cell the **top four links carry about 97% of the sum rate**, and the floors use only about 7% of B_sys.

**O13. With no cap, placement barely matters on the 100 m field. [MEASURED]**
The radio ceiling is B_sys × SE_zenith = **10.213 Mbps** (one UAV directly over one IoT). With no cap, every method at 100 m is within 0.13 Mbps of it:

| Method | Mbps |
| --- | --- |
| random | 10.088 |
| k-means | 10.171 |
| SCA | 10.196 |
| multi-start | 10.199 |
| zenith-subset | 10.199 |

The per-link cap is what makes methods distinguishable, which is why the paper needs O2. Source: `results/run_10mhz_cap_sweep/nocap/100m/n100_all_summary.csv`.

**O14. At a 10% cap with I = 10, bandwidth is forced to equal share. [MEASURED, by construction]**
Ten links times a 1 MHz cap is exactly B_sys, so the LP must give every link exactly 1 MHz. The bandwidth stage has no freedom left: no knapsack and no leftover piling. This makes the cap10 row a clean control for "is the zenith gain a bandwidth-allocation artifact?" (see O27).

This holds only at I = 10. On the I-axis the cap is no longer tight, because I × 1 MHz exceeds B_sys.

**O15. The AoDT constraint is active at the sum-rate optimum. [MEASURED]**
In every LP-scored cell with cap ≥ 12% or no cap, the mean of max-AoDT is exactly T_k = 2.800 s, with a standard deviation of about 10⁻¹⁵ s across 100 seeds. This holds for every method, including random.
The LP squeezes the weakest links in each process down to their AoDT floor and spends the rest on rate. At cap10 (equal share) the AoDT is slack: 0.99 s at 100 m and 1.24 s at 500 m.
For a digital-twin paper this matters: **maximizing sum rate drives synchronization to exactly the deadline, never better.** Sources: `n100_all_summary.csv` (`mean_max_AoDT_s`) for every cap.

**O16. Arrival rate and CPU speed hardly move the objective. [MEASURED]**
- Over λ = 1.0 to 3.5 /s, SCA moves by 0.008 Mbps (100 m) and 0.137 Mbps (500 m).
- Over f_j = 0.5 to 2.5×10⁸, SCA moves by 0.008 Mbps (100 m) and 0.066 Mbps (500 m).

Both are small next to the gaps between placement methods. Source: `docs/results_new.md` sweep tables.

---

## 4. Headline comparison (10 MHz, 25% cap, n = 100) [MEASURED]

Sources: `results/run_10mhz_cap25/{100m,500m}/n100_all_summary.csv`, `results/run_10mhz_controls/analysis.txt`. Every method is 100% feasible.

| Method | 100 × 100 m | 500 × 500 m |
| --- | --- | --- |
| random | 9.950 ± 0.119 | 6.665 ± 1.026 |
| k-means | 10.110 ± 0.048 | 8.478 ± 0.655 |
| PSO (equal-share fitness, see O33) | 10.120 ± 0.045 | 9.241 ± 0.276 |
| SCA (Algorithm 1, k-means init) | 10.172 ± 0.026 | 9.418 ± 0.394 |
| multi-start SCA | 10.184 ± 0.013 | 9.589 ± 0.248 |
| p-median (k-medoids) + SCA | 10.187 ± 0.010 | 9.679 ± 0.192 |
| **zenith-subset SCA** | **10.188 ± 0.010** | **9.682 ± 0.192** |

Same-seed gaps:

| Comparison | 100 m | 500 m |
| --- | --- | --- |
| SCA − random | +0.223 ± 0.122, 100/100 | +2.753 ± 1.117, 100/100 |
| SCA − k-means | +0.062 ± 0.037, 100/100 | +0.940 ± 0.508, 100/100 |
| zenith − SCA | +0.015 ± 0.021, W/L/T 79/11/10, **9 practical** | **+0.264 ± 0.298**, W/L/T 81/5/14, **65 practical** |
| multi-start − SCA | +0.011 ± 0.019, 8 practical | +0.171 ± 0.274, 45 practical |
| zenith − multi-start | +0.004 ± 0.006, 0 practical | **+0.094 ± 0.138**, W/L/T 73/7/20, 48 practical |

**O17. The ranking is identical on both fields:** zenith-subset ≳ p-median > multi-start > SCA > PSO ≈ k-means > random.

**O18. The 100 m field is close to saturation.**
SCA reaches 99.6% of the 10.213 Mbps ceiling, and zenith-subset adds only 0.15%. The 500 m field, which is the paper's own field, is where method choice matters: zenith-subset adds +2.8% over SCA and reaches 94.8% of the ceiling. **Lead the results section with the 500 m field.**

**O19. Moving from 100 m to 500 m costs each method a different amount.**

| Method | Drop (Mbps) |
| --- | --- |
| random | 3.285 |
| k-means | 1.633 |
| SCA | 0.754 |
| multi-start | 0.595 |
| zenith-subset | 0.505 |

The better the placement, the less it loses to field size.

---

## 5. Sweeps (10 MHz, 25% cap) [MEASURED]

Source: `docs/results_new.md`, `results/run_10mhz_cap25/{100m,500m}/campaign_all.csv`.

**O20. UAV count J.**
- The zenith − SCA gap at 500 m peaks at J = 2:

  | J | Zenith − SCA (Mbps) |
  | --- | --- |
  | 1 | +0.205 |
  | 2 | **+0.325** |
  | 3 | +0.264 |
  | 4 | +0.198 |
  | 5 | +0.114 |

- At 100 m the gap is at most 0.031.
- SCA − random shrinks with J: from +0.609 to +0.160 at 100 m, and from +2.995 to +2.434 at 500 m.

**O21. IoT count I.** SCA falls as I grows: 9.426 to 8.721 at 500 m over I = 10 to 30. Zenith-subset stays above SCA at every I. At 500 m the gap is 0.270 at I = 10, then 0.210, 0.147, 0.105, and 0.127 at I = 15, 20, 25, 30.

**O22. AoDT threshold T_k.**
- At T_k = 0.8 s, every method is 0% feasible on both fields under nearest association.
- From T_k = 1.2 s upward, zenith-subset has the highest feasible mean at 500 m.
- At 500 m with T_k = 1.2 s, random is only 78% feasible.

**O23. T_k = 0.8 s is a limit of the frozen-association solver, not of Problem (P). [LEGACY 8.8 MHz]**
- Splitting a process group N_k across UAVs adds UAV-to-UAV forwarding. Queueing plus forwarding (Q + T_u2u ≈ 0.894 s) already exceeds 0.8 s even at infinite rate.
- Process-cohesive association, which keeps each N_k on one UAV, is feasible on **40/40** geometries. SCA with a cohesive candidate is feasible on **20/20**, at 8.393 Mbps. Zenith-subset SCA with the cohesive candidate is also feasible on 20/20, at **8.430 Mbps**.
- **Price of synchronization:** the throughput cost of meeting 0.8 s by grouping has a median of **0.57 Mbps** (mean 0.59, p10–p90 0.38–0.81, n = 40). The spread follows the summed SE loss (r ≈ 0.92), not the number of IoTs moved (r ≈ 0.14).

This is the clearest quantitative instance of the accuracy-versus-synchronization tension that the paper motivates. Source: `results_old.md` §2.8. It is worth rerunning at 10 MHz.

---

## 6. Where the zenith-subset gain comes from

**O24. The choice of subset matters, not just starting at some IoT or having more starts. [MEASURED, 500 m, 25%]**

| Variant (all polished by the same SCA, all keep-best) | Mean | − zenith-subset | − SCA |
| --- | --- | --- | --- |
| zenith-subset (enumerate all C(I, J) IoT subsets, score each with one LP, polish top-3) | 9.682 | — | +0.264 |
| p-median (k-medoids) subset | 9.679 | −0.003 (1 higher, 29 lower, 70 ties) | +0.261 |
| random IoT subset | 9.501 | −0.182 (0 higher) | +0.083 (p = 0.058) |
| 120 random continuous candidate points | 9.587 | −0.095 (2 higher) | +0.169 |
| multi-start SCA | 9.589 | −0.094 | +0.171 |

- Hovering over a random set of IoTs does not help: it is not significant against SCA.
- Restricting candidates to zenith points beats an equal number of continuous candidates.
- The well-chosen subset is essentially the **p-median of the IoT positions**.

Source: `results/run_10mhz_controls/analysis.txt`.

**O25. Zenith-subset is, in effect, a p-median initializer for SCA.**
- In the 25% headline, the p-median result is never meaningfully above zenith-subset (0 higher at 100 m, 1 at 500 m), and the two tie on 68–70 of 100 seeds.
- Under the proportional radio (O30) they pick the same subset on 18/20 seeds at 100 m and 15/20 at 500 m.
- At J = 2, 500 m, the p-median loses more: −0.048, 49 lower.

**Name the method for what it is: a p-median / zenith-subset initialization of SCA.** Do not present it as a new algorithm (see §11).

**O26. Most of the gain is in the starting point, not the polish. [MEASURED]**
From the `sca_anchor` diagnostics, 25% cap:

| Field | One LP at the best zenith subset, minus frozen SCA | Mean SCA polish displacement of the winner |
| --- | --- | --- |
| 500 m | +0.225 Mbps of the +0.264 total | **6.5 m** (max 12.2 m) |
| 100 m | +0.014 Mbps | **1.2 m** |

- The zenith layout is already close to an SCA stationary point.
- The kept solution comes from a zenith subset on 94/100 seeds at 500 m. On the rest, the frozen-SCA candidate was better.
- Under tighter caps the polish does more work: 23–27 m of movement and about +0.18–0.23 Mbps at 500 m for caps of 10–15%.

Source: `n100_sca_anchor.json` diagnostics (`lp_best_Mbps`, `frozen_Mbps`, `winner_polish_mean_disp_m`, `winner_kind`).

**O27. The gain is not an artifact of bandwidth piling. [MEASURED + REPORTED]** Two independent checks:
1. **Equal-share control (cap10, O14).** Bandwidth is fixed at 1 MHz per link, yet zenith-subset still gains **+0.251 Mbps at 500 m** (W/L/T 67/3/30, 53 practical). That is the largest relative gain in the sweep, at 3.5% of SCA.
2. **Bandwidth-proportional noise (O30).** The leftover no longer piles onto the cap (max link share 0.10–0.16), yet the gain persists at **+0.304 Mbps at 500 m**.

The knapsack structure (O12) explains why solutions sit at the cap. It does **not** explain the placement gain.

**O28. Geometric explanation (first-order). [analytic + REPORTED]**
Take two IoTs a distance s apart, a UAV at height H, and the low-SNR, distance-only channel of O9 (rate ∝ 1/d²). Hovering over one IoT beats the midpoint when
1/H² + 1/(H² + s²) > 2/(H² + s²/4), which holds exactly when **s > √2·H** (≈ 141 m at H = 100 m).

k-means puts each UAV at a cluster centroid, which is the midpoint-type choice. On a 100 m field most intra-cluster distances are below √2·H, so the centroid is fine and the gain is small. On a 500 m field they are not, so hovering wins. This matches O18 and O20.

Side-over-height sweep, gain of p-median over SCA as a % of SCA's rate, 5 seeds:

| side/H | noise ×1 | noise ×10 | noise ×0.1 |
| --- | --- | --- | --- |
| 1.0 | 0.78% | 0.41% | 1.03% |
| 1.67 (H = 300) | 1.16% | 0.87% | — |
| 2.0 | 2.02% | 1.06% | 2.63% |
| 2.5 (H = 200) | 2.94% | 1.92% | — |
| 3.0 | 3.00% | 1.59% | 2.76% |
| 5 | 3.90% | 2.10% | — |
| 8 | 4.78% | 2.62% | — |
| 10 (H = 50) | 3.03% | 1.52% | — |
| 16.7 (H = 30) | 2.61% | 1.36% | — |

How to read it:
- The relative gain rises with side/H up to about 8.
- Higher noise (lower SNR) lowers the relative gain. Less noise raises absolute Mbps but not the relative gain.
- The drop past side/H ≈ 8 is not established: one seed dominates (+3.02 at H = 30). Another seed is negative at every point (−0.004 to −0.33).
- At noise ×0.1, the 500 m and 800 m fields are infeasible.
- **Rerun with 20 seeds and report median/IQR before putting this figure in the paper.**

Regenerate with `python scripts/analyze/mechanism_side_over_h.py --n-runs 20`, which writes `results/mechanism_side_over_h.{json,png}`.

---

## 7. Per-link cap sweep (10 MHz, n = 100, J = 3) [MEASURED except where marked]

Source: `results/run_10mhz_cap_sweep/<cap>/<field>/n100_*.json`. The 25% row is from `results/run_10mhz_cap25/`. The paired statistics were recomputed for this file. Every cell is 100% feasible.

### 500 × 500 m

| Cap | random | k-means | SCA | multi-start | zenith | zenith − SCA (W/L/T, practical) | % of SCA | zenith − multi-start (practical) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 7.891 | 9.373 | 9.829 | 9.900 | 9.898 | +0.069 (52/13/35, 36) | 0.70% | **−0.002** (9) |
| 10% (equal share) | 4.598 | 6.600 | 7.092 | 7.274 | 7.344 | +0.251 (67/3/30, 53) | 3.54% | +0.069 (33) |
| 12% | 5.126 | 7.171 | 7.946 | 8.138 | 8.210 | +0.265 (70/7/23, 56) | 3.33% | +0.072 (30) |
| 15% | 5.680 | 7.678 | 8.640 | 8.817 | 8.903 | +0.263 (74/6/20, 62) | 3.04% | +0.086 (35) |
| 18% | TBD | TBD | TBD | TBD | TBD | **[PENDING]** | | |
| 20% | TBD | TBD | TBD | TBD | TBD | **[PENDING]** | | |
| 22% | TBD | TBD | TBD | TBD | TBD | **[PENDING]** | | |
| 25% | 6.665 | 8.478 | 9.418 | 9.589 | 9.682 | +0.264 (81/5/14, 65) | 2.81% | +0.094 (48) |

### 100 × 100 m

| Cap | random | k-means | SCA | multi-start | zenith | zenith − SCA (W/L/T, practical) | % of SCA | zenith − multi-start (practical) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 10.088 | 10.171 | 10.196 | 10.199 | 10.199 | +0.003 (50/15/35, 0) | 0.03% | −0.000 (0) |
| 10% (equal share) | 9.416 | 9.874 | 9.879 | 9.940 | 9.944 | **+0.065** (70/0/30, 34) | 0.66% | +0.004 (2) |
| 12% | 9.623 | 9.973 | 10.025 | 10.058 | 10.065 | +0.039 (63/8/29, 26) | 0.39% | +0.006 (4) |
| 15% | 9.770 | 10.037 | 10.110 | 10.127 | 10.132 | +0.022 (69/13/18, 17) | 0.22% | +0.005 (1) |
| 18% | 9.851 | 10.070 | 10.141 | 10.154 | 10.159 | +0.018 (82/8/10, 8) | 0.18% | +0.005 (0) |
| 20% | TBD | TBD | TBD | TBD | TBD | **[PENDING]** | | |
| 22% | TBD | TBD | TBD | TBD | TBD | **[PENDING]** | | |
| 25% | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 | +0.015 (79/11/10, 9) | 0.15% | +0.004 (0) |

Observations from the finished rows:

**O29a. At 500 m, a binding cap gives a flat zenith − SCA gain.** For every binding cap measured (10–25%) the gain is about **+0.25 to +0.27 Mbps**. With no cap it drops to +0.069. The relative gain rises as the cap tightens, from 2.8% to 3.5%, because SCA's own rate falls.

**O29b. At 100 m, the gain grows steadily as the cap tightens.** It goes from +0.003 (no cap) to +0.015 (25%), +0.022 (15%), +0.039 (12%), and +0.065 (10%). The practical win count rises from 0 to 34 of 100.

**O29c. With no cap, zenith-subset only ties multi-start.** The difference is −0.002 at 500 m and −0.000 at 100 m, and multi-start is slightly ahead on paired seeds (at 500 m, zenith-subset wins 23, loses 40, ties 37). The edge over multi-start exists only when a cap binds. Write it that way.

**O29d. With equal-share bandwidth on the 100 m field, SCA barely improves on its k-means start.**
- SCA − k-means is +0.005 (3/100 practical), against +0.025 to +0.073 at the other caps.
- Multi-start (+0.061) and zenith-subset (+0.065) both escape.
- This points to the k-means centroid being near a local optimum of the equal-share objective. That mechanism is not verified. Report what was observed, not the cause.

**O29e. Wall-clock per scenario is stable across caps.**

| Method | Seconds per scenario |
| --- | --- |
| SCA | 0.7–1.2 |
| multi-start | 3.1–4.4 |
| zenith-subset | 2.1–3.2 |

So zenith-subset is **cheaper than multi-start** on every finished cell.

**O29f. Statistical significance.** With ties dropped and a normal approximation, zenith − SCA has p < 10⁻⁸ in every finished cell. Zenith − multi-start has p < 10⁻² in every capped cell. The two exceptions are at no cap: p = 0.035 at 100 m, favoring multi-start, and p = 0.24 at 500 m. Recompute exact p-values with the project's analysis script for the camera-ready tables.

**[PENDING] cap25 rerun inside the sweep.** It is queued in `results/run_10mhz_cap_sweep/progress.txt`. When it finishes it should reproduce `results/run_10mhz_cap25/` exactly. Report it as a determinism check, or note any difference.

**[PENDING] Campaign axes per cap.** At the time of writing, `cap18/100m/campaign_sca_anchor` is running. The J-axis zenith − SCA values already available from `campaign_all.csv` are:

| Cap, field | J = 1 | J = 2 | J = 3 | J = 4 | J = 5 |
| --- | --- | --- | --- | --- | --- |
| 10%, 500 m | +0.043 | +0.182 | +0.251 | +0.256 | **+0.349** |
| 12%, 500 m | +0.047 | +0.222 | +0.265 | +0.183 | +0.145 |
| 15%, 500 m | +0.058 | **+0.306** | +0.263 | +0.148 | +0.152 |
| 25%, 500 m | +0.205 | **+0.325** | +0.264 | +0.198 | +0.114 |
| none, 500 m | +0.001 | +0.054 | +0.069 | +0.056 | +0.055 |

The equal-share row keeps growing with J instead of peaking at J = 2.

---

## 8. Robustness to the noise model

**O30. Bandwidth-proportional noise. [REPORTED, 20 seeds, 10 MHz, 25% cap]**
Here SNR = p·g/(N₀·B) with N₀ = σ²/(B_sys/I). The rate becomes concave in B, and the allocation is solved with CVXPY `rel_entr` (`src/uavdt/analysis/bw_proportional.py`).

| Field and radio | zenith | p-median | SCA | zenith − SCA | Max link share |
| --- | --- | --- | --- | --- | --- |
| 100 m, proportional | 9.898 | 9.896 | 9.844 | +0.054 (9/20 > 0.05) | 0.104 |
| 100 m, fixed σ² | 10.184 | 10.182 | 10.176 | +0.008 | 0.25 |
| 500 m, proportional | 7.381 | 7.358 | 7.077 | **+0.304**, p = 1×10⁻⁴ | 0.155 |
| 500 m, fixed σ² | 9.635 | 9.621 | 9.495 | +0.140 | 0.25 |

- Under the proportional radio, the optimum no longer pushes any link to the cap.
- The zenith subset chosen under fixed σ², evaluated under the proportional radio, still gains +0.255 at 500 m.
- The SCA in this script runs with `dynamic_assignment=True`, unlike the headline SCA. Say so in the paper.

Regenerate with `python scripts/analyze/compare_bw_proportional_noise.py --n-runs 20 --areas 100 500`, which writes `results/bw_proportional_noise.json`.

**O31. Polishing both starts under the proportional radio. [REPORTED, 8 seeds, Nelder–Mead]**
- At 500 m, the start gap of +0.656 shrinks to **+0.242** after polishing (8/8 seeds; p = 0.0078, the smallest possible with n = 8). The polished p-median reaches 7.438 against 7.196 for polished k-means, even though k-means moves 50–80 m during polish.
- At 100 m, local search closes the gap (−0.003).
- **The basin difference survives local search at 500 m but not at 100 m.**

Regenerate with `python scripts/analyze/polish_proportional_radio.py --n-runs 8 --areas 500 100`. More seeds would help.

---

## 9. Learning-based methods (negative results worth one subsection)

**O32. No learned method tested beats the zenith-subset / p-median initialization. [LEGACY 8.8 MHz]**

| Method | Result vs reference |
| --- | --- |
| TD3 Algorithm 2 (paper) | −0.030 vs SCA (100 m), −0.718 vs SCA (500 m); 0/29 sweep points above SCA (O11) |
| Residual TD3 on top of SCA (±10 m moves) | +0.0032 vs SCA, 4/20 wins, 0 losses; essentially neck and neck |
| CMA-ES polish of SCA | +0.0075 vs SCA, 12/20 wins |
| Association search at frozen SCA q (Experiment C) | +0.001; flat |
| Learned proposal policy, 500 m, 25% cap, 20 seeds | greedy 8.062; 24 proposals 8.363, against SCA 8.300, multi-start 8.399, zenith one-LP 8.443, zenith + polish **8.490** |
| Same, untrained network with 24 proposals | 8.209 |

- The zenith enumeration takes about 0.03 s per layout, against about 1 s for 24 learned proposals.
- At T_k = 0.8 s the learned search is feasible only when association is forced to the processing UAV, and then it ties the untrained network (8.324).

What to conclude: at I = 10, J = 3 the combinatorial space (C(10, 3) = 120 subsets) is small enough to enumerate exactly. **Learning has nothing to amortize at this size.** Any learning claim needs a regime where enumeration is too expensive, roughly C(I, J) > 10⁵, and that has not been run.

Sources: `results_old.md` §2.9 and §2.10; your learned-policy runs.

---

## 10. PSO

**O33. PSO is handicapped in this implementation. [MEASURED, code]**
PSO's inner fitness, `_fitness_equal_share` in `src/uavdt/placement/pso.py`, scores particles with equal-share bandwidth. The campaign then re-scores the final PSO layout with the capped LP. So PSO optimizes a different objective from the one it is graded on.
- With 8.8 MHz and a 15% cap, the LP-minus-equal-share gap was +0.146 for k-means and +0.110 for PSO. Source: `results_old.md` §3.4.
- The cap10 cell (O14) is the one setting where equal share **is** the scored objective, so PSO there would be an unhandicapped comparison. **[PENDING: PSO has not been run at cap10.]**

**O34. PSO never beats zenith-subset on any seed. [MEASURED, n100, 10 MHz]**

| Cap, field | PSO | PSO − SCA (seeds higher) | PSO − zenith (seeds higher) |
| --- | --- | --- | --- |
| none, 100 m | 10.174 | −0.023 (5) | −0.026 (0) |
| none, 500 m | 9.639 | −0.190 (10) | −0.259 (0) |
| 12%, 100 m | 10.014 | −0.012 (21) | −0.051 (0) |
| 12%, 500 m | 7.992 | **+0.046** (27) | −0.218 (0) |
| 15%, 100 m | 10.063 | −0.047 (9) | −0.069 (0) |
| 15%, 500 m | 8.617 | −0.023 (29) | −0.286 (0) |
| 25%, 100 m | 10.120 | −0.052 (3) | −0.068 (0) |
| 25%, 500 m | 9.241 | −0.177 (26) | −0.441 (0) |

- PSO beats SCA in the mean only at 12%, 500 m.
- PSO beats SCA at J = 1, 500 m: +0.015, 19/100 seeds.
- In the 25% campaign, PSO is never significantly above SCA (BH-FDR, 0/24 points on both fields).

Either rerun PSO with an LP fitness, or keep it in the paper with the handicap stated.

Sources: `results/run_10mhz_controls/{cap}/{field}/n100_pso.json` paired with the sweep's SCA and zenith files; `analysis.txt`.

---

## 11. Claims not to make (threats to validity)

1. **Do not claim the zenith-subset or anchoring idea as a new algorithm.** IoT-anchored, k-medoids and p-median initialization of UAV placement exists in prior work (Xie/Xu/Zhang arXiv 1801.04545; Shen et al. TSP 2020; El Hammouti IoT-J 2021; MDPI Telecom 2022; Sensors 2022 PMC9698383). The contribution is the reproduction, the audit, and the regime analysis: when and why centroid-initialized SCA loses.
2. **Do not quote the "upper bound" or "gap closed" numbers.**
   - `leftover_dump_upper_bound` in `sca_anchor.py` maximizes only over layouts with a UAV directly above an IoT, so it is not a bound on (P).
   - Zenith-subset lands **above** this "bound" on 6/100 seeds (100 m) and 3/100 seeds (500 m) (`analysis.txt`).
   - Drop "2.4% gap" and "closes 41% of the gap".
3. **Do not report p-values for zenith-subset vs its own frozen SCA candidate.** Keep-best makes it never worse by construction. The p-values above are against the **separately solved** SCA column, where losses can still occur (0–13 per 100 across the finished cells).
4. **Zenith ≥ p-median at the one-LP stage holds by construction**, because the p-median subset is one of the enumerated subsets. Do not present that as an empirical win.
5. **Do not say "more noise increases the gain".** O28 shows the reverse for relative gain.
6. **Small samples:** the polish study is 8 seeds and the side/H study is 5 seeds. Rerun both at 20 or more seeds before citing them as figures.
7. **Fix the protocol mix.** TD3, the T_k = 0.8 cohesive result, the learned policy, and the Fig. 11 / Eq. (17) audit are at 8.8 MHz. The headline is 10 MHz. Either label each table or rerun.
8. **The per-link cap is the authors' extension.** Every result that separates the methods depends on it (O13). State that it is not part of Problem (P), and show the no-cap row.
9. **The SCA solver is fixed by design.** Association and processing are frozen after initialization, and `src/uavdt/sca/` was not modified. Any comparison with "dynamic" SCA (O30) must say which variant was used.

---

## 12. Pending results checklist

| Item | Status | Fills |
| --- | --- | --- |
| Cap sweep: 18% at 500 m; 20% and 22% on both fields | Queued on the server (`run_10mhz_cap_sweep/progress.txt`) | §7 tables, O29 |
| Cap sweep: 18% at 100 m, campaign axes | `campaign_sca_anchor` running | §7 J-axis table |
| Cap sweep: 25% rerun | Queued | Determinism check against `run_10mhz_cap25` |
| Side/H mechanism at 20 seeds, relative gain with median/IQR | Not started | O28 figure |
| Proportional-noise and polish JSONs | Not on disk; regenerate | O30, O31 |
| PSO at cap10 (unhandicapped), or PSO with LP fitness | Not started | O33 |
| TD3, T_k = 0.8 cohesive, and Fig. 11 at 10 MHz | Optional | O10, O11, O23 |
| Exact Wilcoxon p-values for §7 | After the sweep finishes | O29f |

---

## 13. Artifact index

| Path | What it holds |
| --- | --- |
| `results/run_10mhz_cap25/{100m,500m}/` | Headline n100 bank and campaign sweeps (25% cap) |
| `results/run_10mhz_cap_sweep/<cap>/{100m,500m}/` | Cap sweep, n100 and campaign |
| `results/run_10mhz_controls/analysis.txt` | PSO, p-median, random-subset, and continuous controls; PSO at other caps |
| `docs/results_new.md` | 10 MHz headline tables and sweeps |
| `docs/results_old.md` | 20 kHz audit (§0), radians (§0.2), T_k = 0.8 / SCA-joint (§2.8), residual/C (§2.9), TD3 (§2.10), PSO proxy (§3.4), Fig. 11 (§8) |
| `docs/novelty.md` | Zenith-subset method note and its "do not write" list |
| `src/uavdt/sca_anchor.py` | Zenith-subset SCA: enumeration, LP scoring, top-K polish, keep-best |
| `src/uavdt/analysis/bw_proportional.py` | Proportional-noise allocator |
| `scripts/analyze/{compare_bw_proportional_noise,polish_proportional_radio,mechanism_side_over_h}.py` | Robustness scripts for §8 and O28 |
