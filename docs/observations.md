# Observations for the paper

Snapshot: 2026-10-24. Recomputed from the code in `src/uavdt/` and the result JSON/CSV under `results/`. Each observation has an ID (O1, O2, ...) so it can be referenced from the draft. Rows added or replaced on 2026-10-24 are the dynamic-anchor twins, the frozen-assignment SCA bank, TD3 and PSO across the cap family, residual TD3, Fig. 11, and the 20-seed noise checks.

Status tags:

- **[MEASURED]**: the artifact is on disk and the number was recomputed for this file. The 2026-10-24 pass recomputed the rows named in the snapshot line. Earlier rows were recomputed on 2026-10-20.
- **[REPORTED]**: the run was done and its numbers were recorded, but the JSON is no longer under `results/`. Regenerate it with the listed command before citing.
- **[LEGACY 8.8 MHz]**: measured on the older 8.8 MHz protocol and not rerun at 10 MHz. Say so in the paper or rerun it.
- **[PENDING]**: not started, or the artifact is not on disk. The placeholder is `TBD`.

Conventions, unless a row says otherwise:

- Seeds 1–100 from the frozen scenario banks `data/scenario_bank/n100_i10_j3_{100m,500m}.json`.
- Defaults: I = 10, J = 3, H = 100 m, λ = 2/s, T_k = 2.8 s, f_j = 2×10⁸, B_sys = 10 MHz.
- The AoDT axis is 1.2, 1.6, 2.0, 2.4, 2.8, 3.2 s. T_k = 0.8 s is not a sweep point. Older campaign JSON still stores it; those scores are not used.
- All methods are scored by the same `evaluate()`.
- A "practical" win is a paired gain above **0.05 Mbps**. A tie is |Δ| ≤ 1 bit/s (10⁻⁶ Mbps). W/L/T counts use that tie rule.
- Paired tests are the two-sided Wilcoxon signed-rank in `scripts/lib/paired_winrate.py`. Exact zeros are dropped. A gap below 1 bit/s that is not an exact zero stays in the test, so the nonzero count can exceed W+L.
- When every nonzero gap has the same sign, the one-sided p is half the two-sided p. `docs/results_new.md` prints that one-sided tail for some anchor rows. The p-values in this file are two-sided.
- Means are over 100 seeds. ± is the sample standard deviation.

Two anchor artifacts must not be pooled without a label:

- **Dynamic anchor** is `results/run_10mhz_cap25/{100m,500m}/n100_sca_anchor.json` (2026-10-16; byte-identical to `n100_sca_anchor_dyn.json`), the 100 m campaign (2026-10-17), the 500 m campaign `campaign_sca_anchor_dyn.json` (2026-10-24), and `results/run_10mhz_cap_sweep/<cap>/{100m,500m}/n100_sca_anchor_dyn.json` for no cap and for 10–22%.
- **Frozen anchor** is `n100_sca_anchor_frozen.json`, every sweep file named `n100_sca_anchor.json`, the 500 m campaign file `campaign_sca_anchor.json` (2026-10-08), and the medoid / random-subset / continuous controls. Association and processing stay at the initializer.

---

## 1. Setup facts the paper must state

These are deliberate departures from Khalaf et al., or quantities the paper does not give. A reviewer will look for each of them.

| ID | Item | Value used here | Paper | Source |
| --- | --- | --- | --- | --- |
| O1 | System bandwidth `B_sys` | **10 MHz** (headline); 8.8 MHz in the legacy runs | Table II lists 20 kHz | `src/uavdt/config.py` (`PRIMARY_B_SYS_HZ`) |
| O2 | Per-link cap `max_bw_share` | **25%** (2.5 MHz/link) headline; sweep over none, 10, 12, 15, 18, 20, 22, 25% | Not in Problem (P) | `config.py`, `sca/cvx_problem.py` |
| O3 | Field | 100 × 100 m **and** 500 × 500 m | 500 × 500 m | `results/run_10mhz_cap25/` |
| O4 | Task size `S_i`, CPU cycles `L` | S = 12,000 bytes (96,000 bit), L = 3.75×10⁶ cycles | Not specified | `config.py` (`EXTERNAL_TASK_SIZE_BYTES`) |
| O5 | Angle unit in Eq. (4) | Radians, as Eq. (4) is written | Ambiguous | `channel.py` (`los_angle_unit="rad"`) |
| O6 | Noise | Fixed σ² = 10⁻⁴ W (`sigma = 0.01`), so SNR = p·g/σ² does not depend on B | Eq. (6) as written | `channel.py::snr` |
| O7 | Association and processing | **Not one policy.** See below. | Algorithm 1 is ambiguous | `experiments/methods.py`, `sca_multistart.py` |

**O7. Assignment policy depends on the method name. [MEASURED, code + artifacts]**

`SCASettings.dynamic_assignment` defaults to `False`. The runners then override it:

- `run_method("sca")` and `run_method("sca_anchor")` set `dynamic_assignment=True`. After each accepted (q, B) step, `a_ij` and `b_ij` may be updated by a true-gated block-coordinate step (`sca/assignment.py`). This is outside the convex subproblem.
- `run_method("sca_multistart")` forces `dynamic_assignment=False` (`sca_multistart.py`).
- `run_method("sca_anchor_frozen")` forces `False`.

What the stored headline banks actually did:

| Method | Flag | 100 m | 500 m |
| --- | --- | --- | --- |
| SCA | on, 100/100 | association changed on **43/100**; processing on **44/100**; ≥1 update accepted on **71/100** (172 accepts) | association **37/100**; processing **37/100**; accepted on **59/100** (112 accepts) |
| dynamic anchor | on, 100/100 | association changed on **8/100**; processing on **30/100**; accepted on **35/100** | association **8/100**; processing **36/100**; accepted on **38/100** |
| multi-start | off, 0/100 | no reassignment | no reassignment |
| frozen anchor | off, 0/100 | no reassignment | no reassignment |

Cap-sweep SCA is dynamic on all 16 cap×field banks. The sweep file named `n100_sca_anchor.json`, and multi-start, are frozen on all 16. The dynamic twin is `n100_sca_anchor_dyn.json` for no cap and for 10–22% (O35). The 25% dynamic bank is the headline file.

**Rate cost of SCA's own accepted updates. [MEASURED, 25% cap, n = 100]**

`results/run_10mhz_cap25/{100m,500m}/n100_frozen_sca.json` is the same SCA with the flag off. Dynamic SCA minus that file:

| Field | Mean | W/L/T | Practical | p |
| --- | --- | --- | --- | --- |
| 100 m | +0.004 ± 0.020 | 58/4/38 | 1 | 2.9×10⁻¹² |
| 500 m | +0.024 ± 0.067 | 53/1/46 | 19 | 5.7×10⁻¹² |

The frozen file changes association and processing on **0/100** seeds. Mean wall-clock is 0.88 s versus 1.07 s at 100 m, and 1.03 s versus 0.96 s at 500 m. The zenith−SCA gap at 500 m (+0.266) is about ten times the assignment gap on that field.

**O1 to O3 are why the absolute Mbps numbers here cannot be compared with Figs. 6–10 of the original paper.** State this once, early, and present the work as a model-consistent comparison of methods, not a numeric replication.

---

## 2. Audit findings on the original paper

**O8. Table II bandwidth cannot produce the paper's figures. [MEASURED, model-free]**
Eq. (6) plus constraint (27) give R_sum ≤ B_sys·log₂(1 + SNR_max). With B_sys = 20 kHz the ceiling is **0.997 Mbps even at SNR = 10¹⁵**, while Figs. 6–10 report 7–14 Mbps. Under the 20 kHz reading every method is 0% feasible, at about 0.02 Mbps on the 100 m field and 0.012–0.013 Mbps on the 500 m field. Matching Fig. 6's 8.8 Mbps would need about 862 kHz per link, 43× the stated value. Source: `docs/results_old.md` §0, §7 (copy-ready text there).

**O9. With radians, the LoS term in Eq. (4) is essentially constant. [MEASURED]**
With the Table II constants a = 9.61 and b = 0.16 and the angle in radians, P_LoS ≈ 0.023–0.028 at every elevation. The channel is effectively free-space path loss plus a constant (about 20.4 dB), so it depends on distance only. The zenith SNR is 1.03.
If degrees are used instead, the zenith SNR is 90.5 and the default-point rate is 57.23 Mbps instead of 8.91 Mbps (8.8 MHz). That is inconsistent with every Mbps axis in the paper.
Source: `results_old.md` §0.2.

**O10. Eq. (17) ranks heterogeneous arrival rates below uniform-slow, which contradicts the Fig. 11 narrative. [MEASURED at 10 MHz; the original audit is LEGACY 8.8 MHz]**
The event simulator gives fast < heterogeneous < slow, which matches the paper's text. Eq. (17) does not: because it is limited by the slowest λ and penalizes Σλ, it scores the heterogeneous case as worse than uniform-slow. The code matches the published formula, so this is a question about what the authors intended, not a code defect. Source of the original audit: `results_old.md` "Second audit finding", §8.1.

The 10 MHz rerun is `results/run_10mhz_cap25/fig11/{100m,500m}/fig11.json` (k-means, 25% cap, 20 seeds, J = 1…5). At J = 3:

| Field | Score | Fast | Slow | Heterogeneous |
| --- | --- | --- | --- | --- |
| 100 m | Eq. (17) | 0.984 s | 1.730 s | 1.872 s |
| 100 m | FCFS simulator | 1.441 s | 3.118 s | 2.142 s |
| 500 m | Eq. (17) | 1.091 s | 1.838 s | 2.007 s |
| 500 m | FCFS simulator | 1.524 s | 3.193 s | 2.210 s |

On **20/20** seeds, at every J from 1 to 5, on both fields, Eq. (17) ranks heterogeneous older than uniform-slow and the simulator ranks uniform-slow older than heterogeneous. Fast is youngest on both scores.

**O11. Algorithm 2 TD3 loses to SCA at 10 MHz, on both fields and on every campaign point. [MEASURED]**

The run is `results/run_10mhz_td3/` (4/4 jobs `ok`) and the copy under `results/run_10mhz_cap25/{100m,500m}/n100_td3.json`. Preset `alg2`, 7000 steps, k-means initialization, policy export, CUDA. Code defaults that the paper does not publish: `discount = 0`, `uav_init = "kmeans"`, `total_steps = 7000` (`td3/settings.py`). The 8.8 MHz TD3 numbers in `results_old.md` §2.10 are the same sign and are not the headline.

Frozen bank, 25% cap, I = 10, J = 3, 100% feasible, max AoDT pinned at 2.800 s:

| Field | TD3 | SCA | SCA − TD3 | TD3 − k-means |
| --- | --- | --- | --- | --- |
| 100 m | 10.130 ± 0.042 | 10.172 ± 0.026 | **+0.043 ± 0.031**, 97/3/0, 33 practical, p = 2.7×10⁻²⁷ | **+0.019 ± 0.017**, 96/4/0, 6 practical, p = 8.1×10⁻²⁶ |
| 500 m | 8.559 ± 0.658 | 9.418 ± 0.394 | **+0.859 ± 0.502**, 100/0/0, 96 practical, p = 1.6×10⁻³⁰ | **+0.081 ± 0.061**, 99/1/0, 64 practical, p = 3.2×10⁻³⁰ |

- TD3 − random is +0.180 ± 0.126 (96/4/0, 88 practical) at 100 m and +1.894 ± 1.202 (95/5/0, 94 practical) at 500 m.
- The J = 3 campaign is a fresh train. It sits 0.0005 Mbps under the 100 m bank (per-seed |Δ| up to 0.062) and 0.0025 Mbps above the 500 m bank (per-seed |Δ| up to 0.172).
- On the 24 unique points stored in the campaigns, SCA − TD3 has positive mean and BH-FDR q < 0.05 on **24/24** on both fields. One of those points is the unused T_k = 0.8 s row. The mean exceeds 0.05 Mbps on **3/24** at 100 m (J = 1, J = 2, and that unused row) and on **24/24** at 500 m.
- SCA − TD3 shrinks with J. At 100 m, +0.202 ± 0.126 at J = 1 (92 practical) to +0.009 ± 0.009 at J = 5 (0 practical, 6 losses, p = 7.1×10⁻¹⁷). At 500 m, +1.660 ± 0.881 at J = 1 (99 practical) to +0.205 ± 0.194 at J = 5 (76 practical, 1 loss).
- It also shrinks with I. At 500 m, +0.845 ± 0.509 at I = 10 to +0.404 ± 0.282 at I = 30 (98 practical).
- λ and CPU barely move the gap. SCA − TD3 stays inside +0.042 to +0.044 on the 100 m λ grid and inside +0.796 to +0.866 on the 500 m λ grid.
- Bank wall-clock is **69 s/seed** at 100 m and **71 s/seed** at 500 m. Campaign wall-clock is **57 s** (median 56 s) at 100 m and **78 s** (median 71 s) at 500 m, against about 1 s for SCA.
- Dynamic anchor is ahead of TD3 on every seed of every campaign point (28/28 points at 100 m; 27/27 reported points at 500 m, AoDT from 1.2 s). The 500 m anchor campaign used for that check is now `campaign_sca_anchor_dyn.json`. The frozen file `campaign_sca_anchor.json` remains on disk.
- The same Algorithm 2 preset on the other caps is an n100 bank, not a campaign. See O36.

---

## 3. Structure of the problem

These explain why the numbers look the way they do. Put them in a "problem structure" or "discussion" subsection.

**O12. At frozen q, the bandwidth subproblem is a linear program with a closed-form solution. [MEASURED, analytic]**
Because SNR does not depend on B (O6), rate = B·SE is linear in B. With per-link floors, the per-link cap, and Σ B ≤ B_sys, the optimum is a bounded fractional knapsack:

1. Give every link its floor, max(R_min/SE, S/(SE·slack)).
2. Fill the remaining bandwidth greedily by SE, each link up to the cap.

The leftover bandwidth therefore piles onto the ⌈1/c⌉ highest-SE links (4 links at c = 25%). In the headline cell the **top four links carry about 97% of the sum rate**, and the floors use only about 7% of B_sys. That 97% figure was measured earlier and was not recomputed in this pass: the saved runs do not store the bandwidth matrix.

**O13. With no cap, placement barely matters on the 100 m field. [MEASURED]**
The radio ceiling is B_sys × SE_zenith = **10.213 Mbps** (one UAV directly over one IoT; stored `radio_ceiling_Mbps` is 10.213295). With no cap, every method at 100 m is within 0.13 Mbps of it:

| Method | Mbps |
| --- | --- |
| random | 10.088 |
| k-means | 10.171 |
| SCA | 10.196 |
| multi-start | 10.199 |
| zenith-subset (frozen polish) | 10.199 |

The per-link cap is what makes methods distinguishable, which is why the paper needs O2. Source: `results/run_10mhz_cap_sweep/nocap/100m/n100_*.json`.

**O14. At a 10% cap with I = 10, bandwidth is forced to equal share. [MEASURED, by construction]**
Ten links times a 1 MHz cap is exactly B_sys, so the LP must give every link exactly 1 MHz. The bandwidth stage has no freedom left: no knapsack and no leftover piling. This makes the cap10 row a clean control for "is the zenith gain a bandwidth-allocation artifact?" (see O27).

This holds only at I = 10. On the I-axis the cap is no longer tight, because I × 1 MHz exceeds B_sys.

**O15. The AoDT constraint is active at the sum-rate optimum, except under equal share. [MEASURED]**
On every cap×field bank except 10%, the mean of max-AoDT is 2.800 s for SCA, with a standard deviation of about 10⁻¹⁵ s across 100 seeds. This holds for random, k-means, multi-start, anchor, and TD3 at the 25% default point.
The LP squeezes the weakest links in each process down to their AoDT floor and spends the rest on rate. At cap10 (equal share) the AoDT is slack: **0.994 s** at 100 m and **1.244 s** at 500 m for the sweep methods. PSO and Algorithm 2 TD3 match that slack: PSO **0.993 s** and **1.249 s**, TD3 **0.994 s** and **1.135 s**.
For a digital-twin paper this matters: **maximizing sum rate drives synchronization to exactly the deadline, never better**, once the cap leaves the allocator any freedom.

**O16. Arrival rate and CPU speed hardly move the objective. [MEASURED]**
From the 25% campaigns:

| Axis | 100 m SCA | 500 m SCA | 100 m TD3 | 500 m TD3 |
| --- | --- | --- | --- | --- |
| λ = 1.0 to 3.5 /s | 0.007 Mbps | 0.137 Mbps | 0.006 Mbps | 0.071 Mbps |
| f_j = 0.5 to 2.5×10⁸ | 0.008 Mbps | 0.066 Mbps | 0.003 Mbps | 0.033 Mbps |

Both are small next to the gaps between placement methods.

**O35. Turning dynamic assignment on inside the zenith polish barely moves the rate. [MEASURED]**
Same enumeration, same seeds. Dynamic anchor minus frozen anchor:

| Field | Mean | W/L/T | Practical | p |
| --- | --- | --- | --- | --- |
| 100 m | +0.0001 ± 0.0002 | 31/0/69 | 0 | 6.5×10⁻⁷ |
| 500 m | +0.0019 ± 0.0047 | 31/0/69 | 0 | 1.5×10⁻¹¹ |

Per-seed |Δ| reaches 0.0014 Mbps at 100 m and 0.025 Mbps at 500 m. The placement gain over SCA (O17) is not an assignment-mode artifact. What does change is the win/loss count: dynamic anchor keep-bests against the dynamic SCA, so it has **0 losses** against the SCA column. Frozen anchor keep-bests against a frozen SCA, which is a different reference, and it still loses 11 seeds at 100 m and 5 at 500 m. Do not describe those losses as the dynamic anchor failing.

The same A/B now covers every cap, both fields (`n100_sca_anchor_dyn.json` beside the frozen sweep file; the 25% row is the headline pair above). Dynamic minus frozen is at most **+0.004 Mbps** (500 m, 20%). The practical count is **0/100** on 14 of the 16 cells and **1/100** on the other two (500 m at 18% and at 20%). Two-sided p is below 0.05 on every cell, including no cap at 100 m (p = 0.046, mean +0.0001). Dynamic versus the stored dynamic SCA has **0 losses on all 16 cells**. Mean wall-clock of the dynamic polish is **3.0–5.6 s** per scenario, against **2.1–4.4 s** for the frozen polish. The 500 m dynamic campaign is the same story at every reported sweep point: mean shift at most +0.016 Mbps (T_k = 1.2 s), 0 losses against SCA (O20).

---

## 4. Headline comparison (10 MHz, 25% cap, n = 100) [MEASURED]

Sources: `results/run_10mhz_cap25/{100m,500m}/n100_*.json`, `results/run_10mhz_controls/cap25/`, `results/run_10mhz_td3/`. Every method is 100% feasible. Anchor in this table is the **dynamic** bank unless the row says frozen. PSO is the equal-share-fitness implementation (O33).

| Method | 100 × 100 m | 500 × 500 m |
| --- | --- | --- |
| random | 9.950 ± 0.119 | 6.665 ± 1.026 |
| k-means | 10.110 ± 0.048 | 8.478 ± 0.655 |
| PSO | 10.120 ± 0.045 | 9.241 ± 0.276 |
| TD3 (Algorithm 2) | 10.130 ± 0.042 | 8.559 ± 0.658 |
| SCA (k-means init, dynamic assignment) | 10.172 ± 0.026 | 9.418 ± 0.394 |
| multi-start SCA (frozen assignment) | 10.184 ± 0.013 | 9.589 ± 0.248 |
| frozen zenith-subset | 10.188 ± 0.010 | 9.682 ± 0.192 |
| **dynamic zenith-subset** | **10.188 ± 0.010** | **9.684 ± 0.192** |

Same-seed gaps:

| Comparison | 100 m | 500 m |
| --- | --- | --- |
| SCA − random | +0.223 ± 0.122, 100/0/0, 95 practical | +2.753 ± 1.117, 100/0/0, 100 practical |
| SCA − k-means | +0.062 ± 0.037, 100/0/0, 56 practical | +0.940 ± 0.508, 100/0/0, 97 practical |
| SCA − TD3 | +0.043 ± 0.031, 97/3/0, 33 practical | +0.859 ± 0.502, 100/0/0, 96 practical |
| dynamic zenith − SCA | +0.015 ± 0.021, 80/0/20, **9 practical**, p = 1.7×10⁻²⁴ | **+0.266 ± 0.298**, 84/0/16, **65 practical**, p = 8.1×10⁻²⁸ |
| frozen zenith − SCA | +0.015 ± 0.021, 79/11/10, 9 practical | +0.264 ± 0.298, 81/5/14, 65 practical |
| multi-start − SCA | +0.011 ± 0.019, 58/20/22, 8 practical, p = 2.4×10⁻⁹ | +0.171 ± 0.274, 52/20/28, 45 practical, p = 1.5×10⁻⁸ |
| dynamic zenith − multi-start | +0.004 ± 0.006, 76/12/12, **0 practical** | **+0.096 ± 0.138**, 79/3/18, **48 practical** |

The 500 m two-sided p uses 91 nonzero gaps. Dropping the sub-bit gaps leaves 84, and the one-sided p on those 84 is about 5.4×10⁻²⁶, which is the 5.2×10⁻²⁶ printed in `results_new.md`. The 100 m one-sided p on 80 exact wins is 8.3×10⁻²⁵. Cite one convention.

**O17. The order of the headline methods is the same on both fields.**
Dynamic zenith ≳ frozen zenith > multi-start > SCA > TD3 > k-means > random.
PSO does not sit in one place in that order. At 100 m it is between TD3 and k-means (10.120). At 500 m it is between SCA and TD3 (9.241), **+0.682 ± 0.577** above TD3 (94/6/0, 89 practical, p = 3.9×10⁻²⁶) and still below SCA (−0.177 ± 0.344, 26 seeds higher). At this 25% point, PSO is above dynamic zenith on **0/100** seeds on both fields. The only seeds where it is higher are at 100 m and 10% (O34), and none of those clear 0.05 Mbps.

**O18. The 100 m field is close to saturation.**
SCA reaches 99.6% of the 10.213 Mbps ceiling, and dynamic zenith adds 0.15% (9/100 practical). Wilcoxon significance at 100 m is cheap: SCA beats k-means on 100/100 seeds and only 56 of those clear 0.05 Mbps. The 500 m field, which is the paper's own field, is where method choice matters: dynamic zenith adds +2.8% over SCA, 65/100 practical, and reaches 94.8% of the ceiling. **Lead the results section with the 500 m field.**

**O19. Moving from 100 m to 500 m costs each method a different amount.**

| Method | Drop (Mbps) |
| --- | --- |
| random | 3.285 |
| k-means | 1.633 |
| TD3 | 1.571 |
| PSO | 0.879 |
| SCA | 0.754 |
| multi-start | 0.595 |
| frozen zenith | 0.505 |
| dynamic zenith | 0.503 |

The better the placement, the less it loses to field size. TD3 loses about as much as k-means. PSO, despite the equal-share handicap, loses less than TD3.

---

## 5. Sweeps (10 MHz, 25% cap) [MEASURED]

Source: `results/run_10mhz_cap25/{100m,500m}/campaign_*.json` and `results/run_10mhz_td3/cap25/`. Both anchor campaigns are the dynamic polish. The 500 m file is `campaign_sca_anchor_dyn.json` (27 points, AoDT from 1.2 s). Its J = 3 mean is **9.684**, matching the dynamic bank. The frozen campaign `campaign_sca_anchor.json` is still on disk; its largest mean shift versus the dynamic file is **+0.016 Mbps** at T_k = 1.2 s.

**O20. UAV count J.**
- Dynamic-anchor campaign at 500 m, 25% cap. The zenith − SCA gap peaks at J = 2, and the dynamic polish has 0 losses at every J:

  | J | Zenith − SCA (Mbps) |
  | --- | --- |
  | 1 | +0.205 (43/100, 57 ties, 31 practical) |
  | 2 | **+0.326** (80/100, 20 ties, 65 practical) |
  | 3 | +0.266 (84/100, 16 ties, 65 practical) |
  | 4 | +0.200 (91/100, 9 ties, 73 practical) |
  | 5 | +0.114 (90/100, 10 ties, 59 practical) |

  Against the frozen campaign the J = 3 shift is +0.002 Mbps and J = 1 is an exact tie on 100/100 seeds. BH-FDR on the 23 unique reported points is **23/23**, every mean above 0.05 Mbps. T_k = 0.8 s is not in this file.

- At 100 m and 25%, the dynamic-anchor gap is at most **+0.031** (J = 2). Across the cap sweep the 100 m gap is larger under a tight cap: **+0.065** at 10%, J = 3.
- SCA − random shrinks with J: from +0.609 to +0.160 at 100 m, and from +2.995 to +2.434 at 500 m.
- SCA − TD3 also shrinks with J (O11). TD3 − k-means at 100 m, J = 5 is only +0.006 ± 0.007 (0 practical).

**O21. IoT count I.** SCA falls as I grows: 9.426 to 8.721 at 500 m over I = 10 to 30. Dynamic zenith stays above SCA at every I, with 0 losses. At 500 m the paired gap is **+0.275** at I = 10, then **+0.213**, **+0.151**, **+0.109**, and **+0.134** at I = 15, 20, 25, 30 (means 9.701, 9.546, 9.322, 9.089, 8.855). The frozen campaign is 0.003–0.006 Mbps lower. The I = 10 row is a separate solve from the J = 3 row (SCA 9.426 vs 9.418). Use J = 3 as the headline default.

**O22. AoDT threshold T_k, at the 25% cap.** The axis starts at 1.2 s.
- From T_k = 1.2 s upward, every optimized method is 100% feasible. At 500 m, random is only **78%** feasible at T_k = 1.2 s. Anchor has the highest feasible mean on that row.
- Feasibility at T_k = 1.2 s on the 500 m field gets worse as the cap tightens. This is not only a random-placement failure:

  | Cap | random | k-means | SCA | anchor |
  | --- | --- | --- | --- | --- |
  | 10% | 11% | 80% | 87% | 99% |
  | 12% | 33% | 89% | 92% | 100% |
  | 15% | 47% | 95% | 96% | 100% |
  | 18% | 59% | 99% | 99% | 100% |
  | 20% | 67% | 100% | 100% | 100% |
  | 22% | 73% | 100% | 100% | 100% |
  | 25% | 78% | 100% | 100% | 100% |
  | none | 91% | 100% | 100% | 100% |

  At 100 m, T_k = 1.2 s is 100% feasible for every method at every cap. The 10% cap is also the only UAV-count cell below 100%: at 500 m, random is 98% at J = 1 and 99% at J = 2.

**O23. T_k = 0.8 s is a limit of the frozen-association solver, not of Problem (P). [LEGACY 8.8 MHz, not used]**
- Splitting a process group N_k across UAVs adds UAV-to-UAV forwarding. Queueing plus forwarding (Q + T_u2u ≈ 0.894 s) already exceeds 0.8 s even at infinite rate.
- Process-cohesive association, which keeps each N_k on one UAV, is feasible on **40/40** geometries. SCA with a cohesive candidate is feasible on **20/20**, at 8.393 Mbps. Zenith-subset SCA with the cohesive candidate is also feasible on 20/20, at **8.430 Mbps**.
- **Price of synchronization:** the throughput cost of meeting 0.8 s by grouping has a median of **0.57 Mbps** (mean 0.59, p10–p90 0.38–0.81, n = 40). The spread follows the summed SE loss (r ≈ 0.92), not the number of IoTs moved (r ≈ 0.14).

Source: `results_old.md` §2.8. This is not part of the reported sweep. The AoDT axis starts at 1.2 s.

---

## 6. Where the zenith-subset gain comes from

**O24. The choice of subset matters, not just starting at some IoT or having more starts. [MEASURED, 500 m, 25%]**

Controls are frozen-assignment polishes. Compare them with the frozen anchor so the assignment mode matches. Means below are the control's own mean.

| Variant (all polished by SCA, all keep-best) | Mean | − frozen zenith | − SCA |
| --- | --- | --- | --- |
| dynamic zenith (headline bank) | 9.684 | +0.002 (31 higher, 0 lower, 69 ties) | +0.266 |
| frozen zenith | 9.682 | — | +0.264 (81/5/14) |
| p-median (k-medoids) subset | 9.679 | −0.003 (1 higher, 29 lower, 70 ties) | +0.261 |
| random IoT subset | 9.501 | −0.182 (0 higher) | +0.083 (p = 0.058) |
| 120 random continuous candidate points | 9.587 | −0.095 (2 higher) | +0.169 |
| multi-start SCA | 9.589 | −0.094 | +0.171 |

- Hovering over a random set of IoTs does not help: it is not significant against SCA (31/39/30, p = 0.058).
- Restricting candidates to zenith points beats an equal number of continuous candidates.
- The well-chosen subset is essentially the **p-median of the IoT positions**.

Source: `results/run_10mhz_controls/cap25/500m/n100_{medoid,random_anchor,continuous}.json`, paired here against the frozen and dynamic banks.

**O25. Zenith-subset is, in effect, a p-median initializer for SCA.**
- Against the frozen anchor, p-median is never meaningfully above it (0 higher at 100 m, 1 at 500 m). They tie on 68 and 70 of 100 seeds.
- Against the dynamic anchor the same medoid file looks worse (53 lower at 100 m, 49 lower at 500 m) only because the anchor moved by the O35 gap. That is not a new medoid result.
- At J = 2, 500 m, the p-median loses more: −0.048, 49 lower (`n100_medoid_j2`, from `analysis.txt`).

**Name the method for what it is: a p-median / zenith-subset initialization of SCA.** Do not present it as a new algorithm (see §11).

**O26. At 25% most of the zenith gain is the starting point. Under a tight cap the polish does more of it. [MEASURED]**
From the dynamic `n100_sca_anchor.json` diagnostics. `frozen_Mbps` inside that file matches the external SCA column to 10⁻¹⁵ Mbps, so the keep-best reference is the headline SCA.

| Field | One LP at the best zenith subset, minus SCA | Further gain from SCA polish | Mean polish displacement of the winner | Winner came from a zenith subset |
| --- | --- | --- | --- | --- |
| 500 m | **+0.201** of the +0.266 total | +0.065 | **5.8 m** (max 12.2 m) | **90/100** |
| 100 m | **+0.010** of the +0.015 total | +0.005 | **1.0 m** (max 2.9 m) | **80/100** |

- The zenith layout is already close to an SCA stationary point.
- On the remaining seeds the dynamic-SCA candidate was kept.
- The winner's association changed during polish on 6/100 seeds at 100 m and 4/100 at 500 m.
- The older frozen 500 m campaign kept a zenith subset on 94/100 seeds. That file was not re-averaged for displacement in this pass.
- Under tighter caps the polish does more of the rate work. Difference of means on the dynamic banks, 500 m, one LP at the winning subset, then the further polish. Metre displacement was not stored in these files.

  | Cap | One LP − SCA | Further polish | Zenith winner |
  | --- | --- | --- | --- |
  | none | +0.021 | +0.050 | 57/100 |
  | 10% | +0.065 | +0.187 | 84/100 |
  | 12% | +0.029 | +0.237 | 84/100 |
  | 15% | +0.029 | +0.236 | 89/100 |
  | 18% | +0.095 | +0.183 | 86/100 |
  | 20% | +0.149 | +0.139 | 91/100 |
  | 22% | +0.130 | +0.103 | 94/100 |
  | 25% | +0.201 | +0.065 | 90/100 |

  At 100 m the split is +0.010 then +0.006 at 25% (80/100 zenith winners) and +0.016 then +0.051 at 10% (89/100).

**O27. The gain is not an artifact of bandwidth piling. [MEASURED + REPORTED]** Two independent checks:
1. **Equal-share control (cap10, O14).** Bandwidth is fixed at 1 MHz per link, yet frozen zenith still gains **+0.251 Mbps at 500 m** (67/3/30, 53 practical, p = 2.8×10⁻²⁰). That is 3.5% of SCA, and it is not the largest absolute gap in the finished sweep (O29a).
2. **Bandwidth-proportional noise (O30).** The leftover no longer piles onto the cap (max link share 0.10–0.16), yet the gain persists at **+0.304 Mbps at 500 m**.

The knapsack structure (O12) explains why solutions sit at the cap. It does **not** explain the placement gain.

**O28. Geometric explanation (first-order). [analytic + MEASURED, 20 seeds]**
Take two IoTs a distance s apart, a UAV at height H, and the low-SNR, distance-only channel of O9 (rate ∝ 1/d²). Hovering over one IoT beats the midpoint when
1/H² + 1/(H² + s²) > 2/(H² + s²/4), which holds exactly when **s > √2·H** (≈ 141 m at H = 100 m).

k-means puts each UAV at a cluster centroid, which is the midpoint-type choice. On a 100 m field most intra-cluster distances are below √2·H, so the centroid is fine and the gain is small. On a 500 m field they are not, so hovering wins. This matches O18 and O20.

Side-over-height sweep, 20 seeds, now on disk at `results/run_10mhz_cap25/mechanism/side_over_h.json`. K-means SCA is solved under fixed σ², then both layouts are re-scored with proportional noise. N₀ moves as 1/B_eq. The table is the **median** relative gain at the nominal reference (B_eq scale 1). The mean is not the summary: one seed reaches +26.7% at H = 30 m.

| side/H | Geometry | n | Median | IQR | Seeds positive |
| --- | --- | --- | --- | --- | --- |
| 1.00 | 100 m, H = 100 | 20 | 0.23% | 0.04–1.14 | 17/20 |
| 1.67 | 500 m, H = 300 | 20 | −0.02% | −0.63–1.68 | 10/20 |
| 2.00 | 200 m, H = 100 | 20 | 0.64% | 0.11–2.54 | 17/20 |
| 2.50 | 500 m, H = 200 | 20 | 1.31% | −0.11–4.79 | 13/20 |
| 3.00 | 300 m, H = 100 | 20 | 1.65% | 0.03–2.53 | 15/20 |
| 5.00 | 500 m, H = 100 | 20 | **3.22%** | 0.51–6.76 | 17/20 |
| 8.00 | 800 m, H = 100 | 19 | 1.18% | −0.08–8.60 | 13/19 |
| 10.0 | 500 m, H = 50 | 20 | 1.69% | −0.11–8.08 | 14/20 |
| 16.7 | 500 m, H = 30 | 20 | 0.82% | −0.16–6.39 | 13/20 |

How to read it:
- The median rises from side/H near 2 through the headline geometry at side/H = 5. There the IQR is entirely above zero, 14/20 seeds clear 0.05 Mbps, and the mean gap is +0.281 Mbps.
- It does not keep rising past side/H = 5. The old 5-seed means that peaked near side/H = 8 were the outlier seed.
- Raising the reference bandwidth 10× lowers N₀, raises the absolute Mbps gap, and lowers the relative median. At side/H = 5 the high-noise scale (B_eq ×0.1) has no feasible seed.
- At side/H = 1.67 the median is slightly negative (10/20 seeds positive).

The figure script is `python scripts/analyze/mechanism_side_over_h.py --n-runs 20`. The saved copy is the path above, plus `side_over_h.png`.

---

## 7. Per-link cap sweep (10 MHz, n = 100, J = 3) [MEASURED]

Source: `results/run_10mhz_cap_sweep/<cap>/<field>/n100_*.json`. All 16 cap×field banks and their campaigns are on disk. The tables in this section keep the **frozen** polish (`n100_sca_anchor.json`), including the 25% frozen row. The dynamic twin is on disk for every cap (O35) and moves those means by at most 0.004 Mbps. The cap25 SCA bank matches `run_10mhz_cap25` exactly (max per-seed |Δ| = 0).

SCA in this sweep is dynamic. Anchor and multi-start are frozen. Read losses of anchor against SCA with O35: some of them are the reference mismatch, and the matched dynamic-versus-dynamic gap at 25% is +0.002 Mbps.

Every default-point cell (I = 10, J = 3, T_k = 2.8 s) is 100% feasible.

### 500 × 500 m

| Cap | random | k-means | SCA | multi-start | zenith | zenith − SCA (W/L/T, practical) | % of SCA | zenith − multi-start (practical) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 7.891 | 9.373 | 9.829 | 9.900 | 9.898 | +0.069 (52/13/35, 36), p = 8.0×10⁻¹¹ | 0.70% | **−0.002** (9), p = 0.20 |
| 10% | 4.598 | 6.600 | 7.092 | 7.274 | 7.344 | +0.251 (67/3/30, 53), p = 2.8×10⁻²⁰ | 3.54% | +0.069 (33) |
| 12% | 5.126 | 7.171 | 7.946 | 8.138 | 8.210 | +0.265 (70/7/23, 56), p = 5.6×10⁻¹⁸ | 3.33% | +0.072 (30) |
| 15% | 5.680 | 7.678 | 8.640 | 8.817 | 8.903 | +0.263 (74/6/20, 62), p = 8.6×10⁻²⁰ | 3.04% | +0.086 (35) |
| 18% | 6.079 | 8.016 | 9.007 | 9.211 | 9.282 | +0.275 (76/11/13, 63), p = 2.7×10⁻¹⁸ | 3.05% | +0.070 (39) |
| 20% | 6.281 | 8.184 | 9.150 | 9.353 | 9.434 | **+0.285** (82/6/12, 72), p = 1.1×10⁻²² | 3.11% | +0.081 (48) |
| 22% | 6.462 | 8.321 | 9.320 | 9.489 | 9.551 | +0.232 (84/3/13, 69), p = 2.0×10⁻²⁴ | 2.49% | +0.062 (47) |
| 25% frozen | 6.665 | 8.478 | 9.418 | 9.589 | 9.682 | +0.264 (81/5/14, 65), p = 2.4×10⁻²² | 2.81% | +0.094 (48) |
| 25% dynamic | 6.665 | 8.478 | 9.418 | 9.589 | 9.684 | +0.266 (84/0/16, 65), p = 8.1×10⁻²⁸ | 2.83% | +0.096 (48) |

### 100 × 100 m

| Cap | random | k-means | SCA | multi-start | zenith | zenith − SCA (W/L/T, practical) | % of SCA | zenith − multi-start (practical) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 10.088 | 10.171 | 10.196 | 10.199 | 10.199 | +0.003 (50/15/35, 0), p = 5.1×10⁻¹⁰ | 0.03% | −0.000 (0), p = 0.029 |
| 10% | 9.416 | 9.874 | 9.879 | 9.940 | 9.944 | **+0.065** (70/0/30, 34), p = 6.5×10⁻²⁷ | 0.66% | +0.004 (2) |
| 12% | 9.623 | 9.973 | 10.025 | 10.058 | 10.065 | +0.039 (63/8/29, 26), p = 7.2×10⁻¹⁶ | 0.39% | +0.006 (4) |
| 15% | 9.770 | 10.037 | 10.110 | 10.127 | 10.132 | +0.022 (69/13/18, 17), p = 2.5×10⁻¹⁵ | 0.22% | +0.005 (1) |
| 18% | 9.851 | 10.070 | 10.141 | 10.154 | 10.159 | +0.018 (82/8/10, 8), p = 1.2×10⁻²¹ | 0.18% | +0.005 (0) |
| 20% | 9.889 | 10.086 | 10.152 | 10.168 | 10.173 | +0.021 (80/10/10, 10), p = 1.5×10⁻²⁰ | 0.20% | +0.004 (0) |
| 22% | 9.915 | 10.096 | 10.162 | 10.173 | 10.177 | +0.015 (84/8/8, 3), p = 4.2×10⁻²³ | 0.15% | +0.005 (0) |
| 25% frozen | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 | +0.015 (79/11/10, 9), p = 9.0×10⁻²² | 0.15% | +0.004 (0) |
| 25% dynamic | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 | +0.015 (80/0/20, 9), p = 1.7×10⁻²⁴ | 0.15% | +0.004 (0) |

**O29a. At 500 m, a binding cap keeps a zenith − SCA gain of about +0.23 to +0.29 Mbps.** The finished binding caps (10–25%) are not a flat line. The mean gap is largest at **20% (+0.285, 72/100 practical)** and smallest at **22% (+0.232, 69/100 practical)**. With no cap it drops to +0.069. The relative gain is largest at the 10% equal-share row (3.54%), because SCA's own rate is lowest there, not because the absolute gap is largest.

**O29b. At 100 m, the gain is small and is largest when the cap is tightest.** It goes from +0.003 (no cap, 0 practical) to +0.065 (10%, 34 practical). From 18% to 25% the practical count is 3–10 out of 100. The 20% row (+0.021, 10 practical) is slightly above its two neighbors, so the rise is not perfectly monotone.

**O29c. With no cap, zenith-subset only ties multi-start.** Against the frozen polish the difference is −0.002 at 500 m (p = 0.20; zenith wins 23, loses 40, ties 37) and −0.0004 at 100 m (p = 0.029, favoring multi-start). The dynamic twin is −0.0003 at 100 m (p = 0.091; 28/41/31) and +0.0002 at 500 m (p = 0.60; 30/37/33). The edge over multi-start exists when a cap binds.

**O29d. With equal-share bandwidth on the 100 m field, SCA barely improves on its k-means start in practical terms.**
- SCA − k-means is +0.005 ± 0.020. The sign is positive on 100/100 seeds, and only **3/100** clear 0.05 Mbps. At the other caps the practical count is 18–65.
- Multi-start (+0.061 over k-means, from the means) and zenith (+0.065 over SCA's neighbor) both escape.
- This is consistent with the k-means centroid sitting near a local optimum of the equal-share objective. The sweep does not prove that mechanism. Report the gap, not the cause.

**O29e. Wall-clock per scenario.**

| Solver | Seconds per scenario |
| --- | --- |
| SCA | 0.7–1.2 on every sweep cell; headline bank 1.07 (100 m) and 0.96 (500 m) |
| multi-start | 3.1–4.4 |
| frozen zenith | 2.1–3.2, **below multi-start on all 16 sweep cells** |
| dynamic zenith | **3.0–5.6 s** across all 16 cap×field banks. Headline bank: 4.56 s (100 m) and 5.63 s (500 m) |
| TD3 | 69–71 on the bank; 57 and 78 on the campaigns |

The "zenith is cheaper than multi-start" sentence is true for the frozen polish that fills the cap tables. It is false for the dynamic headline bank.

**O29f. Statistical significance.** Zenith − SCA has two-sided p < 10⁻⁹ on every cell in the tables above, including no cap. Zenith − multi-start is significant on every capped cell. The two no-cap exceptions are p = 0.029 at 100 m (multi-start ahead) and p = 0.20 at 500 m.

**O29g. Where the J-axis gap peaks depends on the cap. [MEASURED]**
Mean zenith − SCA on the cap-sweep campaigns, 500 m. Anchor is the frozen polish. The dynamic 25% campaign is O20; it shifts this 25% row by at most 0.002 Mbps.

| Cap | J = 1 | J = 2 | J = 3 | J = 4 | J = 5 |
| --- | --- | --- | --- | --- | --- |
| 10% | +0.043 | +0.182 | +0.251 | +0.256 | **+0.349** |
| 12% | +0.047 | +0.222 | +0.265 | +0.183 | +0.145 |
| 15% | +0.058 | **+0.306** | +0.263 | +0.148 | +0.153 |
| 18% | +0.160 | **+0.372** | +0.275 | +0.144 | +0.137 |
| 20% | +0.180 | **+0.330** | +0.285 | +0.196 | +0.172 |
| 22% | +0.206 | **+0.320** | +0.232 | +0.144 | +0.112 |
| 25% | +0.205 | **+0.325** | +0.264 | +0.198 | +0.114 |
| none | +0.001 | +0.054 | +0.069 | +0.056 | +0.055 |

The largest cell in this grid is **+0.372 Mbps at 18% and J = 2**. Equal share (10%) is the row that keeps growing through J = 5. Every other binding cap peaks at J = 2. No cap stays under +0.07 at every J. At 100 m the same grid never exceeds +0.065 (10%, J = 3).

---

## 8. Robustness to the noise model

**O30. Bandwidth-proportional noise. [MEASURED, 20 seeds, 10 MHz, 25% cap]**
Here SNR = p·g/(N₀·B) with N₀ = σ²/(B_sys/I). The rate becomes concave in B, and the allocation is solved with CVXPY `rel_entr` (`src/uavdt/analysis/bw_proportional.py`). Source: `results/run_10mhz_cap25/mechanism/bw_proportional_noise.json`.

| Field and radio | zenith | p-median | SCA | zenith − SCA | Max link share |
| --- | --- | --- | --- | --- | --- |
| 100 m, proportional | 9.898 | 9.896 | 9.844 | +0.054 ± 0.079, 17/3/0, 9 practical, p = 0.002 | 0.104 |
| 100 m, fixed σ² | 10.184 | 10.182 | 10.176 | +0.008 ± 0.020, 11/9/0, 2 practical, p = 0.23 | 0.25 |
| 500 m, proportional | 7.381 | 7.358 | 7.077 | **+0.304 ± 0.304**, 17/3/0, 14 practical, p = 1.0×10⁻⁴ | 0.155 |
| 500 m, fixed σ² | 9.635 | 9.621 | 9.495 | +0.140 ± 0.212, 13/7/0, 12 practical, p = 0.015 | 0.25 |

- Under the proportional radio, the optimum no longer pushes any link to the cap.
- The zenith subset chosen under fixed σ², evaluated under the proportional radio, still gains +0.255 at 500 m.
- The SCA in this script runs with `dynamic_assignment=True`, unlike a frozen-anchor comparison. Say so in the paper.

The script is `python scripts/analyze/compare_bw_proportional_noise.py --n-runs 20 --areas 100 500`. The saved copy is the path above.

**O31. Polishing both starts under the proportional radio. [MEASURED, 20 seeds, Nelder–Mead, 200 evaluations]**
Source: `results/run_10mhz_cap25/mechanism/polish_proportional.json`.

- At 500 m the start gap is **+0.596 ± 0.396** (18/2/0, 18 practical, p = 9.5×10⁻⁶). After polishing it is **+0.133 ± 0.196** (17/3/0, 9 practical, p = 8.5×10⁻⁴). Polished p-median is **7.512** against **7.380** for polished k-means. K-means moves **62 m** RMS; p-median moves **25 m**.
- At 100 m the start gap is **+0.015 ± 0.073** (9/11/0, p = 0.65) and the polished gap is **+0.025 ± 0.069** (11/9/0, 3 practical, p = 0.55).
- **The basin difference survives local search at 500 m. At 100 m the polished gap is not distinguishable from zero.**

The script is `python scripts/analyze/polish_proportional_radio.py --n-runs 20 --areas 500 100`. The earlier 8-seed reading (+0.242 after polish at 500 m) was the same sign and a larger gap.

---

## 9. Learning-based methods

**O32. No learned method tested beats the zenith-subset / p-median initialization. [MIXED]**

The Algorithm 2 row is the 10 MHz measurement (O11, and O36 for the other caps). The residual row is the 10 MHz bank below; the 8.8 MHz residual run is kept as a separate line. The remaining rows are still 8.8 MHz.

| Method | Result vs reference |
| --- | --- |
| TD3 Algorithm 2, 10 MHz, n = 100 | −0.043 vs SCA at 100 m (3/100 seeds higher); −0.859 vs SCA at 500 m (0/100). SCA leads on 24/24 campaign points (O11) |
| TD3 Algorithm 2, 8.8 MHz | −0.030 vs SCA (100 m, 20 seeds), −0.718 vs SCA (500 m, 100 seeds); 0/29 sweep points above SCA |
| Residual TD3 on SCA, 10 MHz, 25%, n = 100, ±10 m, frozen association | 100 m: 10.171, **−0.001 ± 0.013** vs SCA (16/53/31, 0 practical, p = 0.010). 500 m: 9.396, **−0.022 ± 0.068** (7/51/42, 2 practical, p = 5.1×10⁻⁸). Leads Algorithm 2 by +0.041 and +0.837. Zenith still leads it by +0.017 and +0.288. Wall 261 s and 231 s. **[MEASURED]** |
| Residual TD3 on top of SCA (±10 m), 8.8 MHz, 20 seeds | +0.0032 vs SCA, 4/20 wins, 0 losses. **[LEGACY 8.8 MHz]** |
| CMA-ES polish of SCA | +0.0075 vs SCA, 12/20 wins. **[LEGACY 8.8 MHz]** |
| Association search at frozen SCA q (Experiment C) | +0.001; flat. **[LEGACY 8.8 MHz]** |
| Learned proposal policy, 500 m, 25% cap, 20 seeds | greedy 8.062; 24 proposals 8.363, against SCA 8.300, multi-start 8.399, zenith one-LP 8.443, zenith + polish **8.490**. **[LEGACY 8.8 MHz]** |
| Same, untrained network with 24 proposals | 8.209 |

- The zenith enumeration takes about 0.03 s per layout, against about 1 s for 24 learned proposals.
- At T_k = 0.8 s the learned search is feasible only when association is forced to the processing UAV, and then it ties the untrained network (8.324).

What to conclude: at I = 10, J = 3 the combinatorial space (C(10, 3) = 120 subsets) is small enough to enumerate exactly. **Learning has nothing to amortize at this size.** Any learning claim needs a regime where enumeration is too expensive, roughly C(I, J) > 10⁵, and that has not been run. The 10 MHz campaign does not change that conclusion: Algorithm 2 TD3 is a consistent but small improvement on k-means, and it does not reach SCA. Residual TD3 starts at SCA and stays within 0.022 Mbps of it (O32 table), still behind the zenith polish.

**O36. On the n100 banks, SCA leads Algorithm 2 TD3 at every cap. [MEASURED]**

Same preset as O11, both fields, caps none and 10–25%. These files are banks, not campaigns. Every cell is 100% feasible. AoDT is 2.800 s except at 10%, where it is slack (0.994 s at 100 m, 1.135 s at 500 m). Source: `results/run_10mhz_td3/<cap>/{100m,500m}/n100_td3.json`.

| Cap | 100 m TD3 | SCA − TD3, 100 m | 500 m TD3 | SCA − TD3, 500 m |
| --- | --- | --- | --- | --- |
| none | 10.189 | +0.008 ± 0.012, 98/2/0, 1 practical | 9.483 | +0.346 ± 0.418, 100/0/0, 63 practical |
| 10% | 9.875 | +0.004 ± 0.020, 86/14/0, 2 practical | 6.651 | +0.441 ± 0.385, 100/0/0, 90 practical |
| 12% | 9.986 | +0.039 ± 0.041, 98/2/0, 28 practical | 7.222 | +0.723 ± 0.432, 100/0/0, 97 practical |
| 15% | 10.052 | +0.058 ± 0.048, 100/0/0, 48 practical | 7.740 | +0.900 ± 0.489, 99/1/0, 98 practical |
| 18% | 10.085 | +0.056 ± 0.044, 99/1/0, 46 practical | 8.088 | +0.918 ± 0.494, 100/0/0, 99 practical |
| 20% | 10.104 | +0.048 ± 0.037, 96/4/0, 37 practical | 8.256 | +0.893 ± 0.490, 100/0/0, 99 practical |
| 22% | 10.115 | +0.047 ± 0.034, 98/2/0, 38 practical | 8.395 | +0.924 ± 0.504, 100/0/0, 99 practical |
| 25% | 10.130 | +0.043 ± 0.031, 97/3/0, 33 practical | 8.559 | +0.859 ± 0.502, 100/0/0, 96 practical |

- At 500 m the SCA lead is above 0.05 Mbps at every cap, and TD3 leads k-means at every cap (45–64 practical from 10% through 25%; 58 practical with no cap).
- At 100 m and 10%, TD3's mean is +0.001 over k-means and k-means is ahead on **74/100** seeds (p = 0.002). That is the equal-share cell. From 12% up, TD3 leads k-means again, with 3–8 practical seeds.
- Dynamic zenith leads TD3 on **100/100** seeds in all 16 cells.
- Bank wall-clock is 66–210 s per seed. The 25% banks are 69 s and 71 s.
- At 500 m, PSO leads this TD3 at every cap (O34). At 100 m, PSO leads it at 10% and TD3 leads from 20% up.

Sources: O11; `results_old.md` §2.9 and §2.10; the learned-policy runs.

---

## 10. PSO

**O33. PSO is handicapped in this implementation. [MEASURED, code]**
PSO's inner fitness, `_fitness_equal_share` in `src/uavdt/placement/pso.py`, scores particles with equal-share bandwidth. The campaign then re-scores the final PSO layout with the capped LP. So PSO optimizes a different objective from the one it is graded on.
- With 8.8 MHz and a 15% cap, the LP-minus-equal-share gap was +0.146 for k-means and +0.110 for PSO. Source: `results_old.md` §3.4.
- The cap10 cell (O14) is the one setting where equal share **is** the scored objective. That bank is now on disk (O34): at 100 m PSO leads SCA by **+0.057** (p = 1.0×10⁻⁵, 50 seeds each way); at 500 m the mean is **+0.082** and the signed-rank test is not (p = 0.76).

**O34. PSO is above the zenith polish on 3 seeds, none of them by 0.05 Mbps. [MEASURED]**

Paired against the dynamic zenith bank. The frozen polish gives the same "0 seeds higher" result except at 100 m and 10%, where 5 seeds are higher and still 0 are practical. Every cell is 100% feasible. p-values are two-sided.

| Cap, field | PSO | PSO − SCA | PSO − dynamic zenith |
| --- | --- | --- | --- |
| none, 100 m | 10.174 | −0.023 (5 higher), p = 2.4×10⁻²² | −0.026 (0) |
| none, 500 m | 9.639 | −0.190 (10), p = 3.2×10⁻¹⁹ | −0.261 (0) |
| 10%, 100 m | 9.936 | **+0.057 ± 0.096** (50/50), 28 practical, p = 1.0×10⁻⁵ | −0.010 (3 higher, 0 practical) |
| 10%, 500 m | 7.174 | **+0.082 ± 0.284** (34/66), 31 practical, p = 0.762 | −0.170 (0) |
| 12%, 100 m | 10.014 | −0.012 (21), p = 2.4×10⁻⁴ | −0.051 (0) |
| 12%, 500 m | 7.992 | **+0.046** (27), p = 0.088 | −0.221 (0) |
| 15%, 100 m | 10.063 | −0.047 (9), p = 2.3×10⁻¹⁹ | −0.069 (0) |
| 15%, 500 m | 8.617 | −0.023 (29), p = 0.009 | −0.288 (0) |
| 18%, 100 m | 10.089 | −0.052 (3), p = 4.4×10⁻²⁴ | −0.071 (0) |
| 18%, 500 m | 8.929 | −0.077 (27), p = 9.3×10⁻⁴ | −0.356 (0) |
| 20%, 100 m | 10.101 | −0.051 (6), p = 4.2×10⁻²¹ | −0.072 (0) |
| 20%, 500 m | 9.052 | −0.098 (31), p = 7.4×10⁻⁴ | −0.386 (0) |
| 22%, 100 m | 10.109 | −0.053 (2), p = 1.2×10⁻²³ | −0.068 (0) |
| 22%, 500 m | 9.143 | −0.176 (24), p = 5.5×10⁻⁸ | −0.409 (0) |
| 25%, 100 m | 10.120 | −0.052 (3), p = 1.2×10⁻²³ | −0.068 (0) |
| 25%, 500 m | 9.241 | −0.177 (26), p = 2.4×10⁻⁷ | −0.443 (0) |

- The unhandicapped cell is 10%. At 100 m the signed-rank test favors PSO over SCA even though the seeds split 50/50, because the positive gaps are larger. At 500 m the mean is positive and the test is not.
- PSO's mean is also above SCA at 12% and 500 m, and that test is not significant (p = 0.088).
- PSO beats SCA at J = 1, 500 m, 25% cap: +0.015, 19/100 seeds (`analysis.txt`). That campaign was not rerun for the new caps.
- In the 25% campaign, PSO is never significantly above SCA (BH-FDR, 0/24 points on both fields).
- At 500 m, PSO leads Algorithm 2 TD3 at every cap: +0.155 with no cap, and +0.52 to +0.88 from 10% through 25%. At 100 m and 25%, TD3 leads PSO by 0.010 Mbps (PSO higher on 24/100). At 100 m and 10%, PSO leads TD3 by 0.061 (99/100).

P-median, random-subset, and continuous candidates are now on disk at no cap, 10%, and 25%, both fields (`results/run_10mhz_controls/`). P-median stays within 0.006 Mbps of the dynamic zenith polish on all six cells, with 0 practical seeds above it. A random subset is indistinguishable from SCA with the cap off (p = 0.97 at 100 m, p = 0.38 at 500 m) and at 100 m, 25% (p = 0.11). At 500 m, 25% it remains the O24 result: +0.083 vs SCA (p = 0.058) and 0 seeds above the zenith polish. Continuous candidates sit between SCA and the zenith polish when a cap binds.

Either rerun PSO with an LP fitness, or keep it in the paper with the handicap stated.

Sources: `results/run_10mhz_controls/{cap}/{field}/n100_pso.json` paired with the sweep's SCA and zenith files.

---

## 11. Claims not to make (threats to validity)

1. **Do not claim the zenith-subset or anchoring idea as a new algorithm.** IoT-anchored, k-medoids and p-median initialization of UAV placement exists in prior work (Xie/Xu/Zhang arXiv 1801.04545; Shen et al. TSP 2020; El Hammouti IoT-J 2021; MDPI Telecom 2022; Sensors 2022 PMC9698383). The contribution is the reproduction, the audit, and the regime analysis: when and why centroid-initialized SCA loses.
2. **Do not quote the "upper bound" or "gap closed" numbers.**
   - `leftover_dump_upper_bound` in `sca_anchor.py` maximizes only over layouts with a UAV directly above an IoT, so it is not a bound on (P).
   - The dynamic anchor lands **above** this "bound" on 7/100 seeds (100 m) and 3/100 seeds (500 m).
   - Drop "2.4% gap" and "closes 41% of the gap".
3. **Do not report p-values for zenith-subset vs its own SCA candidate.** Keep-best makes it never worse than the SCA solve inside the same run. The p-values above are against the **separately stored** SCA column. Dynamic anchor has 0 losses against that column. Frozen anchor still has losses, because its internal reference is a frozen-assignment SCA and the stored SCA column is dynamic.
4. **Zenith ≥ p-median at the one-LP stage holds by construction**, because the p-median subset is one of the enumerated subsets. Do not present that as an empirical win.
5. **Do not say "more noise increases the gain".** O28 shows the reverse for relative gain.
6. **Side/H medians have wide IQRs.** The 20-seed study is on disk (O28). Cite the median and the IQR. The mean is outlier-driven, and the high-noise scale is infeasible at the headline geometry. The proportional polish is also 20 seeds (O31).
7. **Label the protocol.** Fig. 11, residual TD3, and Algorithm 2 TD3 are at 10 MHz. CMA-ES and the learned proposal policy are still 8.8 MHz. The T_k = 0.8 s cohesive repair is not part of this sweep.
8. **The per-link cap is the authors' extension.** Every result that separates the methods depends on it (O13, O29c). State that it is not part of Problem (P), and show the no-cap row.
9. **Do not describe "the SCA solver" as one assignment policy.** Headline SCA and dynamic anchor reassign. Multi-start, the frozen sweep anchor (`n100_sca_anchor.json`), and the subset controls do not (O7, O35). The rate consequence of the anchor switch is at most 0.004 Mbps. The win/loss counts are not comparable across the switch.
10. **Do not put PSO and TD3 in one order on both fields.** At 25%, PSO is ahead of TD3 at 500 m and behind it at 100 m (O17). Across caps, PSO leads TD3 at every 500 m cell and at 100 m only while the cap is tight (O34, O36).
11. **Do not report T_k = 0.8 s.** The axis starts at 1.2 s. Older campaign files still contain that point, and every method is 0% feasible there. The 500 m dynamic-anchor campaign does not store it.

---

## 12. Pending results checklist

| Item | Status | Fills |
| --- | --- | --- |
| Cap sweep, all eight caps, both fields, n100 and campaign | **Done.** Artifacts on disk. `run_10mhz_cap_sweep/progress.txt` is stale (13 cap25 lines still say pending) | §7, O29 |
| Cap25 SCA matches the headline SCA bank | **Done.** Max per-seed \|Δ\| = 0 on both fields | Determinism |
| Dynamic anchor bank, both fields, and both 25% campaigns | **Done.** 100 m campaign 2026-10-17. 500 m campaign `campaign_sca_anchor_dyn.json` matches the bank at J = 3 (9.684). The frozen 500 m campaign remains as `campaign_sca_anchor.json` | O17, O20, O35 |
| Algorithm 2 TD3 at 10 MHz, both banks and both campaigns | **Done** | O11 |
| Side/H mechanism at 20 seeds, median and IQR | **Done.** `results/run_10mhz_cap25/mechanism/side_over_h.json` | O28 |
| Proportional-noise and polish, 20 seeds | **Done.** `results/run_10mhz_cap25/mechanism/` | O30, O31 |
| PSO at every cap, including the unhandicapped 10% cell | **Done.** n100 only. The 25% campaign was already on disk. LP-fitness PSO was not run | O33, O34 |
| TD3 n100 at every cap | **Done.** Campaigns at caps other than 25% were not run | O36 |
| Residual TD3 at 10 MHz, 25%, both fields | **Done.** `results/run_10mhz_td3/residual-on-sca/` | O32 |
| Fig. 11 at 10 MHz, 20 seeds, both fields | **Done.** `results/run_10mhz_cap25/fig11/` | O10 |
| T_k = 0.8 s cohesive repair | Dropped. The AoDT axis starts at 1.2 s | O23 |
| Frozen-assignment SCA bank | **Done** at 25%, both fields. `n100_frozen_sca.json` | O7 |
| Dynamic-anchor n100 at every cap, and the 500 m campaign | **Done.** Sweep twins are `n100_sca_anchor_dyn.json`. The 500 m campaign is `campaign_sca_anchor_dyn.json` | O20, O35 |
| Dynamic-anchor campaigns at caps other than 25%, and at 100 m for the new caps | Not run. The 100 m 25% campaign was already dynamic | O29g |

---

## 13. Artifact index

| Path | What it holds |
| --- | --- |
| `results/run_10mhz_cap25/{100m,500m}/n100_sca_anchor.json` | Dynamic zenith bank. Identical to `n100_sca_anchor_dyn.json` |
| `results/run_10mhz_cap25/{100m,500m}/n100_sca_anchor_frozen.json` | Frozen-polish twin of that bank |
| `results/run_10mhz_cap25/{100m,500m}/campaign_*.json` | 25% sweeps. Both anchor campaigns are dynamic. The 500 m dynamic file is `campaign_sca_anchor_dyn.json`; `campaign_sca_anchor.json` is the frozen twin |
| `results/run_10mhz_cap25/{100m,500m}/n100_frozen_sca.json` | SCA with dynamic assignment off |
| `results/run_10mhz_cap25/fig11/{100m,500m}/fig11.json` | Fig. 11 at 10 MHz, 20 seeds |
| `results/run_10mhz_cap25/mechanism/` | Proportional noise, proportional polish, side/H, 20 seeds |
| `results/run_10mhz_td3/residual-on-sca/cap25/{100m,500m}/n100_td3.json` | Residual TD3, 10 MHz, 25% |
| `results/run_10mhz_td3/<cap>/{100m,500m}/n100_td3.json` | Algorithm 2 banks at every cap |
| `results/run_10mhz_cap25/{100m,500m}/n100_td3.json` | Algorithm 2 bank. Byte-identical under `results/run_10mhz_td3/cap25/` |
| `results/run_10mhz_td3/cap25/{100m,500m}/campaign_td3.json` | Algorithm 2 sweep, 28 points, seeds 1–100 |
| `results/run_10mhz_cap_sweep/<cap>/{100m,500m}/n100_sca_anchor.json` | Frozen zenith polish for that cap |
| `results/run_10mhz_cap_sweep/<cap>/{100m,500m}/n100_sca_anchor_dyn.json` | Dynamic twin, no cap and 10–22%. The 25% dynamic bank is the headline file |
| `results/run_10mhz_controls/` | PSO at every cap. P-median, random-subset, and continuous controls at no cap, 10%, and 25% |
| `docs/results_new.md` | 10 MHz write-up. Some anchor p-values there are the one-sided tail; this file uses two-sided |
| `docs/results_old.md` | 20 kHz audit (§0), radians (§0.2), T_k = 0.8 / SCA-joint (§2.8), residual/C (§2.9), 8.8 MHz TD3 (§2.10), PSO proxy (§3.4), Fig. 11 (§8) |
| `docs/novelty.md` | Zenith-subset method note and its "do not write" list |
| `src/uavdt/sca_anchor.py` | Zenith-subset SCA: enumeration, LP scoring, top-K polish, keep-best |
| `src/uavdt/experiments/methods.py` | Where `sca` / `sca_anchor` turn dynamic assignment on, and multi-start does not |
| `src/uavdt/analysis/bw_proportional.py` | Proportional-noise allocator |
| `scripts/lib/paired_winrate.py` | Wilcoxon used for the p-values in this file |
| `scripts/analyze/{compare_bw_proportional_noise,polish_proportional_radio,mechanism_side_over_h}.py` | Robustness scripts for §8 and O28 |
