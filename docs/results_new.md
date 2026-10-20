# Experimental results

**10 MHz, 25% per-link cap, 100 scenarios.** Headline numbers are from `results/run_10mhz_cap25/`. The cap family (no cap, then 10% through 25%) is `results/run_10mhz_cap_sweep/`, 160/160 jobs complete. The TD3 sweep (25% cap, both fields, n100 bank and full campaign) is `results/run_10mhz_td3/`, 4/4 jobs complete. The earlier ledger (8.8 MHz, 20 kHz audit, PSO) is in [`results_old.md`](results_old.md).


| | |
| --- | --- |
| **Headline run** | `results/run_10mhz_cap25/` (20 jobs, 25% cap) |
| **Cap sweep** | `results/run_10mhz_cap_sweep/` (no cap, 10, 12, 15, 18, 20, 22, 25%; both fields; n100 bank and full campaign) |
| **Protocol** | `B_sys` = 10 MHz, seeds 1–100, CVXPY, 30 SCA iterations. Headline cap 25% is 2.50 MHz/link |
| **Fields** | **100 × 100 m** and **500 × 500 m** |
| **Methods** | random, k-means, SCA, `sca_multistart`, `sca_anchor` (dynamic assignment polish), TD3 (Algorithm 2) |
| **TD3 bank** | Both n100 banks at 25%. Algorithm 2, policy export, 7000 steps, k-means init. `results/run_10mhz_cap25/{100m,500m}/n100_td3.json` is byte-identical to `results/run_10mhz_td3/cap25/{100m,500m}/n100_td3.json` |
| **TD3 sweep** | `results/run_10mhz_td3/cap25/{100m,500m}/campaign_td3.json`. Same protocol, seeds 1–100, `--td3-preset alg2`, CUDA. 100 m finished 2026-10-15 (159944 s); 500 m finished 2026-10-17 (219136 s) |
| **Not in these runs** | PSO, TD3 at caps other than 25%, 20 kHz diagnostic |


---

## Conclusion

On the default point (I = 10, J = 3, λ = 2/s, T_k = 2.8 s, f_j = 2×10⁸), every method is 100% feasible on both fields. The ranking is the same on both, and it is the ranking on every feasible point of the TD3 campaign:

**sca_anchor > sca_multistart > SCA > TD3 > k-means > random.**

Algorithm 2 TD3 on the frozen banks (policy export, 7000 steps) is 100% feasible on both fields, with max AoDT pinned at 2.800 s. At 100 m it is **10.130 ± 0.042 Mbps**: SCA leads by **0.043 ± 0.031** (97/100, 33 practical, p = 2.7×10⁻²⁷), and TD3 leads k-means by **0.019 ± 0.017** (96/100, 6 practical). At 500 m it is **8.559 ± 0.658 Mbps**: the SCA lead opens to **0.859 ± 0.502** (100/100, 96 practical, p = 1.6×10⁻³⁰), while TD3 still leads k-means by **0.081 ± 0.061** (99/100, 64 practical) and random by **1.894 ± 1.202** (95/100, 94 practical). The 100→500 m drop is **1.571 Mbps**, next to k-means (1.633) and larger than SCA (0.754). Mean wall-clock on that bank is **69 s** per seed at 100 m and **71 s** at 500 m.

At **100 × 100 m** the radio is close to the zenith ceiling (**10.213 Mbps**). SCA is **10.172 Mbps**, **+0.223 ± 0.122** above random (100/100) and **+0.062 ± 0.037** above k-means (100/100). Zenith-anchor adds **+0.015 ± 0.021** over headline SCA (80/100 wins, 0 losses, 20 ties, 9/100 above 0.05 Mbps). Multi-start adds **+0.011 ± 0.019** (58/100 wins, 8/100 practical). The SCA−random gap shrinks from **+0.609 Mbps** at J = 1 to **+0.160 Mbps** at J = 5.

At **500 × 500 m** placement still matters. SCA is **9.418 Mbps**, **+2.753 ± 1.117** above random and **+0.940 ± 0.508** above k-means (both 100/100). Zenith-anchor is **9.684 Mbps**, **+0.266 ± 0.298** over SCA (84/100 wins, 0 losses, 16 ties, 65/100 practical). Multi-start is **+0.171 ± 0.274** (54/100 wins, 45/100 practical). Moving from 100 m to 500 m costs SCA **0.754 Mbps**, TD3 **1.571 Mbps**, and random **3.285 Mbps**.

On the Algorithm 2 campaign, SCA stays ahead of TD3 on every feasible point, and the lead shrinks as J or I grows. At 100 m it falls from **+0.202 ± 0.126** at J = 1 (92/100 practical) to **+0.009 ± 0.009** at J = 5 (0/100 practical, p = 7.1×10⁻¹⁷). At 500 m it falls from **+1.660 ± 0.881** at J = 1 to **+0.205 ± 0.194** at J = 5, and from **+0.845 ± 0.509** at I = 10 to **+0.404 ± 0.282** at I = 30 (98/100 practical). Campaign wall-clock averages **57 s** per seed at 100 m and **78 s** at 500 m.

Paired tests are Wilcoxon signed-rank on the same seed. Random, k-means, SCA, multi-start, and anchor use the cap25 campaign files. TD3 sweep rows use `results/run_10mhz_td3`. BH-FDR over the 24 unique sweep points (repeated default ticks excluded) is **24/24** for SCA versus random and versus k-means on both fields. The same 24-point test is **24/24** for SCA versus TD3 and for TD3 versus random on both fields. TD3 versus k-means is also **24/24**; at 100 m the mean is positive on 23 of those points and negative on the infeasible T_k = 0.8 s point. λ and CPU move the default-point score by well under 0.1 Mbps. T_k = 0.8 s is 0% feasible for every method on both fields. The IoT-axis I = 10 row was solved separately from the UAV-axis J = 3 row; use J = 3 as the headline default.

`sca_anchor` keep-bests against dynamic one-shot k-means SCA (same assignment policy as headline SCA). Anchor − SCA counts use `n100_sca_anchor.json` vs `n100_sca.json`.

The cap sweep uses the same frozen banks, so each cap is paired on the same 100 layouts. At every cap from 10% to 25%, on both fields, the bank ranking is **sca_anchor > sca_multistart > SCA > k-means > random**, and the default point is 100% feasible. With no per-link cap the anchor−SCA gap shrinks to **+0.003 ± 0.004** at 100 m (0/100 above 0.05 Mbps) and **+0.069 ± 0.112** at 500 m (36/100 practical). Multi-start's mean is then **0.0004 Mbps** above anchor at 100 m (p = 0.029) and **0.002 Mbps** above it at 500 m (p = 0.20). At 500 m the anchor−SCA practical count stays between **53 and 72** out of 100 from 10% through 25%, and is largest at 20% (**+0.285 ± 0.321**, 72/100). At 100 m that count peaks at 10% (**34/100**) and is at most 10/100 from 18% through 25%. T_k = 0.8 s stays 0% feasible at every cap.

---

## Default point (I = 10, J = 3)

The n100 bank summaries (`n100_all_summary.csv`) match this UAV-axis row on both fields.

### 100 × 100 m


| Method | Mean Mbps | Std | Feasible |
| --- | --- | --- | --- |
| **sca_anchor** | **10.188** | 0.010 | 100% |
| sca_multistart | 10.184 | 0.013 | 100% |
| **SCA** | **10.172** | 0.026 | 100% |
| TD3 | 10.130 | 0.042 | 100% |
| k-means | 10.110 | 0.048 | 100% |
| random | 9.950 | 0.119 | 100% |


Spread from anchor to random: **0.238 Mbps**. SCA to random: **0.223 Mbps**. SCA to TD3: **0.043 Mbps**. TD3 to k-means: **0.019 Mbps**. TD3 is the Algorithm 2 policy export on this bank. The sweep's J = 3 retrain averages **10.129** (**0.0005 Mbps** under this bank).

### 500 × 500 m


| Method | Mean Mbps | Std | Feasible |
| --- | --- | --- | --- |
| **sca_anchor** | **9.684** | 0.192 | 100% |
| sca_multistart | 9.589 | 0.248 | 100% |
| **SCA** | **9.418** | 0.394 | 100% |
| TD3 | 8.559 | 0.658 | 100% |
| k-means | 8.478 | 0.655 | 100% |
| random | 6.665 | 1.026 | 100% |


Spread from anchor to random: **3.018 Mbps**. SCA to random: **2.753 Mbps**. SCA to TD3: **0.859 Mbps**. TD3 to k-means: **0.081 Mbps**. The sweep's J = 3 retrain averages **8.561** (**0.0025 Mbps** above this bank).

### Same-seed gaps at J = 3


| Comparison | 100 m | 500 m |
| --- | --- | --- |
| SCA − random | **+0.223 ± 0.122**, 100/100, p = 1.6×10⁻³⁰ | **+2.753 ± 1.117**, 100/100, p = 1.6×10⁻³⁰ |
| SCA − k-means | **+0.062 ± 0.037**, 100/100, p = 1.6×10⁻³⁰ | **+0.940 ± 0.508**, 100/100, p = 1.6×10⁻³⁰ |
| SCA − TD3 | **+0.043 ± 0.031**, 97/100, 3 losses, 0 ties, 33 practical, p = 2.7×10⁻²⁷ | **+0.859 ± 0.502**, 100/100, 0 losses, 0 ties, 96 practical, p = 1.6×10⁻³⁰ |
| TD3 − k-means | **+0.019 ± 0.017**, 96/100, 4 losses, 0 ties, 6 practical, p = 8.1×10⁻²⁶ | **+0.081 ± 0.061**, 99/100, 1 loss, 0 ties, 64 practical, p = 3.2×10⁻³⁰ |
| TD3 − random | **+0.180 ± 0.126**, 96/100, 4 losses, 0 ties, 88 practical, p = 5.0×10⁻²⁶ | **+1.894 ± 1.202**, 95/100, 5 losses, 0 ties, 94 practical, p = 2.0×10⁻²⁵ |
| anchor − SCA | **+0.015 ± 0.021**, 80/100, 20 ties, 9 practical, p = 8.3×10⁻²⁵ | **+0.266 ± 0.298**, 84/100, 16 ties, 65 practical, p = 5.2×10⁻²⁶ |
| multi-start − SCA | **+0.011 ± 0.019**, 58/100, 21 losses, 21 ties, 8 practical | **+0.171 ± 0.274**, 54/100, 22 losses, 24 ties, 45 practical |


Practical means a paired gain above **0.05 Mbps**. Anchor versus SCA: p = 8.3×10⁻²⁵ at 100 m and 5.2×10⁻²⁶ at 500 m. Multi-start versus SCA: p = 2.4×10⁻⁹ at 100 m and 1.5×10⁻⁸ at 500 m. The TD3 rows are paired on each frozen bank against that field's `n100_sca.json`, `n100_kmeans.json`, and `n100_random.json`.

### Field drop at J = 3 (500 m − 100 m)


| Method | 100 m | 500 m | Δ |
| --- | --- | --- | --- |
| sca_anchor | 10.188 | 9.684 | −0.503 |
| sca_multistart | 10.184 | 9.589 | −0.595 |
| SCA | 10.172 | 9.418 | −0.754 |
| TD3 | 10.130 | 8.559 | −1.571 |
| k-means | 10.110 | 8.478 | −1.633 |
| random | 9.950 | 6.665 | −3.285 |


---

## Sweeps

Means are over all 100 seeds. A cell is 100% feasible unless the feasibility note says otherwise. Columns are random, k-means, TD3, SCA, multi-start, zenith-anchor (`sca_anchor`, dynamic assignment polish). The TD3 column is the Algorithm 2 campaign. At J = 3 that train is **10.129** at 100 m and **8.561** at 500 m, against bank means **10.130** and **8.559**. The λ = 2, T_k = 2.8 s, and f_j = 2×10⁸ rows are the same per-seed vector as UAV-axis J = 3. The IoT-axis I = 10 row is a separate solve.

The **100 × 100 m** Anchor column is the finished dynamic-assignment campaign (`results/run_10mhz_cap25/100m/campaign_sca_anchor.json`, 28/28 points). Versus the previous polish the largest mean shift is **+0.0007 Mbps**. At three decimals only three cells move, each by **+0.001**: I = 15 (**10.183**), T_k = 1.2 s (**10.141**), and T_k = 1.6 s (**10.168**). Anchor stays above SCA on every feasible point (gap **+0.006** to **+0.031 Mbps**) and above multi-start except at T_k = 0.8 s, where every method is 0% feasible and anchor ties the infeasible SCA score. The **500 × 500 m** Anchor column was not rerun. At J = 3 it is **9.682**, **0.002 Mbps** under the dynamic n100 bank (**9.684**).

### 100 × 100 m — UAV count J


| J | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.344 | 9.730 | 9.751 | 9.953 | 9.971 | 9.974 |
| 2 | 9.774 | 10.001 | 10.027 | 10.116 | 10.138 | 10.147 |
| 3 | 9.950 | 10.110 | 10.129 | 10.172 | 10.184 | 10.188 |
| 4 | 10.002 | 10.160 | 10.174 | 10.195 | 10.200 | 10.204 |
| 5 | 10.042 | 10.188 | 10.194 | 10.202 | 10.207 | 10.208 |




SCA−random is **+0.609** at J = 1 (98/100) and **+0.160** at J = 5 (100/100). Versus k-means the mean gap falls below 0.05 Mbps at J = 4 and J = 5 and stays FDR-significant (100/100). SCA−TD3 is **+0.202 ± 0.126** at J = 1 (100/100, 92 practical) and **+0.009 ± 0.009** at J = 5 (94/100, 6 losses, 0 practical, p = 7.1×10⁻¹⁷).

### 100 × 100 m — IoT count I

I = 10 on this axis is SCA **10.174**, about **0.001 Mbps** above the UAV-axis J = 3 row. Headline default stays the J = 3 row.


| I | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 9.951 | 10.111 | 10.133 | 10.174 | 10.185 | 10.188 |
| 15 | 10.003 | 10.118 | 10.135 | 10.170 | 10.177 | 10.183 |
| 20 | 10.024 | 10.119 | 10.136 | 10.164 | 10.168 | 10.173 |
| 25 | 10.024 | 10.116 | 10.131 | 10.154 | 10.157 | 10.162 |
| 30 | 10.018 | 10.111 | 10.125 | 10.145 | 10.148 | 10.152 |




SCA declines as I grows. SCA−random at I = 30 is **+0.127** (100/100). TD3 on this axis is **10.133** at I = 10 (**0.003 Mbps** above the UAV-axis J = 3 row) and is **0.007 Mbps** lower at I = 30. SCA−TD3 shrinks from **+0.041 ± 0.035** (96/100, 31 practical) to **+0.020 ± 0.016** (99/100, 2 practical).

### 100 × 100 m — arrival rate λ


| λ (/s) | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.941 | 10.107 | 10.125 | 10.167 | 10.178 | 10.183 |
| 1.5 | 9.947 | 10.109 | 10.128 | 10.171 | 10.182 | 10.186 |
| 2 | 9.950 | 10.110 | 10.129 | 10.172 | 10.184 | 10.188 |
| 2.5 | 9.951 | 10.111 | 10.129 | 10.173 | 10.184 | 10.188 |
| 3 | 9.951 | 10.111 | 10.131 | 10.174 | 10.185 | 10.189 |
| 3.5 | 9.952 | 10.111 | 10.130 | 10.174 | 10.185 | 10.189 |




SCA moves by **0.008 Mbps** from λ = 1.0 to λ = 3.5. TD3 moves by **0.006 Mbps** across the same grid, and SCA−TD3 stays between **+0.042** and **+0.044**.

### 100 × 100 m — AoDT threshold T_k


| T_k (s) | Random | K-means | TD3 | SCA | Multi-start | Anchor | Feasible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 9.416 | 9.874 | 9.859 | 10.118 | 10.145 | 10.118 | **0%** all methods |
| 1.2 | 9.855 | 10.074 | 10.087 | 10.118 | 10.132 | 10.141 | 100% |
| 1.6 | 9.914 | 10.096 | 10.112 | 10.150 | 10.162 | 10.168 | 100% |
| 2 | 9.934 | 10.104 | 10.121 | 10.161 | 10.174 | 10.178 | 100% |
| 2.4 | 9.943 | 10.108 | 10.127 | 10.169 | 10.180 | 10.184 | 100% |
| 2.8 | 9.950 | 10.110 | 10.129 | 10.172 | 10.184 | 10.188 | 100% |
| 3.2 | 9.954 | 10.112 | 10.131 | 10.175 | 10.186 | 10.190 | 100% |




The 0.8 s column is an infeasible score under frozen nearest association. Anchor matches SCA on that row. K-means scores **0.015 ± 0.015** above TD3 there (92/100, p = 1.5×10⁻²⁰). From T_k = 1.2 s the feasible TD3 score rises from **10.087** to **10.131** at 3.2 s, back above k-means.

### 100 × 100 m — UAV CPU f_j


| f_j (×10⁸) | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 9.945 | 10.109 | 10.127 | 10.165 | 10.181 | 10.185 |
| 1 | 9.948 | 10.110 | 10.129 | 10.171 | 10.183 | 10.187 |
| 1.5 | 9.949 | 10.110 | 10.130 | 10.172 | 10.183 | 10.187 |
| 2 | 9.950 | 10.110 | 10.129 | 10.172 | 10.184 | 10.188 |
| 2.5 | 9.950 | 10.110 | 10.129 | 10.173 | 10.184 | 10.188 |




All five CPU rows are 100% feasible. SCA moves by **0.008 Mbps** from 0.5×10⁸ to 2.5×10⁸. TD3 moves by **0.002 Mbps** across the same grid.

### 500 × 500 m — UAV count J


| J | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.965 | 5.271 | 5.300 | 6.959 | 7.071 | 7.164 |
| 2 | 5.598 | 7.182 | 7.247 | 8.626 | 8.830 | 8.952 |
| 3 | 6.665 | 8.478 | 8.561 | 9.418 | 9.589 | 9.682 |
| 4 | 7.184 | 9.254 | 9.342 | 9.808 | 9.910 | 10.006 |
| 5 | 7.553 | 9.735 | 9.782 | 9.987 | 10.065 | 10.101 |




Every J is 100% feasible. Anchor leads at every J. The anchor−SCA gap is **0.205** at J = 1 and **0.113** at J = 5. SCA−random is **2.995** at J = 1 and **2.434** at J = 5. SCA−TD3 is **+1.660 ± 0.881** at J = 1 (100/100, 99 practical) and **+0.205 ± 0.194** at J = 5 (99/100, 1 loss, 76 practical).

### 500 × 500 m — IoT count I

I = 10 on this axis is SCA **9.426**, **0.008 Mbps** above the UAV-axis J = 3 row. TD3 is **8.581**, **0.019 Mbps** above its UAV-axis row (**8.561**).


| I | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 6.679 | 8.483 | 8.581 | 9.426 | 9.601 | 9.696 |
| 15 | 6.972 | 8.521 | 8.595 | 9.333 | 9.460 | 9.543 |
| 20 | 6.995 | 8.479 | 8.549 | 9.171 | 9.248 | 9.318 |
| 25 | 6.797 | 8.389 | 8.461 | 8.979 | 9.029 | 9.084 |
| 30 | 6.470 | 8.246 | 8.317 | 8.721 | 8.801 | 8.848 |




SCA falls by **0.705 Mbps** from I = 10 to I = 30. TD3 is **0.263 Mbps** lower at I = 30 than at I = 10. SCA−TD3 shrinks from **+0.845 ± 0.509** (100/100, 96 practical) to **+0.404 ± 0.282** (100/100, 98 practical). Anchor stays above SCA at every I. SCA versus random and versus k-means stays above 0.05 Mbps on every unique point of this field (24/24).

### 500 × 500 m — arrival rate λ


| λ (/s) | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.552 | 8.420 | 8.512 | 9.308 | 9.490 | 9.579 |
| 1.5 | 6.634 | 8.462 | 8.542 | 9.391 | 9.564 | 9.654 |
| 2 | 6.665 | 8.478 | 8.561 | 9.418 | 9.589 | 9.682 |
| 2.5 | 6.681 | 8.486 | 8.567 | 9.433 | 9.605 | 9.698 |
| 3 | 6.690 | 8.491 | 8.578 | 9.442 | 9.614 | 9.707 |
| 3.5 | 6.697 | 8.495 | 8.583 | 9.445 | 9.621 | 9.714 |




SCA moves by **0.137 Mbps** from λ = 1.0 to λ = 3.5, larger than the 100 m λ sweep and still small next to the placement gaps. TD3 moves by **0.071 Mbps**. SCA−TD3 stays between **+0.796** and **+0.866**.

### 500 × 500 m — AoDT threshold T_k


| T_k (s) | Random | K-means | TD3 | SCA | Multi-start | Anchor | Feasible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 4.598 | 6.600 | 6.624 | 8.620 | 9.014 | 8.620 | **0%** all methods |
| 1.2 | 5.072 | 7.779 | 7.822 | 8.332 | 8.570 | 8.726 | random **78%**; others 100% |
| 1.6 | 6.099 | 8.227 | 8.298 | 8.988 | 9.186 | 9.267 | 100% |
| 2 | 6.435 | 8.364 | 8.438 | 9.229 | 9.398 | 9.483 | 100% |
| 2.4 | 6.581 | 8.434 | 8.518 | 9.342 | 9.513 | 9.605 | 100% |
| 2.8 | 6.665 | 8.478 | 8.561 | 9.418 | 9.589 | 9.682 | 100% |
| 3.2 | 6.720 | 8.507 | 8.589 | 9.468 | 9.645 | 9.736 | 100% |




Anchor matches the infeasible SCA score at T_k = 0.8 s. TD3's infeasible score is **6.624**, **0.024 ± 0.028** above k-means (83/100). From T_k = 1.2 s upward, anchor is the highest feasible mean and TD3 is 100% feasible, including the row where random is 78%.

### 500 × 500 m — UAV CPU f_j


| f_j (×10⁸) | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 6.606 | 8.445 | 8.530 | 9.355 | 9.533 | 9.627 |
| 1 | 6.649 | 8.469 | 8.554 | 9.405 | 9.578 | 9.667 |
| 1.5 | 6.660 | 8.475 | 8.557 | 9.413 | 9.584 | 9.677 |
| 2 | 6.665 | 8.478 | 8.561 | 9.418 | 9.589 | 9.682 |
| 2.5 | 6.668 | 8.479 | 8.564 | 9.421 | 9.592 | 9.685 |




All five rows are 100% feasible. SCA moves by **0.066 Mbps** across this CPU grid. TD3 moves by **0.033 Mbps**.

---

## TD3 sweep

Four jobs in `results/run_10mhz_td3/progress.txt` are `ok`. The n100 files match the headline bank byte for byte. The new runs are the two campaigns: 28 points × 100 seeds on each field, Algorithm 2, policy export, 7000 steps, k-means initialization, `--td3-preset alg2`, CUDA. Mean wall-clock is **57 s** per seed at 100 m (median 56 s; job 159944 s) and **78 s** at 500 m (median 71 s; job 219136 s).

TD3 is 100% feasible on 27/28 points and 0% at T_k = 0.8 s, the same pattern as SCA, k-means, multi-start, and anchor. The feasible order is **sca_anchor > sca_multistart > SCA > TD3 > k-means > random** on both fields. Zenith-anchor is ahead of TD3 on all 100 seeds at every point. The TD3 break in that order is the infeasible T_k = 0.8 s score at 100 m, where k-means is **0.015 ± 0.015** above TD3 (92/100). At 500 m that infeasible score still has TD3 **0.024 ± 0.028** above k-means (83/100).

The J = 3 campaign is a fresh train of the default point, **0.0005 Mbps** under the 100 m bank and **0.0025 Mbps** above the 500 m bank (per-seed |Δ| up to 0.062 and 0.172 Mbps). Its field drop is **1.568 Mbps**. The bank drop is **1.571 Mbps**. Paired SCA − TD3 on that retrain is **+0.043 ± 0.032** (97/100, 3 losses, 33 practical, p = 1.4×10⁻²⁷) at 100 m and **+0.857 ± 0.490** (100/100, 97 practical, p = 1.6×10⁻³⁰) at 500 m.

BH-FDR on the 24 unique points is **24/24** for SCA versus TD3 and for TD3 versus random, both fields, with every mean positive. At 100 m the SCA − TD3 mean is above 0.05 Mbps on 3/24 unique points (J = 1, J = 2, and infeasible T_k = 0.8 s). At 500 m it is above 0.05 Mbps on 24/24. TD3 versus k-means is **24/24** as well. The mean is above 0.05 Mbps on 0/24 unique points at 100 m and on 20/24 at 500 m. At 100 m and J = 5, TD3's lead over k-means is **+0.006 ± 0.007** (77/100, 23 losses, 0 practical, p = 1.4×10⁻¹⁶).

λ and CPU barely move the gap. SCA − TD3 stays inside **+0.042** to **+0.044** across the 100 m λ grid and inside **+0.796** to **+0.866** across the 500 m λ grid.

### SCA − TD3 on the campaign


| Point | 100 m | 500 m |
| --- | --- | --- |
| J = 1 | **+0.202 ± 0.126**, 100/100, 92 practical, p = 1.6×10⁻³⁰ | **+1.660 ± 0.881**, 100/100, 99 practical, p = 1.6×10⁻³⁰ |
| J = 3 | **+0.043 ± 0.032**, 97/100, 3 losses, 33 practical, p = 1.4×10⁻²⁷ | **+0.857 ± 0.490**, 100/100, 97 practical, p = 1.6×10⁻³⁰ |
| J = 5 | **+0.009 ± 0.009**, 94/100, 6 losses, 0 practical, p = 7.1×10⁻¹⁷ | **+0.205 ± 0.194**, 99/100, 1 loss, 76 practical, p = 1.4×10⁻²⁷ |
| I = 10 | **+0.041 ± 0.035**, 96/100, 4 losses, 31 practical, p = 1.6×10⁻²⁵ | **+0.845 ± 0.509**, 100/100, 96 practical, p = 1.6×10⁻³⁰ |
| I = 30 | **+0.020 ± 0.016**, 99/100, 1 loss, 2 practical, p = 3.2×10⁻³⁰ | **+0.404 ± 0.282**, 100/100, 98 practical, p = 1.6×10⁻³⁰ |


I = 10 is the separate IoT-axis solve. J = 3 is the UAV-axis default. Practical means a paired gain above **0.05 Mbps**. Ties are zero on every row.

### TD3 − k-means on the campaign


| Point | 100 m | 500 m |
| --- | --- | --- |
| J = 1 | **+0.022 ± 0.016**, 99/100, 1 loss, 6 practical | **+0.029 ± 0.026**, 95/100, 5 losses, 19 practical |
| J = 3 | **+0.019 ± 0.017**, 98/100, 2 losses, 7 practical | **+0.084 ± 0.065**, 99/100, 1 loss, 59 practical |
| J = 5 | **+0.006 ± 0.007**, 77/100, 23 losses, 0 practical | **+0.047 ± 0.044**, 88/100, 12 losses, 42 practical |
| I = 10 | **+0.022 ± 0.023**, 92/100, 8 losses, 12 practical | **+0.097 ± 0.071**, 98/100, 2 losses, 70 practical |
| I = 30 | **+0.014 ± 0.014**, 95/100, 5 losses, 2 practical | **+0.071 ± 0.054**, 96/100, 4 losses, 59 practical |


---

## Cap family

Same 10 MHz protocol and the same n100 banks as the headline. Caps are per-link fractions of `B_sys`: none, then 10, 12, 15, 18, 20, 22, and 25%. The 25% row matches the headline bank. TD3 was run at 25% only; it is not in this family. Means below are the frozen-bank default point (I = 10, J = 3). A cell is 100% feasible. **Anchor** is zenith-subset SCA with dynamic assignment polish (`n100_sca_anchor.json` at 25%; cap-sweep `sca_anchor` elsewhere until those banks are rerun).

### 100 × 100 m


| Cap | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| none | 10.088 | 10.171 | 10.196 | 10.199 | 10.199 |
| 10% | 9.416 | 9.874 | 9.879 | 9.940 | 9.944 |
| 12% | 9.623 | 9.973 | 10.025 | 10.058 | 10.065 |
| 15% | 9.770 | 10.037 | 10.110 | 10.127 | 10.132 |
| 18% | 9.851 | 10.070 | 10.141 | 10.154 | 10.159 |
| 20% | 9.889 | 10.086 | 10.152 | 10.168 | 10.173 |
| 22% | 9.915 | 10.096 | 10.162 | 10.173 | 10.177 |
| 25% | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 |
Tightening the cap lowers every method. From no cap to 10%, anchor falls **0.255 Mbps** and random falls **0.672 Mbps**. SCA minus k-means is only **+0.005 ± 0.020** at 10% (100/100, **3/100** practical) and is back above 0.05 Mbps from 12% up.

### 500 × 500 m


| Cap | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| none | 7.891 | 9.373 | 9.829 | 9.900 | 9.898 |
| 10% | 4.598 | 6.600 | 7.092 | 7.274 | 7.344 |
| 12% | 5.126 | 7.171 | 7.946 | 8.138 | 8.210 |
| 15% | 5.680 | 7.678 | 8.640 | 8.817 | 8.903 |
| 18% | 6.079 | 8.016 | 9.007 | 9.211 | 9.282 |
| 20% | 6.281 | 8.184 | 9.150 | 9.353 | 9.434 |
| 22% | 6.462 | 8.321 | 9.320 | 9.489 | 9.551 |
| 25% | 6.665 | 8.478 | 9.418 | 9.589 | 9.684 |
From no cap to 10%, anchor falls **2.554 Mbps** and random falls **3.293 Mbps**. SCA minus random is **100/100** and **100/100 practical** at every cap, including no cap (**+1.938 ± 1.072**). SCA minus k-means stays practical on at least 67/100 seeds at every cap.

### Anchor − SCA on the same bank


| Cap | 100 m | 500 m |
| --- | --- | --- |
| none | **+0.003 ± 0.004**, 50/100, 15 losses, 35 ties, 0 practical, p = 5.1×10⁻¹⁰ | **+0.069 ± 0.112**, 52/100, 13 losses, 35 ties, 36 practical, p = 8.0×10⁻¹¹ |
| 10% | **+0.065 ± 0.097**, 70/100, 0 losses, 30 ties, 34 practical, p = 6.5×10⁻²⁷ | **+0.251 ± 0.343**, 67/100, 3 losses, 30 ties, 53 practical, p = 2.8×10⁻²⁰ |
| 12% | **+0.039 ± 0.065**, 63/100, 8 losses, 29 ties, 26 practical, p = 7.2×10⁻¹⁶ | **+0.265 ± 0.356**, 70/100, 7 losses, 23 ties, 56 practical, p = 5.6×10⁻¹⁸ |
| 15% | **+0.022 ± 0.030**, 69/100, 13 losses, 18 ties, 17 practical, p = 2.5×10⁻¹⁵ | **+0.263 ± 0.311**, 74/100, 6 losses, 20 ties, 62 practical, p = 8.6×10⁻²⁰ |
| 18% | **+0.018 ± 0.021**, 82/100, 8 losses, 10 ties, 8 practical, p = 1.2×10⁻²¹ | **+0.275 ± 0.363**, 76/100, 11 losses, 13 ties, 63 practical, p = 2.7×10⁻¹⁸ |
| 20% | **+0.021 ± 0.027**, 80/100, 10 losses, 10 ties, 10 practical, p = 1.5×10⁻²⁰ | **+0.285 ± 0.321**, 82/100, 6 losses, 12 ties, 72 practical, p = 1.1×10⁻²² |
| 22% | **+0.015 ± 0.021**, 84/100, 8 losses, 8 ties, 3 practical, p = 4.2×10⁻²³ | **+0.232 ± 0.281**, 84/100, 3 losses, 13 ties, 69 practical, p = 2.0×10⁻²⁴ |
| 25% | **+0.015 ± 0.021**, 80/100, 0 losses, 20 ties, 9 practical, p = 8.3×10⁻²⁵ | **+0.266 ± 0.298**, 84/100, 0 losses, 16 ties, 65 practical, p = 5.2×10⁻²⁶ |


Practical means a paired gain above **0.05 Mbps**. On the UAV-count campaign, anchor's mean is above SCA at every J from 1 to 5 for every cap. The no-cap edge is under **0.01 Mbps** at 100 m for every J. At 500 m and 10% the edge grows with J, from **+0.043** at J = 1 to **+0.349** at J = 5.

T_k = 0.8 s is 0% feasible for every method at every cap on both fields. At 500 m and T_k = 1.2 s, random feasibility rises as the cap loosens: **11%** at 10%, **33%** at 12%, **47%** at 15%, **59%** at 18%, **67%** at 20%, **73%** at 22%, **78%** at 25%, and **91%** with no cap. The 10% cap is the only cell where a UAV-count point is below 100% feasible aside from that threshold: random is **98%** at J = 1 and **99%** at J = 2.

---

## Files

| Path | Contents |
| --- | --- |
| `server_results/results/run_10mhz_cap25/100m/campaign_all.csv` | 100 m sweep means |
| `server_results/results/run_10mhz_cap25/500m/campaign_all.csv` | 500 m sweep means |
| `server_results/results/run_10mhz_cap25/{100m,500m}/campaign_<method>.json` | Per-seed campaign used for the paired tests |
| `server_results/results/run_10mhz_cap25/{100m,500m}/n100_all_summary.csv` | Frozen-bank default point |
| `results/run_10mhz_cap25/{100m,500m}/n100_sca_anchor.json` | Zenith-anchor n100 bank (dynamic assignment polish) |
| `results/run_10mhz_cap25/{100m,500m}/n100_td3.json` | Algorithm 2 TD3 bank, seeds 1–100. Byte-identical under `results/run_10mhz_td3/cap25/` |
| `results/run_10mhz_td3/cap25/{100m,500m}/campaign_td3.json` | Algorithm 2 TD3 sweep, 28 points, seeds 1–100 |
| `results/run_10mhz_td3/cap25/{100m,500m}/campaign_all.csv` | TD3 sweep means |
| `results/run_10mhz_td3/progress.txt` | TD3 ledger, 4/4 `ok` |
| `server_results/results/run_10mhz_cap25/progress.txt` | Job ledger, all `ok` |
| `results/run_10mhz_cap_sweep/{nocap,cap10,cap12,cap15,cap18,cap20,cap22,cap25}/{100m,500m}/n100_all.json` | Cap-family frozen bank, per seed |
| `results/run_10mhz_cap_sweep/{nocap,cap10,cap12,cap15,cap18,cap20,cap22,cap25}/{100m,500m}/campaign_all.csv` | Cap-family sweep means |
| `results/run_10mhz_cap_sweep/progress.txt` | Cap-sweep ledger, 160/160 `ok` |

Prior audit, 8.8 MHz campaigns, PSO, and the earlier TD3 runs: [`results_old.md`](results_old.md).
