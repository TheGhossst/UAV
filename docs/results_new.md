# Experimental results

**10 MHz, 25% per-link cap, 100 scenarios.** Headline numbers are from `results/run_10mhz_cap25/`. The cap family (no cap, then 10% through 25%) is `results/run_10mhz_cap_sweep/`, 160/160 jobs complete. The 25% TD3 sweep (both fields, n100 bank and full campaign) is `results/run_10mhz_td3/cap25/`. TD3 banks at the other caps, PSO, the subset controls, the dynamic-anchor rerun, frozen-assignment SCA, Fig. 11, and the noise checks are in those same trees. `results/run_10mhz_open/` adds the dynamic-anchor campaigns at the other caps, PSO with an LP particle fitness, the degree reading of Eq. (4), and CMA-ES. The earlier ledger (8.8 MHz, 20 kHz audit) is in [`results_old.md`](results_old.md).


| | |
| --- | --- |
| **Headline run** | `results/run_10mhz_cap25/` (20 jobs, 25% cap) |
| **Cap sweep** | `results/run_10mhz_cap_sweep/` (no cap, 10, 12, 15, 18, 20, 22, 25%; both fields; n100 bank and full campaign) |
| **Protocol** | `B_sys` = 10 MHz, seeds 1–100, CVXPY, 30 SCA iterations. Headline cap 25% is 2.50 MHz/link |
| **Fields** | **100 × 100 m** and **500 × 500 m** |
| **Methods** | random, k-means, SCA, `sca_multistart`, `sca_anchor` (dynamic assignment polish), PSO, TD3 (Algorithm 2), residual TD3 on SCA |
| **TD3 bank** | Both n100 banks at 25%. Algorithm 2, policy export, 7000 steps, k-means init. `results/run_10mhz_cap25/{100m,500m}/n100_td3.json` is byte-identical to `results/run_10mhz_td3/cap25/{100m,500m}/n100_td3.json` |
| **TD3 sweep** | `results/run_10mhz_td3/cap25/{100m,500m}/campaign_td3.json`. Same protocol, seeds 1–100, `--td3-preset alg2`, CUDA. 100 m finished 2026-10-15 (159944 s); 500 m finished 2026-10-17 (219136 s) |
| **Also on disk** | TD3 n100 at every cap (`results/run_10mhz_td3/`); PSO at every cap and the p-median / random-subset / continuous controls at no cap, 10%, and 25% (`results/run_10mhz_controls/`); dynamic-anchor banks beside the frozen sweep anchor; Fig. 11 and the noise checks under `results/run_10mhz_cap25/`; the open follow-up in `results/run_10mhz_open/` |
| **Not in these runs** | TD3 campaigns at caps other than 25% (still to be added), the learned proposal policy, 20 kHz diagnostic |


---

## Conclusion

On the default point (I = 10, J = 3, λ = 2/s, T_k = 2.8 s, f_j = 2×10⁸), every method is 100% feasible on both fields. The ranking is the same on both, and it is the ranking on every feasible point of the TD3 campaign:

**sca_anchor > sca_multistart > SCA > TD3 > k-means > random.**

Algorithm 2 TD3 on the frozen banks (policy export, 7000 steps) is 100% feasible on both fields, with max AoDT pinned at 2.800 s. At 100 m it is **10.130 ± 0.042 Mbps**: SCA leads by **0.043 ± 0.031** (97/100, 33 practical, p = 2.7×10⁻²⁷), and TD3 leads k-means by **0.019 ± 0.017** (96/100, 6 practical). At 500 m it is **8.559 ± 0.658 Mbps**: the SCA lead opens to **0.859 ± 0.502** (100/100, 96 practical, p = 1.6×10⁻³⁰), while TD3 still leads k-means by **0.081 ± 0.061** (99/100, 64 practical) and random by **1.894 ± 1.202** (95/100, 94 practical). The 100→500 m drop is **1.571 Mbps**, next to k-means (1.633) and larger than SCA (0.754). Mean wall-clock on that bank is **69 s** per seed at 100 m and **71 s** at 500 m.

At **100 × 100 m** the radio is close to the zenith ceiling (**10.213 Mbps**). SCA is **10.172 Mbps**, **+0.223 ± 0.122** above random (100/100) and **+0.062 ± 0.037** above k-means (100/100). Zenith-anchor adds **+0.015 ± 0.021** over headline SCA (80/100 wins, 0 losses, 20 ties, 9/100 above 0.05 Mbps). Multi-start adds **+0.011 ± 0.019** (58/100 wins, 8/100 practical). The SCA−random gap shrinks from **+0.609 Mbps** at J = 1 to **+0.160 Mbps** at J = 5.

At **500 × 500 m** placement still matters. SCA is **9.418 Mbps**, **+2.753 ± 1.117** above random and **+0.940 ± 0.508** above k-means (both 100/100). Zenith-anchor is **9.684 Mbps**, **+0.266 ± 0.298** over SCA (84/100 wins, 0 losses, 16 ties, 65/100 practical). Multi-start is **+0.171 ± 0.274** (54/100 wins, 45/100 practical). Moving from 100 m to 500 m costs SCA **0.754 Mbps**, TD3 **1.571 Mbps**, and random **3.285 Mbps**.

On the Algorithm 2 campaign, SCA stays ahead of TD3 on every feasible point, and the lead shrinks as J or I grows. At 100 m it falls from **+0.202 ± 0.126** at J = 1 (92/100 practical) to **+0.009 ± 0.009** at J = 5 (0/100 practical, p = 7.1×10⁻¹⁷). At 500 m it falls from **+1.660 ± 0.881** at J = 1 to **+0.205 ± 0.194** at J = 5, and from **+0.845 ± 0.509** at I = 10 to **+0.404 ± 0.282** at I = 30 (98/100 practical). Campaign wall-clock averages **57 s** per seed at 100 m and **78 s** at 500 m.

Paired tests are Wilcoxon signed-rank on the same seed. Random, k-means, SCA, multi-start, and anchor use the cap25 campaign files. TD3 sweep rows use `results/run_10mhz_td3`. BH-FDR over the 24 unique sweep points (repeated default ticks excluded) is **24/24** for SCA versus random and versus k-means on both fields. The same 24-point test is **24/24** for SCA versus TD3 and for TD3 versus random on both fields. TD3 versus k-means is also **24/24**. λ and CPU move the default-point score by well under 0.1 Mbps. The reported AoDT axis starts at 1.2 s. The IoT-axis I = 10 row was solved separately from the UAV-axis J = 3 row; use J = 3 as the headline default.

`sca_anchor` keep-bests against dynamic one-shot k-means SCA (same assignment policy as headline SCA). Anchor − SCA counts use `n100_sca_anchor.json` vs `n100_sca.json`.

The cap sweep uses the same frozen banks, so each cap is paired on the same 100 layouts. At every cap from 10% to 25%, on both fields, the five sweep methods rank **sca_anchor > sca_multistart > SCA > k-means > random**, and the default point is 100% feasible. The anchor numbers in that comparison are the dynamic-assignment polish. Turning the flag on moves the mean by at most **0.004 Mbps** relative to the frozen polish, and the dynamic polish has **0 losses** against dynamic SCA on every cap. With no per-link cap the anchor−SCA gap shrinks to **+0.003 ± 0.004** at 100 m (0/100 above 0.05 Mbps) and **+0.071 ± 0.110** at 500 m (36/100 practical). Multi-start is then **0.0003 Mbps** above anchor at 100 m (p = 0.091) and **0.0002 Mbps** below it at 500 m (p = 0.60). At 500 m the anchor−SCA practical count stays between **53 and 73** out of 100 from 10% through 25%, and is largest at 20% (**+0.289 ± 0.322**, 73/100). At 100 m that count peaks at 10% (**34/100**, **+0.067 ± 0.098**) and is at most 10/100 from 18% through 25%.

Freezing SCA's own assignment, at the 25% default point, costs **0.004 Mbps** at 100 m (1/100 practical) and **0.024 Mbps** at 500 m (19/100 practical). Algorithm 2 TD3 on these banks stays between k-means and SCA at every cap on the 500 m field. At 100 m and a 10% cap, equal-share PSO leads SCA on the signed-rank test (**+0.057 ± 0.096**, 50 seeds each way, 28/100 practical, p = 1.0×10⁻⁵). On that same cell k-means is ahead of TD3 on 74/100 seeds. At 500 m and 10% the PSO mean is **+0.082** over SCA and the signed-rank test is not (p = 0.76). Residual TD3, a ±10 m residual on an SCA start, finishes **0.001 Mbps** under SCA at 100 m and **0.022 Mbps** under it at 500 m. Eq. (17) still scores heterogeneous arrivals as older than uniform-slow on 20/20 seeds at 10 MHz, on both fields; the event simulator scores them the other way.

The open follow-up keeps those bank results and extends them. Dynamic-anchor campaigns at every cap other than 25% reproduce the bank at J = 3. On seeds where both methods are feasible, anchor has 0 losses at every sweep point. At 500 m the mean gap is above 0.05 Mbps on 22 or 23 of the 23 unique points, and the largest point is T_k = 1.2 s (**+0.35 to +0.45 Mbps**). At 100 m that bar is cleared on 17/23 unique points at a 10% cap and on no unique point at the other caps. Scoring each PSO particle with the frozen-q LP leaves the 10% bank identical to equal share and raises the 25% bank by **+0.056 Mbps** at 100 m and **+0.235 Mbps** at 500 m. The signed-rank result against SCA is still the 100 m, 10% win; the other caps stay within about 0.08 Mbps of SCA, and no LP seed is a practical win over the dynamic zenith polish. Reading Eq. (4) in degrees on the 25% banks moves the 100 m default point from about 10 Mbps to about 65 Mbps and keeps the same method order. The 500 m anchor−SCA gap grows from **+0.266** to **+1.076 ± 1.494** (74/100 practical). CMA-ES on seeds 1–20 adds one practical seed over frozen SCA at 100 m and none at 500 m.

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

The **100 × 100 m** Anchor column is the dynamic-assignment campaign (`results/run_10mhz_cap25/100m/campaign_sca_anchor.json`, 28/28 points). Versus the previous polish the largest mean shift is **+0.0007 Mbps**. At three decimals only three cells move, each by **+0.001**: I = 15 (**10.183**), T_k = 1.2 s (**10.141**), and T_k = 1.6 s (**10.168**). Anchor stays above SCA on every reported point (gap **+0.006** to **+0.031 Mbps**) and above multi-start. The **500 × 500 m** Anchor column is now the same polish (`campaign_sca_anchor_dyn.json`, 27 points, AoDT from 1.2 s). At J = 3 it is **9.684**, matching the dynamic n100 bank. Against the frozen campaign still stored as `campaign_sca_anchor.json`, the largest mean shift is **+0.016 Mbps** (T_k = 1.2 s). J = 1, J = 2, and J = 5 are unchanged at three decimals. Anchor has **0 losses** against SCA on all 27 points, and the mean gap is above 0.05 Mbps on all 23 unique points (BH-FDR 23/23).

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
| 1.2 | 9.855 | 10.074 | 10.087 | 10.118 | 10.132 | 10.141 | 100% |
| 1.6 | 9.914 | 10.096 | 10.112 | 10.150 | 10.162 | 10.168 | 100% |
| 2 | 9.934 | 10.104 | 10.121 | 10.161 | 10.174 | 10.178 | 100% |
| 2.4 | 9.943 | 10.108 | 10.127 | 10.169 | 10.180 | 10.184 | 100% |
| 2.8 | 9.950 | 10.110 | 10.129 | 10.172 | 10.184 | 10.188 | 100% |
| 3.2 | 9.954 | 10.112 | 10.131 | 10.175 | 10.186 | 10.190 | 100% |




The axis starts at 1.2 s. From there the feasible TD3 score rises from **10.087** to **10.131** at 3.2 s, back above k-means.

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
| 3 | 6.665 | 8.478 | 8.561 | 9.418 | 9.589 | 9.684 |
| 4 | 7.184 | 9.254 | 9.342 | 9.808 | 9.910 | 10.008 |
| 5 | 7.553 | 9.735 | 9.782 | 9.987 | 10.065 | 10.101 |




Every J is 100% feasible. Anchor leads at every J, with 0 losses against SCA. The anchor−SCA gap is **+0.205 ± 0.418** at J = 1 (43/100, 57 ties, 31 practical) and **+0.114 ± 0.123** at J = 5 (90/100, 10 ties, 59 practical). It peaks at J = 2 (**+0.326 ± 0.445**, 65 practical). SCA−random is **2.995** at J = 1 and **2.434** at J = 5. SCA−TD3 is **+1.660 ± 0.881** at J = 1 (100/100, 99 practical) and **+0.205 ± 0.194** at J = 5 (99/100, 1 loss, 76 practical).

### 500 × 500 m — IoT count I

I = 10 on this axis is SCA **9.426**, **0.008 Mbps** above the UAV-axis J = 3 row. TD3 is **8.581**, **0.019 Mbps** above its UAV-axis row (**8.561**).


| I | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 6.679 | 8.483 | 8.581 | 9.426 | 9.601 | 9.701 |
| 15 | 6.972 | 8.521 | 8.595 | 9.333 | 9.460 | 9.546 |
| 20 | 6.995 | 8.479 | 8.549 | 9.171 | 9.248 | 9.322 |
| 25 | 6.797 | 8.389 | 8.461 | 8.979 | 9.029 | 9.089 |
| 30 | 6.470 | 8.246 | 8.317 | 8.721 | 8.801 | 8.855 |




SCA falls by **0.705 Mbps** from I = 10 to I = 30. TD3 is **0.263 Mbps** lower at I = 30 than at I = 10. SCA−TD3 shrinks from **+0.845 ± 0.509** (100/100, 96 practical) to **+0.404 ± 0.282** (100/100, 98 practical). Anchor stays above SCA at every I. SCA versus random and versus k-means stays above 0.05 Mbps on every unique point of this field (24/24).

### 500 × 500 m — arrival rate λ


| λ (/s) | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.552 | 8.420 | 8.512 | 9.308 | 9.490 | 9.582 |
| 1.5 | 6.634 | 8.462 | 8.542 | 9.391 | 9.564 | 9.656 |
| 2 | 6.665 | 8.478 | 8.561 | 9.418 | 9.589 | 9.684 |
| 2.5 | 6.681 | 8.486 | 8.567 | 9.433 | 9.605 | 9.699 |
| 3 | 6.690 | 8.491 | 8.578 | 9.442 | 9.614 | 9.709 |
| 3.5 | 6.697 | 8.495 | 8.583 | 9.445 | 9.621 | 9.715 |




SCA moves by **0.137 Mbps** from λ = 1.0 to λ = 3.5, larger than the 100 m λ sweep and still small next to the placement gaps. TD3 moves by **0.071 Mbps**. SCA−TD3 stays between **+0.796** and **+0.866**.

### 500 × 500 m — AoDT threshold T_k


| T_k (s) | Random | K-means | TD3 | SCA | Multi-start | Anchor | Feasible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.2 | 5.072 | 7.779 | 7.822 | 8.332 | 8.570 | 8.742 | random **78%**; others 100% |
| 1.6 | 6.099 | 8.227 | 8.298 | 8.988 | 9.186 | 9.275 | 100% |
| 2 | 6.435 | 8.364 | 8.438 | 9.229 | 9.398 | 9.487 | 100% |
| 2.4 | 6.581 | 8.434 | 8.518 | 9.342 | 9.513 | 9.607 | 100% |
| 2.8 | 6.665 | 8.478 | 8.561 | 9.418 | 9.589 | 9.684 | 100% |
| 3.2 | 6.720 | 8.507 | 8.589 | 9.468 | 9.645 | 9.738 | 100% |




From T_k = 1.2 s upward, anchor is the highest feasible mean and TD3 is 100% feasible, including the row where random is 78%.

### 500 × 500 m — UAV CPU f_j


| f_j (×10⁸) | Random | K-means | TD3 | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 6.606 | 8.445 | 8.530 | 9.355 | 9.533 | 9.630 |
| 1 | 6.649 | 8.469 | 8.554 | 9.405 | 9.578 | 9.669 |
| 1.5 | 6.660 | 8.475 | 8.557 | 9.413 | 9.584 | 9.679 |
| 2 | 6.665 | 8.478 | 8.561 | 9.418 | 9.589 | 9.684 |
| 2.5 | 6.668 | 8.479 | 8.564 | 9.421 | 9.592 | 9.687 |




All five rows are 100% feasible. SCA moves by **0.066 Mbps** across this CPU grid. TD3 moves by **0.033 Mbps**.

---

## TD3 sweep

Four jobs in `results/run_10mhz_td3/progress.txt` are `ok`. The n100 files match the headline bank byte for byte. The new runs are the two campaigns: 28 points × 100 seeds on each field, Algorithm 2, policy export, 7000 steps, k-means initialization, `--td3-preset alg2`, CUDA. Mean wall-clock is **57 s** per seed at 100 m (median 56 s; job 159944 s) and **78 s** at 500 m (median 71 s; job 219136 s).

From T_k = 1.2 s, TD3 is 100% feasible on every reported point, the same pattern as SCA, k-means, multi-start, and anchor. The order is **sca_anchor > sca_multistart > SCA > TD3 > k-means > random** on both fields. Zenith-anchor is ahead of TD3 on all 100 seeds at every reported point.

The J = 3 campaign is a fresh train of the default point, **0.0005 Mbps** under the 100 m bank and **0.0025 Mbps** above the 500 m bank (per-seed |Δ| up to 0.062 and 0.172 Mbps). Its field drop is **1.568 Mbps**. The bank drop is **1.571 Mbps**. Paired SCA − TD3 on that retrain is **+0.043 ± 0.032** (97/100, 3 losses, 33 practical, p = 1.4×10⁻²⁷) at 100 m and **+0.857 ± 0.490** (100/100, 97 practical, p = 1.6×10⁻³⁰) at 500 m.

BH-FDR on the 24 unique points is **24/24** for SCA versus TD3 and for TD3 versus random, both fields, with every mean positive. At 100 m the SCA − TD3 mean is above 0.05 Mbps on J = 1 and J = 2. The stored 24-point family also counts an unused T_k = 0.8 s row. At 500 m it is above 0.05 Mbps on 24/24. TD3 versus k-means is **24/24** as well. The mean is above 0.05 Mbps on 0/24 unique points at 100 m and on 20/24 at 500 m. At 100 m and J = 5, TD3's lead over k-means is **+0.006 ± 0.007** (77/100, 23 losses, 0 practical, p = 1.4×10⁻¹⁶).

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

Same 10 MHz protocol and the same n100 banks as the headline. Caps are per-link fractions of `B_sys`: none, then 10, 12, 15, 18, 20, 22, and 25%. The 25% row matches the headline bank. Means below are the frozen-bank default point (I = 10, J = 3). A cell is 100% feasible. **Anchor** is the dynamic-assignment zenith polish (`n100_sca_anchor.json` at 25%; `n100_sca_anchor_dyn.json` at the other caps). The frozen polish stored as the sweep's `n100_sca_anchor.json` is within **0.004 Mbps** on every cell. TD3 and PSO on these banks are in the sections below.

### 100 × 100 m


| Cap | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| none | 10.088 | 10.171 | 10.196 | 10.199 | 10.199 |
| 10% | 9.416 | 9.874 | 9.879 | 9.940 | 9.946 |
| 12% | 9.623 | 9.973 | 10.025 | 10.058 | 10.065 |
| 15% | 9.770 | 10.037 | 10.110 | 10.127 | 10.133 |
| 18% | 9.851 | 10.070 | 10.141 | 10.154 | 10.159 |
| 20% | 9.889 | 10.086 | 10.152 | 10.168 | 10.173 |
| 22% | 9.915 | 10.096 | 10.162 | 10.173 | 10.177 |
| 25% | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 |

Tightening the cap lowers every method. From no cap to 10%, anchor falls **0.253 Mbps** and random falls **0.672 Mbps**. SCA minus k-means is only **+0.005 ± 0.020** at 10% (100/100, **3/100** practical) and is back above 0.05 Mbps from 12% up.

### 500 × 500 m


| Cap | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| none | 7.891 | 9.373 | 9.829 | 9.900 | 9.900 |
| 10% | 4.598 | 6.600 | 7.092 | 7.274 | 7.344 |
| 12% | 5.126 | 7.171 | 7.946 | 8.138 | 8.212 |
| 15% | 5.680 | 7.678 | 8.640 | 8.817 | 8.905 |
| 18% | 6.079 | 8.016 | 9.007 | 9.211 | 9.285 |
| 20% | 6.281 | 8.184 | 9.150 | 9.353 | 9.438 |
| 22% | 6.462 | 8.321 | 9.320 | 9.489 | 9.553 |
| 25% | 6.665 | 8.478 | 9.418 | 9.589 | 9.684 |

From no cap to 10%, anchor falls **2.556 Mbps** and random falls **3.293 Mbps**. SCA minus random is **100/100** and **100/100 practical** at every cap, including no cap (**+1.938 ± 1.072**). SCA minus k-means stays practical on at least 67/100 seeds at every cap.

### Anchor − SCA on the same bank


| Cap | 100 m | 500 m |
| --- | --- | --- |
| none | **+0.003 ± 0.004**, 48/100, 0 losses, 52 ties, 0 practical, p = 8.9×10⁻¹⁶ | **+0.071 ± 0.110**, 54/100, 0 losses, 46 ties, 36 practical, p = 6.9×10⁻¹⁸ |
| 10% | **+0.067 ± 0.098**, 71/100, 0 losses, 29 ties, 34 practical, p = 1.6×10⁻²⁷ | **+0.252 ± 0.342**, 68/100, 0 losses, 32 ties, 53 practical, p = 5.2×10⁻²⁶ |
| 12% | **+0.040 ± 0.064**, 66/100, 0 losses, 34 ties, 27 practical, p = 8.3×10⁻²⁵ | **+0.267 ± 0.357**, 74/100, 0 losses, 26 ties, 56 practical, p = 5.2×10⁻²⁶ |
| 15% | **+0.022 ± 0.030**, 69/100, 0 losses, 31 ties, 17 practical, p = 8.3×10⁻²⁵ | **+0.265 ± 0.311**, 76/100, 0 losses, 24 ties, 62 practical, p = 1.6×10⁻²⁷ |
| 18% | **+0.018 ± 0.021**, 84/100, 0 losses, 16 ties, 8 practical, p = 1.3×10⁻²⁶ | **+0.278 ± 0.364**, 76/100, 0 losses, 24 ties, 65 practical, p = 1.3×10⁻²⁶ |
| 20% | **+0.021 ± 0.027**, 82/100, 0 losses, 18 ties, 10 practical, p = 5.2×10⁻²⁶ | **+0.289 ± 0.322**, 83/100, 0 losses, 17 ties, 73 practical, p = 4.0×10⁻²⁸ |
| 22% | **+0.015 ± 0.021**, 86/100, 0 losses, 14 ties, 4 practical, p = 1.6×10⁻²⁷ | **+0.233 ± 0.281**, 85/100, 0 losses, 15 ties, 70 practical, p = 5.0×10⁻²⁹ |
| 25% | **+0.015 ± 0.021**, 80/100, 0 losses, 20 ties, 9 practical, p = 8.3×10⁻²⁵ | **+0.266 ± 0.298**, 84/100, 0 losses, 16 ties, 65 practical, p = 5.2×10⁻²⁶ |


Practical means a paired gain above **0.05 Mbps**. The p-values are the one-sided tail. The 25% row at 500 m keeps the headline convention from the default-point table. Dynamic anchor has 0 losses on every row because it keep-bests against dynamic SCA. The frozen polish, paired with that same SCA column, still has losses; its rate is the column above to within 0.004 Mbps.

On the frozen-polish UAV-count campaigns, anchor's mean is above SCA at every J from 1 to 5 for every cap. The no-cap edge is under **0.01 Mbps** at 100 m for every J. At 500 m and 10% the edge grows with J, from **+0.043** at J = 1 to **+0.349** at J = 5. The 25% dynamic campaign agrees and has 0 losses at every J. The dynamic campaigns at the other caps are below.

Most of the 500 m gain is the zenith start when the cap is loose, and the SCA polish when the cap is tight. The split below is a difference of means (one LP at the winning subset, then the further polish). The winner is a zenith subset on the stated fraction of seeds.


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


At 100 m the same split is **+0.010** then **+0.006** at 25% (zenith winner 80/100) and **+0.016** then **+0.051** at 10% (89/100).

At 500 m and T_k = 1.2 s, random feasibility rises as the cap loosens: **11%** at 10%, **33%** at 12%, **47%** at 15%, **59%** at 18%, **67%** at 20%, **73%** at 22%, **78%** at 25%, and **91%** with no cap. The 10% cap is the only cell where a UAV-count point is below 100% feasible: random is **98%** at J = 1 and **99%** at J = 2.

---

## Dynamic-anchor campaigns

`results/run_10mhz_open/anchor-dyn/{nocap,cap10,cap12,cap15,cap18,cap20,cap22}/{100m,500m}/campaign_sca_anchor_dyn.json`. Same protocol as the sweep, both fields, caps other than 25%. Each file has 27 points and starts the AoDT axis at 1.2 s. The sweep's `campaign_sca.json` still has the T_k = 0.8 s point, so the paired comparison uses the 27 shared points. J = 3 matches the dynamic bank in the cap table (same mean, same paired counts). Dynamic assignment is on for 100/100 seeds. Mean wall-clock at J = 3 is **2.7–5.4 s**.

Against the frozen polish stored as the sweep's `campaign_sca_anchor.json`, the largest mean shift is **+0.0017 Mbps** at 100 m and **+0.014 Mbps** at 500 m. Both maxima are at T_k = 1.2 s.

BH-FDR on the one-sided signed-rank test is **23/23** unique points on every cap and both fields. The mean gap is above 0.05 Mbps on **17/23** unique points at 100 m and 10%, and on **0/23** at the other 100 m caps. The largest 100 m point outside that 10% cell is **+0.045 Mbps** (12% cap, J = 2). At 500 m the mean is above 0.05 Mbps on **22/23** unique points with no cap, at 10%, and at 12% (J = 1 is the point underneath: **+0.001**, **+0.043**, **+0.047**), and on **23/23** from 15% through 22%.

500 m, anchor − SCA. Parentheses are seeds above 0.05 Mbps. The J axis has 0 losses. The T_k = 1.2 s column is the largest point at every cap except 10%, where J = 5 is **+0.350**.


| Cap | J = 1 | J = 5 | T_k = 1.2 s | SCA feasible at 1.2 s |
| --- | --- | --- | --- | --- |
| none | +0.001 (1) | +0.055 (37) | +0.392 (63) | 100% |
| 10% | +0.043 (12) | +0.350 (78) | +0.348 (59) | 87% |
| 12% | +0.047 (11) | +0.146 (62) | +0.357 (60) | 92% |
| 15% | +0.058 (11) | +0.153 (60) | +0.412 (68) | 96% |
| 18% | +0.160 (24) | +0.138 (65) | +0.421 (72) | 99% |
| 20% | +0.180 (30) | +0.173 (78) | +0.452 (73) | 100% |
| 22% | +0.206 (31) | +0.113 (70) | +0.426 (71) | 100% |


Dynamic anchor is feasible on **99/100** seeds at 500 m, 10%, T_k = 1.2 s, and on **100/100** at the other cells in that column. The stored rate is lower than SCA on 2 seeds at 10% and on 1 seed at 12, 15, and 18%. Each of those seeds is infeasible for SCA. On seeds where both methods are feasible, the loss count is 0 at every point.

---

## Frozen assignment

`results/run_10mhz_cap25/{100m,500m}/n100_frozen_sca.json` is headline SCA with dynamic assignment off. The stored SCA column has it on. Both are 100% feasible at the 25% default point, with max AoDT at 2.800 s.


| Field | Dynamic SCA | Frozen SCA | Dynamic − frozen |
| --- | --- | --- | --- |
| 100 m | 10.172 | 10.168 | **+0.004 ± 0.020**, 58/100, 4 losses, 38 ties, 1 practical, p = 1.5×10⁻¹² |
| 500 m | 9.418 | 9.394 | **+0.024 ± 0.067**, 53/100, 1 loss, 46 ties, 19 practical, p = 2.8×10⁻¹² |


Association changed on **43/100** seeds at 100 m and **37/100** at 500 m while the flag was on, and on **0/100** with it off. Accepted assignment updates: **71** and **59**. Mean wall-clock is **1.07 s** versus **0.88 s** at 100 m, and **0.96 s** versus **1.03 s** at 500 m. The 500 m zenith gap (**+0.266**) is about ten times this assignment gap.

---

## TD3 on the other caps

Algorithm 2, same n100 banks, every cap, both fields. These are banks, not campaigns. Every cell is 100% feasible. Max AoDT is **2.800 s** except at the 10% cap, where equal share leaves it slack: **0.994 s** at 100 m and **1.135 s** at 500 m. The 25% row is the headline bank.

### 100 × 100 m


| Cap | TD3 | SCA − TD3 | TD3 − k-means |
| --- | --- | --- | --- |
| none | 10.189 | **+0.008 ± 0.012**, 98/100, 2 losses, 1 practical | **+0.018 ± 0.022**, 72/100, 28 losses, 10 practical |
| 10% | 9.875 | **+0.004 ± 0.020**, 86/100, 14 losses, 2 practical | **+0.001 ± 0.012**, 26/100, 74 losses, 1 practical |
| 12% | 9.986 | **+0.039 ± 0.041**, 98/100, 2 losses, 28 practical | **+0.013 ± 0.014**, 96/100, 4 losses, 3 practical |
| 15% | 10.052 | **+0.058 ± 0.048**, 100/100, 0 losses, 48 practical | **+0.015 ± 0.017**, 83/100, 17 losses, 6 practical |
| 18% | 10.085 | **+0.056 ± 0.044**, 99/100, 1 loss, 46 practical | **+0.015 ± 0.016**, 94/100, 6 losses, 3 practical |
| 20% | 10.104 | **+0.048 ± 0.037**, 96/100, 4 losses, 37 practical | **+0.018 ± 0.016**, 96/100, 4 losses, 6 practical |
| 22% | 10.115 | **+0.047 ± 0.034**, 98/100, 2 losses, 38 practical | **+0.019 ± 0.016**, 99/100, 1 loss, 8 practical |
| 25% | 10.130 | **+0.043 ± 0.031**, 97/100, 3 losses, 33 practical | **+0.019 ± 0.017**, 96/100, 4 losses, 6 practical |


### 500 × 500 m


| Cap | TD3 | SCA − TD3 | TD3 − k-means |
| --- | --- | --- | --- |
| none | 9.483 | **+0.346 ± 0.418**, 100/100, 0 losses, 63 practical | **+0.110 ± 0.111**, 75/100, 25 losses, 58 practical |
| 10% | 6.651 | **+0.441 ± 0.385**, 100/100, 0 losses, 90 practical | **+0.050 ± 0.033**, 100/100, 0 losses, 45 practical |
| 12% | 7.222 | **+0.723 ± 0.432**, 100/100, 0 losses, 97 practical | **+0.051 ± 0.042**, 98/100, 2 losses, 39 practical |
| 15% | 7.740 | **+0.900 ± 0.489**, 99/100, 1 loss, 98 practical | **+0.062 ± 0.051**, 97/100, 3 losses, 49 practical |
| 18% | 8.088 | **+0.918 ± 0.494**, 100/100, 0 losses, 99 practical | **+0.072 ± 0.053**, 98/100, 2 losses, 60 practical |
| 20% | 8.256 | **+0.893 ± 0.490**, 100/100, 0 losses, 99 practical | **+0.072 ± 0.057**, 97/100, 3 losses, 57 practical |
| 22% | 8.395 | **+0.924 ± 0.504**, 100/100, 0 losses, 99 practical | **+0.074 ± 0.053**, 98/100, 2 losses, 63 practical |
| 25% | 8.559 | **+0.859 ± 0.502**, 100/100, 0 losses, 96 practical | **+0.081 ± 0.061**, 99/100, 1 loss, 64 practical |


SCA leads TD3 on all 16 cells. The 500 m lead is above 0.05 Mbps on every cap. At 100 m it clears 0.05 Mbps from 15% upward and is **1–2** practical seeds at no cap and at 10%. At 100 m and 10%, k-means is ahead of TD3 on **74/100** seeds (two-sided p = 0.002). Dynamic zenith is ahead of TD3 on **100/100** seeds in every cell. Bank wall-clock is **66–210 s** per seed. The 25% banks are **69 s** and **71 s**.

Residual TD3 (`residual-on-sca`, ±10 m on an SCA start, frozen association, inner LP) at 25%:


| Field | Residual | Residual − SCA | Residual − Algorithm 2 | Zenith − residual | Wall |
| --- | --- | --- | --- | --- | --- |
| 100 m | 10.171 | **−0.001 ± 0.013**, 16/53/31, 0 practical, p = 0.010 | **+0.041 ± 0.026**, 100/0/0, 32 practical | **+0.017 ± 0.026**, 91/0/9, 7 practical | 261 s |
| 500 m | 9.396 | **−0.022 ± 0.068**, 7/51/42, 2 practical, p = 5.1×10⁻⁸ | **+0.837 ± 0.478**, 100/0/0, 97 practical | **+0.288 ± 0.307**, 93/0/7, 71 practical | 231 s |


The residual p-values are two-sided. SCA leads the residual on both fields. The residual still leads Algorithm 2, and the zenith polish still leads the residual.

---

## PSO and the subset controls

PSO's particle fitness is equal share. The saved score is the capped LP, so the 10% row is the one cap where the trained objective and the reported objective match. Every PSO cell below is 100% feasible. Wall-clock is **0.09–0.15 s** per seed.


| Cap | 100 m | PSO − SCA, 100 m | 500 m | PSO − SCA, 500 m |
| --- | --- | --- | --- | --- |
| none | 10.174 | **−0.023 ± 0.021**, 5/91/4, p = 2.4×10⁻²² | 9.639 | **−0.190 ± 0.181**, 10/89/1, p = 3.2×10⁻¹⁹ |
| 10% | 9.936 | **+0.057 ± 0.096**, 50/50/0, 28 practical, p = 1.0×10⁻⁵ | 7.174 | **+0.082 ± 0.284**, 34/66/0, 31 practical, p = 0.762 |
| 12% | 10.014 | **−0.012 ± 0.061**, 21/79/0, p = 2.4×10⁻⁴ | 7.992 | **+0.046 ± 0.285**, 27/73/0, p = 0.088 |
| 15% | 10.063 | **−0.047 ± 0.039**, 9/91/0, p = 2.3×10⁻¹⁹ | 8.617 | **−0.023 ± 0.284**, 29/71/0, p = 0.009 |
| 18% | 10.089 | **−0.052 ± 0.038**, 3/97/0, p = 4.4×10⁻²⁴ | 8.929 | **−0.077 ± 0.373**, 27/73/0, p = 9.3×10⁻⁴ |
| 20% | 10.101 | **−0.051 ± 0.044**, 6/94/0, p = 4.2×10⁻²¹ | 9.052 | **−0.098 ± 0.352**, 31/69/0, p = 7.4×10⁻⁴ |
| 22% | 10.109 | **−0.053 ± 0.042**, 2/98/0, p = 1.2×10⁻²³ | 9.143 | **−0.176 ± 0.321**, 24/76/0, p = 5.5×10⁻⁸ |
| 25% | 10.120 | **−0.052 ± 0.039**, 3/97/0, p = 1.2×10⁻²³ | 9.241 | **−0.177 ± 0.344**, 26/74/0, p = 2.4×10⁻⁷ |


PSO p-values are two-sided. The 100 m, 10% win is a magnitude effect: 50 seeds each way, and the positive gaps are larger (28 of them above 0.05 Mbps). The 500 m, 10% mean is positive and the signed-rank test is not (p = 0.762). PSO is above the dynamic zenith polish on **3/100** seeds at 100 m and 10%, and on **0/100** seeds on every other cap and field. None of those three clear 0.05 Mbps.

At 500 m, PSO leads Algorithm 2 TD3 at every cap, by **0.155 Mbps** with no cap and by **0.52–0.88 Mbps** from 10% through 25%. At 100 m that lead exists at 10% (**+0.061**, 99/100) and is gone by 20%, where TD3 leads by **0.004 Mbps**.

P-median, a random IoT subset, and 120 continuous candidates are frozen-assignment polishes. They are on disk at no cap, 10%, and 25%, both fields. P-median stays within **0.006 Mbps** of the dynamic zenith polish on every one of those six cells, with 0 practical seeds above it. A random subset is indistinguishable from SCA when the cap is off (p = 0.97 at 100 m, p = 0.38 at 500 m) and at 100 m with a 25% cap (p = 0.11). At 500 m and 25% it is **+0.083 ± 0.223** over SCA (p = 0.058) and **0.184 Mbps** under the dynamic zenith polish, with 0 seeds higher. Continuous candidates land between SCA and the zenith polish whenever a cap binds, and within **0.02 Mbps** of the zenith polish when it does not.

### LP particle fitness

`results/run_10mhz_open/pso-lp/`. The swarm scores each particle with the frozen-q bandwidth LP. The exported layout is still rescored with that same LP. Every cell is 100% feasible. Max AoDT is **2.800 s** except at 10%, where it is **0.993 s** at 100 m and **1.249 s** at 500 m, matching the equal-share bank. Mean wall-clock is **1.31–1.53 s** per seed.

At a 10% cap the LP bank and the equal-share bank are the same per-seed vector on both fields (100/100 ties). Elsewhere the LP fitness is above equal share.


| Cap | 100 m, LP − equal share | 500 m, LP − equal share |
| --- | --- | --- |
| none | **+0.022 ± 0.020**, 88/5/7, 10 practical, p = 1.2×10⁻²³ | **+0.181 ± 0.179**, 91/8/1, 81 practical, p = 2.0×10⁻¹⁹ |
| 10% | **0**, 100/100 ties | **0**, 100/100 ties |
| 12% | **+0.038 ± 0.035**, 95/5/0, 26 practical, p = 1.9×10⁻¹⁸ | **+0.021 ± 0.123**, 72/28/0, 29 practical, p = 3.7×10⁻⁴ |
| 15% | **+0.055 ± 0.037**, 97/3/0, 51 practical, p = 8.5×10⁻²⁸ | **+0.052 ± 0.300**, 72/27/1, 62 practical, p = 2.4×10⁻⁵ |
| 18% | **+0.059 ± 0.041**, 97/2/1, 52 practical, p = 8.0×10⁻²⁸ | **+0.147 ± 0.290**, 83/17/0, 76 practical, p = 6.7×10⁻¹⁰ |
| 20% | **+0.061 ± 0.040**, 99/1/0, 54 practical, p = 3.0×10⁻²⁹ | **+0.180 ± 0.247**, 83/17/0, 77 practical, p = 4.4×10⁻¹² |
| 22% | **+0.058 ± 0.039**, 97/3/0, 51 practical, p = 1.7×10⁻²⁸ | **+0.214 ± 0.228**, 86/14/0, 82 practical, p = 6.6×10⁻¹⁵ |
| 25% | **+0.056 ± 0.037**, 99/1/0, 51 practical, p = 7.9×10⁻³⁰ | **+0.235 ± 0.211**, 88/12/0, 84 practical, p = 5.1×10⁻¹⁸ |


Paired with SCA, the 10% row is the equal-share row above. On the other caps the LP mean is within **0.01 Mbps** of SCA at 100 m except at 12% (**+0.026**), and within **0.08 Mbps** at 500 m. The two-sided test favors LP at 100 m and 12% (p = 0.021, 41/59, 18 practical) and at 100 m and 20% (p = 0.011, 48/52, 8 practical). It favors SCA when the cap is off: **−0.001 ± 0.004** at 100 m (21/71/8, p = 9.9×10⁻⁵) and **−0.009 ± 0.096** at 500 m (25/73/2, 16 practical, p = 0.003). The remaining two-sided p-values are 0.13 to 0.78. LP is above the dynamic zenith polish on at most **5/100** seeds in any cell, and on **0/100** seeds by more than 0.05 Mbps.


| Cap | 100 m | LP − SCA, 100 m | 500 m | LP − SCA, 500 m |
| --- | --- | --- | --- | --- |
| none | 10.196 | **−0.001 ± 0.004**, 21/71/8, p = 9.9×10⁻⁵ | 9.820 | **−0.009 ± 0.096**, 25/73/2, p = 0.003 |
| 10% | 9.936 | **+0.057 ± 0.096**, 50/50/0, 28 practical, p = 1.0×10⁻⁵ | 7.174 | **+0.082 ± 0.284**, 34/66/0, p = 0.762 |
| 12% | 10.051 | **+0.026 ± 0.061**, 41/59/0, 18 practical, p = 0.021 | 8.012 | **+0.067 ± 0.292**, 33/67/0, p = 0.56 |
| 15% | 10.118 | **+0.007 ± 0.028**, 33/67/0, p = 0.59 | 8.669 | **+0.029 ± 0.264**, 35/65/0, p = 0.39 |
| 18% | 10.148 | **+0.007 ± 0.018**, 42/58/0, p = 0.13 | 9.076 | **+0.070 ± 0.290**, 39/61/0, p = 0.52 |
| 20% | 10.162 | **+0.010 ± 0.026**, 48/52/0, 8 practical, p = 0.011 | 9.231 | **+0.082 ± 0.295**, 41/59/0, p = 0.25 |
| 22% | 10.167 | **+0.005 ± 0.021**, 41/59/0, p = 0.36 | 9.358 | **+0.038 ± 0.280**, 36/64/0, p = 0.56 |
| 25% | 10.176 | **+0.004 ± 0.021**, 35/65/0, p = 0.43 | 9.476 | **+0.058 ± 0.274**, 35/65/0, 30 practical, p = 0.78 |


The 25% LP campaign matches this bank at J = 3. The LP mean is above equal share on all 23 unique points, both fields. Along J at 100 m the SCA gap runs from **+0.018** at J = 1 (30/65/5, 15 practical) to **−0.002** at J = 5. At 500 m it is **+0.151** at J = 2 (51/49, 43 practical) and **−0.023** at J = 5 (27/73, 20 practical).

---

## Degrees in Eq. (4)

`results/run_10mhz_open/los-deg/cap25/`. The 25% banks, both fields, with `los_angle_unit=deg`: elevation is converted to degrees before Eq. (4), and Table II keeps a = 9.61, b = 0.16. Random, k-means, SCA, multi-start, and dynamic zenith, 100 seeds, paired with the radian headline files. Every cell is 100% feasible and max AoDT is 2.800 s.

The degree reading is a different radio. The 100 m means sit near **65 Mbps**, above the radian zenith ceiling of **10.213 Mbps**. The method order is the radian order.


| Field | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 100 m | 64.721 | 65.001 | 65.122 | 65.140 | 65.143 |
| 500 m | 54.601 | 61.495 | 63.486 | 64.287 | 64.562 |


Anchor − SCA is **+0.021 ± 0.035** at 100 m (86/100, 0 losses, 14 ties, 12 practical, p = 1.0×10⁻²⁸) and **+1.076 ± 1.494** at 500 m (83/100, 0 losses, 17 ties, 74 practical, p = 8.1×10⁻²⁸). Under radians those gaps are **+0.015** and **+0.266**. SCA − random is **+0.401 ± 0.235** at 100 m (100/100, 96 practical) and **+8.886 ± 5.369** at 500 m (98/100, 2 losses, 98 practical). The headline channel remains radians.

---

## CMA-ES at 10 MHz

`results/run_10mhz_open/cmaes/cap25/{100m,500m}/cmaes_polish.json`. Seeds **1–20**, 400 evaluations, σ₀ = 5 m, 25% cap, radian channel. The residual is scored with the frozen-q LP on the SCA association, then Algorithm 1 polishes that layout. The comparison is frozen SCA on the same 20 seeds, so the SCA column is the first 20 seeds of the bank rather than the n = 100 mean. p-values are two-sided. A tie is 1 bit/s.


| Field | SCA | CMA-ES | Polish | CMA − SCA | Polish − SCA |
| --- | --- | --- | --- | --- | --- |
| 100 m | 10.169 | 10.179 | 10.180 | **+0.010 ± 0.021**, 9/11/0, 1 practical, p = 0.20 | **+0.012 ± 0.021**, 10/6/4, 1 practical, p = 0.053 |
| 500 m | 9.486 | 9.486 | 9.486 | **+0.0004 ± 0.0019**, 3/12/5, 0 practical, p = 0.009 | **+0.0004 ± 0.0019**, 4/6/10, 0 practical, p = 0.57 |


The largest CMA gain is **+0.071 Mbps** at 100 m and **+0.008 Mbps** at 500 m. One seed at 100 m clears 0.05 Mbps. None do at 500 m.

---

## Fig. 11 at 10 MHz

`results/run_10mhz_cap25/fig11/{100m,500m}/fig11.json`. K-means placement, 25% cap, 20 seeds, J = 1 through 5. Eq. (17) is the Problem (P) age. The simulator column is FCFS max process age. Lower is younger.

At J = 3:


| Field | Score | Uniform-fast | Uniform-slow | Heterogeneous |
| --- | --- | --- | --- | --- |
| 100 m | Eq. (17) | 0.984 s | 1.730 s | 1.872 s |
| 100 m | Simulator | 1.441 s | 3.118 s | 2.142 s |
| 500 m | Eq. (17) | 1.091 s | 1.838 s | 2.007 s |
| 500 m | Simulator | 1.524 s | 3.193 s | 2.210 s |


On every seed, at every J from 1 to 5, on both fields, Eq. (17) ranks heterogeneous older than uniform-slow, and the simulator ranks uniform-slow older than heterogeneous. Fast is youngest on both scores, 20/20 seeds.

---

## Noise model

The three checks use bandwidth-proportional noise, SNR = p·g / (N₀·B), with N₀ set so an equal-share reference matches the fixed-σ² radio. They are 20 seeds. The JSON is under `results/run_10mhz_cap25/mechanism/`.

**Proportional radio, 25% cap.** The zenith subset still leads SCA after the allocator changes: **+0.054 ± 0.079** at 100 m (17/3/0, 9 practical, p = 0.002, max link share 0.104) and **+0.304 ± 0.304** at 500 m (17/3/0, 14 practical, p = 1.0×10⁻⁴, max link share 0.155). The subset chosen under fixed σ², scored under the proportional radio, still leads SCA by **+0.255** at 500 m. P-median is within 0.023 Mbps of that zenith subset, with 0 practical seeds above it.

**Local polish under that radio** (Nelder–Mead, 200 evaluations). At 500 m the start gap is **+0.596 ± 0.396** (18/2/0) and the polished gap is **+0.133 ± 0.196** (17/3/0, 9 practical, p = 8.5×10⁻⁴). Polished p-median is **7.512** against **7.380** for polished k-means. K-means moves **62 m** RMS; p-median moves **25 m**. At 100 m the polished gap is **+0.025 ± 0.069** (11/9/0, p = 0.55).

**Side / H.** Median relative gain of a p-median layout over k-means SCA, nominal noise, after the proportional re-score. The mean is pulled by single seeds (one seed reaches **+27%** at H = 30 m), so the table is the median and the interquartile range.


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


The median rises from side/H near 2 through the headline geometry at side/H = 5, where the interquartile range sits above zero and **14/20** seeds clear 0.05 Mbps (mean **+0.281 Mbps**). It does not keep rising past that. Raising the reference bandwidth by 10×, which lowers N₀, raises the absolute Mbps gap and lowers the relative median. At side/H = 5 the high-noise scale (reference bandwidth ×0.1) has no feasible seed.

---

## Files

| Path | Contents |
| --- | --- |
| `server_results/results/run_10mhz_cap25/100m/campaign_all.csv` | 100 m sweep means |
| `server_results/results/run_10mhz_cap25/500m/campaign_all.csv` | 500 m sweep means |
| `server_results/results/run_10mhz_cap25/{100m,500m}/campaign_<method>.json` | Per-seed campaign used for the paired tests |
| `server_results/results/run_10mhz_cap25/{100m,500m}/n100_all_summary.csv` | Frozen-bank default point |
| `results/run_10mhz_cap25/{100m,500m}/n100_sca_anchor.json` | Dynamic zenith bank. Identical to `n100_sca_anchor_dyn.json` |
| `results/run_10mhz_cap25/500m/campaign_sca_anchor_dyn.json` | Dynamic zenith campaign, 500 m, 27 points from T_k = 1.2 s |
| `results/run_10mhz_cap25/{100m,500m}/n100_frozen_sca.json` | Headline SCA with dynamic assignment off |
| `results/run_10mhz_cap25/{100m,500m}/n100_td3.json` | Algorithm 2 TD3 bank, seeds 1–100. Byte-identical under `results/run_10mhz_td3/cap25/` |
| `results/run_10mhz_td3/{nocap,cap10,cap12,cap15,cap18,cap20,cap22,cap25}/{100m,500m}/n100_td3.json` | Algorithm 2 banks across the cap family |
| `results/run_10mhz_td3/residual-on-sca/cap25/{100m,500m}/n100_td3.json` | Residual TD3 on SCA, 25% cap |
| `results/run_10mhz_td3/cap25/{100m,500m}/campaign_td3.json` | Algorithm 2 TD3 sweep, 28 points, seeds 1–100 |
| `results/run_10mhz_td3/cap25/{100m,500m}/campaign_all.csv` | TD3 sweep means |
| `results/run_10mhz_td3/progress.txt` | TD3 ledger for the 25% sweep, 4/4 `ok` |
| `server_results/results/run_10mhz_cap25/progress.txt` | Job ledger, all `ok` |
| `results/run_10mhz_cap_sweep/{nocap,cap10,cap12,cap15,cap18,cap20,cap22,cap25}/{100m,500m}/n100_all.json` | Cap-family frozen bank, per seed |
| `results/run_10mhz_cap_sweep/{nocap,cap10,cap12,cap15,cap18,cap20,cap22,cap25}/{100m,500m}/campaign_all.csv` | Cap-family sweep means |
| `results/run_10mhz_cap_sweep/{nocap,cap10,cap12,cap15,cap18,cap20,cap22}/{100m,500m}/n100_sca_anchor_dyn.json` | Dynamic zenith bank beside the frozen sweep anchor |
| `results/run_10mhz_cap_sweep/progress.txt` | Cap-sweep ledger, 160/160 `ok` |
| `results/run_10mhz_controls/{nocap,cap10,cap12,cap15,cap18,cap20,cap22,cap25}/{100m,500m}/n100_pso.json` | PSO bank, equal-share fitness. Subset controls are at no cap, 10%, and 25% |
| `results/run_10mhz_open/anchor-dyn/{nocap,cap10,cap12,cap15,cap18,cap20,cap22}/{100m,500m}/campaign_sca_anchor_dyn.json` | Dynamic zenith campaigns, caps other than 25% |
| `results/run_10mhz_open/pso-lp/{nocap,cap10,cap12,cap15,cap18,cap20,cap22,cap25}/{100m,500m}/n100_pso_lp.json` | PSO bank, LP particle fitness |
| `results/run_10mhz_open/pso-lp/cap25/{100m,500m}/campaign_pso_lp.json` | PSO campaign, LP fitness, 25%, 27 points |
| `results/run_10mhz_open/los-deg/cap25/{100m,500m}/n100_<method>_deg.json` | Eq. (4) in degrees, 25% bank |
| `results/run_10mhz_open/cmaes/cap25/{100m,500m}/cmaes_polish.json` | CMA-ES residual plus SCA polish, seeds 1–20 |
| `results/run_10mhz_cap25/fig11/{100m,500m}/fig11.json` | Fig. 11 at 10 MHz, 20 seeds |
| `results/run_10mhz_cap25/mechanism/` | Proportional-noise, polish, and side/H checks, 20 seeds |

Prior audit and the 8.8 MHz campaigns: [`results_old.md`](results_old.md). The observation ledger is [`observations.md`](observations.md).
