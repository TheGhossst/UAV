# Experimental results

**10 MHz, 25% per-link cap, 100 scenarios.** Numbers below are from the server run in `server_results/results/run_10mhz_cap25/`. The earlier ledger (8.8 MHz, TD3, 20 kHz audit, 15% cap) is in [`results_old.md`](results_old.md).


| | |
| --- | --- |
| **Run** | `server_results/results/run_10mhz_cap25/` (all 20 jobs `ok` in `progress.txt`) |
| **Protocol** | `B_sys` = 10 MHz, per-link cap 25% (2.50 MHz/link), seeds 1–100, CVXPY, 30 SCA iterations |
| **Fields** | **100 × 100 m** and **500 × 500 m** |
| **Methods** | random, k-means, SCA, `sca_multistart`, `sca_anchor` |
| **Not in this run** | PSO, TD3, 15% cap, 20 kHz diagnostic |


---

## Conclusion

On the default point (I = 10, J = 3, λ = 2/s, T_k = 2.8 s, f_j = 2×10⁸), every method is 100% feasible on both fields. The ranking is the same on both:

**sca_anchor > sca_multistart > SCA > k-means > random.**

At **100 × 100 m** the radio is close to the zenith ceiling (**10.213 Mbps**). SCA is **10.172 Mbps**, **+0.223 ± 0.122** above random (100/100) and **+0.062 ± 0.037** above k-means (100/100). Zenith-anchor adds **+0.015 ± 0.021** over the separate SCA campaign (79/100 wins, 11 losses, 9/100 above 0.05 Mbps). Multi-start adds **+0.011 ± 0.019** (58/100 wins, 8/100 practical). The SCA−random gap shrinks from **+0.609 Mbps** at J = 1 to **+0.160 Mbps** at J = 5.

At **500 × 500 m** placement still matters. SCA is **9.418 Mbps**, **+2.753 ± 1.117** above random and **+0.940 ± 0.508** above k-means (both 100/100). Zenith-anchor is **9.682 Mbps**, **+0.264 ± 0.298** over SCA (86/100 wins, **65/100** practical). Multi-start is **+0.171 ± 0.274** (54/100 wins, 45/100 practical). Moving from 100 m to 500 m costs SCA **0.754 Mbps** and random **3.285 Mbps**.

Paired tests are Wilcoxon signed-rank on the same seed, from the five per-method campaign JSON files. BH-FDR over the 24 unique sweep points (repeated default ticks excluded) is **24/24** for SCA versus random and versus k-means on both fields. λ and CPU move the default-point score by well under 0.1 Mbps. T_k = 0.8 s is 0% feasible for every method on both fields. The IoT-axis I = 10 row was solved separately from the UAV-axis J = 3 row; use J = 3 as the headline default.

`sca_anchor` and `sca_multistart` keep the best candidate inside their own run (`best_Mbps` ≥ `frozen_Mbps`). The win/loss counts above compare those outputs with the separately solved `campaign_sca.json` column, so a seed can still land below that SCA column.

---

## Default point (I = 10, J = 3)

The n100 bank summaries (`n100_all_summary.csv`) match this UAV-axis row on both fields.

### 100 × 100 m


| Method | Mean Mbps | Std | Feasible |
| --- | --- | --- | --- |
| **sca_anchor** | **10.188** | 0.010 | 100% |
| sca_multistart | 10.184 | 0.013 | 100% |
| **SCA** | **10.172** | 0.026 | 100% |
| k-means | 10.110 | 0.048 | 100% |
| random | 9.950 | 0.119 | 100% |


Spread from anchor to random: **0.238 Mbps**. SCA to random: **0.223 Mbps**.

### 500 × 500 m


| Method | Mean Mbps | Std | Feasible |
| --- | --- | --- | --- |
| **sca_anchor** | **9.682** | 0.192 | 100% |
| sca_multistart | 9.589 | 0.248 | 100% |
| **SCA** | **9.418** | 0.394 | 100% |
| k-means | 8.478 | 0.655 | 100% |
| random | 6.665 | 1.026 | 100% |


Spread from anchor to random: **3.018 Mbps**. SCA to random: **2.753 Mbps**.

### Same-seed gaps at J = 3


| Comparison | 100 m | 500 m |
| --- | --- | --- |
| SCA − random | **+0.223 ± 0.122**, 100/100, p = 1.6×10⁻³⁰ | **+2.753 ± 1.117**, 100/100, p = 1.6×10⁻³⁰ |
| SCA − k-means | **+0.062 ± 0.037**, 100/100, p = 1.6×10⁻³⁰ | **+0.940 ± 0.508**, 100/100, p = 1.6×10⁻³⁰ |
| anchor − SCA | **+0.015 ± 0.021**, 79/100, 11 losses, 10 ties, 9 practical | **+0.264 ± 0.298**, 86/100, 8 losses, 6 ties, 65 practical |
| multi-start − SCA | **+0.011 ± 0.019**, 58/100, 21 losses, 21 ties, 8 practical | **+0.171 ± 0.274**, 54/100, 22 losses, 24 ties, 45 practical |


Practical means a paired gain above **0.05 Mbps**. Anchor versus SCA: p = 9.0×10⁻²² at 100 m and 2.4×10⁻²² at 500 m. Multi-start versus SCA: p = 2.4×10⁻⁹ at 100 m and 1.5×10⁻⁸ at 500 m.

### Field drop at J = 3 (500 m − 100 m)


| Method | 100 m | 500 m | Δ |
| --- | --- | --- | --- |
| sca_anchor | 10.188 | 9.682 | −0.505 |
| sca_multistart | 10.184 | 9.589 | −0.595 |
| SCA | 10.172 | 9.418 | −0.754 |
| k-means | 10.110 | 8.478 | −1.633 |
| random | 9.950 | 6.665 | −3.285 |


---

## Sweeps

Means are over all 100 seeds. A cell is 100% feasible unless the feasibility note says otherwise. Columns are random, k-means, SCA, multi-start, zenith-anchor.

### 100 × 100 m — UAV count J


| J | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 1 | 9.344 | 9.730 | 9.953 | 9.971 | 9.974 |
| 2 | 9.774 | 10.001 | 10.116 | 10.138 | 10.147 |
| 3 | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 |
| 4 | 10.002 | 10.160 | 10.195 | 10.200 | 10.204 |
| 5 | 10.042 | 10.188 | 10.202 | 10.207 | 10.208 |


SCA−random is **+0.609** at J = 1 (98/100) and **+0.160** at J = 5 (100/100). Versus k-means the mean gap falls below 0.05 Mbps at J = 4 and J = 5 and stays FDR-significant (100/100).

### 100 × 100 m — IoT count I

I = 10 on this axis is SCA **10.174**, about **0.001 Mbps** above the UAV-axis J = 3 row. Headline default stays the J = 3 row.


| I | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 10 | 9.951 | 10.111 | 10.174 | 10.185 | 10.188 |
| 15 | 10.003 | 10.118 | 10.170 | 10.177 | 10.182 |
| 20 | 10.024 | 10.119 | 10.164 | 10.168 | 10.173 |
| 25 | 10.024 | 10.116 | 10.154 | 10.157 | 10.162 |
| 30 | 10.018 | 10.111 | 10.145 | 10.148 | 10.152 |


SCA declines as I grows. SCA−random at I = 30 is **+0.127** (100/100).

### 100 × 100 m — arrival rate λ


| λ (/s) | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 9.941 | 10.107 | 10.167 | 10.178 | 10.183 |
| 1.5 | 9.947 | 10.109 | 10.171 | 10.182 | 10.186 |
| 2.0 | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 |
| 2.5 | 9.951 | 10.111 | 10.173 | 10.184 | 10.188 |
| 3.0 | 9.951 | 10.111 | 10.174 | 10.185 | 10.189 |
| 3.5 | 9.952 | 10.111 | 10.174 | 10.185 | 10.189 |


SCA moves by **0.008 Mbps** from λ = 1.0 to λ = 3.5.

### 100 × 100 m — AoDT threshold T_k


| T_k (s) | Random | K-means | SCA | Multi-start | Anchor | Feasible |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 9.416 | 9.874 | 10.118 | 10.145 | 10.118 | **0%** all methods |
| 1.2 | 9.855 | 10.074 | 10.118 | 10.132 | 10.140 | 100% |
| 1.6 | 9.914 | 10.096 | 10.150 | 10.162 | 10.167 | 100% |
| 2.0 | 9.934 | 10.104 | 10.161 | 10.174 | 10.178 | 100% |
| 2.4 | 9.943 | 10.108 | 10.169 | 10.180 | 10.184 | 100% |
| 2.8 | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 | 100% |
| 3.2 | 9.954 | 10.112 | 10.175 | 10.186 | 10.190 | 100% |


The 0.8 s column is an infeasible score under frozen nearest association. Anchor matches SCA on that row. The feasible score rises from T_k = 1.2 s up to the default plateau.

### 100 × 100 m — UAV CPU f_j


| f_j (×10⁸) | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 9.945 | 10.109 | 10.165 | 10.181 | 10.185 |
| 1.0 | 9.948 | 10.110 | 10.171 | 10.183 | 10.187 |
| 1.5 | 9.949 | 10.110 | 10.172 | 10.183 | 10.187 |
| 2.0 | 9.950 | 10.110 | 10.172 | 10.184 | 10.188 |
| 2.5 | 9.950 | 10.110 | 10.173 | 10.184 | 10.188 |


All five CPU rows are 100% feasible. SCA moves by **0.008 Mbps** from 0.5×10⁸ to 2.5×10⁸.

### 500 × 500 m — UAV count J


| J | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.965 | 5.271 | 6.959 | 7.071 | 7.164 |
| 2 | 5.598 | 7.182 | 8.626 | 8.830 | 8.952 |
| 3 | 6.665 | 8.478 | 9.418 | 9.589 | 9.682 |
| 4 | 7.184 | 9.254 | 9.808 | 9.910 | 10.006 |
| 5 | 7.553 | 9.735 | 9.987 | 10.065 | 10.101 |


Every J is 100% feasible. Anchor leads at every J. The anchor−SCA gap is **0.205** at J = 1 and **0.113** at J = 5. SCA−random is **2.995** at J = 1 and **2.434** at J = 5.

### 500 × 500 m — IoT count I

I = 10 on this axis is SCA **9.426**, **0.008 Mbps** above the UAV-axis J = 3 row.


| I | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 10 | 6.679 | 8.483 | 9.426 | 9.601 | 9.696 |
| 15 | 6.972 | 8.521 | 9.333 | 9.460 | 9.543 |
| 20 | 6.995 | 8.479 | 9.171 | 9.248 | 9.318 |
| 25 | 6.797 | 8.389 | 8.979 | 9.029 | 9.084 |
| 30 | 6.470 | 8.246 | 8.721 | 8.801 | 8.848 |


SCA falls by **0.705 Mbps** from I = 10 to I = 30. Anchor stays above SCA at every I. SCA versus random and versus k-means stays above 0.05 Mbps on every unique point of this field (24/24).

### 500 × 500 m — arrival rate λ


| λ (/s) | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 6.552 | 8.420 | 9.308 | 9.490 | 9.579 |
| 1.5 | 6.634 | 8.462 | 9.391 | 9.564 | 9.654 |
| 2.0 | 6.665 | 8.478 | 9.418 | 9.589 | 9.682 |
| 2.5 | 6.681 | 8.486 | 9.433 | 9.605 | 9.698 |
| 3.0 | 6.690 | 8.491 | 9.442 | 9.614 | 9.707 |
| 3.5 | 6.697 | 8.495 | 9.445 | 9.621 | 9.714 |


SCA moves by **0.137 Mbps** from λ = 1.0 to λ = 3.5, larger than the 100 m λ sweep and still small next to the placement gaps.

### 500 × 500 m — AoDT threshold T_k


| T_k (s) | Random | K-means | SCA | Multi-start | Anchor | Feasible |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 4.598 | 6.600 | 8.620 | 9.014 | 8.620 | **0%** all methods |
| 1.2 | 5.072 | 7.779 | 8.332 | 8.570 | 8.726 | random **78%**; others 100% |
| 1.6 | 6.099 | 8.227 | 8.988 | 9.186 | 9.267 | 100% |
| 2.0 | 6.435 | 8.364 | 9.229 | 9.398 | 9.483 | 100% |
| 2.4 | 6.581 | 8.434 | 9.342 | 9.513 | 9.605 | 100% |
| 2.8 | 6.665 | 8.478 | 9.418 | 9.589 | 9.682 | 100% |
| 3.2 | 6.720 | 8.507 | 9.468 | 9.645 | 9.736 | 100% |


Anchor matches the infeasible SCA score at T_k = 0.8 s. From T_k = 1.2 s upward, anchor is the highest feasible mean.

### 500 × 500 m — UAV CPU f_j


| f_j (×10⁸) | Random | K-means | SCA | Multi-start | Anchor |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 6.606 | 8.445 | 9.355 | 9.533 | 9.627 |
| 1.0 | 6.649 | 8.469 | 9.405 | 9.578 | 9.667 |
| 1.5 | 6.660 | 8.475 | 9.413 | 9.584 | 9.677 |
| 2.0 | 6.665 | 8.478 | 9.418 | 9.589 | 9.682 |
| 2.5 | 6.668 | 8.479 | 9.421 | 9.592 | 9.685 |


All five rows are 100% feasible. SCA moves by **0.066 Mbps** across this CPU grid.

---

## Files

| Path | Contents |
| --- | --- |
| `server_results/results/run_10mhz_cap25/100m/campaign_all.csv` | 100 m sweep means |
| `server_results/results/run_10mhz_cap25/500m/campaign_all.csv` | 500 m sweep means |
| `server_results/results/run_10mhz_cap25/{100m,500m}/campaign_<method>.json` | Per-seed campaign used for the paired tests |
| `server_results/results/run_10mhz_cap25/{100m,500m}/n100_all_summary.csv` | Frozen-bank default point |
| `server_results/results/run_10mhz_cap25/progress.txt` | Job ledger, all `ok` |

Prior audit, 8.8 MHz campaigns, PSO, TD3, and the 15% sensitivity: [`results_old.md`](results_old.md).
