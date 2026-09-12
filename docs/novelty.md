# Zenith-anchor SCA — method note

**Status:** remaining headline probes measured 2026-09-12 (I-axis 500 m, 15% 500 m, \(T_k=0.8\)). Opt-in (`method="sca_anchor"` / `uavdt.sca_anchor`). Khalaf’s Algorithm 1 remains the paper’s solver and the default campaign method. This wrapper is the proposed ranking method at 500 m.

This note is the standalone description of the method: physics, algorithm, knobs, measured stats, and what may honestly be called novelty. Campaign tables and paired readout live in [`RESULTS.md` §2.15](RESULTS.md#215-zenith-anchor-sca-cap-aware-subset-placement) and `results/sca_anchor_cases_analysis.txt`.

---

## 1. The claim, in one paragraph

Every real uplink has a per-device bandwidth limit (carrier bandwidth, channelisation, device RF). A sum-rate objective with **no** per-link cap is the unphysical case: leftover-dump constraint (27) then puts ~95% of `B_sys` on one link, which is the degeneracy [`RESULTS.md` §2.7](RESULTS.md#27-primary-campaign-88-mhz-25-cap) already documents.

Under leftover-dump (27) plus **any** per-link cap fraction \(c\), the frozen-q bandwidth LP dumps leftover Hertz onto about \(k=\lceil 1/c\rceil\) highest spectral-efficiency (SE) links. SE is maximal when a UAV sits at zenith over an IoT. So the placement problem is combinatorial: **which J IoTs the UAVs hover over**, subject to the 10 m UAV separation and QoS floors for everyone else. The top-\(k\) structure is a family, not a 25%-only trick. The paper’s uncapped model is the degenerate \(k=1\) case.

K-means puts UAVs *between* IoTs. Algorithm 1’s first-order Taylor step only pulls toward links that already hold bandwidth. Unstructured multi-start samples extra random / k-means inits and keep-bests. Zenith-anchor SCA instead **enumerates those J-subsets**, scores each with one exact leftover-dump LP, polishes the top-K with unmodified `solve_sca`, and keep-bests against frozen k-means SCA.

It is never worse than one-shot SCA **by construction** (`include_frozen=True`). At 500 m it is a ranking method. At 100 m leftover dump saturates and the same wrapper is a nick.

**Allowed claim.** *Under any per-link bandwidth limit, the capped water-fill makes (P)’s placement a top-\(k\) anchoring problem. K-means is the wrong prior; the structure is zenith-subset selection. The paper’s uncapped model is the degenerate \(k=1\) case.*

**Forbidden claim.** *We solved Khalaf’s Problem (P) better than Algorithm 1 on the paper’s own (uncapped) model.* Hovering over users is also a common UAV placement prior; the load-bearing piece here is coupling that prior to the leftover-dump LP as an exact subset oracle.

---

## 2. Why this radio has that structure

### 2.1 Leftover dump

Constraint (27) is a sum cap on allocated Hertz: \(\sum_{i,j} B_{ij} \le B_{\mathrm{sys}}\). After every associated link has met its QoS floor, leftover spectrum is dumped onto the highest-SE links. That is the same inner LP every method in this repo is scored with (`solve_bandwidth_at_fixed_q` + shared `evaluate()`).

Without a per-link cap the dump is top-1: one zenith link takes almost all leftover Hertz. That is why the no-cap radio collapses to a single-link geometry and why SCA already walks toward zenith from k-means. A per-link cap is the physically standard constraint that *spreads* the dump across \(k=\lceil 1/c\rceil\) links and turns placement into subset selection.

Headline operating point: \(B_{\mathrm{sys}} = 8.8\) MHz, `PRIMARY_MAX_BW_SHARE = 0.25` (2.20 MHz per link, \(k=4\)). Tightening the cap (15%, \(k=7\); 12%, \(k=9\)) binds harder and can invert the SCA vs PSO ranking; 25% is the leftover-dump stress test, not a third of Problem (P). See [`RESULTS.md` §2.7, §2.13, §2.14](RESULTS.md).

### 2.2 SE is maximal at zenith

Path loss and LoS probability are best when horizontal distance is zero (UAV at height \(H\) over the IoT). So the links the leftover-dump LP *wants* to feed are exactly the zenith links. A UAV parked at a k-means centroid — in the gap between several IoTs — is a high-SE link for nobody.

### 2.3 Why Algorithm 1 does not walk there from k-means

SCA linearizes the nonconvex rate around the current \(q\). The first-order step improves links that already hold \(B\). If leftover is sitting on a mediocre geometry, the basin can be 16–72 m from a better zenith layout (Experiment A, [`RESULTS.md` §2.9](RESULTS.md#29-residual-policy-on-sca)). One-shot k-means SCA is a local solver of a nonconvex program from the wrong prior. Multi-start samples more of the same family (random / extra k-means). It finds some distant basins; it does not search the combinatorial set of zenith J-subsets.

### 2.4 Why Experiment C is a different (flat) probe

Experiment C searches association \(a\) at **frozen** SCA \(q\). The LP surface over \(a\) at that \(q\) is essentially flat on default Mbps. The hole is \(q\), not a 1-opt on \(a\). Zenith-anchor searches \(q\) first (subset of IoT xy), then lets nearest \(a\) and the LP follow, then polishes with Algorithm 1.

---

## 3. Algorithm

Implementation: `src/uavdt/sca_anchor.py`. Does **not** edit `src/uavdt/sca/`. Every candidate is scored with the same `evaluate()` as the campaign.

### 3.1 Inputs

| Symbol | Default | Role |
| --- | --- | --- |
| Scenario | I IoTs, J UAVs, field, \(B_{\mathrm{sys}}\), cap | Frozen physics |
| `top_k` | 3 | How many LP-best subsets get a full SCA polish |
| `max_enumerate` | 1000 | Full \(\binom{I}{J}\) enum if at most this many subsets |
| `beam_width` | 10 | Beam width when \(\binom{I}{J}\) exceeds `max_enumerate` |
| `include_frozen` | True | Put one-shot k-means SCA in the keep-best pool |
| `selection` | `enum` | `random` is an ablation control (one random J-subset) |
| SCA settings | 30 iters, \(\varepsilon=10^{-4}\), step 20 m, CVXPY | Unmodified Algorithm 1 |

Default I=10, J=3: \(\binom{10}{3}=120 \le 1000\), so full enumeration. I-axis 500 m sets `max_enumerate=10000`, so \(\binom{32}{3}=4960\) is also full enum (20/20 `enum_mode=full`; mean 4951 LPs / seed after a few jitter-unfittable skips).

### 3.2 Steps (default I=10, J=3)

1. **Enumerate** every J-subset of IoTs. For subset \(S=\{i_1,\ldots,i_J\}\), place UAV \(j\) at zenith over IoT \(i_j\) (same \((x,y)\), \(z=H\)).
2. **Separation repair.** If two host IoTs are closer than \(\theta=10\) m, jitter the later UAV 10 m along the connecting line (opposite direction if the box clips). Skip \(S\) only if the field cannot fit the pair. Headline 25% artifacts used a hard skip; that discarded a non-trivial fraction of 100 m subsets (69/100 n100 layouts skipped at least one; mean 8.7 of 120). At 500 m skip is rare (3/100). New solves jitter.
3. **LP score.** Nearest association \(a\), CPU-stable processing \(b\), one `solve_bandwidth_at_fixed_q`. Drop infeasible LPs. Record `(feasible, sum rate)`.
4. **Rank.** Sort lexicographically by `(feasible, rate)`. Keep the top-K.
5. **Polish.** For each of those K layouts, run unmodified `solve_sca(..., uav_xyz_m=, allocation=)` — Algorithm 1 from that zenith start, not from k-means.
6. **Frozen candidate.** Also run one-shot k-means SCA on the same seed (the headline solver).
7. **Keep-best.** Return the lexicographic winner among {frozen} ∪ {K polishes}. Ties stay with whoever was first; frozen is evaluated first, so exact ties keep frozen.

With `include_frozen=True` the return is never worse than one-shot SCA. That is construction, not an empirical 0-loss on a seed set. `--no-frozen-start` drops the guarantee; campaigns do not use it.

### 3.3 Beam search (large \(\binom{I}{J}\))

When \(\binom{I}{J} >\) `max_enumerate`, grow an ordered subset one IoT at a time:

- At depth \(t < J\), the \(t\) chosen IoTs get zenith UAVs; the remaining \(J-t\) UAVs are padded with k-means and jittered to meet \(\theta\).
- Score that mixed layout with the same LP.
- Keep the best `beam_width` partials and extend.
- At depth \(J\) the layout is pure zenith. Polish still uses only the top-K complete subsets.

This is a heuristic for I=32-class points. On I=10 (`max_enumerate=0`, beam width 10) it matches exhaustive enum to machine precision (n20 100 m and 500 m, max \(|\Delta|=0\); winner identity differs on 1/20 and 3/20 equal-rate ties). The I=32 path is validated on the geometry we can check.

### 3.4 What is *not* in the method

- No change to Algorithm 1’s convex step, association rule, or processing rule.
- No rematch of \(a\) inside a start (same as frozen SCA: nearest at init, then fixed).
- Process-cohesive \(a\) is **off** on every 25% / 15% artifact. The \(T_k=0.8\) s product turns `process_cohesive_candidate=True` and keep-bests a frozen cohesive start plus cohesive \(a\) on every zenith set (§5.9). Do not cite 25% artifacts as feasible at 0.8 s.
- Multi-start’s extra random / k-means inits are **not** in the pool. Anchor can therefore lose a leftover-dump nick to multi-start on a seed (see §5.3). It cannot lose to frozen SCA.

### 3.5 Complexity (default point)

| Piece | Count | Notes |
| --- | --- | --- |
| LP scores | \(\le \binom{I}{J}\) | 120 at I=10, J=3; some skipped only if jitter cannot fit \(\theta\) |
| SCA polishes | `top_k` | default 3 |
| Frozen SCA | 1 | keep-best |
| Wall (measured) | 6.5 s/seed (100 m), 7.3 s/seed (500 m) | ~7× one-shot SCA (~1 s); similar to multi-start’s five SCA solves (5–6 s) |

Algorithm 2 TD3 is ~97 s/seed (~110×) and scores **below** SCA.

---

## 4. What is new vs what is not

### 4.1 Versus this repo (the honest novelty)

| Method | What it searches | Default J=3 100 m | Default J=3 500 m |
| --- | --- | ---: | ---: |
| Random + LP | Unstructured xy | 8.928 | 7.947 |
| K-means + LP | Cluster centroids | 8.901 | 7.474 |
| PSO + LP (external) | Swarm on xy, equal-share inner fitness | 8.905 | 8.122 |
| Frozen SCA (Alg. 1) | Local (q, B) from k-means | **8.946** (headline) | **8.300** (headline) |
| Multi-start SCA | 1 frozen + 2 random + 2 k-means SCA | 8.961 | 8.404 |
| Algorithm 2 TD3 | Residual around k-means, leftover inner B | 8.917 | n100: 7.510 |
| Experiment C | Discrete \(a\) at frozen SCA \(q\) | flat | — |
| **Zenith-anchor SCA** | **J-subsets of IoT xy + LP oracle + SCA polish** | **8.964** | **8.490** |

None of the repo priors enumerates zenith J-subsets. That is the method contribution.

### 4.2 Versus the wider UAV literature

Hovering over users, user-centric UAV placement, and combinatorial UAV–user matching are not new ideas. Do not write “we invented hovering.” Write: leftover-dump plus a per-link cap *makes* the optimum a zenith J-subset, and an exact LP is a cheap oracle for that subset, after which Algorithm 1 is only a local polish.

### 4.3 Versus Khalaf (IEEE TNSM 2026)

Khalaf’s (P) has (27) as a sum cap and Algorithm 1 from k-means. It does **not** have `max_bw_share`. That is the unphysical radio: leftover dump is top-1, SCA already walks to zenith, and subset search should collapse. The per-link cap is not an artefact this reproduction apologises for. It is the physically standard constraint whose absence makes the paper’s model degenerate.

The method is therefore a family over \(c\). 25% (\(k=4\)) is the measured leftover-dump stress test. 15% / 12% / no-cap are the cap-family ablation in §5.7.

---

## 5. Measured results

Operating point unless noted: **8.8 MHz, 25% per-link cap, I=10, \(T_k=2.8\) s, \(\lambda=2\)/s**. Score: Python `evaluate()`. Practical bar: **0.05 Mbps**. Tie tolerance: **1 bit/s** (`SCASettings.improvement_tolerance`), applied in `scripts/analyze_sca_anchor_cases.py`. Wilcoxon `p_greater` is the exact one-sided test in `scripts/paired_winrate.py` and is reported **only** for comparisons that are not keep-best-by-construction (vs multi-start, vs PSO, vs random, vs k-means). Vs SCA the signed vector is \(\ge 0\) on every seed; a one-sided Wilcoxon on it is \(p=2^{-n_{\mathrm{moved}}}\) and is not reported.

All quoted 25% tests are **100% feasible**. Headline campaigns and `n100/eval.json` were **not** overwritten.

Driver: `python scripts/run_sca_anchor_cases.py` (~48 min). Ablations: `python scripts/run_sca_anchor_ablations.py`. Readout: `python scripts/analyze_sca_anchor_cases.py`.

### 5.1 Four default-point tests (J=3)

Means in Mbps.

| Test | Anchor | Multi-start | SCA | PSO | Random | K-means | LP-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n20 100 × 100 m | **8.964** | 8.961 | 8.946 | 8.905 | 8.928 | 8.901 | 8.960 |
| n20 500 × 500 m | **8.490** | 8.404 | 8.300 | 8.122 | 7.947 | 7.474 | 8.443 |
| n100 100 × 100 m | **8.963** | 8.959 | 8.946 | 8.905 | 8.932 | 8.896 | 8.959 |
| n100 500 × 500 m | **8.481** | 8.393 | 8.228 | 8.088 | 7.991 | 7.438 | 8.430 |

n20 100 m / 500 m match the 20-seed headline campaigns’ SCA to 0.000 Mbps (8.946 / 8.300). n100 banks are a different 100-layout draw; do not mix them with the 20-seed tables as if they were the same sample.

### 5.2 Paired deltas (anchor minus other)

Vs SCA: mean \(\Delta\), distribution, practical count. No p-value. “Moved” is \(|\Delta| > 1\) bit/s.

| Test | vs SCA | vs multi-start | vs PSO | vs random |
| --- | --- | --- | --- | --- |
| n20 100 m | **+0.017 ± 0.021**, median +0.008, p10–p90 [+0.000, +0.047], 18/20 moved, 2/20 prac. | +0.003 ± 0.005, 18/20 moved, 0 prac., p=3.8e-6 | +0.059, 20/20, 12/20 prac., p=9.5e-7 | +0.036, 20/20, 4/20 prac., p=9.5e-7 |
| n20 500 m | **+0.190 ± 0.191**, median +0.127, p10–p90 [+0.010, +0.479], 18/20 moved, **14/20 prac.** | +0.087 ± 0.130, 17/20 moved, 0 losses, 9/20 prac., p=7.6e-6 | +0.369, 20/20, **20/20 prac.**, p=9.5e-7 | +0.544, 20/20, 20/20 prac., p=9.5e-7 |
| n100 100 m | **+0.017 ± 0.030**, median +0.010, p10–p90 [+0.000, +0.046], 86/100 moved, 8/100 prac. | +0.004 ± 0.007, 70/100 moved, 15 losses, **0/100 prac.**, p=1.2e-16 | +0.058, 100/100, 51/100 prac., p=7.9e-31 | +0.032, 100/100, 16/100 prac., p=7.9e-31 |
| n100 500 m | **+0.253 ± 0.265**, median +0.169, p10–p90 [+0.000, +0.600], 90/100 moved, **71/100 prac.** | +0.088 ± 0.125, 75/100 moved, 5 losses, **45/100 prac.**, p=4.0e-18 | +0.393, 100/100, **100/100 prac.**, p=7.9e-31 | +0.490, 100/100, 98/100 prac., p=7.9e-31 |

Read the 500 m column as the ranking story. The headline practical count is **71/100 at 500 m**, not the keep-best move count. Read the 100 m column as leftover-dump saturation: everyone is already near the radio ceiling (~8.97 Mbps no-cap), so even a better prior can only nick.

### 5.3 Winner kinds and ties

| Test | Winner = anchor | Winner = frozen SCA |
| --- | ---: | ---: |
| n20 100 m | 18 | 2 (seeds 5, 16) |
| n20 500 m | **20** | 0 |
| n100 100 m | 87 | 13 |
| n100 500 m | 92 | 8 |

Deltas inside the 1 bit/s tie band are ties, not wins or losses. That retires the n20 500 m vs multi-start “loss” of \(-1.5\times 10^{-8}\) Mbps: it is a tie. n20 500 m vs multi-start is 17/20 moved, 3 ties, 0 losses.

n100 100 m vs multi-start has 15 losses, **none** practical. Keep-best is only against frozen SCA. An extra k-means start can still steal a leftover-dump nick that the top-3 zenith polish missed. That is expected and not a construction failure.

### 5.4 LP-only vs polish

The combinatorial prior does most of the work. SCA polish is a local cleanup.

| Test | LP-only mean | Anchor+SCA | SCA | LP − SCA | Polish − LP |
| --- | ---: | ---: | ---: | ---: | ---: |
| n20 100 m | 8.960 | 8.964 | 8.946 | +0.014 | +0.004 |
| n20 500 m | **8.443** | **8.490** | 8.300 | **+0.143** | +0.047 |
| n100 100 m | 8.959 | 8.963 | 8.946 | +0.013 | +0.004 |
| n100 500 m | **8.430** | **8.481** | 8.228 | **+0.202** | +0.051 |

At 500 m, hovering on the right J IoTs and running **one LP** already beats one-shot SCA by ~0.14–0.20 Mbps. The three polishes add another ~0.05. On the n20 500 m re-solves (beam / K=10), the polished winner’s mean UAV displacement is **6.26 m** and the nearest-association set **never** changes (0/20). SCA did not undo the anchors; it walked a few metres toward the 4th dump link.

§5.4 shows the *prior* matters. It does not by itself show that *selection* among IoTs matters — that is the random-anchor control in §5.7.

### 5.5 UAV-count sweep (I=10, 20 seeds)

Wins vs SCA are move-counts under the 1 bit/s tie band, not a significance test.

100 m leftover dump: gap largest at **J=2** (+0.042, 7/20 practical), then shrinks as more UAVs already cover zenith slots.

| J | 100 m anchor | 100 m SCA | Δ vs SCA | moved | prac. | 100 m PSO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.774 | 8.758 | +0.016 | 11/20 | 3 | 8.597 |
| 2 | 8.928 | 8.886 | **+0.042** | 16/20 | 7 | 8.848 |
| 3 | 8.964 | 8.946 | +0.017 | 18/20 | 2 | 8.905 |
| 4 | 8.979 | 8.971 | +0.008 | 20/20 | 0 | 8.942 |
| 5 | 8.982 | 8.977 | +0.005 | 19/20 | 0 | 8.962 |

500 m: the gap is practical at **every** J. Largest at **J=2**.

| J | 500 m anchor | 500 m SCA | Δ vs SCA | moved | prac. | 500 m PSO | 500 m k-means |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.238 | 6.025 | +0.213 | 7/20 | 6 | 6.018 | 4.582 |
| 2 | **7.844** | 7.486 | **+0.358** | 17/20 | **14** | **7.522** | 6.165 |
| 3 | 8.490 | 8.300 | +0.190 | 18/20 | 14 | 8.122 | 7.474 |
| 4 | 8.778 | 8.580 | +0.197 | 18/20 | **16** | 8.413 | 8.110 |
| 5 | 8.871 | 8.765 | +0.106 | 18/20 | 15 | 8.607 | 8.531 |

**Load-bearing geometry: 500 m / J=2.** PSO beats k-means SCA (7.522 vs 7.486). Zenith-anchor **7.844** is +0.358 vs SCA and **+0.322 vs PSO** (20/20, p=9.5e-7). Too few UAVs, too large a field, k-means sits between IoTs, leftover dump still has room. That is the sentence the method is for.

### 5.5.1 IoT-count sweep (J=3, 500 m, 20 seeds, full enum)

Fig. 7 analogue. `max_enumerate=10000`, so every cell is exhaustive \(\binom{I}{3}\), not beam. Absolute rates fall with \(I\) for every method (more QoS floors). The vs-SCA gap stays practical at **every** \(I\); it does not collapse at I=32.

| I | \(\binom{I}{3}\) | Anchor | SCA | PSO | Δ vs SCA | moved | prac. | Δ vs PSO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 120 | **8.490** | 8.300 | 8.122 | **+0.190** | 18/20 | **14** | +0.369 |
| 16 | 560 | **8.239** | 8.065 | 7.892 | +0.173 | 18/20 | 13 | +0.347 |
| 20 | 1140 | **8.054** | 7.885 | 7.745 | +0.169 | 18/20 | 12 | +0.309 |
| 24 | 2024 | **7.849** | 7.664 | 7.500 | +0.185 | 18/20 | **15** | +0.349 |
| 28 | 3276 | **7.611** | 7.484 | 7.332 | +0.127 | 14/20 | 11 | +0.279 |
| 32 | 4960 | **7.351** | 7.201 | 7.106 | +0.150 | 16/20 | 10 | +0.245 |

Winner kind is still mostly the zenith polish (20/20 at I=10; 14/20 at I=28; 16/20 at I=32). Frozen k-means SCA wins more often as \(I\) grows — more IoTs, k-means is likelier to sit near someone — but the mean and the practical count stay with the subset prior. Vs PSO is **20/20** at every \(I\) (p=9.5e-7). Ranking at I=32: **anchor > SCA > PSO > k-means > random**.

### 5.6 Optimality gap and \(K\)

Closed-form leftover-dump upper bound at frozen nearest-\(a\): every IoT gets its QoS floor; leftover Hertz is water-filled onto at most \(k=\lceil 1/c\rceil\) links, each at most \(B_{\mathrm{cap}}\). The J host links are scored at \(\mathrm{SE}_{\mathrm{zenith}}\); remaining dump links use SE to each IoT’s nearest UAV among a host J-subset. The reported bound is the max over subsets, clipped at \(B_{\mathrm{sys}}\cdot\mathrm{SE}_{\mathrm{zenith}}\). Implementation: `leftover_dump_upper_bound` in `uavdt.sca_anchor`. Artifact: `results/sca_anchor_ablations/bound_n100_500m.json`.

| Bank | Bound | Anchor | SCA | Gap (bound − anchor) | Closed of (bound − SCA) |
| --- | ---: | ---: | ---: | ---: | ---: |
| n20 500 m | 8.753 | 8.490 | 8.300 | 0.263 Mbps (**3.0%**) | 34% |
| n100 500 m | 8.732 | 8.481 | 8.228 | 0.251 Mbps (**2.87%**) | 40% per-seed mean |

Radio ceiling is \(B_{\mathrm{sys}}\cdot\mathrm{SE}_{\mathrm{zenith}}=8.988\) Mbps (\(\mathrm{SE}_{\mathrm{zenith}}=1.021\)). The geometry-aware bound is tighter than that ceiling. Anchor+SCA closes about **a third to two-fifths** of the gap between k-means SCA and the bound. One n100 seed sits slightly *above* the bound (polish can raise a remaining link’s SE by leaving zenith). A better continuous heuristic could still take some of the leftover 3%; it is not a 20% hole.

\(K\) is **not** a tuned knob. Reconstructing keep-best(frozen, polish of LP-rank-1) from the stored top-3 polishes:

| Bank | K=1 recon | K=3 | Δ | Identity K>1 | Rate K>1 | Practical K>1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n100 100 m | 8.9631 | 8.9632 | +0.00006 | 35/100 | 26/100 | **0** |
| n100 500 m | 8.4804 | 8.4810 | +0.00056 | 23/100 | 14/100 | **0** |

Winner *identity* often moves (LP-top subsets are nearly tied; median LP gap rank-1 vs rank-2 is ~0). Winner *rate* does not move by a practical amount. K=10 polish on n20 500 m is **8.490**, identical to K=3. \(K\) is not a tuned knob.

### 5.7 Ablations (selection, beam, cap family, degrees-LoS)

These isolate the claims §5.4 cannot. Artifacts under `results/sca_anchor_ablations/`. Driver: `python scripts/run_sca_anchor_ablations.py`.

**Random-anchor control.** Pick 3 random IoTs, one LP, polish that one, keep-best with frozen.

| Test | Random-anchor | Enum | Multi-start | SCA |
| --- | ---: | ---: | ---: | ---: |
| n20 100 m | 8.954 | **8.964** | 8.961 | 8.946 |
| n20 500 m | 8.329 | **8.490** | 8.404 | 8.300 |
| n100 500 m | 8.315 | **8.481** | 8.393 | 8.228 |

At 500 m random-anchor is a nick over SCA (n20 +0.029, 3/20 practical; n100 +0.088) and **loses to multi-start**. Hovering over any IoT is not the method. Enumeration is load-bearing.

**Beam vs exhaustive on I=10.** `max_enumerate=0`, beam width 10. Rate match is exact: n20 100 m 8.963842 = exhaustive; n20 500 m 8.490441 = exhaustive; max \(|\Delta|=0\). Winner kind+combo differs on 1/20 and 3/20 (equal-rate ties). The I=32 beam is a validated heuristic, not an untested stub.

**\(K=10\) polish.** n20 500 m **8.490**, identical to K=3. Combined with the K=1 reconstruction above, \(K \in \{1,3,10\}\) (and therefore all 120) is not a tuned knob.

**Cap family.** Predictions were written down first: no-cap → collapse; 15% → still wins, smaller edge; 12% → the cell where PSO currently beats SCA.

| Cell | Anchor | SCA | PSO | Δ vs SCA | Δ vs PSO |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 m 25% (headline) | 8.964 | 8.946 | 8.905 | +0.017 | +0.059 |
| 100 m 15% | 8.914 | 8.895 | 8.854 | +0.019 | +0.060 |
| 100 m 12% | 8.853 | 8.816 | 8.815 | +0.037 | +0.038 |
| 100 m no-cap | 8.973 | 8.971 | 8.943 | +0.002 | +0.030 |
| **500 m 25% (headline)** | **8.490** | 8.300 | 8.122 | **+0.190** | **+0.369** |
| **500 m 15%** | **7.783** | 7.538 | 7.585 | **+0.244** | **+0.198** |
| **500 m 12%** | **7.171** | 6.900 | 7.020 | **+0.271** | **+0.151** |
| 500 m no-cap | 8.671 | 8.601 | 8.418 | +0.070 | +0.253 |

No-cap 100 m collapses (+0.002 vs SCA, 0/20 practical). No-cap 500 m shrinks to +0.070: SCA does walk toward zenith, but *which* IoT still matters on a large field. At 12% / 500 m, PSO beats SCA (7.020 vs 6.900, 15/20 practical vs SCA for anchor) and zenith-anchor still ranks first (7.171, +0.151 vs PSO, 15/20 practical). At 15% / 500 m the vs-SCA edge is *larger* than at 25% (+0.244, 13/20 practical), not smaller; the vs-PSO edge shrinks as predicted because PSO is a better leftover-dump search when \(k\) is large. The method is a family over \(c\), not a 25%-only trick. n100 500 m confirms both cells: 12% anchor **7.216** / PSO 7.022 / SCA 6.916 (+0.299 / +0.194); 15% anchor **7.816** / PSO 7.556 / SCA 7.542 (+0.274 / +0.260).

Dedicated 15% 500 m artifacts (`sca_anchor_n20_500m_cap15.json`, `n100_500m_cap15/eval_anchor.json`) match those means and add the paired readout:

| Test | Anchor | SCA | PSO | vs SCA | vs PSO |
| --- | ---: | ---: | ---: | --- | --- |
| n20 500 m 15% | **7.783** | 7.538 | **7.585** | +0.244, 17/20 moved, **13/20** prac. | **+0.198**, 20/20, **20/20** prac., p=9.5e-7 |
| n100 500 m 15% | **7.816** | 7.542 | **7.556** | +0.274, 86/100 moved, **72/100** prac. | **+0.259**, 100/100, **97/100** prac., p=7.9e-31 |

PSO beats SCA on the mean at 15% / 500 m. Anchor still ranks first on both samples, and closes every SCA-loss-to-random layout (n20 0/2, n100 0/11). Tighter cap does move the *continuous* optimum off pure zenith: winner polish displacement is **21.1 m** (n20) / **22.4 m** (n100) vs ~6 m at 25%, nearest-\(a\) changes on 5/100 n100 seeds, and K=3 vs K=1 is practical on **5/100** (0/100 at 25%). LP-only is only +0.060 vs SCA on n20 15% (7.598 vs 7.538); the three polishes add **+0.185**. The zenith J-subset remains the right *start*. The leftover-dump bound at frozen zenith \(a\) is slightly *below* the polished rate (−0.16% n20, −0.27% n100) because polish is allowed to leave zenith.

**Degrees-LoS.** Default radio is radians. Under `los_angle_unit="deg"` the LoS/NLoS geometry is a different radio (~57 Mbps). Prediction: zenith anchoring should matter more, not less.

| Test | Anchor | SCA | PSO | Δ vs SCA | prac. vs SCA |
| --- | ---: | ---: | ---: | ---: | ---: |
| n20 500 m / deg | **56.831** | 56.041 | 54.390 | **+0.791** | 14/20 |
| n100 500 m / deg | **56.783** | 55.773 | 54.335 | **+1.010** (median +0.576) | **78/100** |

About 4× the radian +0.190 / +0.253. The largest risk was also the largest upside.

### 5.8 Cost vs the rest of the stack (default J=3)

| Method | Wall / seed | vs SCA mean (100 m) | vs SCA mean (500 m n20) |
| --- | ---: | ---: | ---: |
| Frozen SCA | ~1.0 s | — | — |
| Multi-start SCA | ~5–6 s (~5×) | +0.014 | +0.103 |
| **Zenith-anchor SCA** | **6.5–7.3 s (~7×)** | **+0.017** | **+0.190** |
| Algorithm 2 TD3 | ~97 s (~110×) | **−0.030** | n100 500 m: **−0.718** |

Cheap structured search buys the 500 m ranking. Expensive RL search does not.

### 5.9 \(T_k=0.8\) s with process-cohesive \(a\)

Pass line was: feasible > 0/20 and mean above cohesive SCA-joint **8.393** Mbps. **Passed.** Artifact: `results/sca_anchor_tk08.json`. Flag `process_cohesive_candidate=True` (off on every other artifact). Same 20 campaign seeds, 100 m, 8.8 MHz, 25% cap. Each zenith subset is LP-scored under nearest *and* process-cohesive \(a\) (~224–240 LPs / seed). Keep-best pool: frozen nearest SCA, frozen cohesive SCA, top-K zenith polishes.

| | Feasible | Mean Mbps | Winner |
| --- | ---: | ---: | --- |
| Frozen SCA (nearest \(a\)) | **0/20** | — (AoDT infeasible) | — |
| SCA-joint + process-cohesive rematch | **20/20** | **8.393** | cohesive \(a\), k-means \(q\) |
| **Zenith-anchor + process-cohesive** | **20/20** | **8.430** | 11 zenith / 9 frozen-cohesive |

Paired vs cohesive SCA-joint: **+0.037 ± 0.061**, median 0, 10/20 moved, 0 losses, 10 ties, **6/20 practical**, p_greater=9.8e-4. All 20 winners use `process_cohesive` association. The 9 frozen-cohesive winners sit on the SCA-joint number (mean 8.392, 0/9 practical). The extra 0.037 is the 11 zenith starts (mean 8.461, +0.063 vs SCA-joint, 6/11 practical). Cohesive \(a\) recovers feasibility; zenith \(q\) recovers leftover-dump rate on half the seeds.

Wall **9.1 s/seed** (~2 s above the 25% default, from doubling the LP oracle). This is a separate product: do not cite 25% artifacts as 0.8 s feasible.

---

## 6. How to talk about it

### Copy-ready (honest)

> Under leftover-dump (27) plus any per-link bandwidth cap, the frozen-q LP dumps leftover Hertz onto the top-\(\lceil 1/c\rceil\) SE links, and SE is maximal at zenith. That makes (P)’s placement a zenith-subset problem; the paper’s uncapped model is the degenerate top-1 case. Enumerating those J-subsets, scoring each with one leftover-dump LP, and polishing the top-3 with unmodified Algorithm 1 is never worse than one-shot SCA by construction. At 500 × 500 m / J=3 it scores 8.490 Mbps vs SCA 8.300 and multi-start 8.404 (+0.190 / +0.087; 14/20 practical vs SCA; median Δ vs SCA +0.127, p10–p90 [+0.010, +0.479]). The 100-layout bank is +0.253 Mbps vs SCA (71/100 practical; median +0.169). That closes 40% of the gap from k-means SCA to a leftover-dump upper bound (bound 8.732, gap 2.9%). The same ranking holds on the I-axis through I=32 (full enum, +0.150 at I=32) and at 15% cap (n100 7.816 vs SCA 7.542 / PSO 7.556). With a process-cohesive candidate it is 20/20 feasible at \(T_k=0.8\) s and 8.430 vs cohesive SCA-joint 8.393. At 100 × 100 m leftover dump compresses the same wrapper to +0.017 Mbps. K-means is the wrong prior. This is not a claim that Algorithm 1 is wrong on Khalaf’s uncapped (P).

### Do not write

- “We propose SPA-SCA” — that name is retired; do not revive it.
- “We beat Khalaf / Algorithm 1 on Problem (P).”
- “Hovering over users is a new UAV idea.”
- Headline 100 m +0.017 Mbps as the method win. The method win is 500 m.
- Wilcoxon p-values vs SCA (or vs-SCA “wins” in the J-sweep as if they were a test). Those are \(2^{-n}\) by construction.
- That the default 25% method is feasible at \(T_k=0.8\) s. That product needs `process_cohesive_candidate=True` (§5.9).

---

## 7. What remains open

| Probe | Status |
| --- | --- |
| Random-anchor / beam / \(K=10\) / bound | **Measured** (§5.6–§5.7) |
| Cap family (15% / 12% / no-cap) | **Measured** on n20 both fields and n100 500 m at 12% / 15% (§5.7). Dedicated 15% 500 m paired: still ranks first vs SCA and vs PSO |
| Degrees-LoS | **Measured** on n20 and n100 500 m (§5.7). Edge grows, not shrinks |
| I-axis (I=16…32, J=3, 500 m) | **Measured** (§5.5.1). Full enum through \(\binom{32}{3}=4960\). Gap stays 0.13–0.19 Mbps, practical at every \(I\) |
| \(T_k=0.8\) s + process-cohesive | **Measured** (§5.9). **20/20** feasible, **8.430** vs cohesive SCA-joint **8.393** |
| λ / CPU / AoDT axes | Only the UAV and IoT axes were merged onto the 500 m campaign |

The remaining-before-headline bar in [`RESULTS.md` §2.15](RESULTS.md#215-zenith-anchor-sca-cap-aware-subset-placement) is cleared. Default `--methods` is still `random,kmeans,pso,sca`. Do not overwrite the protected headline JSON.

---

## 8. Code, CLI, artifacts

| Piece | Path |
| --- | --- |
| Solver | `src/uavdt/sca_anchor.py` |
| Method hook | `src/uavdt/experiments/methods.py` (`KNOWN_METHODS`, not in default `METHODS`) |
| CLI | `python -m uavdt sca-anchor --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25` |
| Four tests + J-sweep | `python scripts/run_sca_anchor_cases.py` |
| Ablations | `python scripts/run_sca_anchor_ablations.py` |
| Paired readout | `python scripts/analyze_sca_anchor_cases.py` |
| Tests | `tests/test_sca_anchor.py` (keep-best, sep jitter, I=32 beam, bound, random, opt-in) |
| n20 JSON | `results/sca_anchor_n20.json`, `sca_anchor_n20_500m.json`, `sca_anchor_n20_500m_cap15.json` |
| n100 JSON | `results/n100/eval_anchor.json`, `results/n100_500m_cap25/eval_anchor.json`, `results/n100_500m_cap15/eval_anchor.json` |
| J-sweep merge | `results/campaign_sca_anchor_uavs.json`, `_500m.json` |
| I-sweep merge | `results/campaign_sca_anchor_iots_500m.json` |
| \(T_k=0.8\) | `results/sca_anchor_tk08.json` |
| Analysis | `results/sca_anchor_cases_analysis.txt` / `.json` |
| Bound / ablations | `results/sca_anchor_ablations/` |
| Figures | `results/figures/n100_anchor/`, `n100_500m_anchor/`, `n100_500m_cap15_anchor/`, `anchor_uavs/fig06_sum_rate.png`, `anchor_uavs_500m/fig06_sum_rate.png`, `anchor_iots_500m/fig07_sum_rate.png` |

Campaign / n100 flags: `--methods ...,sca_anchor`, `--anchor-top-k 3`, `--anchor-max-enumerate 1000`, `--anchor-beam-width 10`.

Protected files this method must never overwrite: `results/campaign_8.8mhz_cap25_si12k.json`, `_500m.json`, `results/n100/eval.json`, `results/n100_500m_cap25/eval.json`.

---

## 9. Related notes in this repo

- Frozen SCA and leftover-dump physics: [`RESULTS.md`](RESULTS.md) §0, §2.7
- Multi-start / Experiment A basins: [`RESULTS.md`](RESULTS.md) §2.9
- TD3 bookend: [`RESULTS.md`](RESULTS.md) §2.10
- n100 banks: [`RESULTS.md`](RESULTS.md) §2.12
- This method’s campaign section: [`RESULTS.md`](RESULTS.md) §2.15
- Freeze ledger: [`EXPERIMENTS.md`](EXPERIMENTS.md)
