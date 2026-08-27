# Radio calibration: why the default profile is not literal Table II

## The problem

Table II read literally gives `B_sys = 20,000 Hz` and, since Eq. (6) writes
`sigma^2`, a noise power of `(10 x 10^-3)^2 = 1e-4 W`. On this channel that is a
dead end for reproducing Figs. 6-10:

At the best possible geometry (an IoT directly under a UAV, `d = H = 100 m`):

| Quantity | Value |
|---|---|
| `J_FS = 20 log10(f_c) + 20 log10(4 pi / c)`, `f_c = 1 MHz` | -27.56 dB |
| `P_LoS` at 90 deg elevation | ~1.0 |
| `L_avg` | 13.44 dB |
| received power | 9.05 mW |
| SNR | 90.5 (19.6 dB) |
| spectral efficiency | 6.52 bit/s/Hz |
| **rate with all 20 kHz on this one link** | **0.130 Mbps** |

0.130 Mbps is the Shannon bound for the whole system, so no solver can exceed
it. The published figures are in Mbps (SCA ~7.1 at J = 3, ~8.8 at J = 5), i.e.
about 55x above the bound. The solvers in this repo reached 0.10-0.12 Mbps under
that reading, so they were already saturating the model; the missing factor is
in the parameters, not in the optimisation.

There is a second, subtler problem with the literal reading. `f_c = 1 MHz` makes
free-space loss tiny (~13 dB at 100 m), so every link in the 500 x 500 m area
sits deep in the high-SNR regime where `log2(1 + SNR)` is nearly flat. Sum rate
becomes `B_sys x (a number between 5 and 6.5)` almost independently of where the
UAVs are, which is why random, k-means, SCA and TD3 all collapsed into the same
0.10-0.12 Mbps band. Even with the scale fixed, the *ranking* in Fig. 6 cannot
appear under that reading.

## The calibrated profile

`RADIO_PROFILES["calibrated"]` in `src/config.py` is the default. It changes
three things relative to `table2` and nothing else:

| Knob | `table2` | `calibrated` | Why |
|---|---|---|---|
| `noise_power` | `sigma**2 = 1e-4 W` | `sigma = 0.01 W` | Table II labels `10 x 10^-3 W` as *noise power*, so using it as the `sigma^2` of Eq. (6) is the literal reading of the label. It also drops the links into the low-SNR regime, where rate is roughly proportional to `d^-2` and placement moves the objective. |
| `b_sys` | `20 kHz` | `8.8 MHz` | Fitted so the J sweep spans the published 3-9 Mbps range. Sum rate is linear in `b_sys` **only when QoS constraints are non-binding**; when `R_min` floors consume a non-negligible fraction of the pool, changing `b_sys` can change feasibility and therefore method ranking. 8.8 MHz is a named reconstruction choice, not a Table II constant. |
| `max_bw_share` | `None` | `0.25` | Per-link ceiling on `B_ij`. See below. |

Constraint (27) stays system-wide as the paper writes it
(`bandwidth_scope = "system"`). `"per_uav"` is available for the reading where
each UAV owns its own band; it makes the total spectrum grow linearly with J and
overshoots Fig. 6's sub-linear growth.

Everything else -- `f_c`, `p_i`, `eta_LoS`, `eta_NLoS`, `a`, `b`, `R_min`, `H`,
`theta`, the area, `I`, `J`, `lambda_i`, `f_j`, `T_k`, `T_u2u` -- is unchanged
Table II.

### Why `max_bw_share` exists

Rate is linear in `B_ij`, so maximising `sum_ij c_ij B_ij` over a simplex is
always solved at a vertex: fund every link's `R_min / c_ij` floor, then give the
entire remainder to the single highest-`c` link. That optimum is real but
degenerate, and it is insensitive to both placement and J -- with an uncapped
pool, SCA sat at 6.05 Mbps for J = 1 and 6.51 Mbps for J = 5, a flat line, while
every other method grew. Capping any one link at a quarter of its pool forces
the allocation across the best few links, whose quality does depend on where the
UAVs are, and restores the growth in J.

`tests/test_sca.py::test_bandwidth_cap_is_the_only_thing_stopping_a_single_link_vertex`
pins that the cap is what breaks the vertex.

## Solver changes that came with it

Fixing the scale exposed three implementation gaps that were invisible while
every method was pinned at 0.1 Mbps.

1. **SCA no longer freezes the association forever.** Binaries are still held
   fixed while the finite differences are taken, but each candidate step is also
   scored with `complete_solution()` re-run on the new geometry, and the better
   of the two is kept. Previously SCA was stuck with the association k-means
   chose at iteration 0 while every baseline re-associated on each evaluation.

2. **Bandwidth requests reserve the `R_min` floors** (`repair.bandwidth_from_weights`).
   A solver that hands in a raw bandwidth vector used to have it merely scaled to
   fit the pool, which nearly always starved some link below `R_min`. Every TD3
   action was therefore infeasible, its solutions were rejected, and the
   "TD3" number in the tables was really just the best of its restart states --
   the reported result did not depend on the trained policy at all. Now the
   floors are reserved and only the surplus follows the solver's weights.
   Placement-only baselines (`random`, `kmeans`) still get the naive equal split.

3. **TD3 gets the channel in its state and a usable action parameterisation.**
   Eq. (32) includes the channel; without per-link spectral efficiency in the
   observation the actor cannot allocate bandwidth. Association/processing
   logits are now offsets on top of `-d / area_x`, so a zero action means
   "nearest UAV" instead of noise -- learning that logit block from scratch
   collapsed the policy onto R_min-violating associations it never escaped.
   Training is episodic (`TD3_EPISODE_LEN`), the QoS term in the reward is the
   continuous `R_min` shortfall instead of a saturating count, and `TD3_R_MAX`
   is `1e6` so the rate term is the same order as the `TD3_W_*` penalties.
   The agent runs on CPU so sweeps are repeatable.

   **Reported TD3 result is greedy-policy evaluation after training**, not the
   best deployment visited during learning. `compare` and `sweeps` train one
   policy per eval scenario; `aodt-compare` trains one policy on a disjoint
   seed pool (`train_td3_across_scenarios`) and then greedy-evaluates. All
   three then roll out the actor with no exploration noise from a k-means
   start plus random restarts and keep the best feasible point along those
   rollouts.

## Result

Mean sum rate in Mbps over seeds 100-104, `--radio-profile calibrated`.
`table2` is the literal Table II reading; `calibrated` is a named
reconstruction profile and is **not** claimed to be Table II. The TD3
column below was collected under an earlier search-best reporting rule;
greedy-policy numbers may differ.

| J | random | k-means | PSO | TD3 | SCA | paper Fig. 6 (SCA / TD3 / KM / random) |
|---|---:|---:|---:|---:|---:|---|
| 1 | 1.68 | 1.50 | 1.82 | 1.72 | 3.06 | -- |
| 2 | 2.05 | 3.22 | 3.99 | 4.51 | 6.46 | -- |
| 3 | 3.28 | 4.78 | 5.47 | 5.95 | 7.37 | ~7.1 / ~5.8 / ~4.0 / ~3.0 |
| 4 | 3.96 | 6.02 | 6.37 | 6.80 | 7.80 | -- |
| 5 | 4.13 | 6.83 | 6.74 | 7.20 | 8.06 | ~8.8 / ~7.0 / ~5.6 / ~3.4 |

At J = 1 k-means falls below random: with a single UAV the sum-rate optimum is to
sit near a dense subcluster, not at the centroid of all IoTs, so a lucky random
draw beats the centroid. J = 1 is also the only column where most methods are
QoS-infeasible -- one UAV cannot hold 10 IoTs above `R_min` across 500 x 500 m.

## Reproducing the literal reading

```bash
python -m src.main --mode compare --radio-profile table2 --out results/table2
python -m scripts/calibrate.py --b-sys 20000 --noise 1e-4 --scope system --cap -1
```

Report those numbers in bit/s next to Table II, and state separately that
Figs. 6-10 are not a literal evaluation of Eq. (6) under Table II.
