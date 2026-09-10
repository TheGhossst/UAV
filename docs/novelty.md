# Novelty — beat SCA on Problem (P)

Last updated: 2026-09-10  
Status: **Experiment C measured** (flat at frozen SCA \(q\)). Residual-on-SCA is closed. TD3 Algorithm 2 **measured** (below SCA; does not overtake at large \(I\)).

**Goals (in order):** (1) **beat frozen SCA** on the same `evaluate()` (2) **novelty** that is not another \(q\)-nudge.

Those two fight if “beat SCA” means *higher Mbps at default \(T_k=2.8\) s*. Grouping processes **costs** ~0.6 Mbps. Residual-on-\(q\) **gained** +0.003 Mbps. So the only remaining place to beat SCA on **rate** is the integers \(a_{ij}\) that Algorithm 1 never searches — including when grouping is *not* required.

---

## Goal check — two ways to “beat SCA”

| Meaning | Same \(T_k\), same seeds, `evaluate()` | Does v0 grouping do it? | What actually can |
| --- | --- | --- | --- |
| **A. Higher sum rate** (what a guide usually wants) | Yes, both methods feasible | **No.** At 2.8 s v0 **is** SCA. At 0.8 s grouped rate is **lower** (~8.39 vs infeasible ~8.91) | Search **non-nearest** \(a\), then SCA. Experiment A already: extra inits that **change \(a\)** beat frozen SCA on **13/20** seeds, mean \(+0.013\), two seeds \(>0.05\) |
| **B. Better (P) point** (feasible, then rate) | Feasibility first | **Yes** at \(T_k=0.8\): SCA **0/20**, cohesive **20/20** | Slack-aware grouping. Honest, but it is not a default-Mbps win |

**This note’s proposed method must do A where (P) is already feasible, and B where nearest-\(a\) makes (P) infeasible.** One discrete search, two regimes. Experiment C is **flat at frozen SCA \(q\)** (mean LP \(\Delta=+0.001\) Mbps). Multi-start’s extra \(a\) (Experiment A, \(+0.06\) on seeds 11/19) is a **joint \((q,a)\) basin**, not an \(a\)-only win. Default-Mbps “beat SCA” is a **multi-start** paper if you want rate; grouping remains the clean win for goal B.

Invert the last draft: “match SCA at 2.8 s” is a **failure mode** for goal A, not a feature. Matching is only required as a **regression test** (never *worse* than SCA when grouping is off). The method must still **search** \(a\) at 2.8 s.

---

## 0. What is already closed

| Direction | Result | Implication |
| --- | --- | --- |
| Frozen SCA on \((q,B)\) at nearest \(a\) | Best default-Mbps solver we have (n100, campaigns) | Do not replace Algorithm 1’s convex loop |
| Residual-on-SCA (TD3, \(\pm 10\) m, inner LP) | \(+0.0032\) Mbps, 4 wins / 16 origin ties | No exploitable slack in \(q\) at frozen SCA \(a,b\) |
| CMA-ES residual (field box, same frozen \(a,b\)) | \(+0.0075\) Mbps | Neural net is not load-bearing; field-wide \(q\) still not a method |
| Multi-start SCA (new inits → new nearest \(a\)) | Mean \(+0.013\) Mbps; two seeds \(>0.05\); basins 16–72 m | Better points exist, but they **change association**, not a 10 m nudge |
| Best-SE SCA-joint | Max \(\Delta +0.006\) Mbps vs frozen SCA | Greedy rematch ≠ grouping |
| Process-cohesive \(a\) at \(T_k=0.8\) s | SCA **0/20** feasible; cohesive **20/20**; ~**0.6 Mbps** median rate tax | The discrete block of (P) **does** move the score |

**Load-bearing hole:** Algorithm 1 (and this repo’s frozen SCA) never updates \(a_{ij}, b_{ij}\). Problem (P) is a MINLP. The paper’s own §I tradeoff (accuracy vs synchronization) lives in that discrete block.

---

## 1. Critique of the one-line proposal

**One-liner (v0):** If forwarding would break AoDT, put each process on one UAV; otherwise keep best-SE; then run frozen SCA + the \(B\) LP.

### Claim (falsifiable)

On the same `evaluate()`, slack-aware grouping then SCA is **feasible at \(T_k=0.8\) s** where frozen SCA is not, matches frozen SCA on Mbps at \(T_k=2.8\) s, and traces a rate–deadline Pareto whose step is the measured ~0.6 Mbps grouping tax.

### Mechanism (one idea)

Eq. (17) AoDT for process \(k\) is

\[
\Delta_k = D_{N_k} + \underbrace{\frac{1}{\lambda_{N_k}}\Big(1+\sum_{i\in N_k}\frac{\lambda_i}{\mu}\Big)}_{Q_k}.
\]

Upload \(D_i\) includes \(T_{\mathrm{u2u}}=0.3\) s if \(a_{ij}\neq b_{ij}\). At frozen \(q\), the bandwidth LP needs a **positive slack**

\[
s_i = T_k - Q_k - \mathbf{1}_{\mathrm{fwd}}\,T_{\mathrm{u2u}}.
\]

At default \(\lambda=2/\mathrm{s}\), \(I=10\), \(K=2\), \(\mu\approx 53.3/\mathrm{s}\):

- \(Q_k \approx 0.594\,\mathrm{s}\)
- \(T_k-Q_k \approx +0.206\,\mathrm{s}\) at \(T_k=0.8\)
- \(T_k-Q_k-T_{\mathrm{u2u}} \approx -0.094\,\mathrm{s}\)

So **any forwarded IoT makes the LP infeasible**, independent of rate. Grouping (\(a=b\) for the process, all of \(N_k\) on one UAV) is necessary. Best-SE / nearest **split** \(N_k\) and fail. That is not a solver bug; it is the constraint.

When \(T_k-Q_k \ge T_{\mathrm{u2u}}\) (paper grid: \(T_k\ge 1.2\,\mathrm{s}\) at default \(\lambda,\mu\)), forwarding is legal and best-SE is the rate-optimal greedy. The method **must** coincide with frozen SCA there.

### What is new vs recombined

| Closest prior | Delta |
| --- | --- |
| Khalaf Alg. 1 SCA | They freeze \(a,b\). We search \(a\) using Eq. (17) slack. |
| Khalaf Alg. 2 TD3 | They learn \((q,a,b,B)\) with penalties. We keep SCA for \((q,B)\) and only fix the integers. |
| Wu–Zeng–Zhang BCD+SCA (TWC 2018) | Generic association + trajectory. We have **process constraint (23)** and a **hard \(T_{\mathrm{u2u}}\) cliff**, not a smooth SINR matching game. |
| This repo’s `sca_joint` + `process_cohesive_candidate` | Same physics, tagged as a **probe flag**. Proposal: make it the **solver**, triggered by slack, not by `T_k==0.8`. |

v0 is a **new combination in this problem**, not a new convex algorithm. A reviewer can call it “an if-statement + centroid heuristic.” That is why §2 exists.

### Assumptions and where it breaks

- Eq. (17) is the score (frozen). If the paper intended Eq. (15) / FCFS, the slack cliff moves.
- \(Q_k\) depends only on \(\lambda,\mu,N_k\), not on \(q\). The **decision to group** is geometry-free; **which UAV** to group onto is not.
- Campaign \(K=2\). Enumeration of process→UAV maps is \(J^K\) (9 at \(J=3\), 25 at \(J=5\)).
- At default \(T_k=2.8\,\mathrm{s}\), **v0–v2 grouping is off**, so those versions **cannot beat SCA on Mbps**. Goal A requires **v5** (search \(a\) even when grouping is optional).

### Evidence already in hand (supports the mechanism, not yet the named method)

- Cohesive construction **40/40** feasible at k-means \(q\) (`tk08_cohesive_construction.json`).
- Best-SE joint **0/20**; cohesive candidate **20/20** (`tk08_scajoint_cohesive.json`).
- Grouping tax median **~0.57 Mbps** (p10–p90 ~0.38–0.81) at fixed \(q\), 40 geometries.
- After grouping, SCA still helps: 8.275 → **8.393 Mbps** (UAVs move; \(a\) stays cohesive).

### Weaknesses of v0 (do not ship v0 as the whole thesis)

1. **On the Fig. 9 grid it only differs at \(T_k=0.8\,\mathrm{s}\).** \(1.2\)–\(3.0\,\mathrm{s}\) are “same as SCA.”
2. **Centroid-nearest host** is a hand construction, not an optimized discrete block (`centroid_cohesive_association`).
3. **All-or-nothing per process** is forced by Eq. (17): slack with/without \(T_{\mathrm{u2u}}\) is **shared** by all members of \(N_k\). There is no “partial cohesion” under the written formula.
4. SCA-joint already ran the cohesive flag. Promoting a flag is incremental unless the **trigger** (slack, not a magic \(T_k\)) and the **host search** (not centroid) are the method.

---

## 2. Make it better (ranked increments)

Each step should change a measured number. If it does not, do not add it.

### v1 — Physics trigger, not `T_k == 0.8` (required)

**Rule, per process \(k\):**

```text
if T_k - Q_k < T_u2u:   # forwarding slack negative even at infinite rate
    process k must be cohesive (no 1_fwd)
else:
    best-SE / nearest is allowed
```

\(Q_k\) from `queueing_term_s` (already in `aodt.py` / `cvx_problem.py`).

**Why this is not cosmetic:** \(Q_k\) grows with \(\sum\lambda\) and shrinks with \(\mu\). On the **paper grids** grouping is required at more than one tick:

| Axis | When \(Q_k+T_{\mathrm{u2u}} > T_k\) (default other params) |
| --- | --- |
| \(T_k=0.8\) | Always (default \(\lambda,\mu\)) |
| \(\lambda\) up (Fig. 8) | \(Q_k\) rises; cliff moves toward \(1.2\,\mathrm{s}\) |
| CPU down (Fig. 10, \(0.5\times 10^8\)) | \(\mu\downarrow\), \(Q_k+T_{\mathrm{u2u}}\approx 1.18\,\mathrm{s}\) → **both 0.8 and 1.2** need grouping |
| Fig. 11 hetero \(\lambda\) | \(\lambda_{\min}\) small → \(Q_k\) large → grouping at looser \(T_k\) for the slow process |

**Falsify:** implement the slack test; if it still only flips on the \(T_k=0.8\) row of the default campaign, the “physics trigger” story is weak at \(I=10,J=3,\lambda=2\). Then Fig. 10 low-CPU and Fig. 11 are the plots that must carry it.

**Surprising:** slack test groups at \(T_k=1.2\) on the default row (then \(Q_k\) in code ≠ 0.594 s; check \(\mu,L\)).

### v2 — Search the host UAV, do not use the centroid (this is the discrete method)

When process \(k\) must group, there are **\(J\) hosts** (or \(J^K\) joint maps). Centroid-nearest is one feasible point. The rate-optimal cohesive map is an LP at frozen \(q\):

1. For each process→UAV assignment that satisfies (23) and (24),
2. Set \(a=b\) (no forwarding),
3. `solve_bandwidth_at_fixed_q` + `evaluate()`,
4. Keep the feasible map with best true sum rate,
5. `solve_sca` with **that** frozen \(a,b\).

At \(K=2,J=3\): 9 maps. Cheap. Campaign \(J=5\): 25 maps.

**Ground truth to beat:** cohesive SCA-joint mean **8.393 Mbps** at \(T_k=0.8\), \(I=10,J=3\). If enumeration + SCA is \(\le 8.393\), centroid was already optimal and v2 is an ablation, not a gain.

**Surprising:** best host is often **not** the centroid UAV (leftover-dump wants the process on the UAV that can sit on a high-SE IoT). That would justify v2 as more than bookkeeping.

### v3 — BCD: discrete map, then SCA (keep; do not re-open residual-on-SCA)

After \(a,b\) change, the \((q,B)\) landscape is **not** the one residual-on-SCA searched (that was frozen nearest \(a\)). One SCA from the grouped init is already a measured +0.12 Mbps vs frozen-\(q\) cohesive.

**Do not** add TD3 residual here. Optional ablation: 2–4 extra SCA inits **with \(a,b\) held at the v2 map**. If that \(\Delta\) is still \(<0.05\) Mbps, stop polishing \(q\).

### v4 — Pareto figure as the paper contribution (cheap, high value)

Same method, sweep \(T_k\) (and optionally low CPU / hetero \(\lambda\)). Plot:

- feasible fraction (SCA vs proposed)
- mean **feasible** sum rate
- grouping tax vs \(T_k\)

That is Fig. 9 done **correctly** (frozen SCA’s 0% at 0.8 s is not a model limit). The novelty slide is the **algorithm that traces the feasible frontier of (P)**.

### v5 — Search \(a\) even when grouping is *not* required (**this is how you beat SCA on rate**)

v1–v4 only change the point when the forwarding cliff is on. Goal A is at **loose** \(T_k\), where that cliff is **off**. Frozen SCA still uses **one** map: nearest at k-means \(q\).

**Rule:**

```text
legal(a) = maps that satisfy (21)–(24) and, for each k with
           T_k - Q_k < T_u2u, process k is cohesive.

score(a) = evaluate() after frozen-q B LP
a*       = local search / sample / (if tiny) enumerate over legal(a)
then      solve_sca with a*, b* frozen
```

Neighborhood (fits `rematch.py`): **1-opt** — move one IoT to another UAV; reject if illegal; accept if LP rate improves; optionally 2-opt. At \(I=10,J=3\) a 1-opt step has 20 neighbours. Best-SE is **one** neighbour; that probe already failed (\(+0.006\)). VNS can reach maps nearest never proposes — the same class Experiment A found by re-init (16–72 m basins, **new** \(a\)).

**Why this is the load-bearing increment for beating SCA:**  
Residual searched \(q\) at frozen nearest \(a\) and lost. A searched **new \(a\)** (via new placement init) and won 13/20. v5 searches \(a\) **directly** instead of hoping a random k-means restart.

**Novelty:** not VNS-in-general. The legal set is **Eq. (17) slack + (23)**. No other UAV BCD paper has a \(T_{\mathrm{u2u}}\) cliff that *shrinks* the integer set at tight \(T_k\) and *opens* it at loose \(T_k\).

**Falsified (2026-09-10):** Experiment C — 1-opt + best-SE + 200 random legal maps at **frozen SCA \(q\)**, default 2.8 s / 25% / \(I=10,J=3\), then SCA-polish. Artifact: `results/residual_on_sca/assoc_oracle_n20.json`. Readout **`flat_at_frozen_q`**.

| Result | Meaning | Measured |
| --- | --- | --- |
| Oracle/VNS \(\Delta > 0.05\) Mbps vs frozen SCA | Goal A is real. v5 *is* the method. | **No.** Max polish \(\Delta=+0.035\) (seed 18). 0/20 practical. |
| \(\Delta \in (0.01, 0.05)\) | Beat SCA statistically, not practically. Still publish paired wins; do not headline Mbps. | Polish mean \(+0.0027\); only seed 18 in this band, and only after \(q\) moved. |
| \(\Delta \approx 0\) at frozen \(q\), but Experiment A still \(+0.013\) | The win is **joint** \((q,a)\) (re-init). Method = multi-start SCA with v5 \(a\) at *each* init, keep best. Novelty is weaker (multi-start is old); still beats one-shot k-means SCA. | **This row.** LP mean \(+0.0010\); A’s seeds 11/13/19 untouched. |
| Both flat | **You cannot beat SCA on default Mbps.** Only goal B (\(T_k=0.8\) feasibility) remains. | A is not flat; C is. Rate win exists, but not as SPA-SCA at the SCA point. |

Skip MAPPO, SAC, named swarms, enlarging residual boxes, soft \(a\in[0,1]\).

---

## 3. Named method (what to call it)

Working name: **Slack-aware process association + SCA (SPA-SCA)**.

Block coordinate descent on Problem (P):

| Block | Solver | Status |
| --- | --- | --- |
| \(a,b\) | v1 slack test + v2 host enumeration + `cpu_stable_processing` | **Proposed** |
| \(q,B\) | Frozen Algorithm 1 SCA + exact \(B\) LP | **Reuse** |

TD3 Algorithm 2 stays a **baseline** (paper fill-in). Frozen SCA is the thing to **beat**, not the thing to copy at loose \(T_k\).

**SPA-SCA = v1 cliff + v5 search on the legal set + one SCA.**  
Without v5 it does not beat SCA on rate. Without v1 it does not beat SCA at \(T_k=0.8\) s (SCA is infeasible; a rate comparison is invalid).

### 3.1 Algorithm (SPA-SCA)

Inputs: scenario (IoT \(xy\), \(N_k\), \(\lambda\), \(T_k\)), seed.  
Output: \(q,a,b,B\) scored by `evaluate()`.

```text
Q_k ← queueing_term_s(λ on N_k, μ)
must_group[k] ← (T_k - Q_k < T_u2u)

legal(a): (21)–(24) and cohesive on every must_group process

# Discrete block — NEVER skip this (skipping = cannot beat SCA on rate)
a0 ← nearest or best-SE, repaired by cpu_stable_processing
if must_group: restrict a0 to cohesive hosts (v2: enumerate J^K)
a* ← 1-opt / VNS on legal maps, score = frozen-q LP + evaluate()
     start at a0; keep incumbent if rate improves and feasible

(q, B) ← solve_sca with a*, b* frozen     # same Algorithm 1, better integers
return evaluate()
```

One discrete search, then one SCA. Not rematch-every-iteration (`sca_joint`). Grouping **restricts** the legal set; it does not replace the search.

### 3.2 vs `sca_joint` (do not clone the probe)

| | `sca_joint` (probe) | SPA-SCA (proposed) |
| --- | --- | --- |
| When it groups | Optional flag, always-on cohesive candidate | Only if \(T_k-Q_k < T_{\mathrm{u2u}}\) |
| Host UAV | Centroid-nearest | Enumerate (v2) or centroid (v0) |
| Loop | Rematch **inside** every SCA iteration | Discrete **once**, then frozen SCA |
| Default \(T_k=2.8\) | Best-SE rematch, \(\Delta\approx 0\) | **v5 1-opt/VNS** on legal \(a\) (this is the rate attack) |
| \(T_k=0.8\) | Cohesive flag 20/20 | Same feasibility; v2 host enum + v5 inside the 9 maps |

Without v5, SPA-SCA at 2.8 s **is** SCA. That fails the main goal.

### 3.3 Sentence for the guide

> Frozen SCA is a strong local solver of UAV positions and bandwidth, but it **freezes** IoT–UAV association at nearest neighbour. Residual search around that point does not raise sum rate (we measured +0.003 Mbps). Searching legal \(a\) at that **same** \(q\) also does not (Experiment C: +0.001 Mbps). The default-Mbps gap in Experiment A is a **different basin** (16–72 m, new \(a\)). Where the deadline forces each process onto one UAV, SCA is infeasible and slack-aware grouping is not. That is the remaining clean claim.

---

## 4. Minimal core

**Experiment C is implemented and flat.** Do not promise the guide a default-Mbps SPA-SCA win at frozen SCA \(q\).

**Core A (falsify goal A, ~20 seeds, frozen SCA \(q\)):**  
`search_legal_association(scenario, q_sca) -> (a, b, rate)`  
1-opt from nearest, legal = (21)–(24) + slack cohesive mask, score = LP + `evaluate()`. Then one `solve_sca` on the winner. Compare to frozen SCA.

**Core B (goal B, already almost done):** slack mask + centroid cohesive + SCA at \(T_k=0.8\). Target: 20/20 feasible vs SCA 0/20.

**Do not edit `uavdt.sca`.** New module; call `solve_sca` with frozen integers set to the search incumbent.

| Check | Target |
| --- | --- |
| C at \(T_k=2.8\) | **Measured flat:** mean LP \(\Delta=+0.0010\) Mbps. Stop claiming a frozen-\(q\) rate win. |
| \(T_k=0.8\) cohesive | 20/20 feasible, ~8.39 Mbps |
| Slack at default \(\lambda,\mu\) | \(Q_k+T_{\mathrm{u2u}}\approx 0.894\,\mathrm{s}\) |

**Out of scope for the core:** RL, MATLAB, n100, full Fig. 6–10, residual-on-SCA.

**Ambiguities:** 1-opt vs random sample of maps; how many 1-opt rounds; SCA polish or LP-only for C (LP-only is the cheaper falsification; polish if LP \(\Delta>0\)).

---

## 5. What is still open

**C is done (goal A at frozen \(q\)).** Rate paper at the SCA point is dead. Remaining options: multi-start SCA with v5 \(a\) at *each* init (weaker novelty), change the instance (15% cap / 500 m / \(I=32\)) and rerun C, or tell the guide the clean win is feasibility.

**Then** low-CPU \(\times T_k=1.2\) (goal B on more than one grid tick): prediction SPA feasible, SCA not, if \(Q_k+T_{\mathrm{u2u}}\approx 1.18\,\mathrm{s}\).

**Then** v2 vs centroid at \(T_k=0.8\). If \(\Delta<0.05\) Mbps, centroid is enough for goal B.

---

## 6. Cross-pollination (what transfers)

- **Matching with externalities:** UAV “quota” is CPU (24), not a user count. Coalition = \(N_k\) must share a host when slack is negative.
- **Lexicographic OR:** first feasibility of (30)–(31), then (20). Frozen SCA does the opposite order (rate at a possibly illegal \(a\)).
- **Combinatorial + convex:** same split as PDD/BCD papers, but the integer here is **tiny** (\(J^K\)) so we can be exact on \(a,b\) instead of relaxing.

---

## 7. Unstated cost

- **Grouping** trades ~0.6 Mbps for a deadline. That **beats SCA on (P)** at \(T_k=0.8\) (feasible vs not). It **loses** on raw Mbps. Do not sell 8.39 vs 8.91 as a rate win.
- **v5 search** costs extra LPs (\(O(\text{rounds}\times I J)\)). Cheap at \(I=10\). The cost that matters is **honesty**: leftover-dump 25% / 100 m may still cap practical \(\Delta\) at \(0.02\)–\(0.06\) Mbps even if you win 15/20 seeds (Experiment A’s scale).
- If C is flat, **do not** stretch novelty with SAC/GNN. Change the **instance** (15% cap, 500 m, \(I=32\)) where leftover dump is less of a vertex, and rerun C. Or tell the guide the only clean win is feasibility.

---

## 8. Research log

| Date | Tried | Found | Next |
| --- | --- | --- | --- |
| 2026-09-10 | Residual-on-SCA + CMA-ES | \(q\) at frozen \(a\) is flat | Stop residual as proposed method |
| 2026-09-10 | This note v0 | If-statement + centroid is too thin | v1 slack trigger + v2 host enum |
| 2026-09-10 | This note v1–v3 | Grouping cliff is \(Q_k+T_{\mathrm{u2u}}\) | Necessary for goal B; **insufficient for goal A** |
| 2026-09-10 | Goal check | v0 **matches** SCA at 2.8 s → fails “beat SCA” on rate | **v5** 1-opt on legal \(a\); **Experiment C before** any campaign |
| 2026-09-10 | Experiment C (1-opt + best-SE + 200 random, then SCA polish) | Frozen-\(q\) LP mean \(\Delta=+0.0010\) Mbps (7/20, max +0.0077). Polish mean +0.0027, max +0.035 (seed 18). A’s seeds 11/13/19 untouched. Readout **flat_at_frozen_q**. | Default-Mbps “beat SCA” is **joint \((q,a)\)** (Experiment A), not \(a\) at frozen SCA \(q\). Goal B (\(T_k=0.8\)) remains. |

**When TD3 Alg. 2 finishes:** baseline table only.

**Next:** do not promise a default-Mbps SPA-SCA win. Either multi-start SCA with v5 \(a\) at *each* init (weaker novelty), or tell the guide the clean win is \(T_k=0.8\) s feasibility. Optional: rerun C at 15% cap / 500 m / \(I=32\) where leftover dump is less of a vertex.
