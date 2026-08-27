# UAV-Aided Digital Twin — Implementation & Research Plan

## 0. Project goal

Use the paper's optimization problem as the common mathematical problem, then compare different solution methods.

The paper's objective is

\[
\max \sum_{i=1}^{I}\sum_{j=1}^{J} a_{ij}r_{ij}.
\]

The planned comparison is:

```text
Common paper model
        |
        v
Common evaluator
        |
  +-----+-----+-----+------+
  |           |            |
 Random     K-means       PSO
  |           |            |
  +-----------+------------+
              |
             SCA
              |
             TD3
              |
        Proposed method
```

Every method must eventually use the same scenario generator and evaluator.

## 1. Source-of-truth rule

Use the supplied paper for paper-specific equations and numerical parameters.

Do not silently invent missing values.

The paper's Table II gives the main simulation values: `500 × 500 m²`, 10 IoTs, UAV altitude 100 m, minimum rate 10,000 bit/s, system bandwidth 20,000 Hz, minimum UAV separation 10 m, `T_u2u = 0.3 s`, carrier frequency `1e6 Hz`, task arrival rate `2 tasks/s`, UAV CPU `2e8 cycles/s`, AoDT threshold `2.8 s`, LoS attenuation 1, NLoS attenuation 21, `sigma = 0.01 W`, IoT power 0.2 W, and environment parameters `a=9.61`, `b=0.16`.

The paper's simulation section states that the main scenario is a `500 × 500 m²` field with 10 randomly deployed IoTs, every five monitoring one process, and that plotted data points are averages over 20 random runs.

## 2. Reproduction gaps

The paper defines but does not numerically specify in Table II:

- `S_i`: task size in bytes, needed for upload time;
- `L`: average task size in CPU cycles, needed for `mu_j = f_j / L`.

Therefore the first implementation intentionally does not claim to reproduce the full AoDT/computing model.

The displayed LoS equation also does not explicitly state the angle unit. The code exposes this as a configuration option instead of silently changing the equation.

## 3. Phase 1 — communication model

For one UAV:

\[
q_u=(x_u,y_u,H), \qquad H=100.
\]

For every IoT:

1. distance;
2. LoS probability;
3. LoS path loss;
4. NLoS path loss;
5. average path loss;
6. rate;
7. total sum rate.

Distance:

\[
d_i =
\sqrt{(x_i-x_u)^2+(y_i-y_u)^2+H^2}.
\]

Rate:

\[
r_i =
B_i\log_2\left(
1+\frac{p_i10^{-L_i^{avg}/10}}{\sigma^2}
\right).
\]

The Table II label is `sigma = 0.01 W`, while Eq. (6) uses `sigma^2`, so the code uses:

```python
sigma = 0.01
noise_power = sigma ** 2
```

### First checkpoint

Run:

```bash
python main.py --mode single
```

and verify all 10 link rates and the total sum rate for a fixed UAV at `(250,250,100)`.

## 4. Phase 2 — one-UAV PSO

For one UAV, a particle is:

\[
[x_p,y_p,z_p],\qquad z_p=100.
\]

A population of `N_particles` is:

\[
X=
\begin{bmatrix}
x_1&y_1&100\\
x_2&y_2&100\\
\vdots&\vdots&\vdots\\
x_N&y_N&100
\end{bmatrix}.
\]

The horizontal search bounds are:

\[
0\le x\le500,\qquad0\le y\le500.
\]

Initial algorithm choices:

```text
N_particles = 20
N_iterations = 100
w = 0.7
c1 = 1.5
c2 = 1.5
```

These are algorithm choices, not paper parameters.

For the first controlled experiment, bandwidth is fixed equally:

\[
B_i=B_{sys}/I=2000\text{ Hz}.
\]

The fitness is:

\[
F(x,y)=\sum_i r_i.
\]

The program also reports minimum link rate and QoS violations. This stage is placement-only and is not yet the full Problem (P).

## 5. Phase 3 — multiple seeds

Initial development seeds:

```text
100
101
102
103
104
```

For every seed record:

- best x;
- best y;
- best sum rate;
- minimum link rate;
- QoS violations;
- convergence;
- runtime.

For final paper-style experiments, use 20 runs because the paper reports 20 random runs.

## 6. Phase 4 — bandwidth optimization

The full paper optimizes `B_ij` subject to:

\[
\sum_i\sum_jB_{ij}\le B_{sys}
\]

and:

\[
r_{ij}\ge a_{ij}R_{min}.
\]

After placement-only PSO is validated, bandwidth allocation can become part of the candidate solution.

For one UAV:

\[
[x,y,B_1,\ldots,B_I].
\]

Do not mix this with Phase 2 until the placement-only evaluator is verified.

## 7. Phase 5 — computing and AoDT

Resolve `S_i` and `L`.

Then implement:

\[
\mu_j=f_j/L,
\]

queue stability, upload delay, and

\[
\Delta_{DT_k}
=
\max_{i\in N_k}D_i
+
\frac{1}{\lambda_{N_k}}
\left(
1+
\frac{\sum_{i\in N_k}\lambda_i}{\mu}
\right).
\]

Enforce:

\[
\Delta_{DT_k}\le2.8.
\]

## 8. Phase 6 — multi-UAV PSO

For `J` UAVs, one particle represents:

\[
X_p\in\mathbb R^{J\times3}.
\]

Example:

\[
\begin{bmatrix}
x_1&y_1&100\\
x_2&y_2&100\\
x_3&y_3&100
\end{bmatrix}.
\]

Add the 10 m minimum UAV-separation constraint.

## 9. Phase 7 — association and processing

The full paper uses:

\[
a_{ij}\in\{0,1\}
\]

for association and:

\[
b_{ij}^{k}\in\{0,1\}
\]

for processing assignment.

Eventually implement one association per IoT, processing assignment, same-process consistency, queue stability, bandwidth, QoS, and AoDT.

For PSO, discrete variables need an explicit encoding/repair strategy. Do not implement this until continuous placement works.

## 10. Phase 8 — baselines

Implement the paper's:

- random placement;
- K-means placement.

Evaluate them with the same evaluator.

## 11. Phase 9 — SCA

After the common evaluator is stable:

- reconstruct the paper's SCA formulation;
- use the same parameters and scenarios;
- compare objective and runtime.

The paper states that its SCA implementation uses MATLAB CVX with MOSEK.

## 12. Phase 10 — TD3

Then reconstruct:

- environment;
- state/action definitions;
- reward;
- TD3 training;
- same scenarios;
- same evaluation metrics.

## 13. Phase 11 — novelty

Do not choose the novelty before the baselines work.

Possible directions include:

- constraint-aware PSO;
- hybrid PSO + local search;
- PSO + SCA;
- adaptive PSO;
- PSO initialization for TD3.

The novelty must be justified and measured; “using PSO” alone is not novelty.

## 14. Final comparison

Report at least:

| Method | Sum Rate | AoDT | QoS Violations | Runtime | Std |
|---|---:|---:|---:|---:|---:|
| Random | | | | | |
| K-means | | | | | |
| PSO | | | | | |
| SCA | | | | | |
| TD3 | | | | | |
| Proposed | | | | | |

Plots:

1. convergence;
2. sum rate vs number of UAVs;
3. sum rate vs number of IoTs;
4. sum rate vs AoDT threshold;
5. runtime vs problem size;
6. UAV placement;
7. mean ± standard deviation.

## 15. Immediate TODO

- [x] Parameter document
- [x] Project skeleton
- [x] Scenario generation
- [x] Distance
- [x] LoS probability
- [x] LoS/NLoS path loss
- [x] Average path loss
- [x] Communication rate
- [x] One-position evaluation
- [x] One-UAV PSO
- [x] Multi-seed runner
- [ ] Validate numerical output
- [ ] Resolve `S_i`
- [ ] Resolve `L`
- [ ] Add bandwidth optimization
- [ ] Add AoDT
- [ ] Add multi-UAV PSO
- [ ] Add random baseline
- [ ] Add K-means baseline
- [ ] Reconstruct SCA
- [ ] Reconstruct TD3
- [ ] Design/justify novelty
- [ ] Run final 20-run experiments

## 16. Core principle

```text
Scenario
   |
   v
Common Environment
   |
   v
Common Evaluator
   |
   +--> Random
   +--> K-means
   +--> PSO
   +--> SCA
   +--> TD3
   +--> Proposed
   |
   v
Same metrics
```

Never give different algorithms different mathematical evaluators.
