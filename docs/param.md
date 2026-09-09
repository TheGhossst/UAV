# `param.md` — UAV-Aided Digital Twin IoT Network

> Source: Khalaf et al., *A UAV-Aided Digital Twin Framework for IoT
> Networks With High Accuracy and Synchronization*.
>
> This file separates paper-stated parameters, scenario values,
> optimization variables, derived quantities, and implementation
> assumptions. Values not explicitly specified by the paper are marked
> **NOT SPECIFIED** rather than guessed.

## 1. Core Simulation Parameters


| Parameter                   | Symbol     | Value       | Unit            | Status          | Description                                                                            |
| --------------------------- | ---------- | ----------- | --------------- | --------------- | -------------------------------------------------------------------------------------- |
| Simulation area             | `AREA`     | `500 × 500` | m²              | Paper setup     | Industrial field used for the main simulation.                                         |
| IoT devices                 | `I`        | `10`        | devices         | Paper setup     | Ten IoT devices are randomly deployed.                                                 |
| Physical processes          | `K`        | `2`         | processes       | Paper setup     | Every five IoTs monitor one process.                                                   |
| IoTs per process            | `|N_k|`    | `5`         | devices/process | Paper setup     | Main grouping of IoTs.                                                                 |
| UAVs                        | `J`        | `3`         | UAVs            | Main experiment | Three UAVs are used in several fixed-UAV experiments; the paper also varies UAV count. |
| UAV altitude                | `H`        | `100`       | m               | Table II        | UAV position is `(x_j, y_j, H)`.                                                       |
| Minimum data rate           | `R_min`    | `10,000`    | bit/s           | Table II        | Minimum IoT-UAV communication rate.                                                    |
| Minimum bandwidth allocation | `B_sys`    | `20,000`    | Hz              | Table II        | Table row text is “Minimum.” Constraint (27) uses `B_sys` as a **sum ceiling** (`Σ B_ij ≤ B_sys`). No second bandwidth value is stated. |
| Minimum UAV separation      | `theta`    | `10`        | m               | Table II        | Minimum distance between any two UAVs.                                                 |
| UAV-to-UAV forwarding time  | `T_u2u`    | `0.3`       | s               | Table II        | Time for forwarding a task from an associated UAV to another processing UAV.           |
| Carrier frequency           | `f_c`      | `1 × 10^6`  | Hz              | Table II        | Carrier frequency in the G2A path-loss model.                                          |
| Task arrival rate           | `lambda_i` | `2`         | tasks/s         | Table II        | Mean Poisson task-generation rate per IoT in the main scenario.                        |
| UAV CPU capacity            | `f_j`      | `2 × 10^8`  | cycles/s        | Table II        | Processing capacity of UAV `j`.                                                        |
| Speed of light              | `c`        | `3 × 10^8`  | m/s             | Table II        | Used in the free-space path-loss term.                                                 |
| AoDT threshold              | `T_k`      | `2.8`       | s               | Table II        | Maximum allowed average AoDT.                                                          |
| LoS additional attenuation  | `eta_LoS`  | `1`         | table scale     | Table II        | Additional LoS attenuation term.                                                       |
| NLoS additional attenuation | `eta_NLoS` | `21`        | table scale     | Table II        | Additional NLoS attenuation term.                                                      |
| Noise parameter             | `sigma`    | `0.01`      | W               | Table II        | Table labels this parameter `sigma`; Eq. (6) uses `sigma^2`.                           |
| IoT transmit power          | `p_i`      | `0.2`       | W               | Table II        | IoT transmit power.                                                                    |
| Environment parameter       | `a`        | `9.61`      | —               | Table II        | Parameter in LoS probability.                                                          |
| Environment parameter       | `b`        | `0.16`      | —               | Table II        | Parameter in LoS probability.                                                          |


The paper states that the simulation uses a `500 × 500 m²` field with
ten randomly deployed IoTs, with every five monitoring one physical
process. It also states that each plotted data point is the average of
20 random runs. fileciteturn7file0L36-L57

---



## 2. Noise Notation — Important

The table gives:

 = 0.01 {} 

while Eq. (6) uses:

 . 

Therefore, for a literal implementation of the equation:

 

and:

 . 

Implementation:

```python
sigma = 0.01
noise_power = sigma ** 2
```

The paper’s communication section explicitly identifies `sigma^2` as the
Gaussian white noise power. fileciteturn7file1L385-L405

This notation should be documented in the reproduction because the Table
II label says `sigma` while Eq. (6) contains `sigma^2`.

---



## 3. Geometry



### IoT position

 q_i=(x_i,y_i,0) 

with the main simulation area:

 0x_i,y_i. 

### UAV position

 q_j=(x_j,y_j,H) 

with:

 H=100 {}. 

The paper explicitly models stationary UAVs at `(x_j, y_j, H)`.
fileciteturn7file1L287-L305

---



## 4. Communication Model



### 4.1 Distance

 

fileciteturn7file1L347-L354

### 4.2 Free-space term

 

### 4.3 LoS path loss

 

### 4.4 NLoS path loss

 

### 4.5 LoS probability

 

The paper gives this expression as Eq. (4).
fileciteturn7file1L355-L371

**Angle-unit note:** the displayed equation does not explicitly state
the angle unit. The Table II values `a=9.61` and `b=0.16` need to be
implemented with the intended convention and then validated against the
paper’s results. Do not silently change the equation.

### 4.6 Average path loss

 

fileciteturn7file1L372-L384

### 4.7 Uplink rate

 

where `B_ij` is the bandwidth allocated to the IoT-UAV link and
`sigma^2` is the Gaussian white noise power.
fileciteturn7file1L385-L405

---



## 5. Computing / Queueing Parameters



### Task arrival

Each IoT generates tasks according to a Poisson process:

 

for the main Table II scenario.

### UAV service rate

 

where:

- `f_j` = UAV CPU capacity;
- `L` = average task size in CPU cycles.

The paper gives:

 f_j=2^8 {}. 

`L` **is NOT SPECIFIED in Table II.**

Do not invent a value if exact reproduction is the goal.

### Total workload at UAV `j`

 *{total,j} = *{iN_j}i. 

### Offered load

 {total,j} = . 

The queue must remain stable:

 {total,j}1. 

The paper models each UAV as an M/M/1 queue.
fileciteturn7file1L419-L466

---



## 6. AoDT Parameters



### Process groups

 N_k={k}. 

Main scenario:

```text
K = 2
|N_1| = 5
|N_2| = 5
```



### Effective process update rate

 

### Upload time

If task `i` is processed by its associated UAV:

 D_i=. 

If forwarded:

 D_i=  + T{u2u}. 

with:

 T{u2u}=0.3 {}. 

The paper defines the upload duration this way in Eq. (11).
fileciteturn7file1L502-L554

`S_i`**, the task size in bytes, is NOT SPECIFIED in Table II.**

### Maximum upload time for a process

 D{N_k} = {iN_k}D_i. 

### Average AoDT

 

fileciteturn7file2L630-L694

Constraint:

 

with:

 T_k=2.8 {}. 

---



## 7. Optimization Variables



### 7.1 UAV positions

For UAV `j`:

 (x_j,y_j,H). 

For fixed altitude:

 H=100. 

For `J` UAVs:

 X{UAV} =
. 

### 7.2 Association

 a{ij}{0,1}. 

`a_ij = 1` means IoT `i` is associated with UAV `j`.

### 7.3 Processing assignment

 b{ij}^{k}{0,1}. 

`b_ij^k = 1` means the task from IoT `i` in process group `k` is
processed by UAV `j`.

### 7.4 Bandwidth

 B{ij}. 

System constraint:

 

with:

 B{sys}=20,000 {}. 

The paper’s optimization variables and constraints are given in Problem
(P), Eqs. (20)–(31). fileciteturn7file2L695-L839

---



## 8. Main Objective

The paper’s optimization objective is:

 

This is the common objective that PSO, SCA, TD3, and any proposed method
should ultimately be evaluated against. fileciteturn7file2L736-L742

The paper describes the problem as maximizing the sum rate while
considering computation, AoDT, bandwidth, and system constraints.
fileciteturn7file2L818-L838

---



## 9. Constraint Parameters


| Constraint      | Parameter | Value        | Meaning                         |
| --------------- | --------- | ------------ | ------------------------------- |
| QoS             | `R_min`   | 10,000 bit/s | Minimum rate                    |
| Bandwidth       | `B_sys`   | 20,000 Hz    | Table II: “Minimum bandwidth allocation.” (27): sum cap. |
| UAV separation  | `theta`   | 10 m         | Minimum UAV distance            |
| AoDT            | `T_k`     | 2.8 s        | Maximum process AoDT            |
| UAV height      | `H`       | 100 m        | Fixed altitude                  |
| Queue stability | `mu_j`    | `f_j / L`    | UAV service rate                |
| Forwarding      | `T_u2u`   | 0.3 s        | Inter-UAV task forwarding delay |


---



## 10. PSO Parameters — Your Experiment

These are **not Table II parameters**. They are algorithm parameters for
your proposed comparison study.


| Parameter             | Symbol        | Value       | Status                        |
| --------------------- | ------------- | ----------- | ----------------------------- |
| Number of particles   | `N_particles` | TBD         | Choose and report             |
| Number of iterations  | `N_iter`      | TBD         | Choose and report             |
| Inertia weight        | `w`           | TBD         | Choose/report                 |
| Cognitive coefficient | `c1`          | TBD         | Choose/report                 |
| Social coefficient    | `c2`          | TBD         | Choose/report                 |
| X bounds              | —             | `[0,500]` m | From area                     |
| Y bounds              | —             | `[0,500]` m | From area                     |
| Z                     | `z`           | `100` m     | Fixed by paper                |
| Velocity bounds       | `v_max`       | TBD         | Algorithm choice              |
| Seeds                 | —             | TBD         | Use multiple independent runs |




### One UAV

For one UAV, one particle is:

 

with:

 z_p=100. 

For `N_particles` particles:

 

Each row is **one particle / one candidate UAV position**.

For multiple UAVs, one particle represents a complete deployment and has
shape:

 J. 

---



## 11. Comparison Study

The paper compares:

- SCA-based UAV placement;
- TD3-based UAV placement;
- K-means placement;
- Random placement.

The paper’s simulation section explicitly describes these four
approaches and states that the plotted points are averages over 20
random runs. fileciteturn7file0L38-L57

Your comparison can add:

- PSO-based UAV placement;
- your proposed/novel method.



### Required rule

Every method must use the **same scenario generator and same
evaluator**.

Compare:

 R{sum} 

 {DT_k} 

 

 

 

 

and convergence.

---



## 12. Random Seeds

The paper states:

> Each data point in the figures represents the average of 20 different
> random runs.

For development, a smaller set can be used:

```text
100
101
102
103
104
```

For final experiments, use 20 runs if computationally practical.

Important distinction:

- **scenario seed** → controls IoT deployment and other environment
randomness;
- **algorithm seed** → controls PSO initialization, TD3
initialization/noise, etc.

Do not confuse these with the communication parameter `sigma`.

The exact seed values are **not specified by the paper**.

---



## 13. Parameters That Are Not Numerically Specified


| Quantity                        | Symbol             | Status                          | Why needed                                                |
| ------------------------------- | ------------------ | ------------------------------- | --------------------------------------------------------- |
| Average task size in CPU cycles | `L`                | **NOT SPECIFIED**               | Needed for `mu_j = f_j / L`.                              |
| Task size in bytes              | `S_i`              | **NOT SPECIFIED**               | Needed for upload delay.                                  |
| Base-station coordinates        | `(x_BS,y_BS,z_BS)` | Not numerically specified       | System model includes BS, but download time is neglected. |
| PSO particle count              | `N_particles`      | Not in paper                    | Your algorithm choice.                                    |
| PSO iterations                  | `N_iter`           | Not in paper                    | Your algorithm choice.                                    |
| PSO `w,c1,c2`                   | —                  | Not in paper                    | Your algorithm choice.                                    |
| PSO velocity bound              | `v_max`            | Not in paper                    | Your algorithm choice.                                    |
| Final seed list                 | —                  | Not in paper                    | Paper only states 20 random runs.                         |
| TD3 architecture                | —                  | Not fully specified numerically | Needed for exact reproduction.                            |
| TD3 learning rate               | `eta`              | Not in Table II                 | Needed for exact TD3 implementation.                      |
| TD3 discount                    | `gamma`            | Not in Table II                 | Needed for exact TD3 implementation.                      |
| TD3 policy delay                | `d`                | Not in Table II                 | Needed for exact TD3 implementation.                      |
| TD3 soft update                 | `tau`              | Not in Table II                 | Needed for exact TD3 implementation.                      |
| TD3 exploration noise           | `sigma_RL`         | Not the communication `sigma`   | Must be kept separate.                                    |

These TD3 knobs are now filled in `uavdt.td3.settings.TD3Settings` (Fujimoto et al.
2018 + Algorithm 2 reward weights). They are **not** on `SimConfig` and are
**not** Table II.


---



## 14. Suggested `config.py`

> **Note.** This is the literal Table II extraction and is what
> `--radio-profile table2` runs. It is *not* the shipped default: `B_SYS =
> 20_000` with `NOISE_POWER = SIGMA ** 2` is bounded by Eq. (6)+(27) at
> 0.997 Mbps even for `SNR_max = 10^15`, and at ~0.02 Mbps on the written
> channel (`SNR≈1`), which cannot produce the Mbps-scale Figs. 6-10. `src/config.py`
> defaults to the `calibrated` profile instead; see `docs/calibration.md` for the
> three knobs that differ and why.

```python
# Scenario
AREA_X = 500.0
AREA_Y = 500.0

NUM_IOT = 10
NUM_UAV = 3
NUM_PROCESSES = 2
IOTS_PER_PROCESS = 5

# UAV
UAV_HEIGHT = 100.0
UAV_MIN_DISTANCE = 10.0

# Communication
R_MIN = 10_000.0
B_SYS = 20_000.0
F_C = 1e6
C_LIGHT = 3e8

ETA_LOS = 1.0
ETA_NLOS = 21.0

P_I = 0.2

SIGMA = 0.01
NOISE_POWER = SIGMA ** 2

ENV_A = 9.61
ENV_B = 0.16

# Computing
LAMBDA_I = 2.0
UAV_CPU = 2e8

# AoDT
AODT_THRESHOLD = 2.8
T_U2U = 0.3

# Still unspecified by the paper
TASK_SIZE_BYTES = None
TASK_CYCLES = None
```

---



## 15. Reproduction Checklist

Before claiming reproduction, verify. This milestone uses a **100 × 100 m**
headline field (intentional modification; paper §VII uses 500 × 500 m). A
20 kHz control and optional 8.8 MHz cap25 campaign at 500 m are documented in
`docs/RESULTS.md`.

### Core model (this repo — done)

- [x] Table II parameters are entered exactly (except documented substitutions).
- [x] `sigma = 0.01` is kept separate from `noise_power` (`σ²` in Eq. (6)).
- [x] The radio profile in use is named next to any absolute rate; Table II
      20 kHz as the (27) cap is bounded by Eq. (6)+(27) at 0.997 Mbps even at
      `SNR=10^15` (`docs/RESULTS.md` §0).
- [x] IoT coordinates are generated inside the configured field (**100 × 100 m**
      default; `--area-m 500` for paper-field runs).
- [x] UAV height is fixed at `100 m`.
- [x] Distance, LoS, path loss, rate, and bandwidth constraint (27) implemented.
- [x] `R_min = 10,000 bit/s` is enforced.
- [x] AoDT is calculated using Eq. (17); threshold `T_k = 2.8 s`.
- [x] CPU stability (24) enforced; `S_i` / `L` are external, labeled in config.
- [x] Random, k-means, PSO, SCA, and TD3 use the same `evaluate()` scorer.
- [x] Campaigns use **20 runs** per point; mean and std reported in JSON/CSV.

### Algorithms

- [x] SCA (Algorithm 1) — `src/uavdt/sca/`, CVXPY campaigns, MATLAB spot-check.
- [x] TD3 (Algorithm 2) — `src/uavdt/td3/`. Paper gaps filled in `TD3Settings`
      (not Table II / not `SimConfig`). Opt-in method, not default campaigns.

### Reporting & artifacts

- [x] Paired statistics and FDR documented (`docs/RESULTS.md` §3).
- [x] Paper-style figures from campaign JSON (`scripts/plot_paper_figures.py`).
- [ ] Runtime reported in every published table (optional for writeup).
- [ ] SCA convergence curves saved for every campaign seed (available via
      `python -m uavdt sca --history-json …`; not bundled in campaign JSON).

---



# 16. Key Principle for the Project

The paper’s central optimization problem is:

 

while maintaining:

 

and satisfying:

 

Therefore, the clean research architecture is:

```text
                    Common Paper Model
                           |
                           v
                  Common Evaluator
                           |
          +----------------+----------------+
          |                |                |
         PSO              SCA              TD3
          |                |                |
          +----------------+----------------+
                           |
                           v
                    Same Metrics
                           |
                           v
              Proposed / Novel Method
```

The point of the comparison study is **not** to change the problem for
each algorithm. The algorithms should be different ways of
solving/evaluating the same underlying optimization problem.