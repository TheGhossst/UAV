# TD3 vs Algorithm 2: fidelity notes (diagnosis, not a patch)

**Date:** 30 August 2026  
**Code:** `src/solvers/td3.py` (Algorithm 2 structure)  
**Paper:** Khalaf, Itani, Sharafeddine, IEEE TNSM vol. 23, 2026, p. 3022, Algorithm 2  
**Reproduce:** `python -m scripts.td3_diagnostics --device auto --out results/td3_diagnostics`  
**Raw logs / plots:** `results/td3_diagnostics/` (gitignored with the rest of `results/`; tables below are the persistent copy)

This is an observe-only pass. No config default was retuned. `TD3_EPISODE_LEN`, `TD3_ASSOC_ACTION_SCALE`, `TD3_WARMUP`, reward weights, and radio settings are unchanged. Ablation values were passed as constructor arguments for task 4 only.

**Protocol:** `DEFAULT` config (calibrated radio, **compute off**, I=10, J=3). Task 1 plots use seeds 100–104. Tasks 2–3 use 100–109. Task 4 uses 100–104. These are the existing audit seeds; none were added.

The 7.083 Mbps figure from the live log that motivated this pass is **not** reproduced here: that run was almost certainly `--compute`. Under `DEFAULT` (compute off) seed 100 reports **5.517 Mbps**, matching the earlier audit’s `td3 --td3-steps 7000` line. The structural findings below (who wins in `solve_td3`, whether the actor improves on k-means) do not depend on that y-axis.

---

## Verdict

**There is no evidence that TD3 training currently contributes to the reported Mbps.** On seeds 100–109, every reported `solve_td3` result is the k-means reset **before any actor action** (`start_index=0`, `step=0`, `source=pre_rollout`). The trained greedy rollout never won. Mean reported TD3 is **4.625 Mbps**; the no-op control (k-means + `complete_solution`, no actor) is **4.647 Mbps**. The small per-seed gaps are two different k-means initializations, not policy improvement.

The crashed global training-reward curve is real, but it is not “learning hidden by reset noise.” Re-bucketing by in-episode step shows reward **falling** from reset to step 49 in both early and late training. Warmup (uniform random actions) does not show that crash. Late episodes are not steeper/higher than early ones.

---

## The three deviations (confirmed against Algorithm 2)

IEEE p. 3022, Algorithm 2 is a single `for t = 1 to T do` loop. The state updates only via `s ← s_{t+1}`. There are no episodes, no resets, no warmup gate, and no extra term added to the actor’s association/processing logits.

### 1. Episode resets (not in the algorithm)

**Pseudocode:** one continuous trajectory, `for t = 1 to T do` … `s ← s_{t+1}`.

**Code:** every `TD3_EPISODE_LEN = 50` steps, `train_td3` calls `_episode_start`, which alternates a k-means `env.reset()` with a uniform-random placement.

```373:381:src/solvers/td3.py
def _episode_start(env: UAVAoDTEnv, episode: int) -> np.ndarray:
    """Alternate the k-means start with random starts for exploration."""
    if episode % 2 == 0:
        return env.reset()
    cfg = env.cfg
    xy = np.column_stack(
        [env.rng.uniform(0.0, cfg.area_x, env.j), env.rng.uniform(0.0, cfg.area_y, env.j)]
    )
    return env.reset(uav_xy=xy)
```

```415:418:src/solvers/td3.py
    for t in range(total_steps):
        # Reset before stepping so no buffer transition straddles an episode.
        if episode_len > 0 and t % episode_len == 0:
            state = _episode_start(env, t // episode_len)
```

`TD3_EPISODE_LEN = 50` is set in `src/config.py` line 164. At 7000 steps this is 140 discontinuous segments, not one trajectory. Every reset boundary is a reward transient. This **is** why the global-step log looks like a crash-then-sawtooth. It is **not**, by itself, proof that the policy is or is not learning — that is task 1.

### 2. Distance-prior injected under the actor’s association/processing output (not in the algorithm)

**Pseudocode:** `a_ij ← normalize_association(associate_logits)` — the parsed actor output *is* the logits.

**Code:** `parse_action` adds a nearest-UAV prior `base = -d / area_x`, and the actor only contributes a scaled offset.

```215:234:src/solvers/td3.py
    def _distance_logits(self, uav_xy: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(self.scenario.iot_xy[:, None, :] - uav_xy[None, :, :], axis=-1)
        return -d / max(self.cfg.area_x, 1.0)
    ...
        base = self._distance_logits(xy) if self.distance_prior else np.zeros((i, j))
        xy, a, b, bw = complete_solution(
            self.scenario,
            xy,
            assoc_logits=base + self.assoc_action_scale * assoc_off,
            proc_logits=base + self.assoc_action_scale * proc_off,
```

`TD3_ASSOC_ACTION_SCALE = 0.25` (`src/config.py` line 169). Distances on the 500 m area put `base` in roughly `[-1.4, 0]`. The actor’s tanh head is `[-1, 1]`, times 0.25 → `[-0.25, 0.25]`. That can flip a nearest-UAV assignment only when two UAVs differ by ≲ 125 m. The repair LP in `complete_solution` (shared with every baseline) still builds a feasible bandwidth allocation on top of whatever association comes out.

The `distance_prior` / `assoc_action_scale` constructor arguments default to the same behaviour as before this pass. They exist so task 4 can ablate without editing config defaults.

### 3. Uniform-random warmup (not in the algorithm; lower priority)

**Pseudocode:** `a_t = μ_φ(s_t) + ε` from `t = 1`.

**Code:** the first `TD3_WARMUP = 500` steps (`src/config.py` line 157) sample `env.rng.uniform(-1, 1)` instead of the actor.

```419:422:src/solvers/td3.py
        if t < TD3_WARMUP:
            action = env.rng.uniform(-1.0, 1.0, size=env.action_dim)
        else:
            action = agent.act(state, noise=TD3_NOISE)
```

This is standard Fujimoto et al. 2018 TD3 practice. It is still a deviation from the literal Algorithm 2 and is documented as such. It is not the main suspect for the reported-Mbps question: greedy eval never uses warmup.

---

## Task 1 — within-episode reward (seeds 100–104)

Training reward is logged per `env.step`, so **in-episode step 0 is the first action after a reset**, not the reset’s own `complete_solution` score.

Mean reward at in-episode step 0 vs step 49:

| phase | start | reward @ 0 | reward @ 49 | climb | sum-rate climb (bit/s) |
|---|---|---:|---:|---:|---:|
| warmup | k-means | +4.20 | +3.72 | **−0.49** | −0.39e6 |
| warmup | random | −3.12 | −3.02 | **+0.10** | +0.25e6 |
| early (post-warmup) | k-means | +2.67 | −14.77 | **−17.44** | −2.25e6 |
| early | random | −2.45 | −15.20 | **−12.74** | −0.61e6 |
| late (post-warmup) | k-means | +3.01 | −13.15 | **−16.16** | −2.47e6 |
| late | random | −2.98 | −12.82 | **−9.84** | −0.36e6 |

Plots: `task1_reward_vs_in_episode.png`, `task1_reward_vs_global_step.png`, `task1_reward_climb_vs_episode.png`.

What this shows:

- The global-step curve’s sawtooth after ~step 500 is the 50-step reset cycle: k-means episodes open near +3 Mbps-worth of reward and the (noisy) actor walks them to ~−13.
- Warmup, which uses uniform random actions rather than the actor, stays flat. The crash is therefore **not** “any 50-step trajectory away from k-means.” It starts when the actor takes over (`t ≥ 500`).
- Late training is not a steeper/higher within-episode climb than early training. Both crash. The late k-means climb (−16.2) is essentially the same as the early one (−17.4).
- That is the opposite of “learning masked by reset-boundary noise.” Reset-boundary noise is real (the global plot is unreadable because of it). After accounting for it, the actor is still not improving the start it is handed.

---

## Task 2 — which restart/step produces the reported result (seeds 100–109)

`solve_td3` records a k-means start (`starts[0] = None` → `env.reset()`) **before** the greedy actor loop, and that snapshot is eligible to become `best_result`.

```552:574:src/solvers/td3.py
    for start_index, start in enumerate(starts):
        kind = "kmeans" if start is None else "random"
        state = env.reset(uav_xy=start)
        result = env.last_result
        ...
        snap = _snap(0, "pre_rollout")
        ...
        if _better(result, best_result):
            best_xy, best_result = env.uav_xy.copy(), result
            winner = snap
```

Distribution over 10 seeds:

| winner | count |
|---|---:|
| `source=pre_rollout` (no actor step) | **10 / 10** |
| `source=actor` | **0 / 10** |
| `kind=kmeans`, `start_index=0`, `step=0` | **10 / 10** |
| random restart won | 0 / 10 |

| seed | reported TD3 (Mbps) | feasible | winner |
|---:|---:|---|---|
| 100 | 5.517 | yes | k-means, step 0, pre-rollout |
| 101 | 5.371 | yes | k-means, step 0, pre-rollout |
| 102 | 4.533 | yes | k-means, step 0, pre-rollout |
| 103 | 3.899 | yes | k-means, step 0, pre-rollout |
| 104 | 5.429 | yes | k-means, step 0, pre-rollout |
| 105 | 4.687 | yes | k-means, step 0, pre-rollout |
| 106 | 4.113 | yes | k-means, step 0, pre-rollout |
| 107 | 4.200 | yes | k-means, step 0, pre-rollout |
| 108 | 3.420 | yes | k-means, step 0, pre-rollout |
| 109 | 5.082 | yes | k-means, step 0, pre-rollout |

`gap_td3_minus_best_prerollout_mbps` is **0.0 on every seed**: among the five eval starts, the best *pre-rollout* snapshot is the reported number, and it is always the k-means start. The 20 greedy actor steps per restart never improved on that snapshot.

CSV: `task2_winners.csv`. Per-step traces: `task2_eval_traces.jsonl`.

---

## Task 3 — no-op control vs full TD3 (seeds 100–109)

The no-op is `solve_td3_noop` in `scripts/td3_diagnostics.py`: `n_restarts=1`, `greedy_steps=0` in spirit — a fresh `UAVAoDTEnv.reset()` (k-means + `complete_solution`), no actor, no training. It matches `solve_kmeans` on the same seed (pinned by `test_noop_kmeans_reset_matches_solve_kmeans`). It is a **separate path**; default `solve_td3` is unchanged.

| seed | no-op / k-means (Mbps) | TD3 (Mbps) | TD3 − no-op | TD3 − best pre-rollout |
|---:|---:|---:|---:|---:|
| 100 | 5.124 | 5.517 | +0.393 | 0 |
| 101 | 4.494 | 5.371 | +0.876 | 0 |
| 102 | 3.953 | 4.533 | +0.581 | 0 |
| 103 | 4.219 | 3.899 | −0.320 | 0 |
| 104 | 6.114 | 5.429 | −0.685 | 0 |
| 105 | 4.634 | 4.687 | +0.053 | 0 |
| 106 | 4.113 | 4.113 | 0.000 | 0 |
| 107 | 4.641 | 4.200 | −0.441 | 0 |
| 108 | 4.381 | 3.420 | −0.961 | 0 |
| 109 | 4.797 | 5.082 | +0.285 | 0 |
| **mean** | **4.647** | **4.625** | **−0.022** | **0** |

All no-op and TD3 rows are feasible with `qos=0`.

The per-seed TD3 − no-op gaps are **not** actor contribution. After `train_td3`, `env.rng` has been consumed by warmup uniforms and 140 episode resets, so the eval-time k-means (`starts[0] = None`) is a **different clustering** than `solve_kmeans(seed)`. Sometimes that other k-means is better (seed 101, +0.88 Mbps), sometimes worse (seed 108, −0.96 Mbps). Mean is a wash. CSV: `task3_noop_vs_td3.csv`.

---

## Task 4 — distance-prior scale ablation (seeds 100–104)

Two retrain+eval runs, defaults not edited:

1. `assoc_action_scale = 2.5` (10× the default 0.25), prior still on.
2. `distance_prior = False` (base zeroed; actor logits only), scale left at 0.25.

| seed | default TD3 | scale ×10 | no prior | winner (all three) |
|---:|---:|---:|---:|---|
| 100 | 5.517 | 5.517 | 5.517 | k-means step 0 pre-rollout |
| 101 | 5.371 | 5.371 | 5.371 | k-means step 0 pre-rollout |
| 102 | 4.533 | 4.533 | 4.533 | k-means step 0 pre-rollout |
| 103 | 3.899 | 3.899 | 3.899 | k-means step 0 pre-rollout |
| 104 | 5.429 | 5.429 | 5.429 | k-means step 0 pre-rollout |
| **mean** | **4.950** | **4.950** | **4.950** | |

Reported Mbps is **bit-identical** across the three variants on every seed. That is expected once task 2 is known: the reported number never uses the actor, and eval-time k-means RNG consumption during training does not depend on the actor (after warmup, `env.rng` is only used at episode boundaries). Changing the association head cannot change a number the association head does not produce.

Training dynamics **do** change, which is the actual ablation signal:

| variant | late k-means reward @ step 0 | @ step 49 |
|---|---:|---:|
| default (prior on, scale 0.25) | **+3.01** | −13.15 |
| scale ×10 | **−5.28** | −12.27 |
| no prior | **−5.35** | −15.91 |

Opening the scale (or dropping the prior) makes the **first action after a k-means reset** already bad. The default 0.25 prior is what keeps in-episode step 0 near the k-means score. Un-suppressing the actor’s association head did not produce a visible useful policy; it made the within-episode start worse and still never won greedy eval.

---

## What this does *not* say

- It does not say Algorithm 2 cannot work. It says **this** training loop plus **this** eval protocol currently reports a k-means + `complete_solution` point.
- It does not say `complete_solution` should be removed from TD3. Every solver in this repo uses it; the question was whether the *trained actor* adds anything on top.
- It does not identify a single root cause for the within-episode crash (reward weights vs network capacity vs noise vs the episode reset itself). Task 1 only shows the crash is there late in training, so “just re-bucket the log” is not enough.

---

## Recommended next steps (not done here)

1. **Stop reporting this number as a trained-policy result.** Until eval is changed, “TD3 Mbps” in compare/sweeps is a post-training-RNG k-means + repair draw. If a write-up needs a TD3 column, either label it as such or change eval in a follow-up.
2. **If the question is “can the actor improve k-means?”** run a follow-up that records the best *actor* snapshot only (exclude `step=0` / `pre_rollout` from `best_result`) and compare that to the no-op. Expect it to lose on this codebase given task 1.
3. **If task 1’s crash is the thing to fix:** look at the reward (`V_viol` / R_min shortfall vs a saturating count), whether `TD3_NOISE` plus 10 m steps walk UAVs out of range, and whether 256-wide nets can represent a useful Δx,Δy policy in 6500 updates. That is a separate retune/capacity task. Removing episode resets without addressing the crash would likely make the global log look even worse, not better.
4. **Do not treat raising `TD3_ASSOC_ACTION_SCALE` as a free fix.** Task 4 shows it lets the actor break nearest-UAV on the first step and does not change the reported number under the current eval.
5. **Optional:** repeat tasks 2–3 with `--compute` if the motivating 7.083 Mbps log was from that protocol. The win-source result is about eval structure and is likely the same; the Mbps will not be.

---

## Pytest

| when | command | result |
|---|---|---|
| before this pass | `python -m pytest -v --tb=short` | **54 passed** in 16.25 s |
| after instrumentation (before the diagnostic run) | same | **59 passed** in 2.56 s |
| after the diagnostic run | same | **59 passed** in 2.67 s |

Added tests (observe-only, no change to the default `solve_td3` path): in-episode log fields, eval-trace step-0 tagging, default prior/scale preserved, no-op matches `solve_kmeans`.

---

## Addendum: greedy-rollout shape (Q1) and k-means RNG (Q2)

**Date:** 30 August 2026 (same seeds 100–109, `DEFAULT`, compute off).  
**Reproduce:** `python -m scripts.td3_diagnostics --mode q1q2 --out results/td3_diagnostics`  
**Inputs:** the greedy traces already logged in `task2_eval_traces.jsonl` (every restart, every step 0–20). This mode does not retrain and does not change `solve_td3`.  
**Outputs:** `q1_greedy_steps.csv`, `q1_classifications.csv`, `q1_seed{100–109}_trajectories.png`, `q2_rng_audit.csv`, `q1q2_summary.json`.

No hyperparameter, `_better()`, step-size, or RNG-seeding change was made.

### Paper corroboration for the episode-reset deviation

The first pass cited Algorithm 2’s single `for t = 1 to T` loop. Section VI of the IEEE PDF says the same thing in prose, with no episode length anywhere.

- **VI-A, Eq. (33)** (p. 3020): the action is `A(t) = {(Δx_j(t), Δy_j(t)), a_ij(t), b_ij(t), B_ij(t)}`. Association and processing are listed as actor outputs, not as offsets on a nearest-UAV prior.
- **VI-B** (pp. 3020–3021): “at each time step t the agent observes … and selects a continuous action a_t … This iterative process continues until the policy converges.” One trajectory; no reset, no episode.
- **Complexity** (p. 3021): training cost is `O(T(mB + nB/d))` — linear in the number of training steps `T`. There is no episode-length factor. Inserting 140 independent 50-step fragments is not the complexity the paper wrote down.

### Q1 — why the greedy rollout never beats its starting point

50 restarts (10 seeds × 5). A restart is **FLAT** if every greedy step stays within 0.05 Mbps of step 0 and the violation count never changes. **IMPROVING_BUT_LOSES** is the user’s Q1b win-criterion label: sum rate exceeds step 0 at some point, but every such point has a *higher* violation count, so `_better()` still keeps step 0. **BEATS_OWN_START** is extra: at least one greedy step is preferred to that restart’s own step 0 by `_better()` (fewer violations, or same violations and higher rate). That point can still lose *globally* to another restart’s k-means step 0.

| shape | all 50 | k-means starts (n=10) | random starts (n=40) |
|---|---:|---:|---:|
| FLAT | **0** | 0 | 0 |
| DIP_NO_RECOVER | 17 | **8** | 9 |
| DIP_PARTIAL_RECOVER | 8 | **2** | 6 |
| IMPROVING_BUT_LOSES | 1 | **0** | 1 |
| BEATS_OWN_START | 24 | **0** | 24 |

Mean first greedy step vs that restart’s step 0: **−1.86 Mbps** on k-means starts, **−1.35 Mbps** on random starts. Nothing is flat.

**K-means starts (the ones that become the reported TD3 number):** 10/10 are a dip. 8 never recover; 2 recover some of the trough and then finish *worse* on violations (seeds 104 and 107 end at 6 violations). Rate never exceeds the k-means step-0 value on any of these 10 rollouts, so `_better()` is not the reason k-means step 0 wins. The actor takes a large first step *away* from the good start and, on 8/10 seeds, keeps walking away for 20 noiseless steps. That is not Q1b’s “small early degradation that almost comes back.”

**Random starts:** 24/40 beat *their own* step 0 by `_better()`, almost always by cutting violations on an infeasible random placement. Those repaired points still lose to the k-means pre-rollout on every seed. One random restart (seed 101, restart 2) is IMPROVING_BUT_LOSES: rate climbs above its own feasible start while picking up a violation. That is the Q1b win-criterion pattern, and it is a 1/50 exception, not the reported-result mechanism.

**Which of Q1a / Q1b?** Mixed, with a clean split by start kind.

- The reported result is the k-means start. On that start the actor is not inert (not FLAT) but it has **no useful direction**: it immediately drops ~2 Mbps and does not climb back above step 0. `_better()` never has a candidate to prefer. **Q1b (win criterion / tiny overshoot) does not explain the reported number.**
- The actor is also not “learned nothing at all” in the strongest Q1a sense: from *bad* random starts it often reduces violations. It has a repair reflex for infeasible placements. That never outruns k-means + `complete_solution`.

**Fix ordering:** do the Algorithm 2 fidelity changes first (remove the 50-step episode resets, let actor logits stand alone, shrink/remove warmup). Those are what currently train the policy on a sawtooth of k-means resets it then leaves. Do **not** start by changing `_better()`: it is not why k-means step 0 wins. `Δx, Δy × 10` may amplify the first-step drop, but shrinking it would not reverse a 20-step walk-away from k-means; treat step size as a later knob, not the next patch.

### Q2 — are “K-means” and “TD3” using the same k-means draw?

**No. Not on the compare / sweeps path, and not even when a pre-trained agent is passed in.**

Standalone baseline (`src/solvers/kmeans.py`, used by `src/main.py` `run_compare` / `run_solver`, `src/experiments/sweeps.py`, `src/experiments/aodt_compare.py`):

```40:47:src/solvers/kmeans.py
def solve_kmeans(scenario: Scenario, seed: int = 0, n_uav: int | None = None) -> tuple[np.ndarray, EvalResult, float]:
    j = n_uav if n_uav is not None else scenario.cfg.num_uav
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    uav_xy = kmeans(scenario.iot_xy, j, rng)
    xy, a, b, bw = complete_solution(scenario, uav_xy)
```

`kmeans()`’s first `rng.choice` is the first draw from `default_rng(seed)`.

TD3 eval (`UAVAoDTEnv.reset` when `uav_xy is None`):

```177:179:src/solvers/td3.py
    def reset(self, uav_xy: np.ndarray | None = None) -> np.ndarray:
        if uav_xy is None:
            self.uav_xy = kmeans(self.scenario.iot_xy, self.j, self.rng)
```

`self.rng` is `np.random.default_rng(seed)` at env construction. That *would* match `solve_kmeans` if `reset(None)` were the first use of `self.rng`. After `train_td3` it is not: warmup alone draws `TD3_WARMUP × action_dim` uniforms from `env.rng` (500 × 96 = 48 000 draws at I=10, J=3), and every even episode reset runs another `kmeans(self.rng)`. `compare` / `sweeps` call `solve_td3` without an agent, so eval’s `starts[0] = None` k-means is this depleted generator.

If an agent *is* passed (`aodt-compare`), `solve_td3` still does `env.reset()` once before the restart loop and then `reset(None)` again for start 0 — two k-means draws, still not the baseline’s first draw.

**Measured on seeds 100–109** (standalone `solve_kmeans` re-run; TD3-eval k-means taken from the same traces as the first pass):

| comparison | result |
|---|---|
| standalone k-means vs fresh `UAVAoDTEnv.reset()` (no training) | **identical** on 10/10 seeds (same xy, same Mbps) |
| TD3-eval k-means (post-training `env.rng`) vs standalone | **not the same draw** |

| seed | standalone / fresh-env (Mbps) | TD3-eval k-means (Mbps) | delta |
|---:|---:|---:|---:|
| 100 | 5.124 | 5.517 | +0.393 |
| 101 | 4.494 | 5.371 | +0.876 |
| 102 | 3.953 | 4.533 | +0.581 |
| 103 | 4.219 | 3.899 | −0.320 |
| 104 | 6.114 | 5.429 | −0.685 |
| 105 | 4.634 | 4.687 | +0.053 |
| 106 | 4.113 | 4.113 | 0.000 |
| 107 | 4.641 | 4.200 | −0.441 |
| 108 | 4.381 | 3.420 | −0.961 |
| 109 | 4.797 | 5.082 | +0.285 |
| **mean** | **4.647** | **4.625** | **−0.022** |
| **mean \|delta\|** | | | **0.459** |
| **range** | | | **[−0.961, +0.876]** |

The signed mean is a wash. The per-seed gaps are **unsystematic** (5 TD3-eval k-means better, 4 worse, 1 tie) and the same size as the historical “TD3 vs K-means” gaps in the audit tables. That is a **separate correctness bug**: every past K-means-vs-TD3 comparison in this repo mixed “different k-means RNG draw” with “different solver.” It is independent of whether the actor learns. Do not patch seeding in this pass; when it is patched, re-run the K-means column against a TD3 eval that uses the *same* first `default_rng(seed)` draw.

### Recommendation (as of the Q1/Q2 pass; item 2 is now done)

1. **Next code change:** Algorithm 2 fidelity on the training loop (drop episode resets, drop the distance prior under `a_ij`/`b_ij`, drop or shrink warmup). Q1 says the reported winner is a k-means point the greedy actor only degrades; Q1b’s `_better()` story is not why. **Done as an opt-in variant** (`fidelity_mode=True`); default `train_td3` is unchanged. See “Track B” below.
2. **Eval-time k-means RNG (Q2):** patched. See the next section.

### Pytest (this addendum)

| when | command | result |
|---|---|---|
| before Q1/Q2 work | `python -m pytest -v --tb=short` | **59 passed** |
| after (observation-only; no new tests) | same | **59 passed** |

---

## Patch: eval-time k-means RNG (Q2)

**Date:** 30 August 2026  
**Scope:** `UAVAoDTEnv.reset(..., eval_kmeans=True)` and the `solve_td3` restart loop (`start is None`). Training is untouched (`train_td3`, `_episode_start`, warmup, episode length, distance prior, `_better()`, Δx/Δy).  
**Reproduce tables:** `python results/run_20260830_rng_fixed/_run_comparisons.py`  
**New outputs:** `results/run_20260830_rng_fixed/` (old audit copies in `results/run_20260830/` were not overwritten)

`reset(uav_xy=None, eval_kmeans=True)` draws k-means from a **fresh** `np.random.default_rng(self.seed)` — the env’s construction seed, the same first draw `solve_kmeans(scenario, seed)` uses. `self.rng` is not advanced. Training resets still call `reset()` with the default `eval_kmeans=False`.

`train_td3` reward and sum-rate logs for seed 100, 32 steps, `device=cpu` are **bit-identical** before vs after this patch.

### Pytest

| when | command | result |
|---|---|---|
| before the patch (new tests already added) | `python -m pytest -v --tb=short` | **2 failed, 59 passed** in 2.86 s |
| after the patch | same | **61 passed** in 2.95 s |

The two failures were `test_solve_td3_eval_kmeans_matches_solve_kmeans_after_training` (post-training start_index=0 xy ≠ `solve_kmeans` on seed 100) and `test_eval_kmeans_does_not_advance_training_rng` (`eval_kmeans` kwarg did not exist). Both pass after the patch. The matching test covers seeds 100–104, train-then-eval and a passed-in agent.

### Re-run: `compare --paper-runs --with-td3 --compute` (seeds 100–119)

Old: `results/run_20260830/compare_20seed/`. New: `results/run_20260830_rng_fixed/compare_20seed/`. Side-by-side: `results/run_20260830_rng_fixed/before_after_compare_20seed.md`.

Random / K-means / placement PSO / SCA means are **unchanged** (3.812 / 6.429 / 7.588 / 7.143). Only TD3 moves.

| | old TD3 | new TD3 | K-means |
|---|---:|---:|---:|
| mean Mbps | 6.528 | **6.512** | 6.429 |
| mean − K-means | +0.099 | +0.083 | 0 |

17 / 20 seeds now have TD3 = K-means bit-for-bit. The leftover +0.083 Mbps is three seeds (105, 108, 118) where a non-k-means eval start still wins.

**Ranking:** **PSO 7.588 > SCA 7.143 > TD3 6.512 > K-means 6.429 > Random 3.812** — same order as the audit’s “TD3 beats K-means” claim; the gap shrinks from 0.099 to 0.083 Mbps.

### Re-run: `aodt-compare` default I=10, J=3 (seeds 100–119)

Old: `results/run_20260830/aodt/`. New: `results/run_20260830_rng_fixed/aodt/` (default table only; J/I sweeps not repeated). Side-by-side: `results/run_20260830_rng_fixed/before_after_aodt_default.md`.

| | old TD3 (pooled) | new TD3 (pooled) | K-means |
|---|---:|---:|---:|
| mean Mbps | 6.708 | **6.512** | 6.429 |
| mean − K-means | +0.279 | +0.083 | 0 |

**Ranking:** **SCA 7.143 > TD3 6.512 > K-means 6.429 > PSO-joint 4.959 > Random 3.812** — still “TD3 beats K-means”; the gap shrinks from 0.279 to 0.083 Mbps. The audit’s 6.708 figure was almost entirely the Q2 RNG mismatch.

The new per-seed TD3 column is **identical** between per-seed training (`compare`) and the pooled policy (`aodt-compare`), including the three seeds that beat K-means. That is what you would expect if those winners are random-restart `complete_solution` points (`default_rng(seed+17)`), not the trained actor.

J ≠ 3 and I ≠ 10 TD3 cells in the original AoDT sweeps were **not** re-run; treat those TD3 numbers as still Q2-contaminated.

---

## Track B: Algorithm 2 fidelity variant (opt-in, diagnostic)

**Date:** 31 August 2026  
**Code:** `train_td3(..., fidelity_mode=True)` / `solve_td3(..., fidelity_mode=True)` in `src/solvers/td3.py`. Default `fidelity_mode=False` is the compare/sweeps path and is unchanged.  
**Reproduce:** `python -m scripts.td3_diagnostics --mode alg2 --device auto --out results/td3_diagnostics/alg2_fidelity`  
**Raw logs / plots:** `results/td3_diagnostics/alg2_fidelity/` (does not overwrite the default-path diagnostics)

This is still a diagnostic. Config defaults (`TD3_EPISODE_LEN`, `TD3_ASSOC_ACTION_SCALE`, `TD3_WARMUP`, reward weights, noise, hidden size, LRs) were not retuned. The variant only isolates the three structural deviations.

### What the variant changes

| knob | default `train_td3` | `fidelity_mode=True` |
|---|---|---|
| episode resets | every 50 steps, alternating k-means / random | **none** after the t=0 Initialize reset; `s ← s_{t+1}` for all T |
| `a_ij` / `b_ij` | `base = -d/area_x` plus `0.25 ×` actor | actor logits **directly** (`distance_prior=False`, scale **1.0**) |
| warmup | 500 uniform-random actions | **0** |

The old prior remains available: `fidelity_mode=True, distance_prior=True, assoc_action_scale=0.25`.

**Warmup reasoning:** dropped, not shrunk. Alg. 2 takes `a_t = μ_φ(s_t)+ε` from t=1. `train_step` already no-ops until the buffer has `TD3_BATCH_SIZE` (128) samples, so a 500-step uniform phase is the extra deviation. `TD3_NOISE` on the actor from t=0 still explores. Greedy eval never used warmup on either path.

Scale 1.0 is part of un-mediating the logits (deviation 2), not a separate hyperparameter retune: with the prior off and scale left at 0.25 the actor would still only move association/processing by `±0.25`.

**Track A (eval-time k-means RNG) has landed.** `solve_td3` start_index=0 uses `reset(..., eval_kmeans=True)`. The K-means comparison below is the same first `default_rng(seed)` draw as `solve_kmeans`.

### Task 1 — training-reward trend (seeds 100–104, no episode boundaries)

Equal-length windows of the continuous 7000-step trajectory:

| window | mean reward | mean sum rate (Mbps) |
|---|---:|---:|
| first half (t = 0–3499) | −22.93 | 0.596 |
| second half (t = 3500–6999) | −18.39 | 0.751 |
| first 2000 | −21.68 | 0.644 |
| last 2000 | −18.52 | 0.726 |

Late − early = **+4.54 reward** and **+0.156 Mbps**. That is a real, small lift inside a still-crashed band. Training sum rate stays **below 1 Mbps**; standalone k-means on these seeds is ~4.6 Mbps. Seed 100’s first training step is already −11.9 reward / 0.79 Mbps — the Initialize k-means is abandoned on the first noisy actor action (no distance prior), and the rest of the 7000 steps never return to it.

Plots: `task1_reward_vs_global_step.png`, `task1_reward_vs_50step_bin.png`. There is no sawtooth, because there are no resets. There is also no climb toward a k-means-quality operating point.

### Task 2 — who wins greedy eval (seeds 100–109)

| winner | count |
|---|---:|
| `source=pre_rollout` (no actor step) | **10 / 10** |
| `source=actor` | **0 / 10** |
| `kind=kmeans`, `start_index=0`, `step=0` | **10 / 10** |

The trained actor never won. Same structure as the default path.

### Task 3 — vs no-op and K-means (Track A in effect)

| seed | no-op / k-means (Mbps) | fidelity TD3 (Mbps) | TD3 − k-means |
|---:|---:|---:|---:|
| 100 | 5.124 | 5.124 | 0 |
| 101 | 4.494 | 4.494 | 0 |
| 102 | 3.953 | 3.953 | 0 |
| 103 | 4.219 | 4.219 | 0 |
| 104 | 6.114 | 6.114 | 0 |
| 105 | 4.634 | 4.634 | 0 |
| 106 | 4.113 | 4.113 | 0 |
| 107 | 4.641 | 4.641 | 0 |
| 108 | 4.381 | 4.381 | 0 |
| 109 | 4.797 | 4.797 | 0 |
| **mean** | **4.647** | **4.647** | **0** |

No-op, standalone K-means, and reported TD3 are **bit-identical** on every seed. That is Track A doing its job: the reported number *is* k-means + `complete_solution`. The actor contributes nothing on top.

### Q1 — greedy-rollout shape from k-means starts

50 restarts (10 seeds × 5). Same classifier as the default-path addendum.

| shape | all 50 | k-means starts (n=10) | random starts (n=40) |
|---|---:|---:|---:|
| FLAT | 0 | 0 | 0 |
| DIP_NO_RECOVER | 28 | **8** | 20 |
| DIP_PARTIAL_RECOVER | 15 | **2** | 13 |
| IMPROVING_BUT_LOSES | 1 | **0** | 1 |
| BEATS_OWN_START | 6 | **0** | 6 |

Mean first greedy step vs that restart’s step 0:

| | default path (prior pass) | fidelity variant |
|---|---:|---:|
| k-means starts | **−1.86 Mbps** | **−3.44 Mbps** |
| random starts | −1.35 Mbps | −1.84 Mbps |
| k-means BEATS_OWN_START | 0 / 10 | **0 / 10** |
| random BEATS_OWN_START | 24 / 40 | **6 / 40** |

The “consistent first step away from k-means” **did not go away**. It got larger. 10/10 k-means rollouts are still a dip (8 never recover, 2 recover some of the trough and finish worse). `_better()` still never sees a candidate. The random-start repair reflex **weakened** (24/40 → 6/40), and those six still lose globally to the k-means pre-rollout.

Per-seed k-means first-step (Mbps): 100 −4.45, 101 −3.94, 102 −3.11, 103 −2.96, 104 −5.29, 105 −3.54, 106 −3.19, 107 −1.18, 108 −2.69, 109 −4.02.

### Verdict

Matching Algorithm 2’s control flow **did not fix the underlying problem**. It changed the shape of training (no sawtooth; a small late-vs-early reward lift from −22.9 to −18.4) without changing what `solve_td3` reports (k-means step 0, 10/10) and while making the k-means-start first step **worse** (−1.86 → −3.44 Mbps).

A plausible reason the k-means degradation got worse, not better: without even-episode k-means resets, the policy almost never sees a k-means state after t=0. Training lives in the crashed ~0.6 Mbps region; greedy eval then starts at k-means, which is out of distribution. The default path’s resets were not producing a useful k-means policy either, but they at least kept that state in the buffer. Un-mediating `a_ij`/`b_ij` (no prior, scale 1.0) also matches the first-pass task 4 ablation: opening the association head from a good start is immediately destructive.

This is not “TD3 works now” and not quite “nothing moved.” What moved is training-curve geometry and the size of the first greedy step. What did **not** move is the reported result and the fact that the actor has no useful direction from a good start.

**Next investigation (not done here):** reward design (`V_viol` / R_min shortfall vs a saturating count, `TD3_R_MAX`) and whether 256-wide nets plus `Δx,Δy × 10` can represent a stay-near-k-means policy at all. Do not treat raising `TD3_ASSOC_ACTION_SCALE` or restoring episode resets as the next patch; this pass already ran the open-logits, no-reset combination.

### Pytest (this pass)

| when | command | result |
|---|---|---|
| before the variant | `python -m pytest -v --tb=short` | **61 passed** in 17.18 s |
| after (additive tests; default path unchanged) | same | **65 passed** in 3.09 s |

New tests pin continuous `start_kinds`, `distance_prior=False` / scale 1.0, actor-from-t=0, default warmup still in effect, and the old prior still selectable under `fidelity_mode=True`.

