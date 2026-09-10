# UAV project — what to do next

Last updated: 2026-09-10

Quick reference. Details in `docs/RESULTS.md` §2.9 and `docs/EXPERIMENTS.md`.

**§2.9 layout:** scripts → `scripts/experiments/residual_on_sca/`; JSON → `results/residual_on_sca/`. Old `scripts/run_*.py` shims still work.

---

## In progress

- [ ] **TD3 Alg 2 training** (~5 h left) — paper reproduction preset (default / `--td3-preset alg2`)
  - k-means init, full action space, penalty reward, policy export
  - Not the same as residual-on-SCA (see below)

---

## Done — residual-on-SCA evaluation (2026-09-10)

- [x] **Experiment B** — `results/residual_on_sca/residual_td3_heldout_21_40.json`  
  Readout: **neck_and_neck**. Mean Δ = +0.0032 Mbps, 4/20 wins, 0 losses, construction OK.

- [x] **CMA-ES control** — `results/residual_on_sca/cmaes_polish_heldout_21_40.json`  
  Polish mean Δ = +0.0075 Mbps, 12/20 wins. CMA-ES finds more local gains than ±10 m TD3.

---

## Optional (after core runs)

- [ ] Add TD3 to paper-style campaign sweeps:

  ```powershell
  python -m uavdt campaign --axis all --methods random,kmeans,pso,sca,td3 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20
  ```

- [x] Update `docs/RESULTS.md` §2.9 with Experiment B + CMA-ES + tie-audit (2026-09-10)
- [ ] Runtime in published tables (optional per `docs/param.md`)

---

## Done — no action needed

- SCA (frozen), baselines (random / k-means / PSO), primary campaigns (8.8 MHz / 25% cap)
- Experiment A — SCA multistart (seeds 1–20, measured 2026-09-09)
- Residual-on-SCA **code** (`--td3-preset residual-on-sca`) — implemented + tested
- CMA-ES polish **code** — implemented
- SCA-joint probe, Fig. 11, AoDT compare, n100 figures

---

## Skip

- **Experiment C** (association oracle) — gated off; multistart A was not flat
- VNS / BCD — docs say unlikely path
- Chasing 1 Mbps win at 100 m / 25% cap — headroom is ~0.03–0.05 Mbps

---

## Cheat sheet: two TD3 presets

| | Alg 2 (paper) | Residual-on-SCA (proposed) |
|---|---|---|
| CLI | `--td3-preset alg2` (default) | `--td3-preset residual-on-sca` |
| Start | k-means | **SCA solution** |
| Action | move + assoc + process | **Δxy only** (±10 m) |
| a, b | recomputed | **frozen SCA** |
| Bandwidth | leftover heuristic | **frozen-q LP** |
| Reward | Alg 2 penalty | **feasible Mbps** |

Residual-on-SCA = TD3 that nudges SCA positions locally; zero action reproduces SCA exactly.
