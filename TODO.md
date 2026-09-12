# UAV project — what to do next

Last updated: 2026-09-12

Quick reference. Details in `docs/RESULTS.md` §2.9–§2.10 and `docs/EXPERIMENTS.md`.

**§2.9 layout:** scripts → `scripts/experiments/residual_on_sca/`; JSON → `results/residual_on_sca/`. Old `scripts/run_*.py` shims still work.

**§2.10 artifacts:** `results/campaign_8.8mhz_cap25_td3.json`, `results/n100/eval_td3.json`, `results/n100_500m_cap25/eval_td3.json`. Do not cite `*_SNAPSHOT_INVALID_*`.

---

## In progress

- *(none)*

---

## Done — fine B_sys × cap search (2026-09-12)

- [x] 144/144 cells, 7.1–8.8 MHz × {none,10,12,15,18,20,22,25}%, no TD3
- [x] Null confirmed: at fixed cap, J=3 spread ∝ `B_sys` (r²=1.000 except 25%)
- [x] Perfect 42 = all 12% + all 15% + 18% at 8.3–8.8 MHz. Score-best 8.8/12% is an artifact — do not promote
- [x] 10% PSO best 18/18. Headline 8.8/25% unchanged (spread 0.046, p=0.123)
- [x] `scripts/analyze_bw_fine_search.py` → `results/bw_fine_7p1_8p8/analysis.txt` · `docs/RESULTS.md` §2.13

---

## Done — multi-start four default-point cases (2026-09-11)

- [x] Driver `scripts/run_sca_multistart_cases.py` (does not overwrite headline campaigns / n100/eval.json)
- [x] 20-seed 100 m: reused `sca_multistart_n20.json` — vs SCA **+0.014** (13/20), 2/20 practical
- [x] 20-seed 500 m: `sca_multistart_n20_500m.json` — vs SCA **+0.103** (11/20), **8/20** practical, 20/20 vs random
- [x] n100 100 m: `results/n100/eval_multistart.json` — vs SCA **+0.013** (66/100), 6/100 practical
- [x] n100 500 m (the “n500” bank): `results/n100_500m_cap25/eval_multistart.json` — vs SCA **+0.165** (62/100), **48/100** practical
- [x] Analysis `scripts/analyze_sca_multistart_cases.py` → `results/sca_multistart_cases_analysis.txt`
- Frozen SCA stays the headline solver. 15% cap not rerun.

---

## Done — multi-start SCA method (2026-09-11)

- [x] `uavdt.sca_multistart` / `method="sca_multistart"` (keep-best extra inits; does not edit frozen SCA)
- [x] Default-point eval `results/sca_multistart_n20.json`: vs SCA **+0.014** (13/20), vs random **+0.033** (19/20). Closes 4/5 random-loss seeds; seed 18 still −0.003 Mbps.

---

## Done — 7 MHz campaigns (2026-09-11)

- [x] `--axis all`, 20 seeds, methods random/k-means/PSO/SCA (**no TD3**)
- [x] No cap / 15% / 25% → `results/campaign_7mhz_n20.json`, `_cap15_n20.json`, `_cap25_n20.json`
- [x] SCA at J=3: **7.132 / 7.073 / 7.115** Mbps; linear vs 8.8 MHz to **0.001–0.003 Mbps**
- [x] `docs/RESULTS.md` §2.11

---

## Done — TD3 Algorithm 2 full eval (2026-09-10)

- [x] College-server CUDA run finished (`results-server` → citable names in `results/`)
- [x] Policy export on every seed; official Mbps == `policy_export`
- [x] Default J=3: TD3 **8.917** Mbps, TD3−SCA **−0.030** (2/20, p<0.001)
- [x] **0/29** sweep points with TD3 mean > SCA; gap vs SCA shrinks as I grows but does not change sign
- [x] `docs/RESULTS.md` §2.10 + `python scripts/analyze_td3_vs_methods.py`
- [x] 500 m n100 TD3 (2026-09-11): **7.510 Mbps**, TD3−SCA **−0.718** (0/100). Stays with k-means.

---

## Done — residual-on-SCA evaluation (2026-09-10)

- [x] **Experiment B** — `results/residual_on_sca/residual_td3_heldout_21_40.json`
  Readout: **neck_and_neck**. Mean Δ = +0.0032 Mbps, 4/20 wins, 0 losses, construction OK.

- [x] **CMA-ES control** — `results/residual_on_sca/cmaes_polish_heldout_21_40.json`
  Polish mean Δ = +0.0075 Mbps, 12/20 wins. CMA-ES finds more local gains than ±10 m TD3.

---

## Optional (after core runs)

- [x] Add TD3 to the paper-style 25% grid — done via `run_td3_full_eval.py` (baselines reused; does not overwrite `campaign_8.8mhz_cap25_si12k.json`)
- [x] Update `docs/RESULTS.md` §2.9 with Experiment B + CMA-ES + tie-audit (2026-09-10)
- [x] Experiment C + §2.9 readout (2026-09-10; `assoc_oracle_n20.json`, **flat_at_frozen_q**)
- [ ] Runtime in published tables (optional per `docs/param.md`; Alg. 2 vs SCA wall-clock is in §2.10)

---

## Done — no action needed

- SCA (frozen), baselines (random / k-means / PSO), primary campaigns (8.8 MHz / 25% cap)
- Experiment A — SCA multistart (seeds 1–20, measured 2026-09-09)
- Experiment C — association oracle at frozen SCA \(q\) (seeds 1–20, measured 2026-09-10). Readout **flat_at_frozen_q**.
- Residual-on-SCA **code** (`--td3-preset residual-on-sca`) — implemented + tested
- CMA-ES polish **code** — implemented
- SCA-joint probe, Fig. 11, AoDT compare, n100 figures
- TD3 Algorithm 2 **measured** (policy export; §2.10)

---

## Skip

- VNS / BCD as a default-Mbps method — Experiment C is flat at frozen SCA \(q\) (mean LP \(\Delta=+0.001\) Mbps)
- Chasing 1 Mbps win at 100 m / 25% cap — headroom is ~0.03–0.05 Mbps
- Citing `SNAPSHOT_INVALID` TD3 files — old best-snapshot bookkeeping

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
| Default J=3 vs SCA | **−0.030 Mbps** (2/20) | **+0.003 Mbps** (4/20, 16 origin ties) |

Residual-on-SCA = TD3 that nudges SCA positions locally; zero action reproduces SCA exactly.
