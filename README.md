# UAV-aided digital twin simulator

Fresh implementation of the Khalaf et al. (IEEE TNSM, 2026) **system
model**, with a 100 × 100 m field and configurable `B_sys`.

The paper is the source of **methodology and Table II parameters**,
except area, bandwidth, `S_i`, and `L`. Published Mbps figures are
**not** a target: Table II’s 20 kHz is infeasible under this model’s
QoS/AoDT floors (~102 kHz required) and cannot produce the paper’s
7–14 Mbps plots (Eq. (6) bounds 20 kHz at ~0.13 Mbps). Headline
results are **2.4 MHz and 8.8 MHz**, each with no per-link cap and
with a 25% per-link cap (`max_bw_share=0.25`, an **EXTERNAL
PARAMETER**, not Problem (P)).

TD3 is not in this milestone.

See `docs/REPRODUCTION.md` §4.1 for the 20 kHz substitution, and
`docs/EXPERIMENTS.md` for the frozen-SCA experimental campaign
(Figs. 6–10 axes, Random/K-means/PSO, CVX/MOSEK spot-check).
**Current results and audit:** `docs/RESULTS.md`.

```text
pip install -r requirements.txt
$env:PYTHONPATH="src"   # PowerShell; use export PYTHONPATH=src on Unix
python -m pytest
python -m uavdt evaluate --seed 1 --bandwidth 20000 --placement random
python -m uavdt evaluate --bandwidth-preset 2.4mhz --placement kmeans
python -m uavdt evaluate --bandwidth-preset 8.8mhz
python -m uavdt sca --seed 1 --bandwidth-preset 2.4mhz --solver matlab
python -m uavdt sca --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --solver matlab
python -m uavdt sca-seq-debug --seed 1 --bandwidth-preset 2.4mhz --solver matlab
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --n-runs 5 --solver cvxpy
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz
python -m uavdt aodt-compare --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --placement kmeans
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20
```
