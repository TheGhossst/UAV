# UAV-aided digital twin simulator

Fresh implementation of the Khalaf et al. (IEEE TNSM, 2026) **system
model**, with a 100 × 100 m field and configurable `B_sys`.

The paper is the source of truth. This tree does **not** carry the old
calibrated-radio reconstruction. SCA and TD3 are not in this milestone.

See `docs/REPRODUCTION.md` for the inspection report, equations,
parameter table, and ambiguities.

```text
pip install -r requirements.txt
$env:PYTHONPATH="src"   # PowerShell; use export PYTHONPATH=src on Unix
python -m pytest
python -m uavdt evaluate --seed 1 --bandwidth 20000 --placement random
python -m uavdt evaluate --bandwidth-preset 2.4mhz --placement kmeans
python -m uavdt evaluate --bandwidth-preset 8.8mhz
```
