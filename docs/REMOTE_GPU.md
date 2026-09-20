# Run remaining experiments on college machine (SSH, no sudo, no tmux)

Target host: **`cse2015@192.168.24.82`** (campus/LAN — your PC must reach that IP).

Remaining work is **CPU** (CVXPY + campaigns); the GPU is optional but the remote box is fine for long runs.

## What is left (as of last sync plan)

1. Finish **`sca_anchor` high-stat** — likely **`aodt` @ n200, 500 m** (checkpoint may exist).
2. **`n200` PPT bank eval** (no TD3): `run_n200_ppt_eval.py --skip-td3`.
3. Optional: plots + `latex/ppt` (can do on your laptop after pull).

---

## 1. One-time setup on the remote (SSH in)

```bash
ssh cse2015@192.168.24.82
```

```bash
mkdir -p ~/UAV ~/venvs ~/UAV/results
python3 -m venv ~/venvs/uavdt
source ~/venvs/uavdt/bin/activate
python -m pip install -U pip wheel
```

After you copy the repo (step 2), on the remote:

```bash
cd ~/UAV
source ~/venvs/uavdt/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
python -m pytest -q --tb=no -x   # quick smoke; optional
```

No `sudo` required. If `python3 -m venv` fails, try `python3.11 -m venv` or ask admin for a module load line on your cluster.

---

## 2. Copy project **to** the remote (from your Windows laptop)

**Upload (one zip, one `scp` password):**

```powershell
cd C:\code\UAV
.\scripts\sync_to_college_gpu.ps1
```

Then **one SSH session** to unzip (second password):

```bash
ssh cse2015@192.168.24.82
mkdir -p ~/UAV
unzip -o ~/uav_sync.zip -d ~/UAV
```

Stop the local long-running Python job **before** sync if a checkpoint is still being written.

**Upload laptop → college (one password):**

```powershell
cd C:\code\UAV
.\scripts\push_uav_to_college.ps1
```

**Download college → laptop (one password):**

```powershell
.\scripts\pull_uav_from_college.ps1
```

Results only (smaller zip):

```powershell
.\scripts\pull_uav_from_college.ps1 -ResultsOnly
```

`sync_from_college_gpu.ps1` / `sync_to_college_gpu.ps1` still work but may ask for the password twice.

**Tip:** Set up SSH keys (`ssh-keygen` then `ssh-copy-id cse2015@192.168.24.82`) so you are not prompted on every command.

---

## 3. Run on remote **without tmux** (`nohup`)

```bash
ssh cse2015@192.168.24.82
cd ~/UAV
source ~/venvs/uavdt/bin/activate
chmod +x scripts/remote/college_gpu_resume.sh scripts/remote/college_gpu_nohup.sh

# Foreground (if SSH stays open):
./scripts/remote/college_gpu_resume.sh

# Detached (survives SSH disconnect):
./scripts/remote/college_gpu_nohup.sh
tail -f results/remote_resume_latest.log
```

If `screen` exists (often without sudo): `screen -S uav` then run the resume script inside.

---

## 4. Copy **results back** to your laptop

```powershell
cd C:\code\UAV
.\scripts\sync_from_college_gpu.ps1
```

Then locally: replot / rebuild PPT if needed:

```powershell
$env:PYTHONPATH="src"
python scripts/plot/plot_sweep_n_stat_comparison.py
python scripts/campaigns/run_n200_ppt_eval.py --skip-td3 --plot-only   # if eval JSON already pulled
```

---

## 5. Resume logic

`college_gpu_resume.sh` runs, in order:

- `run_sca_anchor_highstat.py` for **n200 @ 500 m**, axes **aodt** (skips completed files; resumes `.checkpoint.json`).
- `run_n200_ppt_eval.py --skip-td3`
- `plot_sweep_n_stat_comparison.py` + `plot_paper_figures.py`

To run the full orchestrator instead:

```bash
python -u scripts/orchestration/run_full_regeneration_no_td3.py --resume --from-step anchor_highstat --log results/remote_resume_latest.log
```

---

## 6. Troubleshooting

| Issue | Fix |
|--------|-----|
| `ssh: connect refused` | On campus VPN/LAN; ping `192.168.24.82` |
| `pip` / build errors | `pip install cvxpy` may need `gcc`; ask admin or use `pip install cvxpy --only-binary :all:` |
| Job dies when SSH closes | Use `college_gpu_nohup.sh`, not bare `python` |
| Checkpoint mismatch | Re-copy `results/campaign_sca_anchor_*checkpoint*` from laptop |

**Never commit passwords** or put them in scripts. Use SSH keys: `ssh-copy-id cse2015@192.168.24.82` (if allowed).
