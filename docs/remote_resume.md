# Resume no-TD3 regen on a remote Linux host (SSH)

The remaining work is **CPU-bound** (CVXPY + PSO + `sca_anchor`). A GPU server helps mainly as a **dedicated machine**, not because CUDA accelerates this pipeline.

## 1. Stop the local run

On Windows, stop the regen so checkpoints are not being written mid-copy:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'run_full_regeneration|run_highstat|run_sca_anchor' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## 2. Pack (Windows, repo root)

```powershell
.\scripts\pack_resume_bundle.ps1
```

This creates `results\uavdt_resume_bundle.tar.gz` (code + banks + finished campaigns + anchor checkpoints).

## 3. Copy to the server

```powershell
scp results\uavdt_resume_bundle.tar.gz cse2015@192.168.24.82:~/
```

Or with rsync (Git Bash / WSL), full tree:

```bash
rsync -avz --progress \
  --exclude '.git' --exclude '__pycache__' --exclude '.pytest_cache' \
  /c/code/UAV/ cse2015@192.168.24.82:~/UAV/
```

You must be on the college network (or VPN) for `192.168.24.82`.

## 4. Unpack and run (SSH)

```bash
ssh cse2015@192.168.24.82
mkdir -p ~/UAV && cd ~/UAV
tar -xzf ~/uavdt_resume_bundle.tar.gz -C ~/UAV
chmod +x scripts/remote_resume_linux.sh
# If you see `bash\r: No such file or directory` (Windows line endings):
sed -i 's/\r$//' scripts/remote_resume_linux.sh

The script creates `~/UAV/.venv` (Ubuntu PEP 668 blocks system `pip install`).
If `python3 -m venv` fails: `sudo apt install python3-venv python3-full`

# Background (no tmux): survives SSH disconnect
./scripts/remote_resume_linux.sh --background
tail -f results/remote_regen.log

# Or foreground (keep SSH open until done)
# ./scripts/remote_resume_linux.sh
```

Resume starts at **`anchor_highstat`** (skips finished n20/n100/n200 baseline campaigns and completed anchor JSON). Incomplete axes continue from `*.checkpoint.json`.

## 5. Copy results back

```powershell
scp -r cse2015@192.168.24.82:~/UAV/results/*.json C:\code\UAV\results\
scp -r cse2015@192.168.24.82:~/UAV/results/figures C:\code\UAV\results\
scp cse2015@192.168.24.82:~/UAV/latex/ppt/main.pdf C:\code\UAV\latex\ppt\
```

Or rsync the whole `results/` and `latex/ppt/` trees.

## Verify after merge

```powershell
$env:PYTHONPATH = "src"
python scripts/plot/plot_sweep_n_stat_comparison.py
python scripts/plot/plot_ppt_all_methods_bank.py
```
