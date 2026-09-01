# Full Part A audit pipeline -> results/run_20260831/
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$OUT = "results/run_20260831"
$LOG = "$OUT/logs/full_audit.log"
New-Item -ItemType Directory -Force -Path "$OUT/logs" | Out-Null

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts  $msg" | Tee-Object -FilePath $LOG -Append
}

Log "=== full audit start ==="

Log "compare 5-seed + TD3"
python -m src.main --mode compare --with-td3 --compute --particles 20 --iters 100 --td3-steps 7000 --out "$OUT/compare_5seed" --no-status-sync --log-file "$OUT/logs/compare_5seed.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "compare 20-seed + TD3"
python -m src.main --mode compare --paper-runs --with-td3 --compute --particles 20 --iters 100 --td3-steps 7000 --out "$OUT/compare_20seed" --no-status-sync --log-file "$OUT/logs/compare_20seed.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "compare table2 5-seed + TD3"
python -m src.main --mode compare --radio-profile table2 --with-td3 --compute --particles 20 --iters 100 --td3-steps 7000 --out "$OUT/table2_5seed" --no-status-sync --log-file "$OUT/logs/table2_5seed.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "aodt-compare (20 seeds)"
python -m src.main --mode aodt-compare --particles 20 --iters 100 --td3-steps 7000 --out "$OUT" --no-status-sync --log-file "$OUT/logs/aodt_compare.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "constraint sweeps (lambda, Tk, fj) without TD3"
python -c @"
from pathlib import Path
import argparse
from src.config import DEFAULT
from src.experiments.sweeps import sweep_lambda, sweep_aodt, sweep_cpu, plot_summaries
cfg = DEFAULT.with_compute()
args = argparse.Namespace(paper_runs=True, with_td3=False, particles=20, iters=100, td3_steps=7000)
out = Path('$OUT/sweeps_constraints')
out.mkdir(parents=True, exist_ok=True)
sweep_lambda(cfg, args, out)
sweep_aodt(cfg, args, out)
sweep_cpu(cfg, args, out)
plot_summaries(out)
print('wrote constraint sweeps to', out)
"@
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "sweeps paper-runs + TD3 (all axes)"
python -m src.main --mode sweeps --paper-runs --compute --with-td3 --particles 20 --iters 100 --td3-steps 7000 --out "$OUT/sweeps" --no-status-sync --log-file "$OUT/logs/sweeps.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "calibrate calibrated profile"
python -m scripts.calibrate --methods random kmeans sca --js 1 3 5 2>&1 | Tee-Object -FilePath "$OUT/logs/calibrate.log" -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "calibrate table2 profile"
python -m scripts.calibrate --b-sys 20000 --noise 1e-4 --scope system --cap -1 --methods random kmeans sca --js 1 3 5 2>&1 | Tee-Object -FilePath "$OUT/logs/calibrate_table2.log" -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "placement analysis"
New-Item -ItemType Directory -Force -Path "results/aodt" | Out-Null
Copy-Item -Force "$OUT/aodt/*" "results/aodt/"
python scripts/placement_analysis.py 2>&1 | Tee-Object -FilePath "$OUT/logs/placement_analysis.log" -Append
Copy-Item -Force "results/aodt/placement_analysis.md" "$OUT/aodt/placement_analysis.md"

Log "status sync"
python -m src.status_sync --results $OUT 2>&1 | Tee-Object -FilePath "$OUT/logs/status_sync.log" -Append

Log "=== full audit complete ==="
