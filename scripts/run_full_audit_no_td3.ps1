# Part A audit without TD3 -> results/run_20260901/
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$OUT = "results/run_20260901"
$LOG = "$OUT/logs/full_audit_no_td3.log"
New-Item -ItemType Directory -Force -Path "$OUT/logs" | Out-Null

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts  $msg" | Tee-Object -FilePath $LOG -Append
}

Log "=== full audit (no TD3) start ==="

Log "pytest"
python -m pytest -q --tb=short 2>&1 | Tee-Object -FilePath "$OUT/logs/pytest.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "compare 5-seed (no TD3)"
python -m src.main --mode compare --compute --particles 20 --iters 100 --out "$OUT/compare_5seed" --no-status-sync --log-file "$OUT/logs/compare_5seed.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "compare 20-seed (no TD3)"
python -m src.main --mode compare --paper-runs --compute --particles 20 --iters 100 --out "$OUT/compare_20seed" --no-status-sync --log-file "$OUT/logs/compare_20seed.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "compare table2 5-seed (no TD3)"
python -m src.main --mode compare --radio-profile table2 --compute --particles 20 --iters 100 --out "$OUT/table2_5seed" --no-status-sync --log-file "$OUT/logs/table2_5seed.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "aodt-compare (20 seeds, no TD3)"
python -m src.main --mode aodt-compare --skip-td3 --particles 20 --iters 100 --out "$OUT" --no-status-sync --log-file "$OUT/logs/aodt_compare.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "sweeps paper-runs (no TD3, all axes)"
python -m src.main --mode sweeps --paper-runs --compute --particles 20 --iters 100 --out "$OUT/sweeps" --no-status-sync --log-file "$OUT/logs/sweeps.log"
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

Log "=== full audit (no TD3) complete ==="
