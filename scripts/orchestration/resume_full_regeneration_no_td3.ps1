# Resume no-TD3 full regen from campaign checkpoints (kill stale job first).
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
Set-Location $PSScriptRoot\..

$log = "results/full_regeneration_no_td3_live.log"
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "Stopping prior python jobs for this repo (if any)..."
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'run_full_regeneration_no_td3|run_highstat_campaign' } |
    ForEach-Object {
        Write-Host "  stop PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 2

Write-Host "Resuming from highstat checkpoints -> $log"
python -u scripts/orchestration/run_full_regeneration_no_td3.py --resume --log $log
if ($LASTEXITCODE -ne 0) { throw "resume failed with exit $LASTEXITCODE" }
