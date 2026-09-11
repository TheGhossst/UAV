# Full §VII-axis campaigns at 7 MHz: no cap, 15%, 25%. No TD3.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
Set-Location $PSScriptRoot\..

New-Item -ItemType Directory -Force -Path results | Out-Null

$configs = @(
    @{ cap = $null; out = "results/campaign_7mhz_n20.json" },
    @{ cap = 0.15;  out = "results/campaign_7mhz_cap15_n20.json" },
    @{ cap = 0.25;  out = "results/campaign_7mhz_cap25_n20.json" }
)

foreach ($c in $configs) {
    $capArg = if ($null -eq $c.cap) { @() } else { @("--max-bw-share", $c.cap) }
    Write-Host "=== Starting 7mhz cap=$($c.cap) -> $($c.out) ===" -ForegroundColor Cyan
    python -m uavdt campaign `
        --axis all `
        --bandwidth-preset 7mhz `
        @capArg `
        --methods random,kmeans,pso,sca `
        --n-runs 20 `
        --seed-start 1 `
        --solver cvxpy `
        --out $c.out
    if ($LASTEXITCODE -ne 0) { throw "Campaign failed: $($c.out)" }
}

Write-Host "All 7 MHz campaigns complete." -ForegroundColor Green
