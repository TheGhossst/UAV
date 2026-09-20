# Run full campaign for bandwidth presets (no cap + 8.8 MHz 25% primary + 15% sensitivity)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
Set-Location $PSScriptRoot\..

New-Item -ItemType Directory -Force -Path results | Out-Null

$configs = @(
    @{ preset = "20khz";   cap = $null; out = "results/campaign_20khz.json" },
    @{ preset = "20khz";   cap = 0.25;  out = "results/campaign_20khz_cap25.json" },
    @{ preset = "2.4mhz";  cap = $null; out = "results/campaign_2.4mhz.json" },
    @{ preset = "2.4mhz";  cap = 0.25;  out = "results/campaign_2.4mhz_cap25.json" },
    @{ preset = "8.8mhz";  cap = $null; out = "results/campaign_8.8mhz_n20.json" },
    @{ preset = "8.8mhz";  cap = 0.25;  out = "results/campaign_8.8mhz_cap25_n20.json" },
    @{ preset = "8.8mhz";  cap = 0.15;  out = "results/campaign_8.8mhz_cap15_n20.json" }
)

foreach ($c in $configs) {
    $capArg = if ($null -eq $c.cap) { @() } else { @("--max-bw-share", $c.cap) }
    Write-Host "=== Starting $($c.preset) cap=$($c.cap) -> $($c.out) ===" -ForegroundColor Cyan
    python -m uavdt campaign `
        --axis all `
        --bandwidth-preset $c.preset `
        @capArg `
        --methods random,kmeans,pso,sca `
        --n-runs 20 `
        --seed-start 1 `
        --solver cvxpy `
        --out $c.out
    if ($LASTEXITCODE -ne 0) { throw "Campaign failed: $($c.out)" }
}

Write-Host "All campaigns complete." -ForegroundColor Green
