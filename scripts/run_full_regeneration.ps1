# Full test + experiment regeneration (docs/RESULTS.md §9)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
Set-Location $PSScriptRoot\..

$log = "results/full_regeneration_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
New-Item -ItemType Directory -Force -Path results | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

Log "=== pytest ==="
python -m pytest -q --tb=no
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

Log "=== check_bsys_20khz ==="
python scripts/check_bsys_20khz.py --n-runs 20 --out results/check_bsys_20khz.json

Log "=== primary campaign 8.8mhz cap25 si12k ==="
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 --methods random,kmeans,pso,sca --n-runs 20 --seed-start 1 --solver cvxpy --out results/campaign_8.8mhz_cap25_si12k.json

Log "=== init repair ==="
python scripts/rerun_init_repair_points.py

Log "=== cap15 sensitivity ==="
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.15 --methods random,kmeans,pso,sca --n-runs 20 --seed-start 1 --solver cvxpy --out results/campaign_8.8mhz_cap15_n20.json

Log "=== 500m field test ==="
python -m uavdt campaign --axis all --bandwidth-preset 8.8mhz --max-bw-share 0.25 --methods random,kmeans,pso,sca --n-runs 20 --seed-start 1 --solver cvxpy --area-m 500 --out results/campaign_8.8mhz_cap25_si12k_500m.json

Log "=== bandwidth preset sweep ==="
& "$PSScriptRoot\run_all_bandwidth_campaigns.ps1"

Log "=== fig11 cap25 ==="
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --n-runs 20 --seed-start 1 --out results/fig11_8.8mhz_cap25_si12k.json

Log "=== fig11 cap15 ==="
python -m uavdt fig11 --bandwidth-preset 8.8mhz --max-bw-share 0.15 --n-runs 20 --seed-start 1 --out results/fig11_8.8mhz_cap15.json

Log "=== spot-validate ==="
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-iterations 30
python -m uavdt spot-validate --seed 1 --bandwidth-preset 8.8mhz --max-bw-share 0.25 --max-iterations 30

Log "=== sca-joint probe ==="
python scripts/run_sca_joint_campaign.py

Log "=== tk08 followup ==="
python scripts/run_tk08_followup.py

Log "=== analysis ==="
python scripts/paired_winrate.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/paired_winrate.py results/campaign_8.8mhz_cap15_n20.json
python scripts/analyze_sca_vs_random_losses.py results/campaign_8.8mhz_cap25_si12k.json
python scripts/analyze_campaigns.py
python scripts/plot_paper_figures.py
python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap15_n20.json --fig11 results/fig11_8.8mhz_cap15.json --out-dir results/figures/cap15
python scripts/plot_paper_figures.py --campaign results/campaign_8.8mhz_cap25_si12k_500m.json --out-dir results/figures/500m

Log "=== DONE ==="
