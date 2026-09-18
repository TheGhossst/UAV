Set-Location $PSScriptRoot\..
$env:PYTHONPATH = "src"
python scripts\run_highstat_campaign.py --n-runs 100 --area-m 500
python scripts\run_highstat_campaign.py --n-runs 200 --area-m 100
python scripts\run_highstat_campaign.py --n-runs 200 --area-m 500
python scripts\run_sca_anchor_highstat.py --n-runs 100 --area-m 100 --axis all
python scripts\run_sca_anchor_highstat.py --n-runs 100 --area-m 500 --axis all
python scripts\run_sca_anchor_highstat.py --n-runs 200 --area-m 100 --axis all
python scripts\run_sca_anchor_highstat.py --n-runs 200 --area-m 500 --axis all
python scripts\plot_sweep_n_stat_comparison.py
