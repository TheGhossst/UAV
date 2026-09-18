# Zenith-anchor sweeps for n=100 and n=200 (all axes, both fields). Checkpoint-safe.
Set-Location $PSScriptRoot\..
$env:PYTHONPATH = "src"
foreach ($n in @(100, 200)) {
  foreach ($area in @(100, 500)) {
    python scripts\run_sca_anchor_highstat.py --n-runs $n --area-m $area --axis all
  }
}
python scripts\plot_sweep_n_stat_comparison.py
Set-Location latex\ppt
latexmk -pdf -g -interaction=nonstopmode main.tex
