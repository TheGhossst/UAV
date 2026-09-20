# Waits for 100 m n200 TD3, refreshes PPT figures, runs 500 m TD3, rebuilds slides.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$eval = "results\n200\eval_td3.json"
Write-Host "Waiting for $eval (td3 n=200)..."
while ($true) {
    if (Test-Path $eval) {
        $n = python -c "import json;print(json.load(open(r'$eval'))['by_method']['td3']['n'])" 2>$null
        if ($n -eq "200") { break }
    }
    Start-Sleep -Seconds 120
}

Write-Host "100 m TD3 complete. Replotting..."
python scripts\run_n200_ppt_eval.py --plot-only --only 100m

Write-Host "Starting 500 m TD3..."
python scripts\run_n200_ppt_eval.py --only-td3 --only 500m

Write-Host "Final PPT figures..."
python scripts\run_n200_ppt_eval.py --plot-only

Set-Location latex\ppt
latexmk -pdf -interaction=nonstopmode main.tex
Write-Host "Done: latex\ppt\main.pdf"
