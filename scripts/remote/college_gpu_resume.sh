#!/usr/bin/env bash
# Finish anchor high-stat + n200 PPT on a remote Linux host (no sudo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"
LOG="${ROOT}/results/remote_resume_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" "${ROOT}/results/remote_resume_latest.log" 2>/dev/null || true

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [[ ! -d "${ROOT}/src/uavdt" ]]; then
  echo "missing ${ROOT}/src/uavdt — sync repo first" >&2
  exit 1
fi

PY="${PY:-python3}"
log "=== college_gpu_resume ROOT=$ROOT ==="

log "=== anchor high-stat n200 @ 500 m (resume checkpoints) ==="
for axis in aodt lambda iots uavs; do  # skips axes whose merged JSON already exists
  out="${ROOT}/results/campaign_sca_anchor_${axis}_n200_500m.json"
  if [[ -f "$out" ]]; then
    log "skip existing $out"
    continue
  fi
  log "axis=$axis area=500 n_runs=200"
  "$PY" -u scripts/campaigns/run_sca_anchor_highstat.py --n-runs 200 --area-m 500 --axis "$axis" 2>&1 | tee -a "$LOG"
done

log "=== n200 PPT bank eval (no TD3) ==="
"$PY" -u scripts/campaigns/run_n200_ppt_eval.py --skip-td3 2>&1 | tee -a "$LOG"

log "=== plots ==="
"$PY" scripts/plot/plot_sweep_n_stat_comparison.py 2>&1 | tee -a "$LOG"
"$PY" scripts/plot/plot_paper_figures.py 2>&1 | tee -a "$LOG"
"$PY" scripts/plot/plot_ppt_all_methods_n100.py 2>&1 | tee -a "$LOG" || true
if [[ -f scripts/plot/plot_ppt_n100_500m.py ]]; then
  "$PY" scripts/plot/plot_ppt_n100_500m.py 2>&1 | tee -a "$LOG" || true
fi

log "=== DONE ==="
