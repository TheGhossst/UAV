#!/usr/bin/env bash
# Detached resume (no tmux): survives SSH logout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results
LOG="${ROOT}/results/remote_resume_latest.log"
nohup bash scripts/remote/college_gpu_resume.sh >>"$LOG" 2>&1 &
echo "started PID $!  log=$LOG"
echo "monitor: tail -f $LOG"
