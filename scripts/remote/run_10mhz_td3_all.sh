#!/usr/bin/env bash
# TD3 on a GPU server. Same nohup ledger as run_10mhz_cap_sweep_all.sh.
#
# Algorithm 2 preset: k-means residual moves, leftover inner bandwidth,
# penalty reward, noise-free policy export, 7000 steps. Training uses CUDA.
# The environment step stays on CPU.
#
# Does not need tmux. --background uses nohup and setsid so the queue keeps
# running after SSH logout. Finished JSON is skipped. Unfinished jobs continue
# from *.checkpoint.json. Every start/ok/fail line is appended to jobs.jsonl.
#
# Install PyTorch in .venv first (not in requirements.txt):
#   .venv/bin/pip install -e ".[td3]"
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
#   .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
#
# From the repo root on Linux:
#   bash scripts/remote/run_10mhz_td3_all.sh --background
#   tail -f results/run_10mhz_td3/current.log
#   bash scripts/remote/run_10mhz_td3_all.sh --status
#   bash scripts/remote/run_10mhz_td3_all.sh --stop
#
# Resume is the same command. Headline cap only (100 m and 500 m, n100 plus
# the campaign axes). Finished cap-25 n100 banks already in
# results/run_10mhz_cap25/{100,500}m/n100_td3.json are copied in and skipped.
#   bash scripts/remote/run_10mhz_td3_all.sh --background --caps 25
#
# One campaign:
#   bash scripts/remote/run_10mhz_td3_all.sh --background \
#     --caps 25 --areas 100 --experiments campaign
#
# Another GPU:
#   bash scripts/remote/run_10mhz_td3_all.sh --background --device cuda:1
#
# If the server checks the file out with Windows line endings:
#   sed -i 's/\r$//' scripts/remote/run_10mhz_td3_all.sh \
#     scripts/remote/run_10mhz_cap_sweep_all.sh

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SOURCE_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/run_10mhz_td3_all.sh [options]

  --background          detach with nohup/setsid (survives SSH logout)
  --status              print the job ledger and output checks
  --stop                stop the detached run (finished checkpoints are kept)
  --dry-run             print the python commands and exit
  --device NAME         cpu, cuda, cuda:N, or auto (default cuda)
  --td3-preset NAME     alg2 (default) or residual-on-sca
  --areas LIST          100,500 or 100x100,500x500
  --experiments LIST    n100,campaign (default: n100 then campaign)
  --caps LIST           none,10,12,15,18,20,22,25 (percent; none = no per-link cap)
  --n-runs N            seeds per sweep point (default 100)
  --n-scenarios N       frozen bank size (default 100)
  --plots               also write n100 figures
  --force               redo selected jobs instead of skipping complete ones
  -h, --help            show this help

Logs:
  results/run_10mhz_td3/jobs.jsonl    resume ledger (start/ok/fail)
  results/run_10mhz_td3/run.log       full output
  results/run_10mhz_td3/current.log   symlink to run.log
  results/run_10mhz_td3/progress.txt  one line per job

The full cap grid retrains every point. --caps 25 is the headline protocol.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export UAVDT_SWEEP_OUT="${ROOT}/results/run_10mhz_td3"
export UAVDT_SWEEP_HELPER="${ROOT}/scripts/remote/run_10mhz_td3_jobs.py"
export UAVDT_SWEEP_METHODS="${UAVDT_SWEEP_METHODS:-td3}"
export UAVDT_TD3_DEVICE="${UAVDT_TD3_DEVICE:-cuda}"
export UAVDT_TD3_PRESET="${UAVDT_TD3_PRESET:-alg2}"
export UAVDT_SWEEP_LAUNCHER="scripts/remote/run_10mhz_td3_all.sh"
export UAVDT_SWEEP_LOG_HINT="results/run_10mhz_td3/current.log"
export UAVDT_SWEEP_BANNER="10 MHz TD3"

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      UAVDT_TD3_DEVICE="${2:?--device needs a value}"
      export UAVDT_TD3_DEVICE
      shift 2
      ;;
    --td3-preset)
      UAVDT_TD3_PRESET="${2:?--td3-preset needs a value}"
      export UAVDT_TD3_PRESET
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ ${#args[@]} -eq 0 ]]; then
  exec bash "${SOURCE_DIR}/run_10mhz_cap_sweep_all.sh"
fi
exec bash "${SOURCE_DIR}/run_10mhz_cap_sweep_all.sh" "${args[@]}"
