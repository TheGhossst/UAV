#!/usr/bin/env bash
# Resumable queue for the still-open 10 MHz rows.
# Same nohup ledger as run_10mhz_missing_all.sh. Own output tree.
#
# Finished JSON is skipped. Placement jobs continue from *.checkpoint.json.
# CMA-ES has no checkpoint: a kill restarts that one script.
# Nothing is written into run_10mhz_cap25, run_10mhz_cap_sweep, run_10mhz_td3,
# run_10mhz_controls, run_10mhz_missing, or results/residual_on_sca.
# T_k = 0.8 s is not in this queue. Cap 25% TD3 and dynamic-anchor campaigns
# already exist and are not repeated.
#
# From the repo root on Linux:
#   bash scripts/remote/run_10mhz_open_all.sh --background
#   tail -f results/run_10mhz_open/current.log
#   bash scripts/remote/run_10mhz_open_all.sh --status
#   bash scripts/remote/run_10mhz_open_all.sh --stop
#
# Resume is the same --background command.
#
# CPU-only subset (no TD3 campaigns):
#   bash scripts/remote/run_10mhz_open_all.sh --background \
#     --groups anchor-dyn-campaign,pso-lp,los-deg,cmaes
#
# Print every command without running:
#   bash scripts/remote/run_10mhz_open_all.sh --dry-run
#
# The learned proposal policy is not queued: its trainer is not in this repo.
#
# If the server checks the file out with Windows line endings:
#   sed -i 's/\r$//' scripts/remote/run_10mhz_open_all.sh

set -euo pipefail

SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SOURCE")/../.." && pwd)"
HELPER="${ROOT}/scripts/remote/run_10mhz_open_jobs.py"
OUT="${ROOT}/results/run_10mhz_open"
LEDGER="${OUT}/jobs.jsonl"
RUNLOG="${OUT}/run.log"
PIDFILE="${OUT}/pid"
CURRENT_JOB_FILE="${OUT}/current_job"
INVOCATION="${OUT}/invocation.json"
LAUNCHER="scripts/remote/run_10mhz_open_all.sh"
LOG_HINT="results/run_10mhz_open/current.log"

cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ -z "${PY:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PY="${ROOT}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    PY="python"
  fi
fi

JOB_GROUPS="anchor-dyn-campaign,pso-lp,los-deg,cmaes,td3-campaign"
N_RUNS=100
ANALYSIS_N=20
DEVICE="${UAVDT_TD3_DEVICE:-cuda}"
TD3_PRESET="${UAVDT_TD3_PRESET:-alg2}"
FORCE=0
BACKGROUND=0
WORKER=0
STATUS=0
STOP=0
DRY_RUN=0
EXPLICIT=0
CURRENT_JOB=""
CURRENT_CHILD=""
CURRENT_MIRROR=""

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/run_10mhz_open_all.sh [options]

  --background          detach with nohup/setsid (survives SSH logout)
  --status              print the job ledger and output checks
  --stop                stop the detached run (finished checkpoints are kept)
  --dry-run             print the python commands and exit
  --groups LIST         anchor-dyn-campaign,pso-lp,los-deg,cmaes,td3-campaign
                        (default all of those). all is the same list.
  --n-runs N            seeds for placement and TD3 (default 100)
  --analysis-n N        seeds for CMA-ES (default 20)
  --device NAME         cpu, cuda, or cuda:N (default cuda; TD3 campaigns only)
  --td3-preset NAME     alg2 (default) for td3-campaign
  --force               redo selected jobs instead of skipping complete ones
  -h, --help            show this help

Logs:
  results/run_10mhz_open/jobs.jsonl
  results/run_10mhz_open/run.log
  results/run_10mhz_open/current.log
  results/run_10mhz_open/progress.txt

Default queue is 58 jobs. TD3 campaigns are the long GPU part.
Not queued: T_k = 0.8 s, the learned proposal policy (no trainer in this
repo), cap-25 TD3 campaigns, and cap-25 dynamic-anchor campaigns.
EOF
}

selector() {
  SELECTOR=(
    --groups "$JOB_GROUPS"
    --n-runs "$N_RUNS"
    --analysis-n "$ANALYSIS_N"
    --device "$DEVICE"
    --td3-preset "$TD3_PRESET"
  )
}

log() {
  local line
  mkdir -p "$OUT"
  line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$line" | tee -a "$RUNLOG"
}

refresh_progress() {
  selector
  "$PY" "$HELPER" status \
    "${SELECTOR[@]}" \
    --ledger "$LEDGER" \
    --pidfile "$PIDFILE" \
    --current-job "$CURRENT_JOB_FILE" \
    --progress "${OUT}/progress.txt" >/dev/null || true
}

record() {
  local job="$1" status="$2"
  shift 2
  local -a fields=()
  local item
  for item in "$@"; do
    fields+=(--field "$item")
  done
  if [[ ${#fields[@]} -gt 0 ]]; then
    "$PY" "$HELPER" record --ledger "$LEDGER" --job "$job" --status "$status" "${fields[@]}"
  else
    "$PY" "$HELPER" record --ledger "$LEDGER" --job "$job" --status "$status"
  fi
}

prepare_cmd() {
  local job="$1"
  local arg
  selector
  local -a py=("$PY" "$HELPER" argv --job "$job" "${SELECTOR[@]}")
  [[ "$FORCE" == 1 ]] && py+=(--force)
  CMD=("$PY")
  while IFS= read -r arg || [[ -n "${arg:-}" ]]; do
    arg="${arg//$'\r'/}"
    [[ -z "$arg" ]] && continue
    CMD+=("$arg")
  done < <("${py[@]}")
}

kill_descendants() {
  local pid="$1" child kids
  kids="$(ps -o pid= --ppid "$pid" 2>/dev/null || true)"
  for child in $kids; do
    kill_descendants "$child"
    kill -TERM "$child" 2>/dev/null || true
  done
}

already_running() {
  local old
  [[ -f "$PIDFILE" ]] || return 1
  old="$(tr -d '[:space:]' < "$PIDFILE" || true)"
  [[ "$old" =~ ^[0-9]+$ ]] || return 1
  [[ "$old" == "$$" ]] && return 1
  kill -0 "$old" 2>/dev/null
}

cleanup() {
  if [[ -f "$PIDFILE" ]]; then
    local old
    old="$(tr -d '[:space:]' < "$PIDFILE" || true)"
    if [[ "$old" == "$$" ]]; then
      rm -f "$PIDFILE"
    fi
  fi
  rm -f "$CURRENT_JOB_FILE"
}

on_term() {
  trap '' TERM INT
  log "stopped; rerun the same command to resume from the ledger and checkpoints"
  if [[ -n "${CURRENT_CHILD}" ]]; then
    kill_descendants "$CURRENT_CHILD" || true
    kill -TERM "$CURRENT_CHILD" 2>/dev/null || true
  fi
  if [[ -n "${CURRENT_MIRROR}" ]]; then
    kill "$CURRENT_MIRROR" 2>/dev/null || true
  fi
  if [[ -n "${CURRENT_JOB}" ]]; then
    record "$CURRENT_JOB" interrupted "reason=signal" || true
  fi
  cleanup
  exit 143
}

preflight() {
  if ! "$PY" -c "import uavdt, cvxpy" >/dev/null 2>&1; then
    echo "Cannot import uavdt and cvxpy with: $PY" >&2
    echo "On the server, from the repo root:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
  fi
  if "$PY" "$HELPER" needs-cuda --groups "$JOB_GROUPS"; then
    if ! "$PY" -c "import torch" >/dev/null 2>&1; then
      echo "Cannot import torch with: $PY" >&2
      echo "TD3 campaigns need torch. Install it, or drop td3-campaign from --groups." >&2
      exit 1
    fi
    if [[ "$DEVICE" == cuda || "$DEVICE" == cuda:* ]]; then
      if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "CUDA is not available to ${PY} (device ${DEVICE})." >&2
        echo "Or run: bash ${LAUNCHER} --background --device cpu ..." >&2
        exit 1
      fi
    fi
  fi
}

launch_background() {
  local -a forward=(
    --worker
    --groups "$JOB_GROUPS"
    --n-runs "$N_RUNS"
    --analysis-n "$ANALYSIS_N"
    --device "$DEVICE"
    --td3-preset "$TD3_PRESET"
  )
  [[ "$FORCE" == 1 ]] && forward+=(--force)
  mkdir -p "$OUT"
  if already_running; then
    echo "already running (pid $(tr -d '[:space:]' < "$PIDFILE"))"
    echo "status: bash ${LAUNCHER} --status"
    echo "log:    tail -f ${LOG_HINT}"
    exit 0
  fi
  rm -f "$PIDFILE"
  : > "${OUT}/console.log"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$SOURCE" "${forward[@]}" >>"${OUT}/console.log" 2>&1 </dev/null &
  else
    echo "setsid not found; using nohup (SIGHUP is ignored)" >&2
    nohup bash "$SOURCE" "${forward[@]}" >>"${OUT}/console.log" 2>&1 </dev/null &
  fi
  disown >/dev/null 2>&1 || true
  local _
  for _ in $(seq 1 150); do
    if [[ -f "$PIDFILE" ]] && already_running; then
      break
    fi
    if [[ -f "${OUT}/console.log" ]] && grep -Eq "Cannot import uavdt|Cannot import torch|CUDA is not available|unknown group" "${OUT}/console.log"; then
      echo "startup failed; see ${OUT}/console.log" >&2
      exit 1
    fi
    if [[ -f "$PIDFILE" ]] && ! already_running; then
      echo "process exited immediately; see ${OUT}/console.log" >&2
      exit 1
    fi
    sleep 0.2
  done
  if ! already_running; then
    echo "did not stay up; see ${OUT}/console.log" >&2
    exit 1
  fi
  echo "started pid $(tr -d '[:space:]' < "$PIDFILE")"
  echo "log:    tail -f ${LOG_HINT}"
  echo "status: bash ${LAUNCHER} --status"
  echo "stop:   bash ${LAUNCHER} --stop"
  echo "resume: bash ${LAUNCHER} --background"
}

stop_run() {
  if ! already_running; then
    echo "not running"
    rm -f "$PIDFILE"
    exit 0
  fi
  local pid
  pid="$(tr -d '[:space:]' < "$PIDFILE")"
  kill -STOP "$pid" 2>/dev/null || true
  kill_descendants "$pid"
  sleep 1
  kill_descendants "$pid"
  kill -KILL "$pid" 2>/dev/null || true
  sleep 0.2
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PIDFILE" "$CURRENT_JOB_FILE"
  fi
  echo "stopped pid ${pid}"
  echo "resume: bash ${LAUNCHER} --background"
}

show_status() {
  mkdir -p "$OUT"
  selector
  if [[ "$EXPLICIT" != 1 && -f "$INVOCATION" ]]; then
    "$PY" "$HELPER" status \
      --from-invocation "$INVOCATION" \
      --ledger "$LEDGER" \
      --pidfile "$PIDFILE" \
      --current-job "$CURRENT_JOB_FILE" \
      --progress "${OUT}/progress.txt"
  else
    "$PY" "$HELPER" status \
      "${SELECTOR[@]}" \
      --ledger "$LEDGER" \
      --pidfile "$PIDFILE" \
      --current-job "$CURRENT_JOB_FILE" \
      --progress "${OUT}/progress.txt"
  fi
}

run_one() {
  local job="$1" out_rel="$2"
  local bind_rc t0 rc cmd_str safe joblog waited reaped last
  safe="${job//\//_}"
  joblog="${OUT}/logs/${safe}.log"

  selector
  if [[ "$FORCE" != 1 ]]; then
    if "$PY" "$HELPER" complete --job "$job" "${SELECTOR[@]}"; then
      last="$("$PY" "$HELPER" last --ledger "$LEDGER" --job "$job" || true)"
      last="${last//$'\r'/}"
      if [[ "$last" != "ok" ]]; then
        record "$job" ok "out=${out_rel}" "reason=output"
      fi
      log "skip complete ${job}"
      refresh_progress
      return 0
    fi
  else
    "$PY" "$HELPER" discard --job "$job" "${SELECTOR[@]}"
  fi

  set +e
  if [[ "$FORCE" == 1 ]]; then
    "$PY" "$HELPER" bind-settings --job "$job" "${SELECTOR[@]}" --force
  else
    "$PY" "$HELPER" bind-settings --job "$job" "${SELECTOR[@]}"
  fi
  bind_rc=$?
  set -e
  if [[ "$bind_rc" -eq 3 ]]; then
    log "settings mismatch ${job}"
    record "$job" fail "out=${out_rel}" "reason=settings_mismatch"
    refresh_progress
    return 1
  fi
  if [[ "$bind_rc" -ne 0 ]]; then
    log "could not write settings for ${job}"
    record "$job" fail "out=${out_rel}" "reason=settings"
    refresh_progress
    return 1
  fi

  prepare_cmd "$job"
  cmd_str="$(printf '%q ' "${CMD[@]}")"
  CURRENT_JOB="$job"
  printf '%s\n' "$job" > "$CURRENT_JOB_FILE"
  mkdir -p "${OUT}/logs"
  record "$job" start "out=${out_rel}" "cmd=${cmd_str}"
  log "start ${job}"
  log "cmd ${cmd_str}"
  t0=$SECONDS
  set +e
  touch "$joblog"
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL tail -n 0 -f "$joblog" >>"$RUNLOG" &
  else
    tail -n 0 -f "$joblog" >>"$RUNLOG" &
  fi
  CURRENT_MIRROR=$!
  "${CMD[@]}" >>"$joblog" 2>&1 &
  CURRENT_CHILD=$!
  waited=0
  reaped=0
  rc=0
  while kill -0 "$CURRENT_CHILD" 2>/dev/null; do
    if "$PY" "$HELPER" complete --job "$job" "${SELECTOR[@]}" >/dev/null 2>&1; then
      waited=$((waited + 5))
      if [[ "$waited" -ge 60 ]]; then
        log "stopping ${CURRENT_CHILD} after ${job} was written"
        kill_descendants "$CURRENT_CHILD" || true
        kill -TERM "$CURRENT_CHILD" 2>/dev/null || true
        sleep 1
        kill -KILL "$CURRENT_CHILD" 2>/dev/null || true
        wait "$CURRENT_CHILD" 2>/dev/null || true
        rc=0
        reaped=1
        break
      fi
    else
      waited=0
    fi
    sleep 5
  done
  if [[ "$reaped" -eq 0 ]]; then
    wait "$CURRENT_CHILD"
    rc=$?
  fi
  kill "$CURRENT_MIRROR" 2>/dev/null || true
  wait "$CURRENT_MIRROR" 2>/dev/null || true
  CURRENT_CHILD=""
  CURRENT_MIRROR=""
  set -e
  log "command ${job} exit=${rc}"
  if [[ "$rc" -eq 0 ]] && "$PY" "$HELPER" complete --job "$job" "${SELECTOR[@]}"; then
    record "$job" ok "out=${out_rel}" "seconds=$((SECONDS - t0))"
    log "ok ${job} in $((SECONDS - t0))s"
    refresh_progress
    return 0
  fi
  record "$job" fail "out=${out_rel}" "code=${rc}" "seconds=$((SECONDS - t0))"
  log "fail ${job} exit=${rc} after $((SECONDS - t0))s (rerun to resume)"
  refresh_progress
  return 1
}

run_queue() {
  local plan_out line job group out_rel check
  local failed=0
  mkdir -p "${OUT}/logs"
  if already_running; then
    echo "already running (pid $(tr -d '[:space:]' < "$PIDFILE"))" >&2
    exit 1
  fi
  echo $$ > "$PIDFILE"
  ln -sfn "run.log" "${OUT}/current.log" || true
  trap on_term TERM INT
  trap '' HUP
  trap cleanup EXIT

  selector
  "$PY" "$HELPER" save-invocation --path "$INVOCATION" "${SELECTOR[@]}"
  plan_out="$("$PY" "$HELPER" plan "${SELECTOR[@]}")"
  log "=== 10 MHz open results  groups=${JOB_GROUPS}  n_runs=${N_RUNS}  analysis_n=${ANALYSIS_N}  device=${DEVICE} ==="
  log "py=$PY  root=$ROOT"
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    line="${line//$'\r'/}"
    [[ -z "$line" ]] && continue
    IFS=$'\t' read -r job group out_rel check <<< "$line"
    if [[ -z "${job}" || -z "${group}" || -z "${out_rel}" || -z "${check}" ]]; then
      log "bad plan line: ${line}"
      failed=$((failed + 1))
      continue
    fi
    if ! run_one "$job" "$out_rel"; then
      failed=$((failed + 1))
    fi
  done <<< "$plan_out"

  CURRENT_JOB=""
  rm -f "$CURRENT_JOB_FILE"
  refresh_progress
  if [[ "$failed" -ne 0 ]]; then
    log "finished with ${failed} failed job(s); rerun the same command to resume"
    exit 1
  fi
  log "ALL DONE"
}

dry_run() {
  local plan_out line job group out_rel check
  selector
  plan_out="$("$PY" "$HELPER" plan "${SELECTOR[@]}")"
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    line="${line//$'\r'/}"
    [[ -z "$line" ]] && continue
    IFS=$'\t' read -r job group out_rel check <<< "$line"
    prepare_cmd "$job"
    printf '%q ' "${CMD[@]}"
    printf '\n'
  done <<< "$plan_out"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background) BACKGROUND=1; shift ;;
    --worker) WORKER=1; shift ;;
    --status) STATUS=1; shift ;;
    --stop) STOP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; EXPLICIT=1; shift ;;
    --groups) JOB_GROUPS="${2:?--groups needs a value}"; EXPLICIT=1; shift 2 ;;
    --n-runs) N_RUNS="${2:?--n-runs needs a value}"; EXPLICIT=1; shift 2 ;;
    --analysis-n) ANALYSIS_N="${2:?--analysis-n needs a value}"; EXPLICIT=1; shift 2 ;;
    --device) DEVICE="${2:?--device needs a value}"; EXPLICIT=1; shift 2 ;;
    --td3-preset) TD3_PRESET="${2:?--td3-preset needs a value}"; EXPLICIT=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$N_RUNS" =~ ^[0-9]+$ ]] || [[ "$N_RUNS" -lt 1 ]]; then
  echo "--n-runs must be a positive integer" >&2
  exit 2
fi
if ! [[ "$ANALYSIS_N" =~ ^[0-9]+$ ]] || [[ "$ANALYSIS_N" -lt 1 ]]; then
  echo "--analysis-n must be a positive integer" >&2
  exit 2
fi

if [[ "$STATUS" == 1 ]]; then
  show_status
  exit 0
fi
if [[ "$STOP" == 1 ]]; then
  stop_run
  exit 0
fi
if [[ "$DRY_RUN" == 1 ]]; then
  dry_run
  exit 0
fi
preflight
if [[ "$WORKER" != 1 && "$BACKGROUND" == 1 ]]; then
  launch_background
  exit 0
fi
run_queue
