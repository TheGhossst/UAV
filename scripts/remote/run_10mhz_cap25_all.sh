#!/usr/bin/env bash
# 10 MHz / 25% cap, every placement method, 100x100 m and 500x500 m.
#
# Methods:
#   random, kmeans (alias kmean), sca,
#   sca_multistart (aliases multistep, multistart),
#   sca_anchor (aliases zenith, anchor)
#
# Experiments, in this order for each area:
#   n100      frozen 100-layout bank (I=10, J=3)
#   campaign  paper Figs. 6-10 axes (J, I, lambda, AoDT, CPU), 100 seeds
#
# Does not need tmux or termux. --background uses nohup and setsid so the
# queue keeps running after SSH logout.
#
# From the repo root on Linux:
#   bash scripts/remote/run_10mhz_cap25_all.sh --background
#   tail -f results/run_10mhz_cap25/current.log
#   bash scripts/remote/run_10mhz_cap25_all.sh --status
#   bash scripts/remote/run_10mhz_cap25_all.sh --stop
#
# Resume is the same command. Finished jobs are skipped when the output JSON
# matches this protocol. Unfinished jobs continue from *.checkpoint.json.
# Every start/ok/fail line is appended to results/run_10mhz_cap25/jobs.jsonl.
#
# If the server checks the file out with Windows line endings:
#   sed -i 's/\r$//' scripts/remote/run_10mhz_cap25_all.sh
#
# Subset:
#   bash scripts/remote/run_10mhz_cap25_all.sh --background \
#     --areas 100x100 --methods random,kmeans,sca,multistep,zenith
#
# Shorter smoke (not the headline sample size):
#   bash scripts/remote/run_10mhz_cap25_all.sh --dry-run
#   bash scripts/remote/run_10mhz_cap25_all.sh --n-runs 2 --n-scenarios 2 \
#     --areas 100 --methods random

set -euo pipefail

SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SOURCE")/../.." && pwd)"
HELPER="${ROOT}/scripts/remote/run_10mhz_cap25_jobs.py"
OUT="${ROOT}/results/run_10mhz_cap25"
LEDGER="${OUT}/jobs.jsonl"
RUNLOG="${OUT}/run.log"
PIDFILE="${OUT}/pid"
CURRENT_JOB_FILE="${OUT}/current_job"
INVOCATION="${OUT}/invocation.json"

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

N_RUNS=100
N_SCENARIOS=100
METHODS="random,kmeans,sca,sca_multistart,sca_anchor"
AREAS="100,500"
EXPERIMENTS="n100,campaign"
FORCE=0
PLOTS=0
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
Usage: bash scripts/remote/run_10mhz_cap25_all.sh [options]

  --background          detach with nohup/setsid (survives SSH logout)
  --status              print the job ledger and output checks
  --stop                stop the detached run (finished checkpoints are kept)
  --dry-run             print the python commands and exit
  --methods LIST        random,kmeans,sca,multistep,zenith (aliases ok)
  --areas LIST          100,500 or 100x100,500x500
  --experiments LIST    n100,campaign (default: n100 then campaign)
  --n-runs N            seeds per sweep point (default 100)
  --n-scenarios N       frozen bank size (default 100)
  --plots               also write n100 figures
  --force               redo selected jobs instead of skipping complete ones
  -h, --help            show this help

Logs:
  results/run_10mhz_cap25/jobs.jsonl    resume ledger (start/ok/fail)
  results/run_10mhz_cap25/run.log       full output
  results/run_10mhz_cap25/current.log   symlink to run.log
  results/run_10mhz_cap25/progress.txt  one line per job
EOF
}

log() {
  local line
  mkdir -p "$OUT"
  line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$line" | tee -a "$RUNLOG"
}

refresh_progress() {
  "$PY" "$HELPER" status \
    --methods "$METHODS" \
    --areas "$AREAS" \
    --experiments "$EXPERIMENTS" \
    --n-runs "$N_RUNS" \
    --n-scenarios "$N_SCENARIOS" \
    --ledger "$LEDGER" \
    --root "$OUT" \
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
  local kind="$1" area="$2" method="$3" out_rel="$4"
  local argv_out arg
  local -a py=(
    "$PY" "$HELPER" argv
    --kind "$kind"
    --area "$area"
    --method "$method"
    --out "$out_rel"
    --n-runs "$N_RUNS"
    --n-scenarios "$N_SCENARIOS"
  )
  [[ "$PLOTS" == 1 ]] && py+=(--plots)
  [[ "$FORCE" == 1 ]] && py+=(--force)
  argv_out=$("${py[@]}")
  CMD=("$PY")
  while IFS= read -r arg || [[ -n "${arg:-}" ]]; do
    arg="${arg//$'\r'/}"
    [[ -z "$arg" ]] && continue
    CMD+=("$arg")
  done <<< "$argv_out"
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
  # The worker's own pid is not "another" run.
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
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/pip install -r requirements.txt" >&2
    echo "  export PY=\"\$PWD/.venv/bin/python\"" >&2
    exit 1
  fi
}

launch_background() {
  local -a forward=(
    --worker
    --n-runs "$N_RUNS"
    --n-scenarios "$N_SCENARIOS"
    --methods "$METHODS"
    --areas "$AREAS"
    --experiments "$EXPERIMENTS"
  )
  [[ "$FORCE" == 1 ]] && forward+=(--force)
  [[ "$PLOTS" == 1 ]] && forward+=(--plots)
  mkdir -p "$OUT"
  if already_running; then
    echo "already running (pid $(tr -d '[:space:]' < "$PIDFILE"))"
    echo "status: bash scripts/remote/run_10mhz_cap25_all.sh --status"
    echo "log:    tail -f results/run_10mhz_cap25/current.log"
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
    if [[ -f "${OUT}/console.log" ]] && grep -q "Cannot import uavdt" "${OUT}/console.log"; then
      echo "Python environment failed; see ${OUT}/console.log" >&2
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
  echo "log:    tail -f results/run_10mhz_cap25/current.log"
  echo "status: bash scripts/remote/run_10mhz_cap25_all.sh --status"
  echo "stop:   bash scripts/remote/run_10mhz_cap25_all.sh --stop"
  echo "resume: bash scripts/remote/run_10mhz_cap25_all.sh --background"
}

stop_run() {
  if ! already_running; then
    echo "not running"
    rm -f "$PIDFILE"
    exit 0
  fi
  local pid
  pid="$(tr -d '[:space:]' < "$PIDFILE")"
  # Freeze the shell first so it cannot start the next job while children die.
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
  echo "resume: bash scripts/remote/run_10mhz_cap25_all.sh --background"
}

show_status() {
  mkdir -p "$OUT"
  if [[ "$EXPLICIT" != 1 && -f "$INVOCATION" ]]; then
    "$PY" "$HELPER" status \
      --from-invocation "$INVOCATION" \
      --ledger "$LEDGER" \
      --root "$OUT" \
      --pidfile "$PIDFILE" \
      --current-job "$CURRENT_JOB_FILE" \
      --progress "${OUT}/progress.txt"
  else
    "$PY" "$HELPER" status \
      --methods "$METHODS" \
      --areas "$AREAS" \
      --experiments "$EXPERIMENTS" \
      --n-runs "$N_RUNS" \
      --n-scenarios "$N_SCENARIOS" \
      --ledger "$LEDGER" \
      --root "$OUT" \
      --pidfile "$PIDFILE" \
      --current-job "$CURRENT_JOB_FILE" \
      --progress "${OUT}/progress.txt"
  fi
}

run_one() {
  local job="$1" kind="$2" area="$3" method="$4" out_rel="$5"
  local settings last bind_rc t0 rc cmd_str safe joblog waited reaped
  settings="${out_rel%.json}.settings.json"
  safe="${job//\//_}"
  joblog="${OUT}/logs/${safe}.log"

  if [[ "$FORCE" != 1 ]]; then
    if "$PY" "$HELPER" complete \
      --path "$out_rel" \
      --kind "$kind" \
      --method "$method" \
      --area "$area" \
      --n-runs "$N_RUNS" \
      --n-scenarios "$N_SCENARIOS"; then
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
    if [[ -f "${ROOT}/${out_rel}" ]]; then
      mv -f "${ROOT}/${out_rel}" "${ROOT}/${out_rel}.prev"
    fi
    rm -f \
      "${ROOT}/${out_rel%.json}.checkpoint.json" \
      "${ROOT}/${out_rel%.json}.csv" \
      "${ROOT}/${out_rel%.json}_summary.csv" \
      "${ROOT}/${settings}"
  fi

  set +e
  if [[ "$FORCE" == 1 ]]; then
    "$PY" "$HELPER" bind-settings \
      --path "$settings" --kind "$kind" --area "$area" --method "$method" \
      --n-runs "$N_RUNS" --n-scenarios "$N_SCENARIOS" --force
  else
    "$PY" "$HELPER" bind-settings \
      --path "$settings" --kind "$kind" --area "$area" --method "$method" \
      --n-runs "$N_RUNS" --n-scenarios "$N_SCENARIOS"
  fi
  bind_rc=$?
  set -e
  if [[ "$bind_rc" -eq 3 ]]; then
    log "settings mismatch ${job} (see stderr above; not mixing protocols)"
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

  prepare_cmd "$kind" "$area" "$method" "$out_rel"
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
    if [[ -f "${ROOT}/${out_rel}" ]] && "$PY" "$HELPER" complete \
      --path "$out_rel" --kind "$kind" --method "$method" --area "$area" \
      --n-runs "$N_RUNS" --n-scenarios "$N_SCENARIOS" >/dev/null 2>&1; then
      waited=$((waited + 5))
      if [[ "$waited" -ge 15 ]]; then
        log "output ready for ${job}; process ${CURRENT_CHILD} still running (${waited}s)"
      fi
      if [[ "$waited" -ge 60 ]]; then
        log "stopping ${CURRENT_CHILD} after ${job} was written; continuing the queue"
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
  if [[ "$rc" -eq 0 ]] && "$PY" "$HELPER" complete \
    --path "$out_rel" --kind "$kind" --method "$method" --area "$area" \
    --n-runs "$N_RUNS" --n-scenarios "$N_SCENARIOS"; then
    record "$job" ok "out=${out_rel}" "seconds=$((SECONDS - t0))"
    log "ok ${job} in $((SECONDS - t0))s"
    log "merging finished methods"
    "$PY" "$HELPER" merge \
      --methods "$METHODS" --areas "$AREAS" --experiments "$EXPERIMENTS" \
      --n-runs "$N_RUNS" --n-scenarios "$N_SCENARIOS" --root "$OUT" \
      >>"$RUNLOG" 2>&1 || log "merge skipped after ${job}"
    refresh_progress
    return 0
  fi
  record "$job" fail "out=${out_rel}" "code=${rc}" "seconds=$((SECONDS - t0))"
  log "fail ${job} exit=${rc} after $((SECONDS - t0))s (rerun to resume)"
  refresh_progress
  return 1
}

run_queue() {
  local plan_out line job kind area method out_rel
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

  "$PY" "$HELPER" save-invocation \
    --path "$INVOCATION" \
    --methods "$METHODS" \
    --areas "$AREAS" \
    --experiments "$EXPERIMENTS" \
    --n-runs "$N_RUNS" \
    --n-scenarios "$N_SCENARIOS"

  plan_out="$("$PY" "$HELPER" plan \
    --methods "$METHODS" \
    --areas "$AREAS" \
    --experiments "$EXPERIMENTS" \
    --n-runs "$N_RUNS" \
    --n-scenarios "$N_SCENARIOS")"

  log "=== 10 MHz / 25% cap  areas=${AREAS}  methods=${METHODS}  experiments=${EXPERIMENTS}  n_runs=${N_RUNS}  n_scenarios=${N_SCENARIOS} ==="
  log "py=$PY  root=$ROOT"
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    line="${line//$'\r'/}"
    [[ -z "$line" ]] && continue
    IFS=$'\t' read -r job kind area method out_rel <<< "$line"
    if [[ -z "${job}" || -z "${kind}" || -z "${area}" || -z "${method}" || -z "${out_rel}" ]]; then
      log "bad plan line: ${line}"
      failed=$((failed + 1))
      continue
    fi
    if ! run_one "$job" "$kind" "$area" "$method" "$out_rel"; then
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
  local plan_out line job kind area method out_rel
  plan_out="$("$PY" "$HELPER" plan \
    --methods "$METHODS" \
    --areas "$AREAS" \
    --experiments "$EXPERIMENTS" \
    --n-runs "$N_RUNS" \
    --n-scenarios "$N_SCENARIOS")"
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    line="${line//$'\r'/}"
    [[ -z "$line" ]] && continue
    IFS=$'\t' read -r job kind area method out_rel <<< "$line"
    prepare_cmd "$kind" "$area" "$method" "$out_rel"
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
    --plots) PLOTS=1; EXPLICIT=1; shift ;;
    --methods) METHODS="${2:?--methods needs a value}"; EXPLICIT=1; shift 2 ;;
    --areas) AREAS="${2:?--areas needs a value}"; EXPLICIT=1; shift 2 ;;
    --experiments) EXPERIMENTS="${2:?--experiments needs a value}"; EXPLICIT=1; shift 2 ;;
    --n-runs) N_RUNS="${2:?--n-runs needs a value}"; EXPLICIT=1; shift 2 ;;
    --n-scenarios) N_SCENARIOS="${2:?--n-scenarios needs a value}"; EXPLICIT=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$N_RUNS" =~ ^[0-9]+$ ]] || [[ "$N_RUNS" -lt 1 ]]; then
  echo "--n-runs must be a positive integer" >&2
  exit 2
fi
if ! [[ "$N_SCENARIOS" =~ ^[0-9]+$ ]] || [[ "$N_SCENARIOS" -lt 1 ]]; then
  echo "--n-scenarios must be a positive integer" >&2
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
