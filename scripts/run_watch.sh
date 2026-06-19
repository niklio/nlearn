#!/usr/bin/env bash
# run_watch.sh NAME MAX_SECS STALL_SECS -- CMD...
#
# Runs CMD with UNBUFFERED output to logs/NAME.log (real file — pollable with
# Read, unlike a `| tail` pipe which buffers and never flushes). A watchdog
# guarantees the job can never hang silently forever:
#   * hard WALL timeout (MAX_SECS)  -> always terminates, the no-idle backstop
#   * STALL detector: log unchanged for STALL_SECS AND process CPU time flat
#     -> kills (catches the 0%-CPU deadlocks: Metal-HAL faults, wandb atexit
#        network-flush hang). CPU-busy phases (the ~4min IREE compile emit no
#        output but burn CPU) are NOT killed, so no false positives.
# Exit: job's rc, or 124 (wall timeout), or 125 (stall).
set -u
NAME="$1"; MAX="$2"; STALL="$3"; shift 3
[ "${1:-}" = "--" ] && shift
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/$NAME.log"
: > "$LOG"
# WANDB__SERVICE_WAIT bounds wandb's startup/teardown so it can't wedge forever.
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-60}"
# PYTHONUNBUFFERED so step/loss lines flush immediately -> log is pollable live.
PYTHONUNBUFFERED=1 "$@" >>"$LOG" 2>&1 &
JOB=$!
echo "[watch] pid=$JOB log=$LOG max=${MAX}s stall=${STALL}s cmd: $*" | tee -a "$LOG"
START=$(date +%s); LAST_SIZE=0; LAST_CHANGE=$START; LAST_CPU=0
cpu_secs() { ps -o time= -p "$1" 2>/dev/null | awk -F: '{n=NF; s=0; m=1; for(i=n;i>=1;i--){s+=$i*m; m*=60} print int(s)}'; }
while kill -0 "$JOB" 2>/dev/null; do
  sleep 15
  NOW=$(date +%s); SIZE=$(wc -c <"$LOG" 2>/dev/null || echo 0); CPU=$(cpu_secs "$JOB")
  [ -z "$CPU" ] && CPU=$LAST_CPU
  if [ "$SIZE" != "$LAST_SIZE" ]; then LAST_SIZE=$SIZE; LAST_CHANGE=$NOW; fi
  if [ $((NOW-START)) -ge "$MAX" ]; then
    echo "[watch] WALL TIMEOUT ${MAX}s -> kill $JOB" | tee -a "$LOG"
    kill -9 "$JOB" 2>/dev/null; wait "$JOB" 2>/dev/null; exit 124
  fi
  # stall = no new output for STALL secs AND <3s CPU progress this window
  if [ $((NOW-LAST_CHANGE)) -ge "$STALL" ] && [ $((CPU-LAST_CPU)) -lt 3 ]; then
    echo "[watch] STALL: no output ${STALL}s & CPU flat (${LAST_CPU}->${CPU}s) -> kill $JOB" | tee -a "$LOG"
    kill -9 "$JOB" 2>/dev/null; wait "$JOB" 2>/dev/null; exit 125
  fi
  LAST_CPU=$CPU
done
wait "$JOB"; RC=$?
echo "[watch] job exited rc=$RC" | tee -a "$LOG"
exit $RC
