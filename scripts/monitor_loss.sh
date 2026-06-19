#!/usr/bin/env bash
# monitor_loss.sh LOGFILE RUN_PATTERN MIN_STEP RISE
# Watches a training log; tracks the minimum val_loss seen. Once past MIN_STEP,
# if the latest val_loss exceeds that minimum by > RISE, KILLS the run (slow
# divergence — don't burn hours). Also exits when the run process disappears
# (completed) or loss goes nan/inf. Prints why it stopped. Run in background;
# the completion notification re-invokes the agent to decide next steps.
set -u
L="$1"; PAT="$2"; MIN_STEP="${3:-1000}"; RISE="${4:-0.4}"
minval=""
while pgrep -f "$PAT" >/dev/null 2>&1; do
  line=$(grep -E "val_loss: [0-9.]+|loss: (nan|inf)" "$L" 2>/dev/null | tail -1)
  step=$(grep -E "^Step +[0-9]+" "$L" 2>/dev/null | tail -1 | sed -E 's/^Step +([0-9]+).*/\1/')
  if echo "$line" | grep -qE "nan|inf"; then echo "STOP: nan/inf"; break; fi
  v=$(echo "$line" | sed -E 's/.*val_loss: ([0-9.]+).*/\1/')
  if [ -n "$v" ]; then
    if [ -z "$minval" ] || awk "BEGIN{exit !($v < $minval)}"; then minval="$v"; fi
    if [ -n "$step" ] && [ "$step" -ge "$MIN_STEP" ] && \
       awk "BEGIN{exit !($v > $minval + $RISE)}"; then
      echo "ABORT: val_loss $v > min $minval + $RISE at step $step -> killing $PAT"
      pkill -9 -f "$PAT"; break
    fi
  fi
  sleep 30
done
echo "=== monitor done (min val_loss=$minval) ==="; tail -4 "$L"
