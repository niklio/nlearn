#!/bin/bash
# Generalized auto-resuming training supervisor for this Mac mini.
# Survives the recurring Metal-HAL GPU hang by relaunching `python -m nlearn.train
# --resume` from the last checkpoint until "Training complete" or MAX_RESTARTS.
#
# Usage: scripts/supervise.sh <run_name> <steps> <seq> <batch> <grad_accum> <peak_lr> [extra train args...]
#   e.g. scripts/supervise.sh dense123m 400000 1024 2 16 1e-3
set +e
RUN="${1:?run_name}"; STEPS="${2:?steps}"; SEQ="${3:?seq}"; BS="${4:?batch}"
ACCUM="${5:?grad_accum}"; LR="${6:?peak_lr}"; shift 6
EXTRA="$*"
LOG="/tmp/superv_${RUN}.log"
MAX_RESTARTS=100000          # effectively "until done" — it's a marathon
STALL_LIMIT=120             # seconds of no step-progress => declare hang
cd "$(dirname "$0")/.."     # repo root
source .runenv.sh >/dev/null 2>&1
export PYTHONUNBUFFERED=1
export NLEARN_GRAD_ACCUM="$ACCUM"
export NLEARN_CHECKPOINT_EVERY="${NLEARN_CHECKPOINT_EVERY:-50}"
export HF_HUB_DOWNLOAD_TIMEOUT=60
export WANDB_MODE=offline

gpu_ok() { "$VENV_PY" -c "import jax,jax.numpy as j;c=jax.jit(lambda a,b:a@b)(j.ones((8,8)),j.ones((8,8)));c.block_until_ready()" 2>/dev/null; }

echo "[superv $(date +%T)] start run=$RUN steps=$STEPS seq=$SEQ bs=$BS accum=$ACCUM (eff $((BS*ACCUM))) lr=$LR $EXTRA" | tee -a "$LOG"
for attempt in $(seq 1 $MAX_RESTARTS); do
  pkill -9 -f "run-name $RUN" 2>/dev/null; sleep 3
  until gpu_ok; do echo "[superv] GPU not ready, waiting..." | tee -a "$LOG"; sleep 15; done
  echo "[superv $(date +%T)] attempt $attempt" | tee -a "$LOG"
  "$VENV_PY" -u -m nlearn.train --steps "$STEPS" --batch-size "$BS" --seq-len "$SEQ" \
      --peak-lr "$LR" --run-name "$RUN" --resume $EXTRA >>"$LOG" 2>&1 &
  PID=$!
  last_n=-1; stall=0; started=0
  while kill -0 $PID 2>/dev/null; do
    grep -q "Training complete" "$LOG" && break
    n=$(grep -cE "^Step +[0-9]" "$LOG" 2>/dev/null)
    [ "$n" -gt 0 ] && started=1
    if [ "$n" = "$last_n" ]; then stall=$((stall+15)); else stall=0; last_n="$n"; fi
    limit=$STALL_LIMIT; [ "$started" = 0 ] && limit=420   # generous grace for init+compile
    if [ $stall -ge $limit ]; then
      echo "[superv $(date +%T)] HANG (no step progress ${stall}s, n=$n) — killing, will resume" | tee -a "$LOG"
      kill -9 $PID 2>/dev/null; pkill -9 -f "run-name $RUN" 2>/dev/null
      break
    fi
    sleep 15
  done
  wait $PID 2>/dev/null
  if grep -q "Training complete" "$LOG"; then
    echo "[superv $(date +%T)] DONE after $attempt attempt(s)" | tee -a "$LOG"
    break
  fi
  echo "[superv $(date +%T)] segment ended; recovering GPU before resume" | tee -a "$LOG"
  sleep 15
done
