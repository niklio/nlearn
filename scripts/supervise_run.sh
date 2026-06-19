#!/bin/bash
# Supervisor: run a training run to completion, surviving the recurring Metal-HAL GPU
# hang (~every 100 batched steps). Launches train.py --resume; if the process freezes
# (CPU time stops advancing), kills it, lets the GPU recover, and relaunches — which
# resumes from the last checkpoint. Repeats until "Training complete" or MAX_RESTARTS.
#
# Usage: ./supervise_run.sh <run_name> <steps> [peak_lr]
set +e
RUN="${1:-comp_effbs32}"
STEPS="${2:-2000}"
PEAK_LR="${3:-1e-3}"
LOG="/tmp/superv_${RUN}.log"
MAX_RESTARTS=60
STALL_LIMIT=120          # seconds of frozen CPU ⇒ declare hang
cd /Users/nikliolios/nlearn
source iree_env.sh >/dev/null 2>&1
export PYTHONUNBUFFERED=1 NLEARN_GRAD_ACCUM=2 NLEARN_NO_VAL=1 NLEARN_CHECKPOINT_EVERY=50

gpu_ok() { ~/.venvs/iree/bin/python -c "import jax,jax.numpy as j;c=jax.jit(lambda a,b:a@b)(j.ones((8,8)),j.ones((8,8)));c.block_until_ready()" 2>/dev/null; }

echo "[superv $(date +%T)] start run=$RUN steps=$STEPS lr=$PEAK_LR" | tee -a "$LOG"
for attempt in $(seq 1 $MAX_RESTARTS); do
  pkill -9 -f "run-name $RUN" 2>/dev/null; sleep 3
  until gpu_ok; do echo "[superv] GPU not ready, waiting..." | tee -a "$LOG"; sleep 15; done
  echo "[superv $(date +%T)] attempt $attempt" | tee -a "$LOG"
  ~/.venvs/iree/bin/python train.py --steps "$STEPS" --batch-size 16 --seq-len 512 \
      --peak-lr "$PEAK_LR" --run-name "$RUN" --resume >>"$LOG" 2>&1 &
  PID=$!
  # Detect hang by TRAINING-STEP PROGRESS, not CPU time: train.py's background data
  # thread keeps the process CPU ticking even while the main thread is GPU-hung, so
  # cputime is unreliable. Count "Step N" log lines (prints every 10 steps); if that
  # count stops growing, the training loop is stuck. Generous grace before the first
  # step (model init + val-data streaming + first-step compile can take a while).
  last_n=-1; stall=0; started=0
  while kill -0 $PID 2>/dev/null; do
    grep -q "Training complete" "$LOG" && break
    n=$(grep -cE "^Step +[0-9]" "$LOG" 2>/dev/null)
    [ "$n" -gt 0 ] && started=1
    if [ "$n" = "$last_n" ]; then stall=$((stall+15)); else stall=0; last_n="$n"; fi
    limit=$STALL_LIMIT; [ "$started" = 0 ] && limit=300
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
