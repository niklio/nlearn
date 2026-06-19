#!/bin/bash
# Supervisor for the seq=4096 converged run on THIS Mac mini (user agentjohnson).
# Adapted from supervise_run.sh: survives the recurring Metal-HAL GPU hang by
# relaunching train.py --resume from the last checkpoint until "Training complete".
# Uses .runenv.sh (the prebuilt .iree_runtime bundle) instead of iree_env.sh.
#
# Usage: ./supervise_s4096.sh <run_name> <steps_microbatches> [peak_lr]
set +e
RUN="${1:-conv_s4096}"
STEPS="${2:-4000}"          # MICRO-steps; effective updates = STEPS / GRAD_ACCUM
PEAK_LR="${3:-6e-4}"
LOG="/tmp/superv_${RUN}.log"
MAX_RESTARTS=120
STALL_LIMIT=120             # seconds of no step-progress => declare hang
cd /Users/agentjohnson/nlearn
source .runenv.sh >/dev/null 2>&1
export PYTHONUNBUFFERED=1
export NLEARN_GRAD_ACCUM=4          # effective batch = 2 * 4 = 8
export NLEARN_CHECKPOINT_EVERY=50   # checkpoint often so a hang loses <=50 steps
export HF_HUB_DOWNLOAD_TIMEOUT=60
export WANDB_MODE=offline

gpu_ok() { "$VENV_PY" -c "import jax,jax.numpy as j;c=jax.jit(lambda a,b:a@b)(j.ones((8,8)),j.ones((8,8)));c.block_until_ready()" 2>/dev/null; }

echo "[superv $(date +%T)] start run=$RUN steps=$STEPS lr=$PEAK_LR (bs2 x accum4 = eff bs8)" | tee -a "$LOG"
for attempt in $(seq 1 $MAX_RESTARTS); do
  pkill -9 -f "run-name $RUN" 2>/dev/null; sleep 3
  until gpu_ok; do echo "[superv] GPU not ready, waiting..." | tee -a "$LOG"; sleep 15; done
  echo "[superv $(date +%T)] attempt $attempt" | tee -a "$LOG"
  "$VENV_PY" -u train.py --steps "$STEPS" --batch-size 2 --seq-len 4096 \
      --peak-lr "$PEAK_LR" --run-name "$RUN" --resume >>"$LOG" 2>&1 &
  PID=$!
  last_n=-1; stall=0; started=0
  while kill -0 $PID 2>/dev/null; do
    grep -q "Training complete" "$LOG" && break
    n=$(grep -cE "^Step +[0-9]" "$LOG" 2>/dev/null)
    [ "$n" -gt 0 ] && started=1
    if [ "$n" = "$last_n" ]; then stall=$((stall+15)); else stall=0; last_n="$n"; fi
    limit=$STALL_LIMIT; [ "$started" = 0 ] && limit=360   # generous grace for init+compile
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
