#!/bin/bash
# Usage: ./scripts/train_remote.sh [--n_steps 10000 --batch_size 64 ...]
# Kicks off a training run in a detached tmux session on the Mac Mini via Tailscale SSH.
# Requires: `mini` host alias configured in ~/.ssh/config pointing to the Mini's Tailscale hostname.

set -e

SESSION="nlearn-$(date +%H%M)"
REMOTE_LOG="~/nlearn/logs/${SESSION}.log"

ssh mini "mkdir -p ~/nlearn/logs && tmux new-session -d -s '$SESSION' \
  'cd ~/nlearn && python train.py $* 2>&1 | tee $REMOTE_LOG; echo \"--- run finished ---\" >> $REMOTE_LOG'"

echo "Training started in tmux session: $SESSION"
echo ""
echo "  Watch logs:  ssh mini -t 'tmux attach -t $SESSION'"
echo "  Kill safely: ssh mini \"tmux send-keys -t '$SESSION' C-c\""
echo "  Monitor:     https://wandb.ai"
