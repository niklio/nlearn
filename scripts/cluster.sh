#!/bin/bash
set -e

REPO_ROOT="$(cd "${BASH_SOURCE%/*}/.." && pwd)"
CLUSTER_CONF="${REPO_ROOT}/.cluster.conf"

if [ ! -f "$CLUSTER_CONF" ]; then
  echo "Error: .cluster.conf not found. Copy .cluster.conf.example and fill in your cluster details."
  exit 1
fi
source "$CLUSTER_CONF"

# Leaderboard posting: pull in LEADERBOARD_URL/LEADERBOARD_TOKEN so they can be
# forwarded to remote jobs below, regardless of how this script was invoked
# (these are written by the deploy at ~/.config/nlearn/leaderboard.env).
[ -f "$HOME/.config/nlearn/leaderboard.env" ] && source "$HOME/.config/nlearn/leaderboard.env"

: "${CLUSTER_HOST:?CLUSTER_HOST not set in .cluster.conf}"
: "${CLUSTER_USER:?CLUSTER_USER not set in .cluster.conf}"
: "${CLUSTER_DIR:?CLUSTER_DIR not set in .cluster.conf}"

SSH="${CLUSTER_USER}@${CLUSTER_HOST}"
SENTINEL="${REPO_ROOT}/.cluster_ready"

rssh() {
  ssh "$SSH" "export PATH=/opt/homebrew/bin:\$PATH; $*"
}

usage() {
  echo "Usage: ./cluster.sh <command> [args]"
  echo ""
  echo "JOB COMMANDS"
  echo "  <script> [args]"
  echo "      Sync code to the cluster and submit <script> as a queued job."
  echo "      Looks for <script>.py, <script>.sh, or an executable named <script>."
  echo "      All extra args are passed through to the script."
  echo "      Examples:"
  echo "        ./cluster.sh train --n_steps 50000 --batch_size 32"
  echo "        ./cluster.sh data --dataset fineweb-edu --target-tokens 500_000_000"
  echo ""
  echo "  jobs"
  echo "      Show the pueue job queue: all pending, running, and completed jobs."
  echo ""
  echo "  logs <id>"
  echo "      Tail live output from a running or completed job."
  echo "      <id> is the numeric job ID shown by 'jobs'."
  echo "      Example: ./cluster.sh logs 4"
  echo ""
  echo "  cancel <id>"
  echo "      Remove a pending job from the queue (cannot cancel a running job)."
  echo "      Example: ./cluster.sh cancel 4"
  echo ""
  echo "  kill <id>"
  echo "      Kill a running job immediately."
  echo "      Example: ./cluster.sh kill 6"
  echo ""
  echo "CLUSTER MANAGEMENT"
  echo "  sync"
  echo "      Sync local code to the cluster without submitting a job."
  echo "      Excludes: datasets/, checkpoints/, wandb/, __pycache__, *.pkl, .cluster.conf"
  echo ""
  echo "  pull"
  echo "      Rsync checkpoints/ from the cluster back to local."
  echo "      Preserves run-name subdirectory structure."
  echo ""
  echo "  setup"
  echo "      Re-run full cluster bootstrap: installs pueue, creates remote dir,"
  echo "      syncs code, and installs Python dependencies from requirements.txt."
  echo "      Runs automatically on first use of any command."
  echo ""
  echo "SCRIPTS"
  echo "  train [options]"
  echo "      Train the transformer on datasets/fineweb-edu.bin (or Shakespeare fallback)."
  echo "      Logs metrics to W&B and saves checkpoints to checkpoints/."
  echo "      --n_steps <n>               Training steps (default: 5000)"
  echo "      --seq_len <n>               Sequence length (default: 512)"
  echo "      --batch_size <n>            Batch size (default: 16; max ~20 on Metal GPU)"
  echo "      --peak_lr <f>               Peak learning rate (default: 1e-3)"
  echo "      --seed <n>                  Random seed (default: 0)"
  echo "      --run_name <str>            W&B run name (default: auto-generated)"
  echo "      Example: ./cluster.sh train --n_steps 50000 --batch_size 32"
  echo ""
  echo "  generate --prompt <text> --n-tokens <n> [options]"
  echo "      Generate text from a trained checkpoint."
  echo "      --prompt <text>             Input text to continue from  (required)"
  echo "      --n-tokens <n>              Number of new tokens to generate  (required)"
  echo "      --local-checkpoint <path>   Local .pkl checkpoint file"
  echo "      --run-id <id>               W&B run ID to download checkpoint from"
  echo "      --temperature <f>           Sampling temperature (default: 0.8)"
  echo "      --project <str>             W&B project name (default: nlearn-transformer)"
  echo "      Example: ./cluster.sh generate --local-checkpoint checkpoints/step_005000.pkl --prompt 'Hello' --n-tokens 100"
  echo ""
  echo "CONFIGURATION"
  echo "  Cluster connection is read from .cluster.conf (copy from .cluster.conf.example):"
  echo "    CLUSTER_HOST  — hostname or IP of the remote machine"
  echo "    CLUSTER_USER  — SSH username"
  echo "    CLUSTER_DIR   — remote path where code is synced (e.g. ~/nlearn)"
}

sync_code() {
  echo "Syncing code to cluster..."
  rsync -az --exclude='.git' \
            --exclude='datasets/' \
            --exclude='checkpoints/' \
            --exclude='wandb/' \
            --exclude='__pycache__/' \
            --exclude='*.pkl' \
            --exclude='.cluster.conf' \
            --exclude='.cluster_ready' \
            "${REPO_ROOT}/" \
            "${SSH}:${CLUSTER_DIR}/"
  echo "Sync complete."
}

do_setup() {
  echo "Bootstrapping cluster..."
  rssh "brew install pueue && brew services start pueue || true"
  rssh "mkdir -p ${CLUSTER_DIR}"
  sync_code
  rssh "pip3 install -r ${CLUSTER_DIR}/requirements.txt"
  rssh "pueue status"
  touch "$SENTINEL"
  echo "Cluster ready."
}

# Sync code and ensure deps are installed before every command.
# pip install is a no-op for packages already at the right version.
ensure_ready() {
  if [ ! -f "$SENTINEL" ]; then
    do_setup
  else
    sync_code
    rssh "pip3 install -q -r ${CLUSTER_DIR}/requirements.txt"
  fi
}

case "${1:-}" in
  jobs)
    ensure_ready
    rssh "pueue status"
    ;;

  logs)
    rssh "pueue follow ${2:?Usage: ./cluster.sh logs <job-id>}"
    ;;

  cancel)
    rssh "pueue remove ${2:?Usage: ./cluster.sh cancel <job-id>}"
    ;;

  kill)
    rssh "pueue kill ${2:?Usage: ./cluster.sh kill <job-id>}"
    ;;

  sync)
    sync_code
    ;;

  pull)
    echo "Pulling checkpoints from cluster..."
    rsync -az "${SSH}:${CLUSTER_DIR}/checkpoints/" "${REPO_ROOT}/checkpoints/"
    echo "Done."
    ;;

  setup)
    do_setup
    ;;

  ""|help|--help)
    usage
    ;;

  *)
    SCRIPT="$1"
    shift
    ARGS=$(printf '%q ' "$@")

    if [ -f "${REPO_ROOT}/${SCRIPT}.py" ]; then
      CMD="python3 -u ${SCRIPT}.py ${ARGS}"
    elif [ -f "${REPO_ROOT}/${SCRIPT}.sh" ]; then
      CMD="bash ${SCRIPT}.sh ${ARGS}"
    elif [ -f "${REPO_ROOT}/${SCRIPT}" ] && [ -x "${REPO_ROOT}/${SCRIPT}" ]; then
      CMD="./${SCRIPT} ${ARGS}"
    else
      echo "Error: no '${SCRIPT}', '${SCRIPT}.py', or '${SCRIPT}.sh' found in repo root."
      exit 1
    fi

    ensure_ready

    # Write the job as a script file piped through stdin — avoids all quoting issues
    # across the local shell → SSH → remote shell → pueue chain.
    JOB_SCRIPT="/tmp/cluster_job_$(date +%s).sh"
    printf '#!/bin/bash\nexport PATH=/opt/homebrew/bin:$PATH\n%s\n%s\n%s\ncd %s\n%s\n' \
      "${HF_TOKEN:+export HF_TOKEN=${HF_TOKEN}}" \
      "${LEADERBOARD_URL:+export LEADERBOARD_URL=${LEADERBOARD_URL}}" \
      "${LEADERBOARD_TOKEN:+export LEADERBOARD_TOKEN=${LEADERBOARD_TOKEN}}" \
      "${CLUSTER_DIR}" "${CMD}" \
      | ssh "$SSH" "cat > ${JOB_SCRIPT} && chmod +x ${JOB_SCRIPT}"

    echo "Submitting: ${CMD}"
    rssh "pueue add --label '${SCRIPT} ${ARGS}' -- bash ${JOB_SCRIPT}"
    rssh "pueue status"
    ;;
esac
