#!/bin/bash
set -e

REPO_ROOT="$(cd "${BASH_SOURCE%/*}" && pwd)"
CLUSTER_CONF="${REPO_ROOT}/.cluster.conf"

if [ ! -f "$CLUSTER_CONF" ]; then
  echo "Error: .cluster.conf not found. Copy .cluster.conf.example and fill in your cluster details."
  exit 1
fi
source "$CLUSTER_CONF"

: "${CLUSTER_HOST:?CLUSTER_HOST not set in .cluster.conf}"
: "${CLUSTER_USER:?CLUSTER_USER not set in .cluster.conf}"
: "${CLUSTER_DIR:?CLUSTER_DIR not set in .cluster.conf}"

SSH="${CLUSTER_USER}@${CLUSTER_HOST}"

usage() {
  echo "Usage: ./cluster.sh <command> [args]"
  echo ""
  echo "  <script> [args]   Sync code and submit <script>.py (or .sh) as a queued job"
  echo "  jobs              Show queue status"
  echo "  logs <id>         Tail live output for a job"
  echo "  cancel <id>       Remove a pending job"
  echo "  sync              Sync code to cluster without submitting"
  echo "  setup             First-time cluster bootstrap (installs pueue, creates remote dir)"
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
            "${REPO_ROOT}/" \
            "${SSH}:${CLUSTER_DIR}/"
  echo "Sync complete."
}

case "${1:-}" in
  jobs)
    ssh "$SSH" "pueue status"
    ;;

  logs)
    ssh "$SSH" "pueue follow ${2:?Usage: ./cluster.sh logs <job-id>}"
    ;;

  cancel)
    ssh "$SSH" "pueue remove ${2:?Usage: ./cluster.sh cancel <job-id>}"
    ;;

  sync)
    sync_code
    ;;

  setup)
    echo "Bootstrapping cluster..."
    ssh "$SSH" "brew install pueue && brew services start pueue || true"
    ssh "$SSH" "mkdir -p ${CLUSTER_DIR}"
    sync_code
    ssh "$SSH" "pueue status"
    echo "Cluster ready."
    ;;

  ""|help|--help)
    usage
    ;;

  *)
    SCRIPT="$1"
    shift
    ARGS="$*"

    if [ -f "${REPO_ROOT}/${SCRIPT}.py" ]; then
      CMD="python ${SCRIPT}.py ${ARGS}"
    elif [ -f "${REPO_ROOT}/${SCRIPT}.sh" ]; then
      CMD="bash ${SCRIPT}.sh ${ARGS}"
    elif [ -f "${REPO_ROOT}/${SCRIPT}" ] && [ -x "${REPO_ROOT}/${SCRIPT}" ]; then
      CMD="./${SCRIPT} ${ARGS}"
    else
      echo "Error: no '${SCRIPT}', '${SCRIPT}.py', or '${SCRIPT}.sh' found in repo root."
      exit 1
    fi

    sync_code

    FULL_CMD="cd ${CLUSTER_DIR} && ${CMD}"
    echo "Submitting: ${CMD}"
    ssh "$SSH" "pueue add -- bash -c '${FULL_CMD}'"
    ssh "$SSH" "pueue status"
    ;;
esac
