#!/usr/bin/env bash
# Install the leaderboard pre-commit hook into this repo's .git/hooks.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
cd "$ROOT"

if [ ! -d .git ]; then
  echo "No git repo in $ROOT — initializing one so the presubmit hook can run."
  git init -q
fi

chmod +x "$DIR/pre-commit"
ln -sf "$DIR/pre-commit" "$ROOT/.git/hooks/pre-commit"
echo "Installed pre-commit hook: $ROOT/.git/hooks/pre-commit -> $DIR/pre-commit"
echo "Kernel commits (attention.py, gemm_kernel.py, iree_metal/kernels/**) will now"
echo "trigger a background benchmark that posts to the leaderboard."
