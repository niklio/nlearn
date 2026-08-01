#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
WHEEL_ROOT="${REPO_ROOT}/iree_metal/wheel"
IREE_METAL_BUILD_DIR="${IREE_METAL_BUILD_DIR:-$HOME/src/iree/integrations/pjrt/python_packages/iree_metal_plugin/build/cmake}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  if [ -x "$HOME/.venvs/iree/bin/python" ]; then
    PYTHON_BIN="$HOME/.venvs/iree/bin/python"
  else
    echo "Python 3.11+ is required (set PYTHON_BIN)." >&2
    exit 2
  fi
fi

export IREE_METAL_BUILD_DIR
"$PYTHON_BIN" -m pip wheel --no-deps --wheel-dir "${WHEEL_ROOT}/dist" "${WHEEL_ROOT}"
"$PYTHON_BIN" -m zipfile -l "${WHEEL_ROOT}"/dist/iree_metal_preview-*.whl
