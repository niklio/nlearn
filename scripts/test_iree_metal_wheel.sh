#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
WHEEL="${1:-$(ls -1 "${REPO_ROOT}"/iree_metal/wheel/dist/iree_metal_preview-*.whl | tail -1)}"
SMOKE_DIR="$(mktemp -d /tmp/iree-metal-preview-smoke.XXXXXX)"

cleanup() {
  case "$SMOKE_DIR" in
    /tmp/iree-metal-preview-smoke.*) rm -rf "$SMOKE_DIR" ;;
  esac
}
trap cleanup EXIT

uv venv --python 3.11 "$SMOKE_DIR"
uv pip install --python "$SMOKE_DIR/bin/python" "$WHEEL"
"$SMOKE_DIR/bin/iree-metal-doctor"
JAX_PLATFORMS=iree_metal "$SMOKE_DIR/bin/python" - <<'PY'
import jax
import jax.numpy as jnp

device = jax.devices()[0]
assert device.platform == "iree_metal", device
result = jax.jit(lambda x: x @ x)(jnp.eye(8, dtype=jnp.float32))
result.block_until_ready()
assert float(result.sum()) == 8.0
print(f"wheel smoke test passed: jax={jax.__version__} device={device}")
PY
PYTHONPATH="$REPO_ROOT" JAX_PLATFORMS=iree_metal \
  "$SMOKE_DIR/bin/python" "$REPO_ROOT/scripts/smoke_iree_metal_kernels.py"
