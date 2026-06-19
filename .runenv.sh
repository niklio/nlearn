# Source to run nlearn locally on this Mac mini via the shipped prebuilt IREE-Metal
# bundle (.iree_runtime). Mirrors cluster.py::iree_env_preamble exactly.
D="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
B="$D/.iree_runtime"
LLD="$HOME/.venvs/iree/lib/python3.13/site-packages/iree/compiler/_mlir_libs/iree-lld"
export PATH=/opt/homebrew/bin:$PATH
export JAX_PLATFORMS=iree_metal
export PYTHONPATH="$B:$D:$PYTHONPATH"
export NLEARN_IREE_PLUGIN_DYLIB="$B/pjrt_plugin_iree_metal.dylib"
export IREE_PJRT_COMPILER_LIB_PATH="$B/libIREECompiler.dylib"
export NLEARN_FLASH_KERNEL_PATH="$D/iree_metal/kernels/flash_attention.metal"
export NLEARN_GEMM_KERNEL_PATH="$D/iree_metal/kernels/gemm.metal"
export NLEARN_CE_KERNEL_PATH="$D/iree_metal/kernels/cross_entropy.metal"
export IREE_PJRT_IREE_COMPILER_OPTIONS="--iree-llvmcpu-embedded-linker-path=$LLD --iree-metal-compile-to-metallib=false"
export IREE_PJRT_LOG_LEVEL=error
export VENV_PY="$HOME/.venvs/iree/bin/python"
