# Source this to run nlearn on the M3 GPU via the open-source IREE stack:
#   source iree_env.sh && python train.py ...
# (uses the isolated venv ~/.venvs/iree; your global jax-metal env is untouched)

_IREE_BUILD="$HOME/src/iree/integrations/pjrt/python_packages/iree_metal_plugin/build/cmake"
_VENV="$HOME/.venvs/iree"

# Route JAX to the IREE Metal plugin.
export JAX_PLATFORMS=iree_metal

# Use the locally-built compiler that carries the vector.step SPIR-V fix
# (the pip iree-base-compiler lacks it).
export IREE_PJRT_COMPILER_LIB_PATH="$_IREE_BUILD/iree_core/lib/libIREECompiler.dylib"

# The from-source compiler needs lld to serialize the llvm-cpu host helper
# executables; reuse the one bundled with the pip iree package.
export IREE_PJRT_IREE_COMPILER_OPTIONS="--iree-llvmcpu-embedded-linker-path=$_VENV/lib/python3.13/site-packages/iree/compiler/_mlir_libs/iree-lld"

# Quieter logs (set to debug to see compiler/driver detail).
export IREE_PJRT_LOG_LEVEL=error

echo "IREE Metal env set. Python: $_VENV/bin/python"
echo "Note: wrap eager init in jax.jit (e.g. jax.jit(init_model)) to avoid the Sharding custom_call."
