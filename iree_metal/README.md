# Running nlearn on Apple Silicon via IREE

This directory vendors everything needed to reproduce the open-source
JAX-on-Metal backend used by the `iree-metal` branch. It replaces Apple's
abandoned `jax-metal` with the [IREE](https://github.com/iree-org/iree) PJRT
plugin, which (unlike jax-metal) supports custom kernels — the prerequisite for
a native Metal FlashAttention kernel.

IREE has no Metal PJRT plugin upstream and its `metal-spirv` compiler target has
gaps that block this model. The files here are the authored plugin plus the
compiler/build patches that close those gaps. They live outside the IREE tree so
the project is self-contained; apply them to a fresh IREE checkout to rebuild.

## Contents

- `plugin/metal_backend/` — authored `iree_metal` PJRT backend (C++). Templated
  off IREE's Vulkan backend: registers the Metal HAL driver, names the device
  `metal`, sets `--iree-hal-target-device=metal`.
- `plugin/iree_metal_plugin/` — Python package wrapping the backend dylib
  (JAX `jax_plugins` entry point `iree-metal`). Its `__init__.py` honors
  `IREE_PJRT_COMPILER_LIB_PATH` and probes both `.dylib`/`.so`.
- `plugin/patch_protobuf_absl.sh` — build patch (see patch 03).
- `patches/01-spirv-vector-step-lowering.patch` — **the key compiler fix.**
  IREE registers `vector::populateVectorStepLoweringPatterns` for its NVVM/ROCDL/
  CPU paths but not SPIR-V, so `vector.step` (from `jnp.arange`) fails to
  legalize on Metal. Adds it to `SPIRVFinalVectorLowering.cpp`. Upstreamable.
- `patches/02-pjrt-add-metal-subdir.patch` — wires the metal backend into the
  PJRT `src/CMakeLists.txt` behind `IREE_HAL_DRIVER_METAL`.
- `patches/03-protobuf-abseil-patch-cmd.patch` — makes the protobuf FetchContent
  run `patch_protobuf_absl.sh`, fixing abseil emitting `-msse4.1` on arm64
  (Apple clang 17 rejects it).

## Reproduce

```sh
# 1. Clone IREE at the commit matching pip iree-base-compiler==3.11.0
git clone https://github.com/iree-org/iree.git ~/src/iree
cd ~/src/iree
git checkout e4a3b0405d7d23554da26403658d0e8c3c5ecf25
git submodule update --init --depth 1 third_party/

# 2. Drop in the authored plugin files
P=integrations/pjrt
cp -R <nlearn>/iree_metal/plugin/metal_backend        $P/src/iree_pjrt/metal
cp -R <nlearn>/iree_metal/plugin/iree_metal_plugin     $P/python_packages/iree_metal_plugin
cp    <nlearn>/iree_metal/plugin/patch_protobuf_absl.sh $P/cmake/patch_protobuf_absl.sh

# 3. Apply the tracked-file patches
git apply <nlearn>/iree_metal/patches/*.patch

# 4. Isolated venv + matching compiler/runtime, then build the plugin
uv venv ~/.venvs/iree --python 3.13
uv pip install --python ~/.venvs/iree/bin/python "jax==0.6.1" "jaxlib==0.6.1" \
    "iree-base-compiler==3.11.0" "iree-base-runtime==3.11.0" optax tiktoken numpy
CMAKE_OSX_ARCHITECTURES=arm64 MACOSX_DEPLOYMENT_TARGET=13.0 \
  uv pip install --python ~/.venvs/iree/bin/python --no-build-isolation --no-deps -v \
  -e $P/python_packages/iree_metal_plugin

# 5. Build the patched compiler dylib (carries the vector.step fix; ~7k steps)
cd $P/python_packages/iree_metal_plugin/build/cmake
ninja -j4 libIREECompiler.dylib
```

Then `source iree_env.sh` from the repo root and run the model with
`~/.venvs/iree/bin/python`. Wrap eager init in `jax.jit` (e.g.
`jax.jit(init_model)`) to avoid the `Sharding` custom_call legalization error.

## Known backend constraints (baked into the model on this branch)

- **fp16, not bf16** — the Apple `metal-spirv` target lacks bf16.
- **chunked-CE target select uses one-hot multiply**, not advanced-index gather:
  the gather's scatter backward miscompiles under `vmap` on `metal-spirv`.
- The from-source compiler needs `lld` to serialize llvm-cpu host helpers; point
  at the pip-bundled `iree-lld` (see `iree_env.sh`).

Verified 2026-06-14 on an M3 (16GB): full `train_step` compiles and trains on
the GPU (loss 10.06 → 6.43 over 6 steps, ~0.35s/step).
