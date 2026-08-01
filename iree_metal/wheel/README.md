# iree-metal-preview

An experimental, self-contained JAX PJRT backend for Apple Silicon. The wheel bundles the
patched IREE compiler, Metal PJRT plugin, and the current FlashAttention, GEMM, and fused
cross-entropy Metal kernels used by nlearn.

This is a developer preview, not an upstream IREE release. It currently supports only
macOS arm64 and pins the JAX/IREE versions it was built and tested against.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install iree_metal_preview-*.whl
iree-metal-doctor --probe-jax
```

JAX discovers the plugin automatically. To select it explicitly:

```bash
JAX_PLATFORMS=iree_metal python -c 'import jax; print(jax.devices())'
```

## Compatibility contract

- Apple Silicon (`arm64`), macOS 13 or newer
- Python 3.11+
- JAX/JAXlib 0.6.1
- IREE compiler/runtime Python packages 3.11.0

The wheel embeds `build_info.json` with the source revisions and SHA-256 digest of every
native artifact. Run `iree-metal-doctor` to inspect it without loading JAX.

## Known limitations

- The Metal backend and compiler patches are not yet upstream.
- The native wheel is large because it includes the patched IREE compiler.
- Only the shapes and workloads in the nlearn validation suite have been exercised deeply.
- This preview does not promise ABI compatibility with other JAX or IREE versions.
