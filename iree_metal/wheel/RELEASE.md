# Developer-preview release checklist

The wheel layout is intentionally independent of nlearn model code. Parity work may change
the three `.metal` files or rebuild the patched compiler/plugin; a release rebuild picks up
those artifacts without changing packaging code.

## Build

```bash
IREE_METAL_BUILD_DIR=$HOME/src/iree/integrations/pjrt/python_packages/iree_metal_plugin/build/cmake \
  ./scripts/build_iree_metal_wheel.sh
```

The build fails if either dylib, a kernel, or the upstream IREE license is missing. It emits
a macOS-arm64 platform wheel and embeds hashes plus clean/dirty source-tree state in
`build_info.json`.

## Required gates

1. Both source trees are clean, or the dirty diff hashes are deliberately documented.
2. `scripts/test_iree_metal_wheel.sh` passes in a disposable Python 3.11 environment,
   including compiled execution through the bundled GEMM, FlashAttention, and CE kernels.
3. The nlearn numerical parity suite passes against the wheel, not the editable build.
4. Benchmark results identify hardware, macOS, Python, JAX, IREE, model, shape, dtype,
   warmup, repetitions, and source revisions.
5. Test installation on a second Apple Silicon Mac with no IREE source checkout.
6. Replace the current linker ad-hoc signatures with the intended distribution signing
   policy, then verify loading on a quarantined download.
7. Sign the wheel checksum and publish it as a GitHub prerelease asset before considering
   a package-index upload.

## Deferred until parity lands

- Freeze the kernel files from the parity campaign.
- Rebuild both dylibs from a clean, tagged IREE checkout plus the reviewed patch series.
- Add the final supported-shape matrix and benchmark report.
- Decide whether the public name remains `iree-metal-preview` or moves under an IREE-owned
  namespace before any stable release.
