# Native Metal FlashAttention kernel (Phase 4)

Goal: replace the standard O(seq²)-memory attention (`_attention_standard` in
`attention.py`) with a fused, streaming-softmax Metal kernel — the thing
jax-metal made impossible (it rejects custom kernels at legalization) and that
motivated the whole IREE migration.

## Status

| Piece | State |
|---|---|
| Forward kernel (`flash_attention.metal`) | ✅ ABI-conforming; emits `O` and `L` (logsumexp); runs on M3 GPU via IREE, matches reference 2.4e-7 |
| **True backward kernels** (`flash_attention_bwd_dq`, `flash_attention_bwd_dkdv`) | ✅ O(seq)-memory dQ/dK/dV (recompute softmax from `L`, no seq²); validated on M3 GPU vs NumPy reference — dQ 3.6e-7, dK 9.5e-7, dV 3.9e-7 (`test/flash_bwd_test.mlir`, `test/validate_flash_bwd.py`) |
| Compiler: Metal external-object support | ✅ patched (`patches/04`): `MetalSPIRVTarget::serializeExternalExecutable` embeds a provided MSL object into the metal flatbuffer (Metal had no external path; Vulkan did). Compiler rebuilt clean. |
| Metal dispatch ABI | ✅ reverse-engineered (see below + `abi_reference/`) |
| Hand-MLIR trivial dispatch on Metal (iree-run-module) | ✅ `test/spike.mlir` + `spike_mul.metal`: out=2*4=8 on the M3 GPU |
| FlashAttention kernel on Metal via IREE | ✅ `test/flash_test.mlir` + `validate_flash.py`: matches NumPy reference to 2.4e-7 on the GPU |
| Frontend: `stablehlo.custom_call` → `flow.dispatch` | ⏳ patch `StableHLOCustomCalls.cpp` so JAX can invoke it |
| `jax.ffi` wiring in `attention.py` | ⏳ |
| Backward pass (dQ/dK/dV) + custom_vjp | ⏳ deferred until forward binds end-to-end |

## The Metal dispatch ABI (reverse-engineered)

IREE compiles normal kernels with spirv-cross `argument_buffers = true`. A
conforming kernel (see `abi_reference/generated_mul.msl`, dumped from a real
`iree-compile --iree-metal-compile-to-metallib=false` of an elementwise mul) has
this shape — a hand-authored kernel must match it:

```metal
struct spvDescriptorSetBuffer0 {
    device T* _resource_var_0_0_ [[id(0)]];   // binding 0
    device T* _resource_var_0_1_ [[id(1)]];   // binding 1
    device T* _resource_var_0_2_ [[id(2)]];   // binding 2 ...
};
kernel void name(constant spvDescriptorSetBuffer0& set0 [[buffer(0)]],
                 uint3 wgid [[threadgroup_position_in_grid]],
                 uint3 lid  [[thread_position_in_threadgroup]]) {
    uint gid = lid.x + wgid.x * THREADGROUP_SIZE_X;   // global id
    ...
}
```

- Storage-buffer bindings arrive in a **Metal argument buffer** at `[[buffer(0)]]`,
  each binding at `[[id(n)]]`. NOT individual `[[buffer(n)]]` params.
- Push constants (if any) go at buffer index
  `IREE_HAL_METAL_MAX_DESCRIPTOR_SET_COUNT - 1` (the
  `IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX`).
- Threadgroup (local) size comes from the export's `workgroup_size` attr
  (patch 04 reads it; default 64,1,1); the host `count()` region gives the
  number of threadgroups.

## The kernel

`flash_attention.metal` — causal FlashAttention forward. One thread computes one
output row `O[head, q]` by streaming over keys `k<=q`, maintaining the running
max `m` and normaliser `l` (FlashAttention online-softmax recurrence). Never
materialises the seq×seq score matrix. Operates on the `(n_heads, seq_len,
d_head)` tensors the model already passes to `attention()`.

## Binding roadmap (the remaining work, in risk order)

Investigation found **no JAX→custom-kernel path exists out of the box**:
- IREE's custom_dispatch samples invoke kernels via hand-written
  `flow.dispatch @kernel::@main` MLIR — they never start from JAX/StableHLO.
- IREE's StableHLO frontend marks unknown `stablehlo.custom_call` targets
  illegal (only `shape_assertion` / `ProductOfElementaryHouseholderReflectors`
  are handled), so a `jax.ffi` custom_call named `flash_attention` is rejected
  at legalization — the same wall as the `Sharding` annotation.
- The Vulkan custom_dispatch sample provides a **SPIR-V** object; IREE's Metal
  HAL instead loads **MSL source / metallib** (`newLibraryWithSource:`), so the
  object must be a `metal-msl-fb` executable — and **no Metal custom_dispatch
  sample exists** to copy.

So binding requires, roughly:
1. **Wire the metallib as a `hal.executable` with a `metal-msl-fb` object** and a
   `#hal.pipeline.layout` ABI (constants = n_heads/seq_len/d_head/scale,
   bindings = Q,K,V read-only + O). Highest uncertainty — undocumented for Metal.
2. **Patch IREE's StableHLO frontend** (`StableHLOCustomCalls.cpp`) to lower a
   `stablehlo.custom_call @flash_attention` into a `flow.dispatch` /
   `hal.dispatch.extern` of that executable. Requires another compiler rebuild.
3. **Emit the custom_call from `attention.py`** via `jax.ffi.ffi_call` on the
   IREE-Metal platform branch, falling back to `_attention_standard` elsewhere.
4. **Backward**: register a `custom_vjp` and author the gradient kernel (or
   recompute-in-backward). Significant additional MSL.

Steps 1–2 each likely need a ~7k-step compiler rebuild to test, so this is a
multi-iteration effort. Validate the trivial case (a no-op/identity custom
dispatch end-to-end) before relying on the full kernel.

## Reproduce the standalone kernel validation

```sh
xcrun -sdk macosx metal -std=metal3.0 -c flash_attention.metal -o /tmp/fa.air
xcrun -sdk macosx metallib /tmp/fa.air -o /tmp/fa.metallib   # loadable form
```
