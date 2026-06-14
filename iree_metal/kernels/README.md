# Native Metal FlashAttention kernel (Phase 4)

Goal: replace the standard O(seq²)-memory attention (`_attention_standard` in
`attention.py`) with a fused, streaming-softmax Metal kernel — the thing
jax-metal made impossible (it rejects custom kernels at legalization) and that
motivated the whole IREE migration.

## Status

| Piece | State |
|---|---|
| Forward kernel (`flash_attention.metal`) | ✅ authored; compiles to AIR + metallib (`xcrun metal`); online-softmax recurrence verified vs standard causal attention (max abs diff 1.2e-7) |
| Binding into JAX→IREE→Metal | ⏳ designed, not yet implemented (the hard part — see below) |
| Backward pass (dQ/dK/dV) | ⏳ deferred until forward binds end-to-end |

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
