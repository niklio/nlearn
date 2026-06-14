# Binding the FlashAttention kernel into JAX training

Status of the native Metal FlashAttention work and the precise remaining plan to
wire it into the model for training.

## Validated end-to-end (on the M3 GPU, via IREE)

Everything *below* the JAX frontend is proven working:

1. **Metal external-object compiler support** — `patches/04` adds
   `MetalSPIRVTarget::serializeExternalExecutable` (Metal had none). ✅ rebuilt.
2. **ABI-conforming MSL kernel** — `kernels/flash_attention.metal`
   (argument-buffer struct + push constants, streaming online-softmax). ✅
3. **`flow.dispatch` of the external object** — `kernels/test/flash_test.mlir`:
   numerically correct on GPU (2.4e-7 vs NumPy reference). ✅
4. **`hal.dispatch.extern` form** — `kernels/flash_entry_point.mlir` /
   `flash_extern_test.mlir`: the exact IR the binding emits; correct on GPU
   (2.38e-7). ✅

## Remaining work (the JAX frontend binding + training)

### 1. Compiler: a preprocessing pass `custom_call @flash_attention -> dispatch.extern`

The transform-dialect route is **ruled out**: it has no matcher for a
`custom_call` by `call_target_name` (only structural / `has_no_lowering_config`
matchers exist), so it can't distinguish our kernel from JAX's `@Sharding`
custom_calls. So this must be a C++ pass.

- New preprocessing pass (register in the StableHLO input preprocessing
  pipeline, runs before legalization which would otherwise reject the
  custom_call — same wall as the Sharding issue).
- For each `stablehlo.custom_call` with `call_target_name == "flash_attention"`:
  read operand shapes (Q is `(n_heads, seq_len, d_head)`), compute
  `workload = n_heads*seq_len`, and build a `hal.dispatch.extern
  "flash_attention_fwd"` with the layout/count/objects from
  `flash_entry_point.mlir`, specialized to the call's shapes.
- Requires a compiler rebuild; needs the executable-object search path set so
  the `.metal` object resolves (plugin passes it via
  `IREE_PJRT_IREE_COMPILER_OPTIONS`).

### 2. JAX side: emit the custom_call (`attention.py`)

On the IREE-Metal platform branch, replace `_attention_standard` with a
`jax.ffi.ffi_call("flash_attention", out_shape, Q, K, V)` (Q/K/V already
`(n_heads, seq_len, d_head)`), keeping `_attention_standard` for cpu/cuda.

### 3. Backward / training: `jax.custom_vjp`

FlashAttention's true backward (dQ/dK/dV) is a second, much more complex kernel.
**Pragmatic correct approach** (do this first): wrap attention in
`jax.custom_vjp` where
- forward = the custom flash kernel (fast, O(seq) memory), and
- backward = the gradient of `_attention_standard` (recompute; already compiles
  on IREE-Metal).

Both compute the *same function*, so gradients are exact. This gives training
with the fast fused forward without authoring the flash backward kernel. The
optimized backward kernel is a later optimization (its own MSL effort).

### 4. Verify

Forward parity vs `_attention_standard` (done at kernel level); gradient parity
of the `custom_vjp` vs autodiff through `_attention_standard`; then a multi-step
`train.py` run on IREE-Metal with a sane loss curve and step-time vs the
standard-attention baseline.
