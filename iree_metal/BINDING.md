# Binding the FlashAttention kernel into JAX training

Status of the native Metal FlashAttention work and the precise remaining plan to
wire it into the model for training.

## Validated end-to-end (on the M3 GPU, via IREE)

Everything *below* the JAX frontend is proven working — the COMPLETE kernel set:

1. **Metal external-object compiler support** — `patches/04` adds
   `MetalSPIRVTarget::serializeExternalExecutable` (Metal had none). ✅ rebuilt.
2. **ABI-conforming MSL kernels** — `kernels/flash_attention.metal`: forward
   (emits O + L), backward dQ, backward dK/dV (all O(seq) memory). ✅
3. **`flow.dispatch` of the external objects** — `kernels/test/flash_test.mlir`
   (fwd) + `flash_bwd_test.mlir` (fwd+bwd): numerically correct on GPU.
   Forward 2.4e-7; **backward dQ 3.6e-7, dK 9.5e-7, dV 3.9e-7** vs NumPy. ✅
4. **`hal.dispatch.extern` form** — `kernels/flash_entry_point.mlir`: the
   inline-dispatch IR variant (the production pass uses `flow.dispatch`). ✅

So the kernels are usable on Metal via IREE *today* through hand-MLIR.

## JAX binding — DONE

5. **Compiler pass `ConvertFlashAttentionDispatch`** (patches/05-07) lowers
   `stablehlo.custom_call @flash_attention_{fwd,bwd_dq,bwd_dkdv}` to a
   `flow.dispatch` of the external kernel (shape-specialized string-template
   wrapper; object resolved by absolute path from `NLEARN_FLASH_KERNEL_PATH`). ✅
6. **`attention.py`** `_attention_iree_flash` = `jax.ffi.ffi_call` + `jax.custom_vjp`
   (fwd saves Q,K,V,O,L; bwd computes D then dq + dkdv kernels). ✅
7. Forward+gradient validated vs NumPy (2.4e-7); the full model **trains on the
   M3 GPU** through the kernels (~1.3s/step). ✅

### Caveat: a separate metal-spirv fusion bug (not the kernel)

The experimental metal-spirv backend miscompiles some large *composed* graphs
even though every individual op is correct in isolation (verified vs CPU). It
makes `_attention_standard` wrong (1.75) — the flash kernel fixes that — and
makes train.py's 13-chunk cross-entropy drive the loss negative under
overfitting. Workaround: a simple unchunked one-hot CE (see
`train.py:_simple_cross_entropy_loss`, auto-selected on iree_metal) compiles
correctly. Also: gather's backward (scatter) miscompiles under vmap, so use
one-hot select (not `take_along_axis`) in losses.

## Remaining work (the JAX frontend binding + training)

### 1. Compiler: a preprocessing pass `custom_call @flash_attention -> dispatch.extern`

The transform-dialect route is **ruled out**: it has no matcher for a
`custom_call` by `call_target_name` (only structural / `has_no_lowering_config`
matchers exist), so it can't distinguish our kernel from JAX's `@Sharding`
custom_calls. So this must be a C++ pass.

- New preprocessing pass (register in the StableHLO input preprocessing
  pipeline, runs before legalization which would otherwise reject the
  custom_call — same wall as the Sharding issue).
- Match `stablehlo.custom_call` with `call_target_name` in
  {`flash_attention_fwd`, `flash_attention_bwd_dq`, `flash_attention_bwd_dkdv`}.
- **Implementation approach (refined):** rather than build `hal.dispatch.extern`
  op-by-op in C++ (the builder needs the count region + layout + objects attrs
  constructed by hand), **string-template the validated MLIR wrapper** (see
  `flash_entry_point.mlir`), substitute the call's concrete shapes
  (n_heads/seq_len/d_head → tensor sizes, workload), `parseSourceString` it,
  clone the wrapper func into the module (dedup per shape), and replace the
  custom_call with a `func.call`. Minimal hand-built IR.
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
