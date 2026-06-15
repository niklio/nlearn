# nlearn

A from-scratch GPT-style transformer in JAX, trained on Apple Silicon through the
open-source **IREE-Metal** stack with a hand-authored native Metal FlashAttention
kernel (forward + true O(seq) backward). `cluster.py` provisions and runs jobs
(train / generate / submit) on a remote Mac mini.

See [`iree_metal/README.md`](iree_metal/README.md) for the IREE-Metal setup and
[`iree_metal/BINDING.md`](iree_metal/BINDING.md) for how custom Metal kernels are
wired into JAX (the custom_call → `flow.dispatch` compiler pass + `custom_vjp`).

---

# Optimization roadmap

**Goal:** train a long-context model as efficiently as possible on the Mac mini,
writing new Metal kernels where needed to enable SoTA routines.

**Hardware:** Apple **M4**, 10-core GPU, **16 GB** unified memory.

**Two binding constraints (measured during the migration):**
- **Memory** — 16 GB is shared by activations, params, optimizer state, and
  logits, so memory-saving routines matter as much as speed for long context.
- **Matmul throughput** — IREE's `metal-spirv` codegen runs at **~0.5 TFLOPS**
  (true device time). The achievable fp16 peak on this M4 is **~2.9 TFLOPS**
  (measured with MLX, Apple's own tuned GEMM) — *not* the "teens" first assumed.
  Our custom `simdgroup_matrix` GEMM now hits **~2.4 TFLOPS (≈84% of MLX, ~4.8×
  naive)**; see [`iree_metal/kernels/GEMM.md`](iree_metal/kernels/GEMM.md).

**Enabler:** the custom-MSL-kernel → IREE `flow.dispatch` pass + `custom_vjp`
binding is in place, so each new SoTA kernel is a repeatable pattern (author MSL,
add the target name to the pass, wire the VJP).

## P0 — routines that gate long-context training

- [ ] **Tiled FlashAttention kernel (simdgroup_matrix + threadgroup tiles).**
  Current kernel is correct but scalar (one thread/query, no tiling, no matrix
  units). Rewrite FA-2 style (block over Q/K/V tiles in threadgroup memory,
  `simdgroup_matrix` for QKᵀ and PV, online softmax across K-blocks); same for
  the backward kernels. *Biggest "write a Metal driver" win. Effort: high.*
- [~] **Custom Metal GEMM with `simdgroup_matrix`.** *Diagnostic done
  ([`iree_metal/kernels/GEMM.md`](iree_metal/kernels/GEMM.md)): not a flag — IREE's
  Apple target has `mmaCount=0` and spirv-cross can't emit `simdgroup_matrix`, so
  a custom kernel is required.* Kernel built, validated, and **optimized to ~2.4
  TFLOPS** (multi-simdgroup, vectorized half4 staging, BK=64) — **≈84% of MLX's
  ~2.9 TFLOPS peak, ~4.8× IREE naive**. Profiled with `iree-benchmark-module`;
  occupancy (not bandwidth) is the limiter, staging vectorization was the big win.
  **Remaining:** route the model's big matmuls (MLP, projections, lm_head) through
  it via the dispatch pass + dim/transpose handling. *Impact: very high
  (everything). Effort: medium (kernel done; integration left).*
- [ ] **Fused cross-entropy kernel (online-softmax, no logit materialization).**
  The simple CE materializes the `(L × 50257)` logit matrix (~1.6 GB/seq at
  L=8k, ×batch ×fwd+bwd); the chunked CE that avoided it is broken on
  metal-spirv. A fused MSL kernel (loss + dlogits without materializing the row)
  is essential to fit long sequences in 16 GB. *Impact: very high (memory).
  Effort: medium.*

## P1 — enablers (throughput, stability, bigger batch/seq)

- [ ] **Batch-aware attention dispatch — remove `vmap_method="sequential"`.**
  Today each attention call issues `batch_size` separate dispatches. Flatten
  batch into the kernel grid (one dispatch); speeds up throughput and removes the
  dispatch count that triggers the runtime hangs. *Effort: medium.*
- [ ] **Root-cause the Metal HAL runtime hang at high dispatch count.** Forced
  B=2; gates batch size and run length. Likely command-buffer/semaphore
  accumulation in IREE's Metal HAL. *Effort: medium–high, uncertain.*
- [ ] **bf16 support in the `metal-spirv` target.** M4 supports bf16 in HW; IREE
  doesn't codegen it (forces fp16). bf16's wider range removes the need for loss
  scaling and reduces wasted long runs. *Effort: medium (compiler target).*
- [ ] **fp16 dynamic loss scaling (interim, until bf16).** Guards against silent
  gradient underflow over long runs. *Effort: low.*

## P2 — memory headroom & instrumentation

- [ ] **Optimizer-state & activation memory.** Adam moments in bf16 (~2× model
  size of RAM saved); tune `jax.checkpoint` remat granularity for long L.
  *Effort: low–medium.*
- [ ] **AOT metallib.** Install full Xcode on the mini → drop
  `--iree-metal-compile-to-metallib=false`, removing the slow first-step runtime
  kernel compile. *Effort: low.*
- [ ] **Real profiling.** Fix the MFU benchmark (the 0.8 TFLOPS figure) for true
  peak; use Metal capture to target the GEMM/attention work. *Effort: low.*
- [ ] **KV-cache in `generate`.** Inference recomputes full attention per token;
  lower priority given the training focus. *Effort: medium.*

## Recommended first move

Before investing in a GEMM kernel, run the **~half-day diagnostic** (P0 #2 / P2
profiling): dump the spirv-cross MSL for a matmul and check whether it's naive
loops vs `simdgroup_matrix`, and whether any IREE target-feature flag enables
cooperative/matrix ops. That tells you whether the 15–20× is a config fix (cheap,
huge) or requires custom GEMM kernels — and sharpens the whole P0 plan.

> Strategic note: if `metal-spirv` codegen is fundamentally far from the matrix
> units, P0 #1–2 mean reimplementing SoTA GEMM/attention in MSL (what MLX already
> ships). Staying on IREE keeps the JAX/autodiff stack; just go in knowing that.
