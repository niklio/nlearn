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

**Hardware:** Apple **M3**, 10-core GPU, **16 GB** unified memory.

**Two binding constraints (measured during the migration):**
- **Memory** — 16 GB is shared by activations, params, optimizer state, and
  logits, so memory-saving routines matter as much as speed for long context.
- **Matmul throughput** — IREE's `metal-spirv` codegen runs at **~0.5 TFLOPS**
  (true device time). The achievable fp16 peak on this M3 is **~2.9 TFLOPS**
  (measured with MLX, Apple's own tuned GEMM) — *not* the "teens" first assumed.
  Our custom `simdgroup_matrix` GEMM now hits **~2.4 TFLOPS (≈84% of MLX, ~4.8×
  naive)**; see [`iree_metal/kernels/GEMM.md`](iree_metal/kernels/GEMM.md).

**Enabler:** the custom-MSL-kernel → IREE `flow.dispatch` pass + `custom_vjp`
binding is in place, so each new SoTA kernel is a repeatable pattern (author MSL,
add the target name to the pass, wire the VJP).

## P0 — routines that gate long-context training

- [~] **FlashAttention kernel throughput.** Forward kernel **vectorized (float4
  loads + dot/FMA) → 2.4× faster** (6.9 ms → 2.8 ms at H=8,S=512,D=64), correct,
  same ABI (no compiler/JAX changes). *Finding:* a threadgroup-**tiled** variant
  (shared K/V staging) was ~50% *slower* and `simdgroup_matrix` doesn't fit the
  one-thread-per-query layout — occupancy + the GPU cache beat manual staging,
  same lesson as the GEMM sweep. The backward kernels are register/occupancy-bound
  (vectorizing them gave no gain), so they stay scalar. **Remaining (optional):** a
  full FA-2 cooperative-matrix rewrite (restructure to 8×8 tiles, 2D head-aware
  grid) — needs a compiler-pass change for the grid; uncertain payoff given the
  occupancy findings. *Effort: high.*
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

- [~] **Batch-aware attention dispatch — remove `vmap_method="sequential"`.**
  *Deferred (evidence-based).* Its two motivations both evaporated: (1) the hang it
  was meant to fix was the validation loop, not train-step dispatch count (see
  below); (2) it would only help *large*-batch throughput, but measured MFU is best
  at small batch (B=2/4: 59%; B=8: 46%) and memory is never the limit (~2.5 GB at
  B=8) — so we don't want large batch. Step time scales ~linearly with batch
  (constant throughput), so the B=2/4 runs we actually use see ~no benefit. Not
  worth the model-refactor / `custom_vmap` risk. Revisit only if large global batch
  becomes desirable.
- [x] **Root-cause the Metal HAL runtime hang.** *DONE.* Bisected exhaustively:
  the flash kernels are fine in isolation (fwd/bwd/vmap/chained up to seq=512), the
  full train_step is fine, and `train.py` only hung at seq≥256 — the trigger was the
  **validation loss looping over the entire ~100k-token held-out set** (~100+ rapid
  forward passes in one call), which trips Metal HAL accumulation at seq≥256 (works
  at seq=128 where each pass is smaller). Fix: cap validation to `NLEARN_VAL_BATCHES`
  (default 20) in `logging_utils._compute_val_loss`. seq=512 / 10 steps now trains
  with validation, no hang (1.0s/step with GEMM, MFU ~59%).
- [~] **bf16 support in the `metal-spirv` target.** *Deferred.* High-effort,
  uncertain compiler work (bf16 fails at `vector.bitcast` SPIR-V legalization; needs
  new lowering + spirv-cross→MSL support). Its purpose — wider dynamic range so loss
  scaling isn't needed — is covered by the interim below, which is now in place. Not
  a blocker for training; revisit if fp16+loss-scaling proves insufficient.
- [x] **fp16 loss scaling (interim, until bf16).** *DONE.* Static scale (default
  2^10, `NLEARN_LOSS_SCALE`) in `make_train_step`: scale the loss so small gradients
  survive the fp16 backward, unscale before the optimizer update. Static (no
  `lax.cond` overflow-skip — metal-spirv miscompiles control flow). Loss trajectory
  unchanged; guards long runs against silent gradient underflow.

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
