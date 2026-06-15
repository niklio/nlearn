# nlearn

A from-scratch GPT-style transformer in JAX, trained on Apple Silicon through the
open-source **IREE-Metal** stack with a hand-authored native Metal FlashAttention
kernel (forward + true O(seq) backward). `cluster.py` provisions and runs jobs
(train / generate / submit) on a remote Mac mini.

See [`iree_metal/README.md`](iree_metal/README.md) for the IREE-Metal setup and
[`iree_metal/BINDING.md`](iree_metal/BINDING.md) for how custom Metal kernels are
wired into JAX (the custom_call → `flow.dispatch` compiler pass + `custom_vjp`).

---

# Competitive-performance roadmap (current objective)

**Goal:** match the jax-metal reference run `dtjp60zy` (*pretty-monkey-41*) at the
*same* architecture/batch — **val loss ≤ 5.71** with **MFU competitive to jax-metal
(~68%)**. Autonomous, multi-day/week execution.

### Reference target (jax-metal, measured from W&B)

| metric | dtjp60zy (jax-metal) | our best so far |
|---|---|---|
| arch | d512 / 8h / 4L / seq512 / 64M | same |
| batch_size | **32** | 4 |
| steps | **1000** | 10000 |
| peak_lr / warmup | **1e-3** / 200 | 3e-4 / 200 |
| **val loss** | **5.71** | 7.1 |
| **MFU** | **67.9%** | ~57% (vs naive peak) |
| achieved throughput | **~2.4 TFLOPS** | **~0.4 TFLOPS** |
| step_time | 2.6 s (bs32) | ~2 s (bs4) → ~16 s (bs32, sequential) |
| peak mem | 9.1 GB | 3 GB (bs4) |

### Gap analysis (two independent problems)

1. **Loss gap = batch size / LR.** dtjp60zy hit 5.68 in **1000 steps** because bs32
   makes gradients ~8× less noisy, so **peak_lr=1e-3 is stable** and converges fast.
   At bs4, 1e-3 *diverges* (proven) → forced to 3e-4 → plateau at 7.1. **Fix: bs32 +
   peak_lr=1e-3** (likely reproduces ~5.7 directly, if it fits in 16 GB).
2. **MFU gap = dispatch batching (~6×).** Our custom GEMM does 2.4 TFLOPS *standalone*,
   but the model runs under `vmap_method="sequential"`, so bs32 = 32 tiny separate
   dispatches feeding small (512×512) GEMMs where overhead dominates → ~0.4 TFLOPS
   end-to-end. jax-metal batches into single big ops (M=bs·seq=16384). **Fix: batched
   execution — one big dispatch per op, not bs× sequential ones** (the deferred
   "batch-aware dispatch" P1, now the central lever).

### Phases

- **Phase 0 — Reproduce the loss target (de-risk loss, ignore speed).** Run **bs32,
  peak_lr=1e-3, 1000 steps** on the *current* stack (sequential, slow, if it fits).
  *Milestone: val ≤ ~5.8.* Proves the loss is achievable on our kernels and isolates
  loss from throughput. If it OOMs → Phase 1 first.
- **Phase 1 — Memory for bs32 (if Phase 0 OOMs).** The `(bs, seq, vocab)` logits +
  one-hot in `_simple_cross_entropy_loss` are ~3.3 GB *each* at bs32 (≈10 GB with
  backward). Implement a **fused / online-softmax cross-entropy** (no full-logit-row
  materialization). *Milestone: bs32 trains within 16 GB (~9 GB like the reference).*
- **Phase 2 — Throughput / MFU (the central work; close the ~6× gap).** Move off
  `vmap`-sequential to **batched execution** so kernels see big shapes:
  (2a) flatten `(bs,seq,d)→(bs·seq,d)` → each matmul is ONE GEMM with M=16384;
  (2b) one flash dispatch over `(bs·heads,seq,d)`; (2c) batched/fused CE;
  (2d) re-tune kernels at the real model shapes (GEMM was tuned at 2048³).
  *Milestone: end-to-end ≥ ~1.5–2.4 TFLOPS, MFU ≥ ~60%, bs32 step ≤ ~3–4 s.*
- **Phase 3 — Competitive run.** Tune LR/schedule/optimizer to track the reference
  curve; run the 1000-step (and longer) competitive run. *Milestone: **val ≤ 5.71 at
  MFU competitive with jax-metal** — the headline deliverable.*
- **Phase 4 — Surpass / scale (open-ended).** More tokens (loss < 5.6); larger
  model/context; kernels toward MLX-level (register-blocked GEMM, FA-2 simdgroup
  attention); bf16; KV-cache eval.

### Autonomous execution protocol
One experiment at a time; GPU-health check + `pkill` between runs (hung kernels need
recovery time). Monitor via the unbuffered W&B `output.log` + CPU-advance stall
detection (piped stdout is block-buffered — unreliable for liveness). Cap validation
(`NLEARN_VAL_BATCHES`); checkpoint every 1000 steps; recover from latest checkpoint on
hang/crash; watch LR divergence and OOM. Record every result (config → val, MFU, mem,
step) in the log below; commit working changes; capture compiler edits as patches.

### Results log
- (baseline) bs4 / lr3e-4 / 10k → val ~7.1, MFU ~57%, ~0.4 TFLOPS  [run_10k_lr3e4]
- Phase 0: bs32 / lr1e-3 / 1000 → _in progress_

---

# Optimization roadmap (foundational work — mostly done)

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
- [x] **fp16 loss scaling (interim, until bf16).** *Done — and found unnecessary
  for this model.* Implemented a static scale (`NLEARN_LOSS_SCALE`, unscale before
  the optimizer step; no `lax.cond` overflow-skip since metal-spirv miscompiles
  control flow). A 250-step validation run revealed the static scale (2^10) *causes*
  NaN — it overflows fp16 in the backward once gradients grow post-warmup. Crucially
  the model trains **stably in fp16 with no scaling** (no underflow; loss falls
  cleanly to ~7.8 over 150 steps), so the default is now **off**; the knob remains
  for manual use. Proper dynamic scaling (back off on overflow) would need lax.cond
  → tied to the metal-spirv control-flow gap.

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
