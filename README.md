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
*same* architecture/batch on **three axes**: **(1) val loss ≤ 5.71**, **(2) MFU
competitive (~68%)**, and **(3) GPU utilization at parity** (GPU kept busy, not idle
between dispatches — distinct from MFU). Autonomous, multi-day/week execution.

**Measuring GPU utilization:** ground truth = `sudo powermetrics --samplers gpu_power`
("GPU HW active residency"); accessible proxies = `ioreg -r -c IOAccelerator -d 1`
→ "Device Utilization %" (graphics-biased, sample over time) and the
**Σ(kernel GPU time) / step wall-time** ratio (from `iree-benchmark-module` per
dispatch). Must measure jax-metal's util the same way for the parity number.

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

### Gap analysis (three problems)

1. **Loss gap = batch size / LR.** dtjp60zy hit 5.68 in **1000 steps** because bs32
   makes gradients ~8× less noisy, so **peak_lr=1e-3 is stable** and converges fast.
   At bs4, 1e-3 *diverges* (proven) → forced to 3e-4 → plateau at 7.1. **Fix: bs32 +
   peak_lr=1e-3** (likely reproduces ~5.7 directly, if it fits in 16 GB).
2. **MFU gap.** Batched refactor (done) fixed the worst of this, but throughput is
   FLAT ~0.6 TFLOPS across bs8/16 → the GEMM is inefficient at the model's small-K
   shapes (K=512; it was tuned at K=2048). **Fix: re-tune the GEMM at the model's
   real shapes.**
3. **GPU-utilization gap (NEW axis).** Prelim: GPU appears idle most of each step
   (`ioreg` Device Utilization ~6% time-averaged during a live step; needs
   `powermetrics` confirmation) — consistent with the flat 0.6 TFLOPS being
   *overhead/idle*-bound, not compute-bound. Suspects: many small runtime-compiled
   dispatches with host sync/launch gaps between them, the MultiSteps host logic, and
   per-step Python. **Fix: cut GPU idle — fuse/coalesce dispatches, overlap/async
   submission, minimize host↔device sync, reduce per-step host overhead.** This
   overlaps with the MFU work but is measured separately (busy-time, not FLOPS-eff).

### The linchpin

The **batched-execution refactor** delivers *both* goals at once: it removes the
32-way `vmap`-sequential graph (fixing the bs32 miscompile → correctness), feeds the
kernels big batched shapes (one GEMM at M=bs·seq, one flash over bs·heads → MFU), and
reshapes memory into the layout jax-metal used. So it's the backbone of the plan.
Grad-accumulation (bs8 micro-batches ×4) is a **loss-only fallback** if batched bs32
memory proves intractable.

### Day-by-day plan

**Day 1 — Batched-execution refactor (model.py + train.py).** *Correctness first.*
- `gemm_iree`: a `linear(x, W)` that flattens leading dims `(…, K)→(N_tok, K)`, runs
  the 2D GEMM (M=N_tok), reshapes back. (Kernel unchanged.)
- `attention_forward`: operate on `(bs, seq, d)`; reshape `→ (bs·heads, seq, dh)`,
  call `_attention_iree_flash` directly (NO vmap → one dispatch), reshape back.
- `ffn_forward` / `layer_norm` / `embed`: take a leading `bs` dim (last-axis ops are
  already batch-safe; just thread the shape through).
- `model_forward(_features)`: `token_ids` is `(bs, seq)`.
- `train.py`: drop `batched_loss_fn = vmap(...)`; `batch_loss`/CE operate on
  `(bs, seq[, vocab])` directly.
- **Gate D1:** batched model trains correctly at bs4/bs8 — loss curve matches the old
  vmap path (numeric parity on a few steps), each matmul is a single big GEMM
  (verify shapes/dispatch count).

**Day 2 — bs32 correctness + memory.**
- Run batched model at **bs32**. Expectation: the loss-0/nan miscompile is gone (no
  32-way vmap; single batched ops, smaller graph). Confirm loss sane + decreasing.
- Memory: `(bs·seq, vocab)` logits ≈ 3.3 GB + one-hot + backward ≈ ~10 GB → may OOM.
  Fix in order of preference: (a) **vocab-chunked CE in batched form** (select target
  logit without a full one-hot — careful: gather-backward miscompiles, so use a
  metal-spirv-safe reduction); (b) **fused MSL cross-entropy kernel** (loss + dlogits,
  no logit-row materialization — new kernel + dispatch-pass binding + custom_vjp);
  (c) **gradient accumulation** (bs8 micro ×4) as the fallback.
- **Gate D2:** bs32 trains correctly within 16 GB (target ≈ 9 GB), loss decreasing.

**Day 3 — Throughput / MFU / GPU utilization.**
- **Establish the GPU-util baseline & reference:** measure our step's
  Σ(kernel GPU time)/wall-time and `powermetrics` residency; measure a short jax-metal
  run the same way for the parity target.
- Profile the bs32 step with `iree-benchmark-module` on the *real* shapes; attribute
  wall-time to GEMMs / flash / CE / **idle gaps & host overhead**.
- Re-tune the GEMM at the model's shapes (M=16384, K/N ∈ {512, 2048, 50272}) — it was
  tuned at 2048³; sweep tile/BK there. Re-tune flash at `(bs·heads=256, seq=512, dh=64)`.
- **Cut GPU idle:** fuse/coalesce dispatches, overlap/async submission, minimize
  host↔device sync, reduce per-step Python/MultiSteps host overhead.
- **Gate D3:** end-to-end **≥ ~1.5–2.4 TFLOPS, MFU ≥ ~60%, GPU util at parity with
  jax-metal, bs32 step competitive.**

**Day 4 — Competitive run.**
- Match the reference schedule (peak_lr **1e-3**, end_lr 1e-4, warmup 200); confirm
  1e-3 is stable at bs32 (reference says yes). Run **1000 steps**, monitor loss curve
  vs reference, MFU, mem; checkpoint every 1000 / cap val.
- **Gate D4 (HEADLINE):** **val ≤ 5.71 at MFU *and* GPU utilization competitive with
  jax-metal** — the full three-axis verification on our drivers.

**Day 5+ — Harden / surpass.**
- If loss short: tune LR/warmup/init/optimizer to the reference exactly; longer run.
- If MFU short: deeper kernel work — register-blocked GEMM toward MLX's ~2.9 TFLOPS,
  FA-2 simdgroup-matrix attention.
- Stretch: more tokens (loss < 5.6), bf16, larger model/context, KV-cache eval.

### Risks & fallbacks
- *Batched graph still miscompiles* → bisect (proven method); fall back to grad-accum.
- *bs32 CE OOM* → chunked CE → fused CE kernel → grad-accum (in that order).
- *Flash at bs·heads=256 or one huge dispatch misbehaves* → it's one dispatch (not the
  many-dispatch accumulation that caused the val hang), but verify; tile heads if needed.
- *1e-3 diverges at real bs32* (shouldn't — reference used it) → scale LR / longer warmup.
- Every phase has a correctness gate; don't advance until met. Record results below.

### Autonomous execution protocol
One experiment at a time; GPU-health check + `pkill` between runs (hung kernels need
recovery time). Monitor via the unbuffered W&B `output.log` + CPU-advance stall
detection (piped stdout is block-buffered — unreliable for liveness). Cap validation
(`NLEARN_VAL_BATCHES`); checkpoint every 1000 steps; recover from latest checkpoint on
hang/crash; watch LR divergence and OOM. Record every result (config → val, MFU,
**GPU util**, mem, step) in the log below; commit working changes; capture compiler
edits as patches.

### Results log
- (baseline) bs4 / lr3e-4 / 10k → val ~7.1, MFU ~57%, ~0.4 TFLOPS  [run_10k_lr3e4]
- **bs32 feasibility (vmap-sequential): DEAD END.** GEMM-off → OOM
  (`RESOURCE_EXHAUSTED` on the bs32 logits); GEMM-on → loss 0/nan (32-way vmap graph
  miscompiles on metal-spirv) + 52 s/step. ⇒ Phase 0 blocked; Phase 1+2 both required.
- **Key insight:** kernels already accept batched shapes (GEMM takes any M ⇒
  `(bs·seq, d)` in one dispatch; flash loops per-"head" ⇒ `(bs·heads, seq, d)` in one
  dispatch). **Phase 2 = model/train refactor to batched execution** (drop
  vmap-sequential), which fixes miscompile + memory-shape + MFU at once.
- **Phase 2 batched refactor DONE (Day 1):** correct + 2× faster. bs8 batched: loss
  sane, **MFU 31%→90%** (vs naive peak), 0.64 TFLOPS (was 0.4), 2.5 s/step. Single
  big GEMMs replace per-sequence dispatches.
- **Day 2 findings:** batched bs8/16 train fully; **bs16 is the single-pass ceiling**
  (bs24 grad fits but the full train_step OOMs on Adam state; bs32 OOMs — the
  lm_head+CE materializes ~4–5×3.3 GB buffers). True bs32 ⇒ needs `optax.MultiSteps`
  (manual grad-accum OOMs: both micro-graphs stay live) or a fused-CE kernel.
- **Throughput is FLAT ~0.6 TFLOPS across bs8/16** ⇒ the MFU gap (0.6 vs jax-metal
  2.4, ~4×) is **GEMM efficiency at the model's small-K shapes (K=512)**, not batch
  (the kernel was tuned at K=2048). ⇒ Day 3 = re-tune GEMM at model shapes.
- **bs16 / lr1e-3 DIVERGED** (val 7.7→9.2): lr=1e-3 only stable at the reference's
  bs32 averaging. ⇒ true effective-bs32 required for the loss target.
- **Effective bs32 via `optax.MultiSteps` (bs16×2) WORKS** (each step bs16 mem, 2.7 GB):
  loss converging + stable (val 10.94→8.76 @ 50 updates, no divergence). Loss-target
  run `competitive_effbs32` (lr1e-3, 1000 updates, token-matched) → _in progress_.
- **GPU-utilization (3rd axis, NEW):** prelim `ioreg` Device Utilization ~6%
  time-averaged during a live step → GPU likely idle-bound (overhead/dispatch gaps),
  consistent with flat 0.6 TFLOPS. Needs `powermetrics` confirmation + jax-metal
  reference. ⇒ Day 3 also reduces GPU idle (fuse/overlap dispatches, cut host sync).
- **GPU-UTILIZATION investigation (now top priority per user).** True util needs
  `sudo powermetrics` (unavailable to the agent — user can run
  `sudo powermetrics --samplers gpu_power`); `ioreg` Device Utilization % is graphics-
  biased (unreliable). Profiled the bs8 grad step (2250 ms) instead:
  blocks=1624 ms, lm_head+CE=626 ms. GEMM is fine at most shapes (mlp/lm_head 2.0–2.4
  TFLOPS) — only the small **proj 4096×512×512 = 0.58 TFLOPS**; gemms total only
  ~130 ms. So the step is dominated by the **scalar flash backward (~600 ms)** + a
  long tail of **small unfused elementwise/reshape/transpose dispatches** (IREE-Metal
  doesn't fuse like jax-metal's XLA → GPU idles between many small dispatches). fp16
  activations (linear→fp16) were speed-neutral ⇒ not memory-traffic-bound ⇒ it's
  **dispatch/idle-bound**. **Levers (Day-3, reprioritized):** (1) FA-2 flash backward
  on simdgroup_matrix (kills the ~600 ms scalar cost); (2) cut dispatch count / fuse
  the elementwise+reshape tail; (3) get powermetrics busy% to quantify idle.
- **The recurring "~100-step hang" ROOT-CAUSED (the supervisor was a dead end —
  sawtooth, never progressed).** Systematic isolation: NOT validation (off in every
  hang), NOT a memory leak (RSS flat), NOT MultiSteps (hangs without it), NOT a fixed
  threshold (random iter 10–130), and NOT either custom kernel (flash & GEMM each ran
  400 iters clean at the batched shapes). It's the **combined batched model graph
  faulting the Metal HAL** — full fwd+bwd loop (no loader/opt/wandb) hangs ~iter 60.
  Discriminator: **bs16 graph faults within ~60 iters; bs8 graph ran 400 clean** ⇒
  it's the **per-graph working-set size** (bs16 ⇒ M=8192 GEMMs + 1.6 GB×3 logit/
  one-hot/dlogit buffers stress the HAL; the old per-sequence 10k run had tiny
  bs4/M=512 dispatches). **Fix (no supervisor): bs8 micro-batch (stable) +
  MultiSteps ×4 ⇒ effective bs32** — same effective gradient, lr=1e-3 stable. Loss
  run `comp_bs8x4` relaunched this way → _in progress_.

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
