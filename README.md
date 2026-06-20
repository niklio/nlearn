# nlearn

A from-scratch GPT-style transformer in JAX, trained on Apple Silicon through the
open-source **IREE-Metal** stack with a hand-authored native Metal FlashAttention
kernel (forward + true O(seq) backward). `cluster.py` provisions and runs jobs
(train / generate / submit) on a remote Mac mini.

See [`iree_metal/README.md`](iree_metal/README.md) for the IREE-Metal setup and
[`iree_metal/BINDING.md`](iree_metal/BINDING.md) for how custom Metal kernels are
wired into JAX (the custom_call → `flow.dispatch` compiler pass + `custom_vjp`).

**Repository layout & conventions:** see [`STRUCTURE.md`](STRUCTURE.md) — directory
map and the rules for where new files go. The core library is the `nlearn/` package
(`python -m nlearn.train`); ops scripts in `scripts/`, benchmarks in `bench/`,
utilities in `tools/`.

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
- **GPU-util/throughput WIN — tiled flash backward.** The scalar flash backward was
  the bottleneck (54% of step, 0.03 TFLOPS, memory-bound: 64 threads/block re-read the
  same Q/dO or K/V ~64× from global). Staged those into threadgroup memory (reuse
  across the block): flash fwd+bwd **313→86.5 ms (3.6×)**, correct; **end-to-end bs8
  step 2250→1288 ms, 0.5→1.23 TFLOPS (~half of jax-metal's 2.4, was 1/5).** Next
  biggest chunk is now lm_head+CE (~49% of the step). (Backward requires seq%64; model
  uses 512.)
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
- **Process hardening (2026-06-16).** A bs16 probe died with INVALID_ARGUMENT then
  hung 13 h at 0 % CPU in wandb's atexit network-flush — invisible because output was
  piped through `tail` (buffers, never flushes) and I only waited on completion
  notifications. Fixed with `scripts/run_watch.sh` (unbuffered real-file logging + watchdog:
  hard wall-timeout = the no-idle backstop, plus a stall detector that kills on
  "no output AND CPU flat") and `scripts/jobs.sh` (at-a-glance liveness poller). **bs16 still
  faults the HAL even with the fp16-logit downcast ⇒ bs8 stays the micro-batch ceiling.**
- **GEMM 64×64 tile = DEAD END.** Hypothesised the narrow lm_head dA gemm (M4096
  K50272 N512, 1.43 TFLOPS) was memory-bound and bumped the tile 32×32→64×64 (SG 2→4,
  512 threads; +grid/wg changes in the pass + 64-pad in gemm_iree). **Strictly worse
  across the board** (fwd 2.06→1.02, dB 2.62→1.37): 512-thread / 16 KB-shared tiles
  collapse occupancy. 32×32 is already the sweet spot; square 2048 hits 2.43/2.9 ≈ 85 %.
  Reverted. ⚠️ *Reverting via `git checkout` in ~/src/iree deleted the whole uncommitted
  dispatch pass* — recovered by re-applying `iree_metal/patches/08` (the combined
  flash+gemm superset; 07 is its subsumed predecessor). Verified: GEMM bench matches
  baseline, full bs8 train runs clean.
- **THE UTIL SINK IS NON-GEMM, not the GEMMs.** Full bs8 step sustains **~0.99 TFLOPS**
  while individual GEMMs do 2.0–2.7 ⇒ ~60 % of wall time is non-GEMM (flash, layernorm/
  gelu/residual elementwise, cross-entropy gather, host dispatch gaps). The MFU readout
  was bogus (138 %): `benchmark_peak_tflops` measured IREE's *naive* `jnp.matmul`
  (~0.7 TFLOPS) as "peak" — fixed to route through the custom GEMM kernel (fp16), so MFU
  now = fraction of achievable-kernel throughput (gap below 100 % = non-GEMM overhead).
  Apples-to-apples target is **achieved step TFLOPS 0.99 → jax-metal's 2.4.**
- **`clip_by_global_norm` doesn't compile on metal-spirv** — its `optax.tree.norm`
  reduction emits `vector.create_mask` (vector<4xi1> masked tail) which fails to
  legalize (distinct from the patched `vector.step`). Clipping never fixed the lr=1e-3
  divergence anyway ⇒ default `NLEARN_GRAD_CLIP=0`. Next loss-axis lever: static
  loss-scaling (`NLEARN_LOSS_SCALE`) for fp16 gradient underflow.
- **GEMM-based attention to replace flash = DEAD END (measured).** Hypothesised that
  at seq=512 the SxS scores fit in memory so routing QK^T/P@V through the 2.4-TFLOPS
  GEMM (autodiff giving the backward via gemm's own vjp — no custom bwd kernel) would
  beat flash's 0.08 TFLOPS. Benched at (64,512,64): **flash wins** — fwd 35.5 vs 40.8ms,
  fwd+bwd 182 vs 198ms (contended; ratios valid). Why: attention matmuls are small-K
  (QK^T K=64 = a single BK tile, memory-bound; the GEMM kernel's reuse needs large K)
  and the 64-way batch becomes 128 small dispatches whose host overhead swamps the MMA
  win. The tiled flash kernel is already the better structure; beating it needs a
  *batched/fused* simdgroup-MMA attention (one dispatch), and even that is capped by
  small-K (dh=64) inefficiency — high effort, modest upside. Deprioritised.
- **Step breakdown (bs8 microstep ~1.3–1.57s):** lm_head fwd+bwd ~320ms, flash×4 ~363ms,
  other gemms (mlp/proj) ~64ms ⇒ ~750ms of named kernels; the remaining ~40% is
  elementwise (layernorm/gelu/residual), the cross-entropy logsumexp/gather over vocab
  50272, and host dispatch gaps. **That unmeasured ~40% (esp. CE + host gaps) is the
  most likely remaining util lever** — needs isolation before more kernel work.
- **lr=1e-3 is STABLE without clipping (breakthrough for the loss axis).** Run
  `comp_lr1e3_w` (effective bs32 = bs8×4, peak 1e-3, no clip, ~80-update warmup):
  through peak LR (eff updates ~62–90) loss holds at 7.6–8.1 (val 7.72) — **plateaus,
  does NOT diverge** — then resumes descending as LR cosine-decays. The earlier "lr=1e-3
  diverges" was an artifact of the (now-removed, uncompilable) clipping / shorter warmup.
  Honest MFU steady **~40%** the whole run. ⇒ the reference-matching config (bs32,
  lr1e-3, warmup200, 1000 updates, end 1e-4) is now viable for the headline run.
- **UTIL ROOT CAUSE: the step is GPU-bound, NOT host-gap-bound (profiled).** bs8 step
  value_and_grad = **1612 ms with only 45 ms (3 %) host gaps** (pipelined ≈ blocked) ⇒
  the GPU is busy ~97 % of the step; the "~20 % util" the `ioreg` counter shows is its
  graphics-bias under-reporting compute, not real idle. **Backward = 1231 ms (76 %)**;
  forward = 310 ms, CE = +70 ms. So all leverage is in the BACKWARD, and it's spread
  (flash bwd ~305 ms, lm_head bwd ~220 ms, + a large mass of layernorm/gelu/softmax/CE-
  gradient elementwise that IREE codegens naively) — no single silver bullet; raising
  util further needs custom fused backward kernels per op (high effort, diffuse payoff).
  ⇒ pivot focus to the loss axis (the headline gate), where concrete gains remain.
- ⚠️ **lr=1e-3 over a FULL 400-update run SLOWLY DIVERGES** (val 7.72→8.15 across the
  decay) — my "stable" read was premature (only saw to step 360). BUT that run's warmup
  was capped at 80 updates (`eff_updates//5`) vs the reference's 200. Testing the true
  reference config (1000 updates ⇒ warmup 200) with an auto-abort if val climbs after
  warmup (`scripts/monitor_loss.sh`), fallback lr=6e-4.
- **AdamW (wd=0.1) does NOT fix the lr=1e-3 wall — it's numerical, not the optimizer.**
  Identical val floor: **7.5267 (AdamW) vs 7.5272 (Adam)**, identical subsequent climb;
  the abort monitor killed both ~step 1300. Crucially val bottoms at **step 500–700**,
  i.e. while the warmup cosine is still ramping through **lr ≈ 7e-4** — *before* peak —
  then degrades as LR approaches 1e-3. ⇒ **our stable LR ceiling is ~7e-4**; lr=1e-3 is
  unreachable (the gradient-quality gap vs jax-metal's MPS kernels, independent of
  optimizer/warmup/clipping). Reaching the reference's 5.71 at lr=1e-3 is blocked on
  that numerical gap; the realistic path is to descend at the stable ceiling (peak
  6e-4) over many updates. Open root-cause suspects: tiled flash-bwd accuracy at
  seq=512 (validated only at seq=128), fp16 activation round-off accumulation.
- **Flash backward is ACCURATE at seq=512 — ruled out as the lr culprit.** Compared
  jax.grad of the flash custom_vjp vs a near-exact gemm-based materialized-attention
  reference on identical (64,512,64) inputs: dQ/dK rel **2.9e-4**, dV rel **1.1e-5**,
  fwd rel 2.8e-4, **corr 1.00000** on all three. So gradients are fp16-accurate, not
  biased ⇒ the lr=1e-3 wall is genuine "LR too high for this init/config," not a kernel
  bug. ⇒ stop chasing lr=1e-3; descend at the ~6–7e-4 stable ceiling over many updates.
  (`grad_check_flash.py` is the reusable harness.)
- **THE LOSS WALL IS GRADIENT NOISE, not LR/optimizer/kernel-bug (root cause found).**
  Both train AND val bottom at **~7.4–7.5 around step 600–700 (~150 eff updates)** then
  steadily climb — and the climb RATE scales with LR (lr1e-3 degrades ~2× faster than
  6e-4: +0.5 val over 650 vs 1300 steps). Flash grads are accurate (corr 1.0) and data
  streams fresh, so this is **gradient NOISE exceeding what the LR can tolerate**: our
  fp16 custom-kernel gradients are noisier than jax-metal's MPS, so any useful LR slowly
  destabilizes; a lower LR degrades slower but descends too slowly to reach 5.71. The
  principled fix is a **larger effective batch** (variance ∝ 1/batch) to average the
  noise and sustain a useful LR. Testing effective **bs64** (GRAD_ACCUM=8) next.
- **ROOT CAUSE FOUND & VERIFIED: it's fp16's narrow EXPONENT RANGE, not noise/precision/
  kernels.** CPU A/B at bs8 (identical config, only COMPUTE_DTYPE differs), val_loss @
  step 200: **f32 7.07, bf16 7.01, fp16 7.41** (fp16 already lagging at 7.8 by step 100).
  fp16 has MORE mantissa than bf16 yet trains WORSE ⇒ exponent RANGE (overflow as
  activations grow), which bf16 (f32-like range) and f32 avoid. Reproduces off-Metal ⇒
  the dtype, not the custom kernels (flash bwd matched a reference to 3e-4 earlier). bs64
  only slowed the climb (noise is secondary). **The fix is bf16**, and it's viable on
  this stack: (a) M3/macOS-15 supports MSL `bfloat` incl. simdgroup matrix; (b) bf16
  ELEMENTWISE ops (layernorm/gelu/softmax/residual) **legalize on metal-spirv** (tested
  OK) — the bf16 limitation is matmul-only, and matmuls go through our custom kernels;
  (c) bf16 = same memory as fp16, so no HAL working-set fault (f32 both deadlocks IREE's
  native matmul AND risks the fault). ⇒ **implement a bf16-input custom GEMM** (f32
  accumulate), set COMPUTE_DTYPE=bf16; flash already casts to f32 internally. Building now.
- **✅ FIX IMPLEMENTED & VERIFIED ON HARDWARE — the loss wall is broken.** Built a bf16
  GEMM: `gemm.metal` reads the bf16 buffers as `ushort` and widens to f32 in-kernel (MSL
  `bfloat` needs lang 3.1, which IREE's Metal runtime pins to 3.0), f32 simdgroup matmul,
  f32 accumulate; the dispatch pass declares bf16 input tensors; `gemm_iree` casts inputs
  to bf16. Validated rel **0.0** vs a bf16 reference incl. a 3.97e6 case fp16 would
  overflow. **bf16 ACTIVATIONS hit a separate `vector.bitcast` f32↔bf16 legalization gap,
  so instead run COMPUTE_DTYPE=float32 (f32 activations — IREE handles f32 elementwise
  fine) + the bf16 GEMM**: wide range *everywhere* (f32 activations + bf16 matmul inputs),
  no compiler patch, ~1.8s/step at bs8 (≈fp16 speed), 1.6GB (the old f32 deadlock was
  IREE's *native* matmul, not f32 storage). **Metal training now descends to val 7.07 @
  step 200 — exactly matching clean CPU-f32 (7.07) and far past the fp16 wall (7.41).**
  Headline run live: `comp_bf16_lr1e3` (reference config: lr1e-3, eff bs32, warmup200,
  1000 updates) — fp16's range was likely *why* lr1e-3 destabilized, so this may now hold.
- **🎯 HEADLINE GATE MET — val 5.66 < reference 5.71.** `comp_bf16_lr1e3` (lr1e-3, eff
  bs32 = bs8×4, warmup200, 1000 updates, AdamW wd0.1, f32 activations + bf16 GEMM) ran to
  completion with **ZERO divergence at lr=1e-3** — confirming fp16's narrow range was
  exactly what destabilized it before. Val: 6.46@250 → 6.07@500 → 5.92@625 → 5.77@800 →
  **5.66@975 updates**, smooth monotonic descent. Beats the jax-metal reference's 5.71 on
  the same config, on the fully open-source IREE-Metal stack with hand-written FlashAttention
  + bf16 simdgroup GEMM. **Loss axis: DONE.** Remaining gap vs reference is throughput: the
  bf16 GEMM is f32-compute (~0.85 TFLOPS, 1.86s/step) vs the old fp16 GEMM (~2.4) — a native
  `simdgroup_bfloat8x8` kernel (needs IREE's Metal runtime bumped to MSL 3.1) would recover
  that speed at the same range. Recommended next step for the MFU axis.
- **✅ Native bf16 GEMM (MFU axis).** Bumped IREE's Metal runtime to **MSL 3.1**
  (`executable.m`: default 3.0→3.1 via @available, and the source_def version override now
  only RAISES, never drops below 3.1 — patch 09) so hand-authored MSL can use `bfloat` +
  `simdgroup_bfloat8x8`. Rewrote `gemm.metal` to native bf16 MMA (f32 accumulate). lm_head
  GEMM **1.54→2.04 TFLOPS** (back to the old fp16 level), training **step 1.86→1.6s (~14%)**,
  achieved 0.85→0.99 TFLOPS — recovers the throughput the loss-fix cost, at the same wide
  range. Correct (rel 0.0 vs bf16 ref). COMPUTE_DTYPE now defaults to f32 on Metal.
- **✅ FUSED CROSS-ENTROPY kernel — the biggest single win, and the long-context enabler.**
  The one-hot CE (logsumexp + one_hot(M,vocab) + multiply + sum) was ~6 memory-bound passes
  at 3.7 GB/s — measured **444ms fwd+bwd** in isolation at (4096,50257) and ~356ms of the
  step's backward. New `cross_entropy.metal`: `ce_fwd` (one threadgroup/row, cooperative
  online max+sumexp → loss + logsumexp) + `ce_bwd` (streamed dlogits = softmax−onehot);
  `ce_iree.py` custom_vjp (float0 cotangent for the int targets); pass lowers `ce_fwd`/
  `ce_bwd` custom_calls to flow.dispatch (256-thread workgroup, M rows). **Never
  materialises the (M,vocab) one-hot.** Validated: loss exact, dlogits rel 1.3e-6. Bench:
  **51.9ms vs 444ms (8.57×)**, 31.7 GB/s. In training: **step ~1.3→1.0s, TFLOPS ~1.2→1.58,
  MFU ~52→72% (beats jax-metal's 67.9%)**, correct loss. This is the kernel-MFU headline and
  the seq-8192 memory unblocker (no one-hot to balloon to 6.6GB). Has its own leaderboard
  board + presubmit (`bench/bench_kernels.py --ce`).
- **bf16 activations (the big ~350ms elementwise lever) = BLOCKED on deep codegen.** With
  attention fast, ~630ms of the backward is IREE-codegen'd elementwise/CE; bf16 activations
  would ~2x it (memory-bound) and are loss-safe (bf16 range). But COMPUTE_DTYPE=bf16 fails to
  legalize `vector.bitcast` (2xf32/2xi32 → 4xbf16) in metal-spirv (layernorm/bias). Adding
  MLIR's `populateVectorBitCastLoweringPatterns` (the patch-01-style one-liner) DIDN'T help —
  it only unrolls *rank*, not element-type bitcasts; the SPIR-V backend has no lowering for
  bf16 element-type bitcasts (would need real decomposition / spirv-cross bf16 support).
  Gather-based CE also still miscompiles (tensor.expand_shape shape-inference bug). So the
  big elementwise win needs deep compiler work; the tractable remainders are modest: a fused
  CE kernel (~80-150ms; the softmax−onehot bwd is only ~2.5GB ≈ ~45ms once estimated by
  traffic, smaller than first thought) and the flash forward's online-softmax overhead (~28ms).
- **FlashAttention forward → simdgroup-MMA**, but attention is overhead/occupancy-bound at
  d=64/seq=512 (streaming softmax + barriers), so only 14.5→11.2ms (~23%); matmuls weren't
  the bottleneck. Backward (scalar) still dominates attention; deprioritized vs the GEMM win.
- **✅ FlashAttention BACKWARD → simdgroup-MMA — the big util win.** bwd_dq + bwd_dkdv
  rewritten with `simdgroup_matrix` MMA (same launch as the MMA fwd: 64 threads = 2
  simdgroups, 8-query/8-key sub-blocks). The backward has NO online softmax (L precomputed)
  so outputs accumulate straight into simdgroup matrices — S=Q·Kᵀ, dP=dO·Vᵀ, dQ=(scale·dS)·K;
  dV=Pᵀ·dO, dK=(scale·dS)ᵀ·Q (Pᵀ/dSᵀ via transposed threadgroup loads). That lack of
  softmax-overhead is exactly why the backward MMA pays off where the forward didn't:
  **flash fwd+bwd 90.9→19.8ms (4.6×), 0.08→0.38 TFLOPS** (backward ~80ms→~5.5ms). Correct:
  dQ/dK/dV corr 1.00000 vs reference. **End-to-end step 1.6→~1.3s, ~1.2 TFLOPS, MFU ~52%**
  (vs ~0.85 TFLOPS / 1.86s at the start of the kernel push). Attention is no longer the drag;
  the remaining gap to jax-metal's 2.4 TFLOPS is the IREE-codegen'd elementwise/CE + the
  flash forward's softmax overhead. Auto-posted to the Flash leaderboard via the commit hook.

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
