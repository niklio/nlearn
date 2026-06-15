# Metal GEMM — diagnostic + status (roadmap P0 #2)

## Diagnostic (conclusive)

IREE's `metal-spirv` matmul runs at **~0.5 TFLOPS on the M3** (vs ~teens-of-
TFLOPS peak). Root cause, confirmed:

1. **IREE's Apple GPU target advertises no matrix ops** — `getAppleTargetDetails`
   in `KnownTargets.cpp` sets `mmaCount=0`. So IREE never tries cooperative-matrix
   codegen for Metal; it emits naive vectorized-FMA loops (verified by dumping the
   generated MSL: 0 `simdgroup_matrix` ops).
2. **spirv-cross can't emit `simdgroup_matrix`** — the SPIR-V→MSL translator IREE
   uses has zero support for it (`grep simdgroup_matrix spirv_msl.cpp` → 0). So
   even if #1 were fixed, the matrix-unit win can't come through IREE's codegen.

**Conclusion:** the matrix-unit win is impossible via IREE codegen; it must come
from a hand-written MSL kernel bound through the custom-dispatch pass. (This
re-confirms the roadmap framing.)

## What "peak" actually is on this M3

Measured with **MLX** (Apple's own hand-tuned GEMM) on this machine, fp16 2048³:
**~2.9 TFLOPS**. So the achievable matmul peak here is ~3 TFLOPS — **not** the
"teens of TFLOPS" the roadmap originally assumed. The real headroom over IREE's
naive matmul is ~5–6×, not 15–20×.

## Status: `gemm.metal` — ~2.4 TFLOPS (≈84% of MLX, ~4.8× naive)

A `simdgroup_matrix` GEMM (`C[M,N] = A[M,K] @ B[K,N]`, f16 in, f32 accumulate),
threadgroup-memory tiled, bound via the same custom-dispatch path as flash.
**Correct** (exact vs NumPy at 1024³ / 2048³). Config: 32×32 tile, 4 simdgroups
(2×2), 2×2 8×8 sub-tile each, **BK=64**, vectorized (half4) staging.

All numbers below are true device time via `iree-benchmark-module` (2048³ fp16):

| Variant | Correct | TFLOPS | vs MLX |
|---|---|---|---|
| IREE naive matmul | — | 0.51 | 18% |
| simdgroup, scalar staging, BK=32 | ✅ | 1.6 | 55% |
| + double-buffering | ✅ | 1.3 | (worse — halves occupancy) |
| **+ vectorized (half4) staging, BK=64 (current)** | ✅ | **~2.4** | **84%** |
| jax-metal (`jnp.matmul`, the old backend) | — | 2.8 | 97% |
| MLX (achievable peak) | — | 2.9 | 100% |

## vs jax-metal (the backend we migrated off)

jax-metal's `jnp.matmul` hits **~2.8 TFLOPS** — basically MLX-peak, because it
dispatches to Apple's hand-tuned GPU matmul (MPS). So **matmul was never jax-metal's
weakness**; we left jax-metal because it *categorically cannot run custom kernels*
(custom_call rejected at legalization → no FlashAttention; see
`jax-metal-no-custom-call`). The cost of that move, for plain matmul, is the gap
between Apple's library (2.8) and IREE's naive codegen (0.5); this custom kernel
recovers most of it (→2.4, ~86% of jax-metal). The *payoff* — custom FlashAttention
and other fused kernels — is impossible on jax-metal at any speed.

Lessons baked into the kernel:
- **Unroll loops over `simdgroup_matrix` arrays** — Metal can't dynamically index
  matrices in registers; without `#pragma clang loop unroll(full)` it's silently wrong.
- **Register ceiling** — one simdgroup holds ~2×2 accumulators max; 4×4 overflows
  and silently corrupts. Scale via MORE simdgroups, not bigger per-simdgroup tiles.
- **Occupancy dominates, not reuse/bandwidth.** Bigger tiles (64×64) and
  double-buffering both *lost* — they raise threadgroup-memory / thread count and
  cut the number of concurrent threadgroups. Small 32×32 / 128-thread blocks win.
- **Staging was the real bottleneck.** Scalar per-element loads with `e/BK`, `e%BK`
  div/mod capped it at 1.6; switching to **vectorized half4 staging** (rows are
  contiguous in both global and the slab) jumped it to ~2.4 — the single biggest win.
- **BK sweet spot = 64** (swept 16/32/64/128/256): thick enough to amortize
  barriers, thin enough to keep occupancy.

## Remaining gap / next steps

- The last ~16% to MLX would need MLX-level micro-optimization (register-resident
  output blocking, load/compute software pipelining tuned to the M3, swizzled
  threadgroup layout to kill bank conflicts) — diminishing returns.
- **Integration (the remaining piece of "solve #2"):** route the model's big
  matmuls (MLP, projections, lm_head) through this kernel via the dispatch pass —
  a `gemm()` helper emitting an ffi custom_call, or a compiler pass matching
  `linalg.matmul`. Needs dim handling: M,N multiples of 32, K multiple of 64 (pad
  or a scalar tail), and a transpose/`f16`-output path for lm_head.

## Reproduce the sweep / profiling

`iree-benchmark-module --module=g.vmfb --device=metal --function=run_gemm
--input=2048x2048xf16 --input=2048x2048xf16 --benchmark_repetitions=5` gives true
device time (no host round-trip). Config sweep driver: `/tmp/gemmk/sweep.sh`.

## Reproduce

```sh
# dump IREE's naive matmul MSL (shows no simdgroup_matrix)
iree-compile --iree-hal-target-device=metal --iree-metal-compile-to-metallib=false \
  --iree-hal-dump-executable-files-to=/tmp/d gemm.mlir -o /dev/null
# build + run the custom kernel (hand-MLIR dispatch, like the flash tests)
xcrun -sdk macosx metal -std=metal3.0 -c gemm.metal -o /tmp/gemm.air   # compiles
```
