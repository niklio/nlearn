# Metal GEMM — diagnostic + status (roadmap P0 #2)

## Diagnostic (conclusive)

IREE's `metal-spirv` matmul runs at **~0.8 TFLOPS on the M4** (vs ~teens-of-
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

## Status: `gemm.metal`

A `simdgroup_matrix` GEMM (`C[M,N] = A[M,K] @ B[K,N]`, f16 in, f32 accumulate),
threadgroup-memory tiled, bound via the same custom-dispatch path as flash.
**Correct** (exact vs NumPy at 1024³ and 2048³).

Progression (each measured under the same harness):

| Variant | Correct | Note |
|---|---|---|
| IREE naive (baseline) | — | the thing we're beating |
| simple simdgroup (no tg staging) | ✅ | memory-bound, ≈ naive |
| tg-tiled, 1 simdgroup, 2×2 | ✅ | ≈ naive (low occupancy) |
| tg-tiled, 1 simdgroup, 4×4 | ❌ corrupts | register overflow |
| **multi-simdgroup, 2×2 (4 SG), BK=32 (current)** | ✅ | **~2.2× faster than naive** (same harness) |

Lessons baked into the kernel:
- **Unroll loops over `simdgroup_matrix` arrays** — Metal can't dynamically index
  matrices in registers; without `#pragma clang loop unroll(full)` the multi-tile
  accumulation is silently wrong.
- **Register ceiling** — one simdgroup holds ~2×2 accumulators max; 4×4 overflows
  and silently corrupts. So scale via MORE simdgroups, not bigger per-simdgroup tiles.
- **Barrier-bound at small BK** — with BK=8 the K-loop runs 2 barriers × K/8 times
  and dominates; BK=32 (consume the slab in 4 inner 8-deep steps per barrier) helps.
- A bigger 64×64 / 16-simdgroup tile was *slower* than 32×32 / 4-simdgroup —
  occupancy/barrier cost, not reuse, is the current limiter.

## Measurement caveat

These ratios are confounded by host round-trip + dispatch overhead (the harness
does a 16 MB device→host copy per call), so absolute TFLOPS are pessimistic and
the true kernel speedup is likely higher than 2.2×. **Next: measure with
`iree-benchmark-module` (or a device-resident loop)** to get real kernel TFLOPS
before further tuning.

## Remaining work to fully "solve #2"

1. **Proper profiling** (iree-benchmark-module / Metal capture) to get true TFLOPS
   and find the real bottleneck — do this before more tuning.
2. **Autotune** tile/simdgroup/BK config and add **K-loop double-buffering** (stage
   next slab while computing current) to approach the M4's ~teens-TFLOPS peak.
3. **Integration:** route the model's big matmuls (MLP, projections, lm_head)
   through this kernel via the dispatch pass — a `gemm()` helper emitting an ffi
   custom_call, or a compiler pass matching `linalg.matmul`. Handle
   non-multiple-of-block dims (pad or a fallback tail).

## Reproduce

```sh
# dump IREE's naive matmul MSL (shows no simdgroup_matrix)
iree-compile --iree-hal-target-device=metal --iree-metal-compile-to-metallib=false \
  --iree-hal-dump-executable-files-to=/tmp/d gemm.mlir -o /dev/null
# build + run the custom kernel (hand-MLIR dispatch, like the flash tests)
xcrun -sdk macosx metal -std=metal3.0 -c gemm.metal -o /tmp/gemm.air   # compiles
```
