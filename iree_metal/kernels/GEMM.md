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

| Variant | Correct | TFLOPS @2048³ |
|---|---|---|
| IREE naive (baseline) | — | ~0.8 |
| simple simdgroup (no tg staging) | ✅ | ~0.66 (memory-bound) |
| tg-tiled, 1 simdgroup, **2×2 tiles (current)** | ✅ | ~0.94 |
| tg-tiled, 1 simdgroup, 4×4 tiles | ❌ corrupts | ~1.39 |

Two lessons baked into the kernel:
- **Unroll loops over `simdgroup_matrix` arrays** — Metal can't dynamically index
  matrices in registers; without `#pragma clang loop unroll(full)` the multi-tile
  accumulation is silently wrong.
- **Register ceiling** — one simdgroup holds ~2×2 accumulators max; 4×4 overflows
  and silently corrupts. This caps single-simdgroup throughput.

## Remaining work to "solve #2"

1. **Multi-simdgroup tiled GEMM.** Threadgroup = N simdgroups sharing one large
   staged `BM×BK` / `BK×BN` tile; each simdgroup computes a ≤2×2 sub-tile (within
   its register budget). This gets a large reused threadgroup tile + high
   occupancy → target ~8–10+ TFLOPS. Standard high-perf Metal GEMM structure.
2. **K-loop double-buffering** (stage next K-slab while computing current) to hide
   global-load latency.
3. **Integration:** route the model's big matmuls (MLP, projections, lm_head)
   through this kernel via the dispatch pass — either a `gemm()` helper emitting an
   ffi custom_call, or a compiler pass matching `linalg.matmul`. Handle
   non-multiple-of-8 dims (pad or a fallback tail).

## Reproduce

```sh
# dump IREE's naive matmul MSL (shows no simdgroup_matrix)
iree-compile --iree-hal-target-device=metal --iree-metal-compile-to-metallib=false \
  --iree-hal-dump-executable-files-to=/tmp/d gemm.mlir -o /dev/null
# build + run the custom kernel (hand-MLIR dispatch, like the flash tests)
xcrun -sdk macosx metal -std=metal3.0 -c gemm.metal -o /tmp/gemm.air   # compiles
```
