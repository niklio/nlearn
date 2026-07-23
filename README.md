# nlearn

A from-scratch, decoder-only transformer in [JAX](https://github.com/jax-ml/jax), trained
on a 16 GB Mac mini. It's a toy model — the point isn't the model, it's seeing how far you
can push long-context training on constrained Apple hardware using only an open stack: no
CUDA, no PyTorch, and not Apple's (now frozen) `jax-metal` plugin.

The model compiles to Metal through open-source [IREE](https://iree.dev/). Where IREE's
Metal codegen was too slow or ran out of memory, the hot operators are replaced with
hand-written Metal kernels — FlashAttention (forward and an `O(seq)`-memory backward), a
`bfloat16` GEMM, and a fused cross-entropy — each wired back into JAX's autodiff.

## Where it landed

On the same architecture and token budget, the open stack roughly matches a `jax-metal`
baseline run — which was the bar worth clearing, since `jax-metal` is the tuned,
closed-source option on this hardware:

| | `jax-metal` baseline | nlearn (open stack) |
|---|---|---|
| validation loss | 5.71 | 5.66 |
| MFU | 67.9% | ~72% |
| kernel throughput | ~2.4 TFLOP/s | ~1.6 TFLOP/s |
| hardware | Apple M3, 10-core GPU, 16 GB | same |

Throughput is still down on `jax-metal` — that gap is whole-graph fusion, which lives in
the compiler, not the model (see [iree-metal](https://github.com/niklio/iree-metal)).

## The constraints

16 GB of unified memory shared across activations, parameters, optimizer state, and
logits; and IREE's stock `metal-spirv` matmul running at ~0.5 TFLOP/s against a ~2.9
TFLOP/s achievable peak. Long context makes both bind at once, which is what the kernel
work is for.

## The kernels (`iree_metal/kernels/`)

- **FlashAttention, forward + backward.** Tiled, on Apple's `simdgroup_matrix` units, with
  an exact `O(seq)`-memory backward. Gradients check out to `corr = 1.0` against a
  materialized-attention reference. The backward rewrite moved the kernel from 0.08 → 0.38
  TFLOP/s.
- **A `bf16` GEMM** on `simdgroup_bfloat8x8` with f32 accumulation. This one came out of a
  training loss wall that turned out to be `fp16`'s narrow *exponent range*, not its
  precision — a CPU A/B (`f32` 7.07, `bf16` 7.01, `fp16` 7.41 at the same step) isolated
  it. `bf16` inputs with f32 accumulate fixed the wall.
- **A fused cross-entropy** (online-softmax) that never materializes the `(seq × vocab)`
  logit matrix — ~8.6× over the naive path, and what makes longer sequences fit in 16 GB.

Each kernel binds into JAX as a `custom_call` → `flow.dispatch` compiler pass plus a
`custom_vjp`, so it differentiates like a native op. See
[`iree_metal/BINDING.md`](iree_metal/BINDING.md).

## Notes from the build

- The recurring "~100-step hang" was a Metal HAL working-set fault, not a kernel bug —
  both custom kernels ran hundreds of clean iterations in isolation. Bisection pointed at
  per-graph buffer pressure at large batch; micro-batching plus gradient accumulation
  worked around it.
- Several plausible optimizations were measured and dropped: a 64×64 GEMM tile (occupancy
  collapse), GEMM-based attention (host-overhead-bound at small `K`), static `fp16` loss
  scaling (overflows after warmup). The results log in the git history has the numbers.

## Layout

Core library is the `nlearn/` package (`python -m nlearn.train` /
`python -m nlearn.generate`); ops scripts in `scripts/`, benchmarks in `bench/`, utilities
in `tools/`. [`STRUCTURE.md`](STRUCTURE.md) has the full map;
[`iree_metal/README.md`](iree_metal/README.md) has the IREE-Metal setup.
