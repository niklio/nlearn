"""Microbench the live gemm_iree.gemm() on IREE-Metal across the shapes that
dominate a training step. Source iree_env.sh first. Times wall-clock per call
(block_until_ready) over N reps after a warmup; reports TFLOPS."""
import os, sys, time
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from gemm_iree import gemm

# (M, K, N, label) — the lm_head fwd/bwd and a square baseline.
SHAPES = [
    (4096, 512,   50272, "lm_head fwd  (M4096 K512 N50272)"),
    (4096, 50272, 512,   "lm_head dA   (M4096 K50272 N512)"),
    (512,  4096,  50272, "lm_head dB   (M512 K4096 N50272)"),
    (4096, 512,   2048,  "mlp fc1      (M4096 K512 N2048)"),
    (4096, 2048,  512,   "mlp fc2      (M4096 K2048 N512)"),
    (2048, 2048,  2048,  "square 2048"),
]
REPS = 20

for M, K, N, label in SHAPES:
    A = jnp.ones((M, K), jnp.bfloat16)
    B = jnp.ones((K, N), jnp.bfloat16)
    C = gemm(A, B); C.block_until_ready()           # warmup/compile
    t0 = time.perf_counter()
    for _ in range(REPS):
        C = gemm(A, B)
    C.block_until_ready()
    dt = (time.perf_counter() - t0) / REPS
    tflops = 2.0 * M * K * N / dt / 1e12
    print(f"{label:38s} {dt*1e3:7.2f} ms  {tflops:5.2f} TFLOPS")
