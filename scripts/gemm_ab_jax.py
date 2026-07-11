#!/usr/bin/env python3
"""GEMM A/B (JAX side): nlearn custom Metal kernel vs jnp.matmul at deep_hero training shapes.
Run: JAX_PLATFORMS=iree_metal ~/.venvs/iree/bin/python scripts/gemm_ab_jax.py  (after sourcing .runenv.sh)
FLOPs = 2*M*N*K. Reports best-of-N TFLOP/s for each path per shape."""
import os, time, jax, jax.numpy as jnp
from jax import random
from nlearn.kernels.gemm import gemm as nl_gemm

# (M, K, N): the per-step matmuls of deep_hero (bs4*seq1024=4096 tokens, d768, ff3072, vocab50257)
SHAPES = [
    (4096, 768, 768,   "qkv/o proj"),
    (4096, 768, 3072,  "ffn w1/w3"),
    (4096, 3072, 768,  "ffn w2"),
    (4096, 768, 50257, "lm_head"),
]
TRIALS = 30

def bench(fn, *a):
    r = fn(*a); jax.block_until_ready(r)                      # compile
    best = 1e9
    for _ in range(TRIALS):
        t = time.time(); r = fn(*a); jax.block_until_ready(r)
        best = min(best, time.time() - t)
    return best

def main():
    print(f"backend={jax.devices()[0].platform}", flush=True)
    key = random.PRNGKey(0)
    rnd = lambda shp, k: jax.jit(lambda kk: random.normal(kk, shp, jnp.bfloat16))(k)
    for M, K, N, name in SHAPES:
        k1, k2 = random.split(key)
        A = rnd((M, K), k1)
        B = rnd((K, N), k2)
        flops = 2 * M * N * K
        tk = bench(jax.jit(nl_gemm), A, B)
        try:
            tj = bench(jax.jit(jnp.matmul), A, B)
            jstr = f"   jnp.matmul {flops/tj/1e12:5.2f} TFLOP/s"
        except Exception:
            jstr = "   jnp.matmul: unsupported on IREE"
        print(f"  {name:12s} {M}x{K}x{N}:  nlearn-kernel {flops/tk/1e12:5.2f} TFLOP/s{jstr}", flush=True)

if __name__ == "__main__":
    main()
