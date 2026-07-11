#!/usr/bin/env python3
"""GEMM A/B (MLX side): mx matmul at deep_hero training shapes. Run: python3 scripts/gemm_ab_mlx.py
FLOPs = 2*M*N*K, best-of-N TFLOP/s. Pair with gemm_ab_jax.py for the nlearn-kernel-vs-MLX gap."""
import time
import mlx.core as mx

SHAPES = [
    (4096, 768, 768,   "qkv/o proj"),
    (4096, 768, 3072,  "ffn w1/w3"),
    (4096, 3072, 768,  "ffn w2"),
    (4096, 768, 50257, "lm_head"),
]
TRIALS = 30

def bench(A, B):
    f = mx.compile(lambda a, b: a @ b)
    mx.eval(f(A, B))
    best = 1e9
    for _ in range(TRIALS):
        t = time.time(); mx.eval(f(A, B)); best = min(best, time.time() - t)
    return best

def main():
    print(f"backend=MLX device={mx.default_device()}", flush=True)
    for M, K, N, name in SHAPES:
        A = (mx.random.normal((M, K))).astype(mx.bfloat16)
        B = (mx.random.normal((K, N))).astype(mx.bfloat16)
        mx.eval(A, B)
        flops = 2 * M * N * K
        t = bench(A, B)
        print(f"  {name:12s} {M}x{K}x{N}:  MLX {flops/t/1e12:5.2f} TFLOP/s", flush=True)

if __name__ == "__main__":
    main()
