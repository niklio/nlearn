"""Validate the bf16 GEMM (simdgroup_bfloat8x8, f32 accumulate) via the JAX path:
correctness vs a numpy bf16-rounded reference, and that the kernel COMPILES at runtime
(simdgroup_bfloat8x8 on M3/macOS-15). Also a large-magnitude case fp16 couldn't hold
(>65504) to confirm bf16's range. Source iree_env.sh first."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from gemm_iree import gemm
import ml_dtypes  # for a numpy bf16 reference

def check(M, K, N, scale, label):
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * scale).astype(np.float32)
    B = (rng.standard_normal((K, N)) * scale).astype(np.float32)
    # reference: round inputs to bf16 (what the kernel sees), accumulate f32
    Ab = A.astype(ml_dtypes.bfloat16).astype(np.float32)
    Bb = B.astype(ml_dtypes.bfloat16).astype(np.float32)
    ref = Ab @ Bb
    out = np.asarray(gemm(jnp.asarray(A), jnp.asarray(B)), np.float32)
    rel = np.abs(out - ref).mean() / (np.abs(ref).mean() + 1e-8)
    print(f"{label:28s} rel={rel:.3e}  max|ref|={np.abs(ref).max():.3e}  "
          f"out_finite={np.isfinite(out).all()}")

check(64, 128, 64, 0.1, "small")
check(256, 512, 256, 1.0, "medium")
# large magnitude: products ~ (300^2 * K) far exceed fp16 max 65504; bf16 holds it
check(64, 128, 64, 300.0, "large (fp16 would overflow)")
