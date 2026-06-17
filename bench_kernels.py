#!/usr/bin/env python3
"""
bench_kernels.py — benchmark the kernels the nlearn agent is optimizing and
post the results to the leaderboard.

Two boards:
  flashattention  — benchmarks the model's attention op (attention.py:attention),
                    which dispatches to the active implementation (standard today,
                    the native IREE-Metal flash kernel once it is wired in). So
                    this measures exactly what the agent is optimizing.
  gemm            — benchmarks the matmul path (jnp.matmul today; a custom GEMM
                    kernel once routed through it).

Each run is identified by a content hash of the relevant source files, so
re-benchmarking unchanged code updates the same row, while a code change posts
a new row. Intended to be invoked manually or from the kernel presubmit hook.

Usage:
  python bench_kernels.py                 # both boards, default shapes
  python bench_kernels.py --flash         # flash only
  python bench_kernels.py --gemm          # gemm only
  python bench_kernels.py --flash --heads 8 --seq 512 --dhead 64 --trials 50
"""

import argparse
import hashlib
import os
import resource
import time

from leaderboard_client import is_enabled, post_entry

HERE = os.path.dirname(os.path.abspath(__file__))


def _hash_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except FileNotFoundError:
            h.update(b"<missing>")
        h.update(p.encode())
    return h.hexdigest()[:10]


def _peak_mem_mb():
    # ru_maxrss is bytes on macOS, KB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _time_fn(fn, warmup, trials):
    """Return best-of-N wall time (seconds) for a callable returning a jax array."""
    import jax
    for _ in range(warmup):
        jax.block_until_ready(fn())
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# FlashAttention board
# ---------------------------------------------------------------------------

def bench_flash(heads, seq, dhead, dtype_name, warmup, trials):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import attention as attn_mod

    dtype = getattr(jnp, dtype_name)
    rng = np.random.default_rng(0)
    shape = (heads, seq, dhead)
    Qn = rng.standard_normal(shape).astype(np.float32)
    Kn = rng.standard_normal(shape).astype(np.float32)
    Vn = rng.standard_normal(shape).astype(np.float32)
    Q, K, V = (jnp.asarray(x, dtype=dtype) for x in (Qn, Kn, Vn))

    impl = jax.jit(attn_mod.attention)
    latency = _time_fn(lambda: impl(Q, K, V), warmup, trials)

    # Causal flash forward FLOPs: 2 matmuls (QKᵀ, softmax·V), causal ≈ half the
    # seq×seq pairs. 2*(2·S²·D)·0.5 per head = 2·S²·D per head.
    flops = 2 * heads * seq * seq * dhead
    tflops = flops / latency / 1e12

    # Speedup vs a naive dense O(S²) reference (the thing flash replaces).
    def naive():
        scale = jnp.sqrt(jnp.asarray(dhead, dtype=dtype))
        sc = jnp.matmul(Q, K.transpose(0, 2, 1)) / scale
        pos = jnp.arange(seq)
        mask = jnp.where(pos[:, None] >= pos[None, :], 0.0, -jnp.inf).astype(dtype)
        sc = sc + mask[None]
        return jnp.matmul(jax.nn.softmax(sc, axis=-1), V)
    naive_jit = jax.jit(naive)
    naive_latency = _time_fn(naive_jit, warmup, trials)
    speedup = naive_latency / latency if latency > 0 else None

    # Correctness vs a float64 NumPy reference.
    ref = np.zeros_like(Qn)
    scale = np.sqrt(dhead)
    for h in range(heads):
        sc = (Qn[h] @ Kn[h].T) / scale
        sc[np.triu(np.ones((seq, seq)), 1).astype(bool)] = -np.inf
        w = np.exp(sc - sc.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        ref[h] = w @ Vn[h]
    out = np.asarray(impl(Q, K, V), dtype=np.float32)
    max_abs_err = float(np.max(np.abs(out - ref)))

    impl_name = _active_attention_impl(attn_mod)
    src_hash = _hash_files([
        os.path.join(HERE, "attention.py"),
        os.path.join(HERE, "iree_metal", "kernels", "flash_attention.metal"),
    ])
    entry = {
        "id": f"flash-{impl_name}-{src_hash}",
        "name": f"{impl_name} · {src_hash}",
        "status": "done",
        "metrics": {
            "tflops": round(tflops, 4),
            "latency_ms": round(latency * 1e3, 4),
            "speedup": round(speedup, 3) if speedup else None,
            "peak_mem_mb": round(_peak_mem_mb(), 1),
            "max_abs_err": max_abs_err,
            "shape": f"{heads}×{seq}×{dhead}",
            "dtype": dtype_name,
            "impl": impl_name,
        },
    }
    _report_and_post("flashattention", entry)
    return entry


def _active_attention_impl(attn_mod):
    if getattr(attn_mod, "HAS_CUDNN", False):
        return "cudnn"
    if getattr(attn_mod, "USE_IREE_FLASH", False):
        return "iree-flash"
    return "standard"


# ---------------------------------------------------------------------------
# GEMM board
# ---------------------------------------------------------------------------

def bench_gemm(m, n, k, dtype_name, warmup, trials):
    import jax
    import jax.numpy as jnp
    import numpy as np

    dtype = getattr(jnp, dtype_name)
    rng = np.random.default_rng(0)
    # Scale down so f16 inputs stay in range over a large-K accumulation
    # (matches validate_gemm.py); keeps the error metric meaningful.
    An = (rng.standard_normal((m, k)) * 0.1).astype(np.float32)
    Bn = (rng.standard_normal((k, n)) * 0.1).astype(np.float32)
    A, B = jnp.asarray(An, dtype=dtype), jnp.asarray(Bn, dtype=dtype)

    gemm = _active_gemm()
    impl = jax.jit(gemm)
    latency = _time_fn(lambda: impl(A, B), warmup, trials)

    flops = 2 * m * n * k
    tflops = flops / latency / 1e12

    # % of hardware peak (best-effort; same matmul-based probe train.py uses).
    pct_peak = None
    try:
        from logging_utils import benchmark_peak_tflops
        peak = benchmark_peak_tflops(dtype=dtype)
        if peak > 0:
            pct_peak = tflops / peak
    except Exception:
        pass

    out = np.asarray(impl(A, B), dtype=np.float32)
    ref = An @ Bn
    max_abs_err = float(np.max(np.abs(out - ref)))

    impl_name, src_hash = _gemm_identity()
    entry = {
        "id": f"gemm-{impl_name}-{src_hash}-{m}x{n}x{k}",
        "name": f"{impl_name} · {src_hash}",
        "status": "done",
        "metrics": {
            "tflops": round(tflops, 4),
            "pct_peak": round(pct_peak, 4) if pct_peak is not None else None,
            "latency_ms": round(latency * 1e3, 4),
            "shape": f"{m}×{n}×{k}",
            "dtype": dtype_name,
            "max_abs_err": max_abs_err,
            "impl": impl_name,
        },
    }
    _report_and_post("gemm", entry)
    return entry


def _active_gemm():
    """The matmul under test: gemm_iree.matmul, which dispatches the hand-authored
    Metal simdgroup GEMM kernel on IREE-Metal and falls back to jnp.matmul
    elsewhere. So this measures whatever the agent is currently optimizing."""
    import jax.numpy as jnp
    try:
        import gemm_iree
        return gemm_iree.matmul
    except Exception:
        return jnp.matmul


def _gemm_impl_label():
    try:
        import gemm_iree
        return "gemm_iree" if getattr(gemm_iree, "USE_IREE_GEMM", False) else "jnp-fallback"
    except Exception:
        return "jnp"


def _gemm_identity():
    src = os.path.join(HERE, "gemm_iree.py")
    kernel = os.path.join(HERE, "iree_metal", "kernels", "gemm.metal")
    if os.path.exists(src):
        return _gemm_impl_label(), _hash_files([src, kernel])
    return "jnp", _hash_files([os.path.join(HERE, "model.py")])


# ---------------------------------------------------------------------------

def _report_and_post(board, entry):
    m = entry["metrics"]
    print(f"[{board}] {entry['name']}")
    for k, v in m.items():
        print(f"    {k:14} {v}")
    if is_enabled():
        ok = post_entry(board, entry, blocking=True)
        print(f"    → posted: {ok}")
    else:
        print("    → leaderboard disabled (set LEADERBOARD_URL/LEADERBOARD_TOKEN to post)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flash", action="store_true", help="benchmark the attention kernel")
    p.add_argument("--gemm", action="store_true", help="benchmark the GEMM kernel")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--dhead", type=int, default=64)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="float32", help="flash dtype")
    p.add_argument("--gemm-dtype", type=str, default="float16",
                   help="GEMM input dtype (kernel is f16-in / f32-accumulate)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trials", type=int, default=30)
    args = p.parse_args()

    # Default: run both.
    run_flash = args.flash or not (args.flash or args.gemm)
    run_gemm = args.gemm or not (args.flash or args.gemm)

    if run_flash:
        try:
            bench_flash(args.heads, args.seq, args.dhead, args.dtype, args.warmup, args.trials)
        except Exception as e:
            print(f"[flashattention] benchmark failed: {type(e).__name__}: {e}")
    if run_gemm:
        try:
            bench_gemm(args.m, args.n, args.k, args.gemm_dtype, args.warmup, args.trials)
        except Exception as e:
            print(f"[gemm] benchmark failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
