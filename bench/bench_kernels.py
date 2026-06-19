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
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
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
# Cross-entropy: the fused CE (loss + dlogits over (M, vocab) logits) vs the
# one-hot baseline the model uses today. The fused version's win is never
# materialising the (M×vocab) one-hot — so the headline is peak memory (the
# long-context enabler) and latency, not FLOP/s (CE is memory-bound).

def _onehot_ce(logits, tgt):
    """Baseline CE = the current _simple_ce_batched math (logsumexp + one-hot)."""
    import jax, jax.numpy as jnp
    V = logits.shape[-1]
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    oh = jax.nn.one_hot(tgt, V, dtype=logits.dtype)
    correct = jnp.sum(logits * oh, axis=-1)
    return -(correct - lse).mean()


def _active_ce():
    """The CE under test: ce_iree.cross_entropy (the fused kernel) once it exists,
    else the one-hot baseline — so this measures whatever the agent is optimizing."""
    try:
        import ce_iree
        return ce_iree.cross_entropy, "ce_iree"
    except Exception:
        return _onehot_ce, "onehot-baseline"


def _ce_identity(impl_name):
    src = os.path.join(HERE, "ce_iree.py")
    kernel = os.path.join(HERE, "iree_metal", "kernels", "cross_entropy.metal")
    files = [f for f in (src, kernel) if os.path.exists(f)] or [os.path.join(HERE, "train.py")]
    return _hash_files(files)


def bench_ce(m, v, dtype_name, warmup, trials):
    import jax, jax.numpy as jnp, numpy as np
    dtype = getattr(jnp, dtype_name)
    rng = np.random.default_rng(0)
    logits_n = rng.standard_normal((m, v)).astype(np.float32)
    tgt_n = rng.integers(0, v, (m,)).astype(np.int32)
    logits = jnp.asarray(logits_n, dtype=dtype)
    tgt = jnp.asarray(tgt_n)

    ce, impl_name = _active_ce()
    fn = jax.jit(jax.value_and_grad(lambda lg, t: ce(lg, t)))      # loss + dlogits
    base = jax.jit(jax.value_and_grad(lambda lg, t: _onehot_ce(lg, t)))

    latency = _time_fn(lambda: fn(logits, tgt), warmup, trials)
    peak_mem = _peak_mem_mb()   # capture BEFORE the one-hot baseline pollutes peak RSS
    base_latency = _time_fn(lambda: base(logits, tgt), warmup, trials)
    speedup = base_latency / latency if latency > 0 else None

    # Memory-bound throughput: read logits + write dlogits ≈ 2·M·V·bytes.
    bytes_moved = 2 * m * v * (2 if "16" in dtype_name else 4)
    gb_s = bytes_moved / latency / 1e9

    # Correctness: loss vs an f32 one-hot reference.
    ref_loss = float(_onehot_ce(jnp.asarray(logits_n), tgt).astype(jnp.float32))
    out_loss = float(fn(logits, tgt)[0])
    max_abs_err = abs(out_loss - ref_loss)

    src_hash = _ce_identity(impl_name)
    entry = {
        "id": f"ce-{impl_name}-{src_hash}-{m}x{v}",
        "name": f"{impl_name} · {src_hash}",
        "status": "done",
        "metrics": {
            "latency_ms": round(latency * 1e3, 4),
            "speedup": round(speedup, 4) if speedup is not None else None,
            "peak_mem_mb": round(peak_mem, 1),
            "gb_s": round(gb_s, 2),
            "max_abs_err": max_abs_err,
            "shape": f"{m}×{v}",
            "dtype": dtype_name,
            "impl": impl_name,
        },
    }
    _report_and_post("crossentropy", entry)
    return entry


def bench_lce(m, d, v, dtype_name, warmup, trials):
    """Fused linear-CE (lm_head matmul + CE, chunked over vocab) — never materialises
    the (M,V) logits. Headline is peak memory (the long-context enabler) + latency.
    Posts to the same Cross-Entropy board with a distinct id."""
    import jax, jax.numpy as jnp, numpy as np, ce_iree
    dtype = getattr(jnp, dtype_name)
    rng = np.random.default_rng(0)
    X = jnp.asarray((rng.standard_normal((m, d)) * 0.1).astype(np.float32)).astype(dtype)
    W = jnp.asarray((rng.standard_normal((d, v)) * 0.05).astype(np.float32)).astype(dtype)
    tgt = jnp.asarray(rng.integers(0, v, (m,)).astype(np.int32))

    fn = jax.jit(jax.value_and_grad(ce_iree.linear_cross_entropy, (0, 1)))
    latency = _time_fn(lambda: fn(X, W, tgt), warmup, trials)
    peak_mem = _peak_mem_mb()                       # fused never materialises (M,V) logits

    # Speedup vs the materialised lm_head + fused CE (best-effort; may OOM at big shapes).
    speedup = None
    try:
        base = jax.jit(jax.value_and_grad(
            lambda X, W, t: ce_iree.cross_entropy(_active_gemm()(X, W), t), (0, 1)))
        base_lat = _time_fn(lambda: base(X, W, tgt), warmup, max(3, trials // 5))
        speedup = base_lat / latency if latency > 0 else None
    except Exception:
        pass

    # Correctness vs the materialised full-logits CE (best-effort: the materialised
    # matmul can hit the bf16 const-input padding edge case at non-64 vocab — the fused
    # lce avoids it via 4096-wide chunks; correctness is also covered by check_lce.py).
    max_abs_err = None
    try:
        out_loss = float(fn(X, W, tgt)[0])
        ref_loss = float(ce_iree.cross_entropy(_active_gemm()(X, W), tgt))
        max_abs_err = abs(out_loss - ref_loss)
    except Exception:
        pass

    src_hash = _ce_identity("lce")
    entry = {
        "id": f"lce-ce_iree-{src_hash}-{m}x{d}x{v}",
        "name": f"linear-ce · {src_hash}",
        "status": "done",
        "metrics": {
            "latency_ms": round(latency * 1e3, 4),
            "speedup": round(speedup, 4) if speedup is not None else None,
            "peak_mem_mb": round(peak_mem, 1),
            "max_abs_err": max_abs_err,
            "shape": f"{m}×{d}×{v}",
            "dtype": dtype_name,
            "impl": "linear-ce",
        },
    }
    _report_and_post("crossentropy", entry)
    return entry


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
    p.add_argument("--ce", action="store_true", help="benchmark the cross-entropy kernel")
    p.add_argument("--lce", action="store_true", help="benchmark the fused linear-CE (lm_head+CE)")
    p.add_argument("--ce-m", type=int, default=4096, help="CE rows (bs*seq)")
    p.add_argument("--ce-v", type=int, default=50257, help="CE vocab size")
    p.add_argument("--ce-d", type=int, default=512, help="lce hidden dim D")
    p.add_argument("--ce-dtype", type=str, default="float32", help="CE logits dtype")
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

    # Default (no flags): run all.
    any_flag = args.flash or args.gemm or args.ce or args.lce
    run_flash = args.flash or not any_flag
    run_gemm = args.gemm or not any_flag
    run_ce = args.ce or not any_flag
    run_lce = args.lce or not any_flag

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
    if run_ce:
        try:
            bench_ce(args.ce_m, args.ce_v, args.ce_dtype, args.warmup, args.trials)
        except Exception as e:
            print(f"[crossentropy] benchmark failed: {type(e).__name__}: {e}")
    if run_lce:
        try:
            bench_lce(args.ce_m, args.ce_d, args.ce_v, args.ce_dtype, args.warmup, args.trials)
        except Exception as e:
            print(f"[crossentropy] linear-CE benchmark failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
