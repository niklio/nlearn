"""
attention.py — Cross-platform attention dispatch.

Detects the runtime platform once at import and routes to the best
available implementation:

  CUDA + JAX ≥ 0.4.31  →  cuDNN flash attention (fused kernel, no seq² memory)
  Metal / CPU          →  standard matmul (single fused XLA kernel)

Why Metal can't use fused attention kernels:
  CUDA flash attention goes through XLA's custom_call — a compiled graph node
  that dispatches directly to the GPU with zero CPU involvement. jax-metal
  (closed-source, abandoned Oct 2024) never implemented custom_call. The only
  fallback is jax.pure_callback which routes GPU→CPU→GPU, killing performance.
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Platform detection (runs once at import, outside JIT)
# ---------------------------------------------------------------------------

def _detect_platform():
    """Return 'cuda', 'metal', or 'cpu'.

    'metal' covers both Apple's jax-metal plugin (platform "metal") and the
    open-source IREE PJRT plugin targeting the Metal HAL driver (platform
    "iree_metal"). Both lack a fused attention kernel through the standard XLA
    path, so they share the _attention_standard implementation. (A native IREE
    Metal FlashAttention kernel is dispatched separately; see attention().)
    """
    try:
        device = jax.devices()[0]
        platform = device.platform.lower()
        if platform == "gpu" or "cuda" in platform:
            return "cuda"
        elif "metal" in platform:
            return "metal"
    except Exception:
        pass
    return "cpu"


def _has_cudnn_attention():
    """Check if jax.nn.dot_product_attention is available (JAX >= 0.4.31)."""
    return hasattr(jax.nn, "dot_product_attention")


PLATFORM = _detect_platform()
HAS_CUDNN = PLATFORM == "cuda" and _has_cudnn_attention()


# ---------------------------------------------------------------------------
# Standard matmul attention (Metal / CPU)
# ---------------------------------------------------------------------------

def _attention_standard(Q, K, V):
    """
    Full seq×seq matmul attention with causal mask.
    Q, K, V: (n_heads, seq_len, d_head)
    Returns: (n_heads, seq_len, d_head)
    """
    d_head = Q.shape[-1]
    scale = jnp.sqrt(jnp.array(d_head, dtype=Q.dtype))
    scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / scale

    seq_len = Q.shape[1]
    pos = jnp.arange(seq_len)
    mask = jnp.where(pos[:, None] >= pos[None, :],
                     jnp.zeros((), dtype=Q.dtype),
                     jnp.full((), -jnp.inf, dtype=Q.dtype))
    scores = scores + mask[None, :, :]

    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, V)


# ---------------------------------------------------------------------------
# cuDNN flash attention (CUDA only)
# ---------------------------------------------------------------------------

def _attention_cudnn(Q, K, V):
    """
    cuDNN flash attention via jax.nn.dot_product_attention.
    Only available on CUDA with JAX >= 0.4.31.

    Q, K, V: (n_heads, seq_len, d_head)
    Returns: (n_heads, seq_len, d_head)
    """
    n_heads, seq_len, d_head = Q.shape
    Q_r = Q.transpose(1, 0, 2).reshape(1, seq_len, n_heads, d_head)
    K_r = K.transpose(1, 0, 2).reshape(1, seq_len, n_heads, d_head)
    V_r = V.transpose(1, 0, 2).reshape(1, seq_len, n_heads, d_head)

    out = jax.nn.dot_product_attention(
        Q_r, K_r, V_r,
        is_causal=True,
        implementation="cudnn",
    )
    return out[0].transpose(1, 0, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attention(Q, K, V):
    """
    Dispatch to the best attention implementation for the current platform.

    Q, K, V: (n_heads, seq_len, d_head)
    Returns: (n_heads, seq_len, d_head)
    """
    if HAS_CUDNN:
        return _attention_cudnn(Q, K, V)
    return _attention_standard(Q, K, V)


def print_attention_config():
    """Print which attention strategy will be used. Call at startup."""
    print(f"  Platform:  {PLATFORM}")
    if HAS_CUDNN:
        print(f"  Attention: cuDNN flash attention")
    else:
        print(f"  Attention: standard matmul")
