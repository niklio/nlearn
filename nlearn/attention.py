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

import os

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


def _is_iree_metal():
    try:
        return jax.devices()[0].platform.lower() == "iree_metal"
    except Exception:
        return False


# Native Metal FlashAttention kernel is available only on the IREE-Metal backend
# (Apple's jax-metal can't run custom kernels). Disable with NLEARN_DISABLE_FLASH=1.
USE_IREE_FLASH = _is_iree_metal() and os.environ.get("NLEARN_DISABLE_FLASH") != "1"


# ---------------------------------------------------------------------------
# Native Metal FlashAttention (IREE-Metal) via custom kernels + custom_vjp
#
# Forward/backward dispatch to hand-authored MSL kernels (flash_attention.metal)
# embedded as external metal-msl-fb objects. A custom IREE preprocessing pass
# (ConvertFlashAttentionDispatch) lowers these custom_calls to flow.dispatch.
# Kernels run in f32 (the metal-spirv target lacks bf16); inputs are cast at the
# boundary and results cast back. Memory is O(seq) in both directions.
#
# Attention runs under vmap (batched_loss_fn), so the FFI uses
# vmap_method="sequential" — correct under batching; one GPU dispatch per batch
# element. (A batch-flattening fast path is a later optimization.)
# ---------------------------------------------------------------------------

def _flash_fwd_raw(Q, K, V):
    """(n,s,d) f32 -> O (n,s,d), L (n,s) logsumexp. Emits a custom_call the
    IREE pass turns into a dispatch of flash_attention_fwd."""
    n, s, d = Q.shape
    return jax.ffi.ffi_call(
        "flash_attention_fwd",
        (jax.ShapeDtypeStruct((n, s, d), jnp.float32),
         jax.ShapeDtypeStruct((n, s), jnp.float32)),
        vmap_method="sequential",
    )(Q, K, V)


@jax.custom_vjp
def _attention_iree_flash(Q, K, V):
    in_dtype = Q.dtype
    Qf, Kf, Vf = (x.astype(jnp.float32) for x in (Q, K, V))
    O, _ = _flash_fwd_raw(Qf, Kf, Vf)
    return O.astype(in_dtype)


def _flash_fwd_rule(Q, K, V):
    in_dtype = Q.dtype
    Qf, Kf, Vf = (x.astype(jnp.float32) for x in (Q, K, V))
    O, L = _flash_fwd_raw(Qf, Kf, Vf)
    # Residuals must be JAX types only (no Python dtype objects). The cotangent
    # dO carries the output dtype, so the backward recovers in_dtype from it.
    return O.astype(in_dtype), (Qf, Kf, Vf, O, L)


def _flash_bwd_rule(res, dO):
    Qf, Kf, Vf, O, L = res
    in_dtype = dO.dtype
    n, s, d = Qf.shape
    dOf = dO.astype(jnp.float32)
    D = jnp.sum(dOf * O, axis=-1)  # (n, s); D_i = dO_i . O_i
    dQ = jax.ffi.ffi_call(
        "flash_attention_bwd_dq",
        jax.ShapeDtypeStruct((n, s, d), jnp.float32),
        vmap_method="sequential",
    )(Qf, Kf, Vf, dOf, L, D)
    dK, dV = jax.ffi.ffi_call(
        "flash_attention_bwd_dkdv",
        (jax.ShapeDtypeStruct((n, s, d), jnp.float32),
         jax.ShapeDtypeStruct((n, s, d), jnp.float32)),
        vmap_method="sequential",
    )(Qf, Kf, Vf, dOf, L, D)
    return (dQ.astype(in_dtype), dK.astype(in_dtype), dV.astype(in_dtype))


_attention_iree_flash.defvjp(_flash_fwd_rule, _flash_bwd_rule)


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
    if USE_IREE_FLASH:
        return _attention_iree_flash(Q, K, V)
    return _attention_standard(Q, K, V)


def print_attention_config():
    """Print which attention strategy will be used. Call at startup."""
    print(f"  Platform:  {PLATFORM}")
    if HAS_CUDNN:
        print(f"  Attention: cuDNN flash attention")
    elif USE_IREE_FLASH:
        print(f"  Attention: native Metal FlashAttention (IREE custom kernel)")
    else:
        print(f"  Attention: standard matmul")
