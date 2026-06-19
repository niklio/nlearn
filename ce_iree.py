"""ce_iree.py — fused softmax cross-entropy via the custom Metal kernels.

`cross_entropy(logits, targets)` computes mean softmax CE and its gradient through
the hand-authored `cross_entropy.metal` (ce_fwd: per-row logsumexp + loss; ce_bwd:
dlogits = softmax-onehot), bound by `custom_call @ce_fwd/@ce_bwd` that the
ConvertFlashAttentionDispatch pass lowers to flow.dispatch. This never materialises
the (M,vocab) one-hot the old logsumexp+one_hot CE did — the long-context memory
enabler and a big step-time win (the one-hot CE was ~6 memory-bound passes).

Off IREE-Metal (or NLEARN_DISABLE_CE=1) it falls back to the jnp one-hot CE.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp


def _is_iree_metal():
    try:
        return jax.devices()[0].platform.lower() == "iree_metal"
    except Exception:
        return False


USE_IREE_CE = _is_iree_metal() and os.environ.get("NLEARN_DISABLE_CE") != "1"


def _ce_fwd_raw(logits, targets):
    """logits[M,V] f32, targets[M] i32 -> (loss[M] f32, lse[M] f32)."""
    M = logits.shape[0]
    return jax.ffi.ffi_call(
        "ce_fwd",
        (jax.ShapeDtypeStruct((M,), jnp.float32),
         jax.ShapeDtypeStruct((M,), jnp.float32)),
        vmap_method="sequential",
    )(logits, targets)


def _ce_bwd_raw(logits, targets, lse):
    """-> dlogits[M,V] = (softmax - onehot) / M."""
    M, V = logits.shape
    return jax.ffi.ffi_call(
        "ce_bwd",
        jax.ShapeDtypeStruct((M, V), jnp.float32),
        vmap_method="sequential",
    )(logits, targets, lse)


@jax.custom_vjp
def _ce(logits, targets):
    loss, _ = _ce_fwd_raw(logits, targets)
    return loss.mean()


def _ce_fwd_rule(logits, targets):
    loss, lse = _ce_fwd_raw(logits, targets)
    return loss.mean(), (logits, targets, lse)


def _ce_bwd_rule(res, g):
    logits, targets, lse = res
    dlogits = _ce_bwd_raw(logits, targets, lse)   # already (softmax-onehot)/M
    # targets is integer -> its cotangent is the special float0 zero.
    tgt_ct = np.zeros(targets.shape, dtype=jax.dtypes.float0)
    return (g * dlogits, tgt_ct)


_ce.defvjp(_ce_fwd_rule, _ce_bwd_rule)


def cross_entropy(logits, targets):
    """Mean softmax cross-entropy. logits[..., V] / targets[...]; leading dims are
    flattened to (M, V) for the kernel. Off IREE-Metal -> jnp one-hot CE."""
    V = logits.shape[-1]
    if USE_IREE_CE and logits.ndim >= 2:
        lg = logits.reshape(-1, V).astype(jnp.float32)
        tg = targets.reshape(-1).astype(jnp.int32)
        return _ce(lg, tg)
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    oh = jax.nn.one_hot(targets, V, dtype=logits.dtype)
    return -(jnp.sum(logits * oh, axis=-1) - lse).mean()


# ---------------------------------------------------------------------------
# Fused linear cross-entropy: lm_head matmul + CE, chunked over the vocab so the
# full (M, vocab) logits are NEVER materialised — the long-context memory enabler.
# X[M,D] @ W[D,V] -> CE(targets). Each vocab chunk's logits are produced via the
# bf16 GEMM, folded into an online logsumexp, and discarded. The custom_vjp's
# backward RECOMPUTES each chunk (rather than saving it) and accumulates dX, dW;
# peak memory is X + dX + dW + one (M, chunk) tile, not (M, vocab).
_LCE_CHUNK = int(os.environ.get("NLEARN_LCE_CHUNK", "4096"))


def _lce_chunks(V):
    Vp = V + ((-V) % _LCE_CHUNK)
    return Vp, Vp // _LCE_CHUNK


def _lce_mm():
    if USE_IREE_CE:
        import gemm_iree
        return gemm_iree.matmul
    return jnp.matmul


def _lce_loss_lse(X, W, targets):
    """Online (chunked) loss + logsumexp; full logits never materialised."""
    M, D = X.shape; V = W.shape[1]
    Vp, nch = _lce_chunks(V)
    Wp = W if Vp == V else jnp.pad(W, ((0, 0), (0, Vp - V)))
    mm = _lce_mm()
    run_max = jnp.full((M,), -jnp.inf)
    run_sum = jnp.zeros((M,))
    correct = jnp.zeros((M,))
    for i in range(nch):
        cs = i * _LCE_CHUNK
        logits_c = mm(X, Wp[:, cs:cs + _LCE_CHUNK])              # (M, CV) — transient
        valid = (cs + jnp.arange(_LCE_CHUNK)) < V
        masked = jnp.where(valid, logits_c, -jnp.inf)
        nm = jnp.maximum(run_max, masked.max(-1))
        run_sum = run_sum * jnp.exp(run_max - nm) + jnp.exp(masked - nm[:, None]).sum(-1)
        run_max = nm
        in_chunk = (targets >= cs) & (targets < cs + _LCE_CHUNK)
        oh = jax.nn.one_hot(jnp.clip(targets - cs, 0, _LCE_CHUNK - 1),
                            _LCE_CHUNK, dtype=logits_c.dtype)
        correct = jnp.where(in_chunk, jnp.sum(logits_c * oh, -1), correct)
    lse = run_max + jnp.log(run_sum)
    return -(correct - lse).mean(), lse


@jax.custom_vjp
def linear_cross_entropy(X, W, targets):
    """Mean softmax CE of logits = X[M,D] @ W[D,V] against targets[M], without ever
    materialising the (M,V) logits. The seq-length memory unblocker."""
    loss, _ = _lce_loss_lse(X, W, targets)
    return loss


def _lce_fwd_rule(X, W, targets):
    loss, lse = _lce_loss_lse(X, W, targets)
    return loss, (X, W, targets, lse)


def _lce_bwd_rule(res, g):
    X, W, targets, lse = res
    M, D = X.shape; V = W.shape[1]
    Vp, nch = _lce_chunks(V)
    Wp = W if Vp == V else jnp.pad(W, ((0, 0), (0, Vp - V)))
    mm = _lce_mm()
    gM = g / M
    dX = jnp.zeros_like(X)
    dW_chunks = []
    for i in range(nch):
        cs = i * _LCE_CHUNK
        Wc = Wp[:, cs:cs + _LCE_CHUNK]
        logits_c = mm(X, Wc)                                    # recompute (M, CV)
        valid = (cs + jnp.arange(_LCE_CHUNK)) < V
        sm = jnp.where(valid, jnp.exp(logits_c - lse[:, None]), 0.0)   # softmax (0 for pad)
        in_chunk = (targets >= cs) & (targets < cs + _LCE_CHUNK)
        oh = jax.nn.one_hot(jnp.clip(targets - cs, 0, _LCE_CHUNK - 1),
                            _LCE_CHUNK, dtype=logits_c.dtype) * in_chunk[:, None]
        dlogits_c = (sm - oh) * gM                              # (M, CV)
        dX = dX + mm(dlogits_c, Wc.T)                           # (M,CV)@(CV,D) -> (M,D)
        dW_chunks.append(mm(X.T, dlogits_c))                    # (D,M)@(M,CV) -> (D,CV)
    dW = jnp.concatenate(dW_chunks, axis=1)[:, :V]
    tgt_ct = np.zeros(targets.shape, dtype=jax.dtypes.float0)
    return (dX.astype(X.dtype), dW.astype(W.dtype), tgt_ct)


linear_cross_entropy.defvjp(_lce_fwd_rule, _lce_bwd_rule)
