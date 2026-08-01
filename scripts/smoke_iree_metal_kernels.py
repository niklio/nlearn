"""Small numerical gate proving wheel-bundled GEMM, FlashAttention, and CE load."""

import numpy as np
import jax
import jax.numpy as jnp

from nlearn.attention import USE_IREE_FLASH, _attention_iree_flash
from nlearn.kernels.cross_entropy import USE_IREE_CE, cross_entropy
from nlearn.kernels.gemm import USE_IREE_GEMM, gemm


def main():
    assert USE_IREE_GEMM and USE_IREE_FLASH and USE_IREE_CE
    rng = np.random.default_rng(7)

    a = rng.normal(size=(64, 128)).astype(np.float32)
    b = rng.normal(size=(128, 64)).astype(np.float32)
    got = np.asarray(jax.jit(gemm)(jnp.asarray(a), jnp.asarray(b)))
    ref = a.astype(jnp.bfloat16).astype(np.float32) @ b.astype(jnp.bfloat16).astype(np.float32)
    gemm_rel = np.abs(got - ref).mean() / (np.abs(ref).mean() + 1e-8)
    assert gemm_rel < 2e-3, gemm_rel

    q, k, v = [
        rng.normal(scale=0.1, size=(2, 64, 64)).astype(np.float32)
        for _ in range(3)
    ]
    got = np.asarray(
        jax.jit(_attention_iree_flash)(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v))
    )
    scores = np.einsum("nqd,nkd->nqk", q, k) / 8.0
    scores = np.where(np.tril(np.ones((64, 64), bool))[None], scores, -np.inf)
    probabilities = np.exp(scores - scores.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    ref = np.einsum("nqk,nkd->nqd", probabilities, v)
    flash_max = float(np.max(np.abs(got - ref)))
    assert flash_max < 2e-2, flash_max

    logits = rng.normal(size=(32, 128)).astype(np.float32)
    targets = rng.integers(0, 128, size=(32,), dtype=np.int32)
    loss, grad = jax.jit(jax.value_and_grad(cross_entropy))(
        jnp.asarray(logits), jnp.asarray(targets)
    )
    loss.block_until_ready()
    shifted = logits - logits.max(-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(-1, keepdims=True)
    ref_loss = -np.log(probability[np.arange(32), targets]).mean()
    ref_grad = probability
    ref_grad[np.arange(32), targets] -= 1
    ref_grad /= 32
    ce_loss_err = abs(float(loss) - float(ref_loss))
    ce_grad_max = float(np.max(np.abs(np.asarray(grad) - ref_grad)))
    assert ce_loss_err < 1e-4, ce_loss_err
    assert ce_grad_max < 1e-5, ce_grad_max

    print(
        "custom kernel smoke test passed: "
        f"gemm_rel={gemm_rel:.3e} flash_max={flash_max:.3e} "
        f"ce_loss_err={ce_loss_err:.3e} ce_grad_max={ce_grad_max:.3e}"
    )


if __name__ == "__main__":
    main()
