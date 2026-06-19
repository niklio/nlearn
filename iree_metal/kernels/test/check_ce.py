"""Validate the fused CE (ce_iree.cross_entropy) loss + dlogits vs a one-hot
reference, at a small shape and the real (4096, 50257) train shape. Source iree_env.sh."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import ce_iree


def ref_ce(logits, tgt):
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    oh = jax.nn.one_hot(tgt, logits.shape[-1], dtype=logits.dtype)
    return -(jnp.sum(logits * oh, axis=-1) - lse).mean()


for (M, V) in [(512, 1024), (4096, 50257)]:
    rng = np.random.default_rng(0)
    logits_n = rng.standard_normal((M, V)).astype(np.float32)
    tgt_n = rng.integers(0, V, (M,)).astype(np.int32)
    logits = jnp.asarray(logits_n); tgt = jnp.asarray(tgt_n)
    lf, df = jax.jit(jax.value_and_grad(ce_iree.cross_entropy))(logits, tgt)
    lr, dr = jax.jit(jax.value_and_grad(ref_ce))(logits, tgt)
    lf, lr = float(lf), float(lr)
    df, dr = np.asarray(df, np.float32), np.asarray(dr, np.float32)
    rel = np.abs(df - dr).mean() / (np.abs(dr).mean() + 1e-12)
    print(f"M={M} V={V}: loss fused={lf:.6f} ref={lr:.6f} |d|={abs(lf-lr):.2e}  "
          f"dlogits rel={rel:.2e} max={np.abs(df-dr).max():.2e} finite={np.isfinite(df).all()}")
