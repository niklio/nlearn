#!/usr/bin/env python3
"""Numeric correctness + speed gate for flash_attention.metal, against a NumPy host reference
(causal attention, loss=O.sum()). ANY kernel edit must PASS before the hero resumes.
Run: JAX_PLATFORMS=iree_metal ~/.venvs/iree/bin/python scripts/flash_validate.py"""
import time, numpy as np, jax, jax.numpy as jnp
from jax import random
from nlearn.attention import _attention_iree_flash
BH, S, DH = 48, 1024, 64
SCALE = 1.0 / (DH ** 0.5)

def np_ref(Q, K, V):
    Q, K, V = [np.asarray(x, np.float64) for x in (Q, K, V)]
    s = np.einsum('nqd,nkd->nqk', Q, K) * SCALE
    mask = np.tril(np.ones((S, S), bool))
    s = np.where(mask[None], s, -np.inf)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s); p /= p.sum(-1, keepdims=True)
    O = np.einsum('nqk,nkd->nqd', p, V)
    # backward for loss = O.sum() (dO = 1)
    dO = np.ones_like(O)
    dV = np.einsum('nqk,nqd->nkd', p, dO)
    dP = np.einsum('nqd,nkd->nqk', dO, V)
    dS = p * (dP - (dP * p).sum(-1, keepdims=True))
    dS = np.where(mask[None], dS, 0.0)
    dQ = np.einsum('nqk,nkd->nqd', dS, K) * SCALE
    dK = np.einsum('nqk,nqd->nkd', dS, Q) * SCALE
    return O, dQ, dK, dV

def main():
    rng = np.random.default_rng(0)
    Qn, Kn, Vn = [rng.standard_normal((BH, S, DH)).astype(np.float32) for _ in range(3)]
    Oref, dQr, dKr, dVr = np_ref(Qn, Kn, Vn)
    Q, K, V = jnp.asarray(Qn), jnp.asarray(Kn), jnp.asarray(Vn)
    Ok = np.asarray(jax.jit(_attention_iree_flash)(Q, K, V))
    ferr = float(np.abs(Ok - Oref).max())
    gk = jax.jit(jax.grad(lambda q, k, v: _attention_iree_flash(q, k, v).sum(), (0, 1, 2)))(Q, K, V)
    jax.block_until_ready(gk)
    gerr = max(float(np.abs(np.asarray(gk[i]) - r).max()) for i, r in enumerate((dQr, dKr, dVr)))
    f = jax.jit(_attention_iree_flash); f(Q, K, V)
    best = min((lambda: (time.time(), jax.block_until_ready(f(Q, K, V)))[0])() and 0 or 1e9 for _ in range(0)) if False else 1e9
    for _ in range(30):
        t = time.time(); jax.block_until_ready(f(Q, K, V)); best = min(best, time.time() - t)
    tflops = 2 * BH * S * S * DH / best / 1e12
    ok = ferr < 2e-2 and gerr < 5e-2
    print(f"fwd max|err|={ferr:.2e}  grad max|err|={gerr:.2e}  fwd {best*1000:.2f}ms {tflops:.2f} TFLOP/s  => {'PASS' if ok else 'FAIL'}", flush=True)

if __name__ == "__main__":
    main()
