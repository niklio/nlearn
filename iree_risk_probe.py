"""
iree_risk_probe.py — Minimal per-feature probes for the IREE Metal backend.

Run with: JAX_PLATFORMS=iree_metal python iree_risk_probe.py

Each probe exercises ONE risk area identified in the migration plan, in
isolation, so a failure points at a specific unsupported op/feature rather than
getting lost in the full training graph. Mirrors the Phase 3 risk list:
  bf16 compute, fp16 compute, gather, threefry RNG, jax.checkpoint remat,
  and a small matmul/softmax baseline.
"""

import traceback
import numpy as np
import jax
import jax.numpy as jnp
from jax import random


def _run(name, fn):
    try:
        out = fn()
        jax.block_until_ready(out)
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        msg = str(e).strip().splitlines()
        head = msg[0] if msg else repr(e)
        print(f"[FAIL] {name}: {head[:200]}")
        return False


def probe_matmul_f32():
    x = jnp.ones((128, 128), dtype=jnp.float32)
    return jax.jit(lambda a: (a @ a).sum())(x)


def probe_softmax():
    x = random.normal(random.PRNGKey(0), (32, 128), dtype=jnp.float32)
    return jax.jit(lambda a: jax.nn.softmax(a, axis=-1))(x)


def probe_bf16_matmul():
    x = jnp.ones((128, 128), dtype=jnp.bfloat16)
    return jax.jit(lambda a: (a @ a).astype(jnp.float32).sum())(x)


def probe_fp16_matmul():
    x = jnp.ones((128, 128), dtype=jnp.float16)
    return jax.jit(lambda a: (a @ a).astype(jnp.float32).sum())(x)


def probe_gather():
    # Mirrors chunked-CE: chunk_logits[arange, safe_idx]
    table = random.normal(random.PRNGKey(1), (64, 100), dtype=jnp.float32)
    idx = jnp.arange(64) % 100
    return jax.jit(lambda t, i: t[jnp.arange(64), i])(table, idx)


def probe_threefry_rng():
    def f(seed):
        k = random.PRNGKey(seed)
        k1, k2 = random.split(k)
        return random.normal(k1, (256,)) + random.categorical(k2, jnp.ones((256, 10)))
    return jax.jit(f)(0)


def probe_remat():
    # jax.checkpoint (remat) as used per-block in model_forward_features
    def block(p, x):
        return jnp.tanh(x @ p)
    p = jnp.ones((32, 32), dtype=jnp.float32)
    x = jnp.ones((8, 32), dtype=jnp.float32)
    g = jax.grad(lambda p, x: jax.checkpoint(block)(p, x).sum())(p, x)
    return g


def probe_vmap_grad():
    # vmap + value_and_grad, as in batched_loss_fn / loss_and_grad_fn
    def loss(p, v):
        return jnp.sum((v @ p) ** 2)
    p = jnp.ones((16, 16), dtype=jnp.float32)
    batch = jnp.ones((8, 4, 16), dtype=jnp.float32)
    batched = jax.vmap(loss, in_axes=(None, 0))
    return jax.jit(jax.grad(lambda p, b: jnp.mean(batched(p, b))))(p, batch)


if __name__ == "__main__":
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())
    print("-" * 60)
    results = {}
    for name, fn in [
        ("matmul_f32", probe_matmul_f32),
        ("softmax", probe_softmax),
        ("bf16_matmul", probe_bf16_matmul),
        ("fp16_matmul", probe_fp16_matmul),
        ("gather", probe_gather),
        ("threefry_rng", probe_threefry_rng),
        ("remat_grad", probe_remat),
        ("vmap_grad", probe_vmap_grad),
    ]:
        results[name] = _run(name, fn)
    print("-" * 60)
    n_pass = sum(results.values())
    print(f"{n_pass}/{len(results)} probes passed")
