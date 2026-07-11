#!/usr/bin/env python3
"""Backend A/B: standard-JAX deep transformer (24L, d=768) fwd+bwd, IREE-metal vs jax-metal.

Purpose: get a REAL number for 'jax-metal is faster on the deep model'. Uses only standard jnp ops
(no IREE custom kernels — those are IREE-only), same code on both backends, so it isolates the
COMPILER/BACKEND difference on the scan+remat deep structure. Reports raw TFLOP/s + tokens/s + step
time (backend-independent; avoids the noisy hardware-peak benchmark that makes MFU% incomparable).

Run under each backend with its own venv + JAX_PLATFORMS:
  JAX_PLATFORMS=iree_metal ~/.venvs/iree/bin/python scripts/bench_backend_ab.py     (jax 0.6)
  JAX_PLATFORMS=metal      ~/.venvs/jaxmetal/bin/python scripts/bench_backend_ab.py  (jax 0.4)
"""
import os, time, functools
import jax, jax.numpy as jnp
from jax import random

D = int(os.environ.get("AB_D", 768)); H = int(os.environ.get("AB_H", 12))
L = int(os.environ.get("AB_L", 24)); FF = int(os.environ.get("AB_FF", 3072)); V = 50257
SEQ = int(os.environ.get("AB_SEQ", 1024)); BS = int(os.environ.get("AB_BS", 4))
GROUP = int(os.environ.get("AB_GROUP", 3)); STEPS = int(os.environ.get("AB_STEPS", 12))
DT = jnp.bfloat16

def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-5) * g + b

def block(p, x):
    h = ln(x, p['g1'], p['b1'])
    q = (h @ p['wq']).reshape(BS, SEQ, H, D // H).transpose(0, 2, 1, 3)
    k = (h @ p['wk']).reshape(BS, SEQ, H, D // H).transpose(0, 2, 1, 3)
    v = (h @ p['wv']).reshape(BS, SEQ, H, D // H).transpose(0, 2, 1, 3)
    a = (q @ k.transpose(0, 1, 3, 2)) * ((D // H) ** -0.5)
    mask = jnp.tril(jnp.ones((SEQ, SEQ), DT))
    a = jnp.where(mask[None, None] == 0, jnp.finfo(DT).min, a)
    a = jax.nn.softmax(a, -1)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(BS, SEQ, D) @ p['wo']
    x = x + o
    h = ln(x, p['g2'], p['b2'])
    gate = jax.nn.silu(h @ p['w1']) * (h @ p['w3'])
    return x + gate @ p['w2']

def forward(params, tok):
    x = params['emb'][tok]
    blocks = params['blocks']
    ng = L // GROUP
    grouped = jax.tree_util.tree_map(lambda a: a.reshape(ng, GROUP, *a.shape[1:]), blocks)
    def body(carry, gp):
        for i in range(GROUP):
            carry = block(jax.tree_util.tree_map(lambda a: a[i], gp), carry)
        return carry, None
    body = jax.checkpoint(body, prevent_cse=False)
    x, _ = jax.lax.scan(body, x, grouped)
    x = ln(x, params['gf'], params['bf'])
    return x @ params['emb'].T

def loss_fn(params, tok, tgt):
    logits = forward(params, tok.astype(jnp.int32)).astype(jnp.float32)
    return -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), tgt[..., None], -1).mean()

def init(key):
    ks = random.split(key, 20)
    def blk(i):
        s = 0.02
        return dict(
            g1=jnp.ones(D, DT), b1=jnp.zeros(D, DT), g2=jnp.ones(D, DT), b2=jnp.zeros(D, DT),
            wq=random.normal(ks[i % 15 + 1], (D, D), DT) * s, wk=random.normal(ks[(i + 1) % 15 + 1], (D, D), DT) * s,
            wv=random.normal(ks[(i + 2) % 15 + 1], (D, D), DT) * s, wo=random.normal(ks[(i + 3) % 15 + 1], (D, D), DT) * s,
            w1=random.normal(ks[(i + 4) % 15 + 1], (D, FF), DT) * s, w3=random.normal(ks[(i + 5) % 15 + 1], (D, FF), DT) * s,
            w2=random.normal(ks[(i + 6) % 15 + 1], (FF, D), DT) * s)
    blocks = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[blk(i) for i in range(L)])
    return dict(emb=random.normal(ks[0], (V, D), DT) * 0.02,
                gf=jnp.ones(D, DT), bf=jnp.zeros(D, DT), blocks=blocks)

def main():
    key = random.PRNGKey(0)
    params = jax.jit(init)(key)   # jit init: IREE rejects eager random.normal (Sharding custom_call)
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
    tok = jax.jit(lambda k: random.randint(k, (BS, SEQ), 0, V))(key)
    tgt = jax.jit(lambda k: random.randint(k, (BS, SEQ), 0, V))(random.split(key)[1])
    grad = jax.jit(jax.value_and_grad(loss_fn))
    tokens = BS * SEQ
    flops = 6 * nparams * tokens          # standard fwd+bwd MFU numerator (excludes attention + remat recompute)
    plat = jax.devices()[0].platform
    print(f"backend={plat} jax={jax.__version__} params={nparams/1e6:.1f}M", flush=True)
    times = []
    for i in range(STEPS):
        t = time.time()
        l, g = grad(params, tok, tgt)
        jax.block_until_ready(g)
        dt = time.time() - t
        times.append(dt)
        if i < 2:
            print(f"  step {i} (warmup/compile) {dt:.2f}s loss={float(l):.2f}", flush=True)
    steady = sorted(times[2:])[: max(1, len(times[2:]) - 1)]  # drop compile + slowest outlier
    avg = sum(steady) / len(steady)
    print(f"  RESULT backend={plat}: {avg:.3f}s/step  {tokens/avg:.0f} tok/s  "
          f"{flops/avg/1e12:.2f} TFLOP/s  (n={len(steady)})", flush=True)

if __name__ == "__main__":
    main()
