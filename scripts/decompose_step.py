#!/usr/bin/env python3
"""Decompose the deep_hero training step: forward-features vs full-loss vs fwd+bwd vs apply.
Shows where the ~5.3s/step goes (transformer body / CE / backward+remat / optimizer apply), so the
efficiency campaign attacks the biggest slice. Run under the deep env (see CMD in launch_deep_hero.py):
  NLEARN_SCAN_LAYERS=1 NLEARN_N_LAYERS=24 NLEARN_UNROLL_GROUP=3 NLEARN_MANUAL_ACCUM=1 NLEARN_OPTIMIZER=adamw \
  JAX_PLATFORMS=iree_metal ~/.venvs/iree/bin/python scripts/decompose_step.py
"""
import os, time, jax, jax.numpy as jnp
from jax import random

BS, SEQ, TRIALS = 4, 1024, 15

def _t(fn, *a, n=TRIALS):
    r = fn(*a); jax.block_until_ready(r)                      # compile
    best = 1e9
    for _ in range(n):
        s = time.time(); r = fn(*a); jax.block_until_ready(r); best = min(best, time.time() - s)
    return best

def main():
    from nlearn.model import init_model
    from nlearn.train import batch_loss, make_accum_steps
    from nlearn.optim import make_optimizer
    key = random.PRNGKey(0)
    params = jax.jit(init_model)(key)
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
    batch = jax.jit(lambda k: random.randint(k, (BS, SEQ + 1), 0, 50257))(key)
    tokens = BS * SEQ
    print(f"params={nparams/1e6:.1f}M  tokens/microbatch={tokens}", flush=True)

    t_feat = None                             # forward-only jit aborts on IREE (remat'd scan, no bwd)
    t_loss = _t(jax.jit(batch_loss), params, batch)
    grad_fn = jax.jit(jax.value_and_grad(batch_loss))
    t_grad = _t(grad_fn, params, batch)

    # optimizer apply (eager mean + jit opt.update + eager apply_updates), like make_accum_steps
    opt = make_optimizer(params)
    _, apply_fn = make_accum_steps(opt)
    opt_state = opt.init(params)
    _, g = grad_fn(params, batch)
    def apply_once(p, s, gr):
        return apply_fn(p, s, gr)
    t_apply = _t(lambda: apply_once(params, opt_state, g), n=10)

    print(f"\n  full loss (fwd+CE): {t_loss*1000:7.1f} ms   (forward pass + fused lm_head-CE, no grad)", flush=True)
    print(f"  loss+grad (F+B)  : {t_grad*1000:7.1f} ms   (bwd+remat = {(t_grad-t_loss)*1000:.1f} ms, {(t_grad-t_loss)/t_loss:.2f}x the fwd)", flush=True)
    print(f"  optimizer apply  : {t_apply*1000:7.1f} ms   (eager mean+update+apply, once per accum)", flush=True)
    print(f"\n  => micro-step (grad) ~{t_grad*1000:.0f} ms; measured live step ~5300 ms is with data+logging+rest overhead", flush=True)
    print(f"  => full step TFLOP/s (6*N*T/grad) = {6*nparams*tokens/t_grad/1e12:.2f}", flush=True)

if __name__ == "__main__":
    main()
