"""
train_flash_local.py — Local N-step training on IREE-Metal with the native
Metal FlashAttention kernel, no network (real tiktoken tokenizer over a local
text corpus). Proves forward+backward+optimizer flow through the custom kernel.

Run:  source iree_env.sh && python train_flash_local.py [n_steps]
"""
import sys
import time
import types
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)

# Stub wandb (network) so we can import the real training code unchanged.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
import tiktoken

from nlearn.model import init_model, model_forward, VOCAB_SIZE
import nlearn.attention as A


# Simple full-vocab cross-entropy. NOTE: train.py's memory-efficient 13-chunk
# CE is numerically wrong on the experimental metal-spirv backend (a fusion/
# codegen bug in the large unrolled loop; every individual op is correct), which
# drives the loss negative under overfitting. The unchunked CE compiles
# correctly. Uses one-hot select instead of gather (gather's backward is a
# scatter, which also miscompiles under vmap on metal-spirv).
def simple_ce(params, token_ids):
    logits = model_forward(params, token_ids[:-1])              # (seq, vocab)
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    oh = jax.nn.one_hot(token_ids[1:], VOCAB_SIZE, dtype=logits.dtype)
    cor = jnp.sum(logits * oh, axis=-1)
    return -(cor - lse).mean()


_batched_loss = jax.vmap(simple_ce, in_axes=(None, 0))


def _batch_loss(params, batch):
    return jnp.mean(_batched_loss(params, batch))


def make_train_step(optimizer):
    loss_and_grad = jax.value_and_grad(_batch_loss)

    def train_step(params, opt_state, batch):
        loss, grads = loss_and_grad(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return jax.jit(train_step)

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
B, L = 2, 64    # small batch/seq: fewer sequential flash dispatches (runtime-hang mitigation)

print(f"backend: {jax.default_backend()}  attention: "
      f"{'native Metal FlashAttention' if A.USE_IREE_FLASH else 'standard'}")

# Build a varied local token stream (no network). Repeat a moderately long,
# varied passage so the model has real structure to learn over 100 steps.
enc = tiktoken.get_encoding("gpt2")
paragraph = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories of artificial beings endowed with intelligence by master craftsmen. "
    "Modern machine learning trains large neural networks on vast corpora of text, "
    "adjusting millions of parameters by gradient descent to predict the next token. "
    "On Apple Silicon, a fused FlashAttention kernel keeps memory linear in the "
    "sequence length by never materialising the full attention matrix. "
)
toks = np.array(enc.encode(paragraph * 60), dtype=np.int32)
stride = B * (L + 1)
n_batches = len(toks) // stride
print(f"corpus: {len(toks):,} tokens -> {n_batches} distinct batches")

def get_batch(i):
    off = (i % n_batches) * stride
    return jnp.asarray(toks[off:off + stride].reshape(B, L + 1))

params = jax.jit(init_model)(random.PRNGKey(0))
n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
print(f"params: {n_params:,}")
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)
step_fn = make_train_step(optimizer)

losses = []
step_times = []
for i in range(N_STEPS):
    batch = get_batch(i)
    t0 = time.time()
    params, opt_state, loss = step_fn(params, opt_state, batch)
    jax.block_until_ready(loss)
    dt = time.time() - t0
    losses.append(float(loss))
    if i > 0:
        step_times.append(dt)
    if i % 10 == 0 or i == N_STEPS - 1:
        print(f"step {i:3d}: loss={float(loss):.4f}  ({dt:.2f}s)", flush=True)

print("-" * 56)
print(f"first loss {losses[0]:.4f} -> last loss {losses[-1]:.4f}")
print(f"finite: {bool(np.all(np.isfinite(losses)))}  "
      f"decreasing: {losses[-1] < losses[0]}")
if step_times:
    print(f"median step time (post-compile): {np.median(step_times):.2f}s")
