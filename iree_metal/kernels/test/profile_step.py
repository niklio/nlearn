"""Isolate where the bs8 training step's wall time goes — specifically the
unmeasured ~40% (cross-entropy + elementwise + host dispatch gaps). For each
stage, time BLOCKED-each vs PIPELINED (enqueue REPS, block once): if pipelined/REPS
<< blocked, host dispatch gaps dominate (GPU idle waiting on the CPU to enqueue).

Stages: forward (logits) | forward+CE (full loss) | value_and_grad (full step math).
CE cost = (forward+CE) - forward. Source iree_env.sh first."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from model import init_model, model_forward
import train as T

BS, SEQ = 8, 512
REPS = 20
params = jax.jit(init_model)(jax.random.PRNGKey(0))
batch = jnp.asarray(np.random.default_rng(0).integers(0, 50000, (BS, SEQ + 1), dtype=np.int32))

fwd  = jax.jit(lambda p, b: model_forward(p, b[:, :-1]))
loss = jax.jit(T.batch_loss)
grad = jax.jit(jax.value_and_grad(T.batch_loss))

def timeit(fn, label):
    o = fn(params, batch); jax.block_until_ready(o)            # warmup/compile
    # blocked-each
    t0 = time.perf_counter()
    for _ in range(REPS):
        jax.block_until_ready(fn(params, batch))
    blocked = (time.perf_counter() - t0) / REPS
    # pipelined: enqueue all, block once
    t0 = time.perf_counter()
    outs = [fn(params, batch) for _ in range(REPS)]
    jax.block_until_ready(outs)
    pipe = (time.perf_counter() - t0) / REPS
    print(f"{label:18s} blocked {blocked*1e3:7.2f} ms  pipelined {pipe*1e3:7.2f} ms  "
          f"host-gap {(blocked-pipe)*1e3:6.2f} ms ({(1-pipe/blocked)*100:4.0f}%)")
    return blocked

f = timeit(fwd,  "forward(logits)")
l = timeit(loss, "forward+CE")
g = timeit(grad, "value_and_grad")
print(f"\nCE-only (loss-fwd) ~ {(l-f)*1e3:.1f} ms   backward ~ {(g-l)*1e3:.1f} ms")
