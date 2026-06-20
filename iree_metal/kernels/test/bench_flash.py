"""Microbench the custom Metal FlashAttention (fwd, and fwd+bwd via custom_vjp)
at the real training shape. bs8 folds batch into heads -> (bs*heads, seq, dh) =
(64, 512, 64). Source iree_env.sh first. Reports ms/call and the flash FLOPs
(~4*n*s^2*d for fwd causal, ~2.5x that for fwd+bwd) as TFLOPS."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from nlearn.attention import _attention_iree_flash

N, S, D = 64, 512, 64   # (bs8 * 8 heads, seq, d_head)
REPS = 30
# numpy init (eager jax.random.normal emits the unsupported Sharding custom_call)
rng = np.random.default_rng(0)
Q = jnp.asarray(rng.standard_normal((N, S, D), dtype=np.float32).astype(np.float16))
K = jnp.asarray(rng.standard_normal((N, S, D), dtype=np.float32).astype(np.float16))
V = jnp.asarray(rng.standard_normal((N, S, D), dtype=np.float32).astype(np.float16))

fwd = jax.jit(_attention_iree_flash)
# scalar loss to drive the backward through custom_vjp
def _loss(Q, K, V):
    return jnp.sum(_attention_iree_flash(Q, K, V).astype(jnp.float32))
fwdbwd = jax.jit(jax.value_and_grad(_loss, argnums=(0, 1, 2)))

def bench(fn, label, flop):
    out = fn(Q, K, V); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = fn(Q, K, V)
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / REPS
    print(f"{label:22s} {dt*1e3:7.2f} ms  {flop/dt/1e12:5.2f} TFLOPS")

fwd_flop = 4.0 * N * S * S * D * 0.5         # causal ~half
bench(fwd, "flash fwd", fwd_flop)
bench(fwdbwd, "flash fwd+bwd", fwd_flop * 3.5)  # bwd ~2.5x fwd
