"""Gradient-quality check: does the tiled flash backward match a near-exact
reference at the TRAINING seq=512? (It was only validated at seq=128.) Reference =
gemm-based materialized causal attention (same math, f32 softmax, exact matmuls).
Compare dQ/dK/dV from jax.grad of each on identical inputs. A large mismatch would
explain the lr=1e-3 wall (biased gradients). Source iree_env.sh first."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from nlearn.kernels.gemm import gemm
from nlearn.attention import _attention_iree_flash

N, S, D = 64, 512, 64
scale = 1.0 / (D ** 0.5)
mask = jnp.triu(jnp.full((S, S), -1e30, jnp.float32), k=1)

def attn_ref(Q, K, V):                       # materialized reference (per-batch gemm)
    outs = []
    for i in range(N):
        s = gemm(Q[i], K[i].T) * scale + mask
        p = jax.nn.softmax(s.astype(jnp.float32), axis=-1).astype(jnp.float16)
        outs.append(gemm(p, V[i]))
    return jnp.stack(outs)

# NOTE: flash applies its own 1/sqrt(d) scale internally; ref scales explicitly above.
def loss_flash(Q, K, V): return jnp.sum(_attention_iree_flash(Q, K, V).astype(jnp.float32))
def loss_ref(Q, K, V):   return jnp.sum(attn_ref(Q, K, V).astype(jnp.float32))

rng = np.random.default_rng(0)
mk = lambda: jnp.asarray((rng.standard_normal((N, S, D)) * 0.1).astype(np.float16))
Q, K, V = mk(), mk(), mk()

gf = jax.jit(jax.grad(loss_flash, (0, 1, 2)))
gr = jax.jit(jax.grad(loss_ref,   (0, 1, 2)))
df = gf(Q, K, V); dr = gr(Q, K, V)
jax.block_until_ready(df); jax.block_until_ready(dr)

for name, a, b in zip("dQ dK dV".split(), df, dr):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    denom = np.abs(b).mean() + 1e-8
    print(f"{name}: mean|flash-ref| {np.abs(a-b).mean():.4e}  rel {np.abs(a-b).mean()/denom:.3e}  "
          f"max|ref| {np.abs(b).max():.3e}  corr {np.corrcoef(a.ravel(), b.ravel())[0,1]:.5f}")
# also compare the forward outputs
of = np.asarray(_attention_iree_flash(Q, K, V), np.float32)
orf = np.asarray(attn_ref(Q, K, V), np.float32)
print(f"fwd: mean|flash-ref| {np.abs(of-orf).mean():.4e}  rel {np.abs(of-orf).mean()/(np.abs(orf).mean()+1e-8):.3e}")
