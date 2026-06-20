"""Compare GEMM-based materialized attention vs the custom flash kernel at the
training shape (64,512,64). Hypothesis: at seq=512 the SxS scores matrix fits in
memory, so flash's O(seq) trick buys nothing; routing QK^T and P@V through the
2.4-TFLOPS simdgroup GEMM (with autodiff giving the backward through gemm's own
vjp) should crush the flash kernel's 0.08 TFLOPS. Source iree_env.sh first."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from nlearn.kernels.gemm import gemm
from nlearn.attention import _attention_iree_flash

N, S, D = 64, 512, 64
REPS = 30
scale = 1.0 / (D ** 0.5)
mask = jnp.triu(jnp.full((S, S), -1e30, jnp.float32), k=1)  # causal

def attn_gemm(Q, K, V):
    """Materialized causal attention, all matmuls via the GEMM kernel. Loops the
    2D gemm over the batch (kernel is 2D-only); softmax stays elementwise/f32."""
    outs = []
    for i in range(N):
        s = gemm(Q[i], K[i].T) * scale + mask     # (S,S) scores
        p = jax.nn.softmax(s.astype(jnp.float32), axis=-1).astype(jnp.float16)
        outs.append(gemm(p, V[i]))                # (S,D)
    return jnp.stack(outs)

rng = np.random.default_rng(0)
mk = lambda: jnp.asarray(rng.standard_normal((N, S, D)).astype(np.float16))
Q, K, V = mk(), mk(), mk()

def bench(fn, label):
    out = fn(Q, K, V); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = fn(Q, K, V)
    jax.block_until_ready(out)
    print(f"{label:28s} {(time.perf_counter()-t0)/REPS*1e3:7.2f} ms")

g_loss = lambda Q, K, V: jnp.sum(attn_gemm(Q, K, V).astype(jnp.float32))
f_loss = lambda Q, K, V: jnp.sum(_attention_iree_flash(Q, K, V).astype(jnp.float32))
bench(jax.jit(attn_gemm),               "gemm-attn fwd")
bench(jax.jit(_attention_iree_flash),   "flash fwd")
bench(jax.jit(jax.grad(g_loss, (0,1,2))), "gemm-attn fwd+bwd")
bench(jax.jit(jax.grad(f_loss, (0,1,2))), "flash fwd+bwd")
