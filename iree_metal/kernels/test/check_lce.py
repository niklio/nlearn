"""Validate fused linear-CE (ce_iree.linear_cross_entropy) loss + dX + dW vs a
pure-numpy f32 reference (logits = X@W on host, then standard CE). The numpy
reference never touches the IREE matmul, so it can't hit the bf16 const-padding
crash at large V -- this lets us exercise the MULTI-CHUNK fused path (V > _LCE_CHUNK)
in isolation. Source iree_env.sh."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import nlearn.kernels.cross_entropy as ce_iree
CASES = os.environ.get("LCE_CASES", "256,64,2050;512,256,12000")


def ref_np(X, W, targets):
    X = np.asarray(X, np.float32); W = np.asarray(W, np.float32)
    tgt = np.asarray(targets, np.int64)
    M = X.shape[0]
    logits = X @ W
    m = logits.max(axis=1, keepdims=True)
    z = np.exp(logits - m)
    lse = (m[:, 0] + np.log(z.sum(axis=1)))
    correct = logits[np.arange(M), tgt]
    loss = float((lse - correct).mean())
    sm = z / z.sum(axis=1, keepdims=True)
    oh = np.zeros_like(logits); oh[np.arange(M), tgt] = 1.0
    dlogits = (sm - oh) / M
    dX = dlogits @ W.T
    dW = X.T @ dlogits
    return loss, dX, dW


for spec in CASES.split(";"):
    M, D, V = (int(x) for x in spec.split(","))
    rng = np.random.default_rng(0)
    X = jnp.asarray((rng.standard_normal((M, D)) * 0.1).astype(np.float32))
    W = jnp.asarray((rng.standard_normal((D, V)) * 0.05).astype(np.float32))
    tgt = jnp.asarray(rng.integers(0, V, (M,)).astype(np.int32))
    nchunks = (V + 4096 - 1) // 4096
    print(f"--- M={M} D={D} V={V} ({nchunks} chunks): tracing+running fused ...", flush=True)

    fn = jax.jit(jax.value_and_grad(ce_iree.linear_cross_entropy, (0, 1)))
    lf, (dXf, dWf) = fn(X, W, tgt)
    lf = float(lf)
    dXf = np.asarray(dXf); dWf = np.asarray(dWf)
    print(f"    fused returned: loss={lf:.6f} finite_dX={np.isfinite(dXf).all()} "
          f"finite_dW={np.isfinite(dWf).all()}", flush=True)

    lr, dXr, dWr = ref_np(X, W, tgt)
    rel = lambda a, b: float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean()
                             / (np.abs(np.asarray(b, np.float32)).mean() + 1e-12))
    print(f"    RESULT loss fused={lf:.6f} ref={lr:.6f} |d|={abs(lf-lr):.2e}  "
          f"dX rel={rel(dXf,dXr):.2e}  dW rel={rel(dWf,dWr):.2e}", flush=True)

print("ALL CASES DONE", flush=True)
