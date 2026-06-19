"""Feed the ACTUAL transformer features (model_forward_features at step 0, seq8192
bs1) into linear_cross_entropy and locate where the gradient goes NaN. The random-X
check passes; the full-model fused path NaNs -- so the trigger is the real X. This
dumps X stats and checks finiteness of the fused loss / dX / dW, and compares to a
pure-numpy f32 reference on the same real X."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import ce_iree
from model import init_model, model_forward_features, D_MODEL, VOCAB_SIZE

SEQ = int(os.environ.get("RX_SEQ", "8192"))

params = jax.jit(init_model)(jax.random.PRNGKey(0))
rng = np.random.default_rng(0)
# a real-ish token batch (vocab ids); training uses fineweb but token *values* in
# [0,VOCAB) are what matters for the feature scale, and embeddings are seeded.
ids = jnp.asarray(rng.integers(0, VOCAB_SIZE, (1, SEQ + 1)).astype(np.int32))
input_ids = ids[:, :-1]
target_ids = ids[:, 1:]

x = model_forward_features(params, input_ids)            # (1, SEQ, D_MODEL)
X = x.reshape(-1, D_MODEL)
W = params['lm_head']
tgt = target_ids.reshape(-1)
Xn = np.asarray(X, np.float32)
print(f"X shape={Xn.shape} min={Xn.min():.3f} max={Xn.max():.3f} "
      f"absmax={np.abs(Xn).max():.3f} std={Xn.std():.3f} finite={np.isfinite(Xn).all()}", flush=True)
Wn = np.asarray(W, np.float32)
print(f"W absmax={np.abs(Wn).max():.4f} std={Wn.std():.4f} finite={np.isfinite(Wn).all()}", flush=True)

fn = jax.jit(jax.value_and_grad(ce_iree.linear_cross_entropy, (0, 1)))
lf, (dXf, dWf) = fn(X, W, tgt)
dXf = np.asarray(dXf); dWf = np.asarray(dWf)
print(f"FUSED loss={float(lf):.6f} finite_dX={np.isfinite(dXf).all()} "
      f"finite_dW={np.isfinite(dWf).all()} dX_absmax={np.abs(dXf[np.isfinite(dXf)]).max():.3e} "
      f"dW_absmax={np.abs(dWf[np.isfinite(dWf)]).max():.3e}", flush=True)
if not np.isfinite(dXf).all():
    rows = np.where(~np.isfinite(dXf).all(axis=1))[0]
    print(f"  dX non-finite rows: {len(rows)} e.g. {rows[:8]}", flush=True)
if not np.isfinite(dWf).all():
    cols = np.where(~np.isfinite(dWf).any(axis=0))[0]
    print(f"  dW non-finite cols: {len(cols)} e.g. {cols[:8]}", flush=True)

# numpy f32 reference on the SAME real X
logits = Xn @ Wn
m = logits.max(1, keepdims=True); z = np.exp(logits - m)
lse = m[:, 0] + np.log(z.sum(1))
tn = np.asarray(tgt, np.int64); M = Xn.shape[0]
print(f"REF lse min={lse.min():.3f} max={lse.max():.3f}; logits absmax={np.abs(logits).max():.3f}", flush=True)
print("DONE", flush=True)
