"""Take the FULL gradient of the fused-lmhead batched loss w.r.t. ALL params at
seq8192 bs1 and report which param-tree leaves are non-finite. This backprops
through the whole transformer (flash bwd, RoPE, layernorm) + lce, exactly like a
real training step -- pinpointing where the step-0 NaN originates. Compares fused
vs materialized on the same params+batch. Source iree_env.sh."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import nlearn.kernels.cross_entropy as ce_iree
from nlearn.model import init_model, model_forward_features, model_forward, D_MODEL, VOCAB_SIZE
import nlearn.kernels.gemm as gemm_iree
SEQ = int(os.environ.get("RX_SEQ", "8192"))

params = jax.jit(init_model)(jax.random.PRNGKey(0))
# Detach params from the init executable's output buffers (host round-trip) and
# drop all compiled executables, so the big transformer-grad executable is the
# only one alive -- dodges the IREE executable-destructor (buffer-deinit) abort.
params = jax.tree.map(lambda a: np.asarray(a), params)
jax.clear_caches()
params = jax.tree.map(lambda a: jnp.asarray(a), params)
rng = np.random.default_rng(0)
ids = jnp.asarray(rng.integers(0, VOCAB_SIZE, (1, SEQ + 1)).astype(np.int32))
input_ids = ids[:, :-1]; target_ids = ids[:, 1:]


def loss_fused(p):
    x = model_forward_features(p, input_ids)
    return ce_iree.linear_cross_entropy(x.reshape(-1, D_MODEL), p['lm_head'],
                                        target_ids.reshape(-1))


def loss_mat(p):
    logits = model_forward(p, input_ids)
    return ce_iree.cross_entropy(logits, target_ids)


def report(name, fn):
    print(f"=== {name} ===", flush=True)
    l, g = jax.jit(jax.value_and_grad(fn))(params)
    print(f"  loss={float(l):.6f}", flush=True)
    flat, _ = jax.tree_util.tree_flatten_with_path(g)
    bad = []
    for path, leaf in flat:
        leaf = np.asarray(leaf)
        if not np.isfinite(leaf).all():
            ks = jax.tree_util.keystr(path)
            nfrac = float((~np.isfinite(leaf)).mean())
            bad.append((ks, leaf.shape, nfrac))
    if not bad:
        print("  ALL gradients finite", flush=True)
    else:
        for ks, shp, nf in bad:
            print(f"  NaN/Inf grad: {ks} shape={shp} frac={nf:.3f}", flush=True)


which = os.environ.get("GN_WHICH", "both")
if which in ("both", "fused"):
    report("FUSED", loss_fused)
if which in ("both", "mat"):
    report("MATERIALIZED", loss_mat)
print("DONE", flush=True)
os._exit(0)   # skip Python teardown -> avoids the IREE executable-destructor abort
