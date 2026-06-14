"""Run the hand-authored flash_attention_fwd MSL kernel on Metal via IREE and
check numerical parity vs a NumPy reference. Requires a patched iree-compile
(Metal external-object support, patch 04) to have produced flash.vmfb:

  iree-compile --iree-hal-target-device=metal \
    --iree-hal-executable-object-search-path=<dir> flash_test.mlir -o flash.vmfb
"""
import sys
import numpy as np
import iree.runtime as rt

H, S, D = 2, 16, 8
np.random.seed(1)
Q = np.random.randn(H, S, D).astype(np.float32)
K = np.random.randn(H, S, D).astype(np.float32)
V = np.random.randn(H, S, D).astype(np.float32)
scale = 1.0 / np.sqrt(D)

def ref(Q, K, V):
    out = np.zeros_like(Q)
    for h in range(H):
        sc = (Q[h] @ K[h].T) * scale
        sc[np.triu(np.ones((S, S)), 1).astype(bool)] = -np.inf
        w = np.exp(sc - sc.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
        out[h] = w @ V[h]
    return out

vmfb = sys.argv[1] if len(sys.argv) > 1 else "flash.vmfb"
ctx = rt.SystemContext(config=rt.Config("metal"))
ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, open(vmfb, "rb").read()))
out = np.asarray(ctx.modules.flash["flash"](
    Q.reshape(-1), K.reshape(-1), V.reshape(-1))).reshape(H, S, D)
diff = float(np.max(np.abs(out - ref(Q, K, V))))
print(f"RESULT max_abs_diff={diff:.2e} allclose={np.allclose(out, ref(Q,K,V), atol=1e-4)}")
