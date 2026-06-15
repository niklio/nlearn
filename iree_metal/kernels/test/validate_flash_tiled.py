"""Validate the TILED flash_attention_fwd (seq_len % 64 == 0, d_head=64) on Metal
via IREE vs a NumPy causal-attention reference, including the logsumexp L.

  iree-compile --iree-hal-target-device=metal --iree-metal-compile-to-metallib=false \
    --iree-hal-executable-object-search-path=<kernels-dir> flash_test_tiled.mlir -o ft.vmfb
  python validate_flash_tiled.py ft.vmfb
"""
import sys
import numpy as np
import iree.runtime as rt

H, S, D = 2, 128, 64
np.random.seed(1)
Q = np.random.randn(H, S, D).astype(np.float32)
K = np.random.randn(H, S, D).astype(np.float32)
V = np.random.randn(H, S, D).astype(np.float32)
scale = 1.0 / np.sqrt(D)

def ref(Q, K, V):
    out = np.zeros_like(Q)
    L = np.zeros((H, S), np.float32)
    for h in range(H):
        sc = (Q[h] @ K[h].T) * scale
        sc[np.triu(np.ones((S, S)), 1).astype(bool)] = -np.inf
        mx = sc.max(-1, keepdims=True)
        e = np.exp(sc - mx)
        out[h] = (e / e.sum(-1, keepdims=True)) @ V[h]
        L[h] = (mx[:, 0] + np.log(e.sum(-1)))
    return out, L

vmfb = sys.argv[1] if len(sys.argv) > 1 else "ft.vmfb"
ctx = rt.SystemContext(config=rt.Config("metal"))
ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, open(vmfb, "rb").read()))
O, L = ctx.modules.flash["flash"](Q.reshape(-1), K.reshape(-1), V.reshape(-1))
O = np.asarray(O).reshape(H, S, D)
L = np.asarray(L).reshape(H, S)
Oref, Lref = ref(Q, K, V)
od = float(np.max(np.abs(O - Oref)))
ld = float(np.max(np.abs(L - Lref)))
print(f"RESULT O max_abs_diff={od:.2e} L max_abs_diff={ld:.2e} "
      f"allclose={np.allclose(O, Oref, atol=1e-4) and np.allclose(L, Lref, atol=1e-4)}")
