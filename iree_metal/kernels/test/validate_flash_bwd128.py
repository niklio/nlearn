"""Validate the full flash forward+backward on Metal via IREE vs a NumPy
analytic reference (full-matrix standard attention). Requires flashbwd.vmfb
built from flash_bwd_test.mlir with a patched iree-compile (patches 04). Run
the kernels and compare O, dQ, dK, dV. All match to ~1e-6 (float32 eps)."""
import sys
import numpy as np
import iree.runtime as rt

H, S, Dh = 2, 128, 64
scale = 1.0 / np.sqrt(Dh)
np.random.seed(2)
Q = np.random.randn(H, S, Dh).astype(np.float32)
K = np.random.randn(H, S, Dh).astype(np.float32)
V = np.random.randn(H, S, Dh).astype(np.float32)
dO = np.random.randn(H, S, Dh).astype(np.float32)

vmfb = sys.argv[1] if len(sys.argv) > 1 else "flashbwd.vmfb"
ctx = rt.SystemContext(config=rt.Config("metal"))
ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, open(vmfb, "rb").read()))
m = ctx.modules.fa

O_k, L_k = m["run_fwd"](Q.reshape(-1), K.reshape(-1), V.reshape(-1))
O_k = np.asarray(O_k); L_k = np.asarray(L_k)
Dvec = (dO.reshape(H, S, Dh) * O_k.reshape(H, S, Dh)).sum(-1).reshape(-1).astype(np.float32)  # D_i = dO_i.O_i
dQ_k = np.asarray(m["run_dq"](Q.reshape(-1), K.reshape(-1), V.reshape(-1),
                              dO.reshape(-1), L_k, Dvec)).reshape(H, S, Dh)
r = m["run_dkdv"](Q.reshape(-1), K.reshape(-1), V.reshape(-1), dO.reshape(-1), L_k, Dvec)
dK_k = np.asarray(r[0]).reshape(H, S, Dh); dV_k = np.asarray(r[1]).reshape(H, S, Dh)

# NumPy analytic reference (full-matrix standard causal attention + backward).
mask = np.triu(np.ones((S, S), bool), 1)
dQr = np.zeros_like(Q); dKr = np.zeros_like(K); dVr = np.zeros_like(V); Or = np.zeros_like(Q)
for h in range(H):
    s = (Q[h] @ K[h].T) * scale; s[mask] = -np.inf
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    Or[h] = p @ V[h]; dVr[h] = p.T @ dO[h]
    dp = dO[h] @ V[h].T; ds = p * (dp - (dp * p).sum(-1, keepdims=True))
    dQr[h] = (ds @ K[h]) * scale; dKr[h] = (ds.T @ Q[h]) * scale

ok = all(np.allclose(a, b, atol=1e-4) for a, b in
         [(O_k.reshape(H, S, Dh), Or), (dQ_k, dQr), (dK_k, dKr), (dV_k, dVr)])
sys.stdout.write("O %.2e | dQ %.2e | dK %.2e | dV %.2e | allclose=%s\n" % (
    np.max(np.abs(O_k.reshape(H, S, Dh) - Or)), np.max(np.abs(dQ_k - dQr)),
    np.max(np.abs(dK_k - dKr)), np.max(np.abs(dV_k - dVr)), ok))
sys.stdout.flush()
