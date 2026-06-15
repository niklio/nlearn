"""Validate gemm.metal (simdgroup_matrix GEMM) on Metal via IREE vs NumPy.
Build gemm.vmfb from gemm_test.mlir with a patched iree-compile (patch 04),
--iree-metal-compile-to-metallib=false, and --iree-hal-executable-object-search-
path pointing at the dir with gemm.metal. M=N=K=1024 (must be multiples of 8;
the test MLIR's count region divides by the BMxBN block, currently 16)."""
import sys
import numpy as np
import iree.runtime as rt

M = N = K = 1024
np.random.seed(0)
A = (np.random.randn(M, K) * 0.1).astype(np.float16)
B = (np.random.randn(K, N) * 0.1).astype(np.float16)
ref = A.astype(np.float32) @ B.astype(np.float32)

vmfb = sys.argv[1] if len(sys.argv) > 1 else "gemm.vmfb"
ctx = rt.SystemContext(config=rt.Config("metal"))
ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, open(vmfb, "rb").read()))
C = np.asarray(ctx.modules.g["run_gemm"](A, B))
print("gemm max_abs_diff=%.2e allclose=%s" % (
    np.max(np.abs(C - ref)), np.allclose(C, ref, atol=2e-2)))
