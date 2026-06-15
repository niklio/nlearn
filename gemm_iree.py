"""gemm_iree.py — route matmuls through the custom Metal simdgroup GEMM kernel.

On the IREE-Metal backend, `gemm(A, B)` dispatches the hand-authored
`kernels/gemm.metal` (f16 in, f32 accumulate, ~2.4 TFLOPS vs IREE's naive
~0.5) via a `custom_call @gemm` that the ConvertFlashAttentionDispatch compiler
pass lowers to a `flow.dispatch`. The kernel requires M,N multiples of 32 and K a
multiple of 64, so `_gemm_padded` zero-pads to those and slices the result back
(zero rows/cols contribute nothing). A `custom_vjp` gives the matmul gradient
(dA = dC·Bᵀ, dB = Aᵀ·dC), itself routed through the kernel.

Off IREE-Metal (or with NLEARN_DISABLE_GEMM=1) `gemm` is a plain `A @ B`.
"""
import os
import jax
import jax.numpy as jnp


def _is_iree_metal():
    try:
        return jax.devices()[0].platform.lower() == "iree_metal"
    except Exception:
        return False


USE_IREE_GEMM = _is_iree_metal() and os.environ.get("NLEARN_DISABLE_GEMM") != "1"


def _ceil(x, m):
    return ((x + m - 1) // m) * m


def _gemm_raw(A, B):
    """A[M,K] f16 @ B[K,N] f16 -> C[M,N] f32, dims already conforming."""
    M = A.shape[0]
    N = B.shape[1]
    return jax.ffi.ffi_call(
        "gemm",
        jax.ShapeDtypeStruct((M, N), jnp.float32),
        vmap_method="sequential",
    )(A, B)


def _gemm_padded(A, B):
    """A[M,K] @ B[K,N] -> C[M,N] f32, padding to the kernel's block multiples."""
    M, K = A.shape
    N = B.shape[1]
    Mp, Kp, Np = _ceil(M, 32), _ceil(K, 64), _ceil(N, 32)
    Af = A.astype(jnp.float16)
    Bf = B.astype(jnp.float16)
    if Mp != M or Kp != K:
        Af = jnp.pad(Af, ((0, Mp - M), (0, Kp - K)))
    if Kp != K or Np != N:
        Bf = jnp.pad(Bf, ((0, Kp - K), (0, Np - N)))
    C = _gemm_raw(Af, Bf)
    return C[:M, :N]


@jax.custom_vjp
def gemm(A, B):
    """Differentiable A[M,K] @ B[K,N] -> C[M,N] (f32) via the Metal GEMM kernel."""
    return _gemm_padded(A, B)


def _gemm_fwd(A, B):
    return _gemm_padded(A, B), (A, B)


def _gemm_bwd(res, dC):
    A, B = res
    dCf = dC.astype(jnp.float16)
    dA = _gemm_padded(dCf, B.T.astype(jnp.float16))   # (M,N)@(N,K) -> (M,K)
    dB = _gemm_padded(A.T.astype(jnp.float16), dCf)   # (K,M)@(M,N) -> (K,N)
    return dA.astype(A.dtype), dB.astype(B.dtype)


gemm.defvjp(_gemm_fwd, _gemm_bwd)


def matmul(A, B):
    """A @ B routed through the Metal GEMM kernel on IREE-Metal, else jnp.matmul.

    Safe drop-in for `@`: only plain 2D×2D matmuls go to the kernel (it has no
    batch dim); anything else falls back to jnp.matmul.
    """
    if USE_IREE_GEMM and getattr(A, "ndim", None) == 2 and getattr(B, "ndim", None) == 2:
        return gemm(A, B)
    return jnp.matmul(A, B)
