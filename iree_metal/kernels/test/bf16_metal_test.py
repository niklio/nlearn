"""Does IREE metal-spirv legalize bf16 ELEMENTWISE ops (the ones IREE codegens:
layernorm/gelu/softmax/residual)? Matmuls go through our custom kernels, so only
elementwise bf16 needs to work for a bf16 activation path. Source iree_env.sh."""
import numpy as np, jax, jax.numpy as jnp

def ln_gelu(x, g, b):
    # layernorm + gelu + residual — representative bf16 elementwise graph (no matmul)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    h = (x - mu) * jax.lax.rsqrt(var + 1e-5) * g + b
    h = 0.5 * h * (1 + jnp.tanh(0.7978845608 * (h + 0.044715 * h**3)))
    return x + h

for dt in (jnp.bfloat16, jnp.float16):
    try:
        x = jnp.asarray(np.random.randn(8, 512, 512).astype(np.float32)).astype(dt)
        g = jnp.ones((512,), dt); b = jnp.zeros((512,), dt)
        out = jax.jit(ln_gelu)(x, g, b); out.block_until_ready()
        print(f"{jnp.dtype(dt).name}: OK  out.dtype={out.dtype} mean={float(out.astype(jnp.float32).mean()):.4f}")
    except Exception as e:
        print(f"{jnp.dtype(dt).name}: FAILED — {str(e).splitlines()[0][:140]}")
