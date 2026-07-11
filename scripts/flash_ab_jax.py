import time, jax, jax.numpy as jnp
from jax import random
from nlearn.attention import _attention_iree_flash
BH,S,DH=48,1024,64
key=random.PRNGKey(0)
mk=lambda k: jax.jit(lambda kk: random.normal(kk,(BH,S,DH),jnp.float32))(k)
Q,K,V=mk(key),mk(random.split(key)[1]),mk(random.split(key,3)[2])
fwd=jax.jit(lambda q,k,v:_attention_iree_flash(q,k,v))
gfn=jax.jit(jax.grad(lambda q,k,v: _attention_iree_flash(q,k,v).sum(),argnums=(0,1,2)))
def t(fn,n=30):
    r=fn(Q,K,V); jax.block_until_ready(r); b=1e9
    for _ in range(n):
        s=time.time(); jax.block_until_ready(fn(Q,K,V)); b=min(b,time.time()-s)
    return b
ff=2*BH*S*S*DH  # causal fwd ~2*bh*s^2*d
tf=t(fwd); tg=t(gfn)
print(f"nlearn-flash: fwd {tf*1000:.2f}ms {ff/tf/1e12:.2f} TFLOP/s | fwd+bwd {tg*1000:.2f}ms {3.5*ff/tg/1e12:.2f} TFLOP/s",flush=True)
