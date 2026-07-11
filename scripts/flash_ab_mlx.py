import time, mlx.core as mx
B,H,S,DH=4,12,1024,64
mk=lambda: mx.random.normal((B,H,S,DH)).astype(mx.float32)
Q,K,V=mk(),mk(),mk(); mx.eval(Q,K,V)
sc=DH**-0.5
fwd=mx.compile(lambda q,k,v: mx.fast.scaled_dot_product_attention(q,k,v,scale=sc,mask="causal"))
def gsum(q,k,v): return mx.fast.scaled_dot_product_attention(q,k,v,scale=sc,mask="causal").sum()
gfn=mx.compile(mx.grad(gsum,argnums=(0,1,2)))
def t(fn,n=30):
    mx.eval(fn(Q,K,V)); b=1e9
    for _ in range(n):
        s=time.time(); mx.eval(fn(Q,K,V)); b=min(b,time.time()-s)
    return b
ff=2*B*H*S*S*DH
tf=t(fwd); tg=t(gfn)
print(f"MLX-SDPA:     fwd {tf*1000:.2f}ms {ff/tf/1e12:.2f} TFLOP/s | fwd+bwd {tg*1000:.2f}ms {3.5*ff/tg/1e12:.2f} TFLOP/s",flush=True)
