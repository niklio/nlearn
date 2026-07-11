#!/usr/bin/env python3
"""MLX reference benchmark: deep transformer fwd+bwd, mirroring nlearn's real architecture.

The NORTH-STAR number for the open-IREE stack to close in on. Same dims as deep_hero (24L, d=768,
seq1024, bs4), fused attention (mx.fast.scaled_dot_product_attention == nlearn's flash kernel),
SwiGLU MLP, tied embeddings, fwd+bwd via mx.value_and_grad, mx.compile'd step. Reports raw TFLOP/s
+ tok/s using the SAME 6*N*T formula as bench_backend_ab.py so numbers are directly comparable.

Run: python3 scripts/bench_mlx.py     (uses the base python3 that has mlx)
"""
import os, time, math
import mlx.core as mx
import mlx.nn as nn

D = int(os.environ.get("AB_D", 768)); H = int(os.environ.get("AB_H", 12))
L = int(os.environ.get("AB_L", 24)); FF = int(os.environ.get("AB_FF", 3072)); V = 50257
SEQ = int(os.environ.get("AB_SEQ", 1024)); BS = int(os.environ.get("AB_BS", 4))
STEPS = int(os.environ.get("AB_STEPS", 12))
DT = mx.bfloat16
HD = D // H; SCALE = HD ** -0.5

def ln(x, g, b):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + 1e-5) * g + b

def block(p, x):
    h = ln(x, p['g1'], p['b1'])
    q = (h @ p['wq']).reshape(BS, SEQ, H, HD).transpose(0, 2, 1, 3)
    k = (h @ p['wk']).reshape(BS, SEQ, H, HD).transpose(0, 2, 1, 3)
    v = (h @ p['wv']).reshape(BS, SEQ, H, HD).transpose(0, 2, 1, 3)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal")
    o = o.transpose(0, 2, 1, 3).reshape(BS, SEQ, D) @ p['wo']
    x = x + o
    h = ln(x, p['g2'], p['b2'])
    return x + (nn.silu(h @ p['w1']) * (h @ p['w3'])) @ p['w2']

def forward(P, tok):
    x = P['emb'][tok]
    for i in range(L):
        x = block(P['blk'][i], x)
    x = ln(x, P['gf'], P['bf'])
    return x @ P['emb'].T

def loss_fn(P, tok, tgt):
    logits = forward(P, tok).astype(mx.float32)
    return nn.losses.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="mean")

def init():
    s = 0.02
    def lin(a, b): return mx.random.normal((a, b)).astype(DT) * s
    blk = [dict(g1=mx.ones((D,), DT), b1=mx.zeros((D,), DT), g2=mx.ones((D,), DT), b2=mx.zeros((D,), DT),
               wq=lin(D, D), wk=lin(D, D), wv=lin(D, D), wo=lin(D, D),
               w1=lin(D, FF), w3=lin(D, FF), w2=lin(FF, D)) for _ in range(L)]
    return dict(emb=mx.random.normal((V, D)).astype(DT) * s,
                gf=mx.ones((D,), DT), bf=mx.zeros((D,), DT), blk=blk)

def tree_size(t):
    if isinstance(t, mx.array): return t.size
    if isinstance(t, dict): return sum(tree_size(v) for v in t.values())
    if isinstance(t, (list, tuple)): return sum(tree_size(v) for v in t)
    return 0

def main():
    mx.random.seed(0)
    P = init()
    nparams = tree_size(P)
    tok = mx.random.randint(0, V, (BS, SEQ))
    tgt = mx.random.randint(0, V, (BS, SEQ))
    mx.eval(P, tok, tgt)
    vg = mx.value_and_grad(loss_fn)
    step = mx.compile(vg)
    tokens = BS * SEQ
    flops = 6 * nparams * tokens
    print(f"backend=MLX params={nparams/1e6:.1f}M device={mx.default_device()}", flush=True)
    times = []
    for i in range(STEPS):
        t = time.time()
        l, g = step(P, tok, tgt)
        mx.eval(l, g)
        dt = time.time() - t
        times.append(dt)
        if i < 2:
            print(f"  step {i} (warmup/compile) {dt:.2f}s loss={float(l):.2f}", flush=True)
    steady = sorted(times[2:])[: max(1, len(times[2:]) - 1)]
    avg = sum(steady) / len(steady)
    print(f"  RESULT backend=MLX: {avg:.3f}s/step  {tokens/avg:.0f} tok/s  "
          f"{flops/avg/1e12:.2f} TFLOP/s  (n={len(steady)})", flush=True)

if __name__ == "__main__":
    main()
