// flash_attention.metal — causal FlashAttention forward + true backward (MSL)
//
// Fused streaming-softmax attention with O(seq) memory in BOTH directions (no
// seq x seq matrix is ever materialised). This is the property jax-metal could
// never give us; IREE's Metal HAL loads MSL. See kernels/README.md + BINDING.md.
//
// IREE Metal dispatch ABI (spirv-cross argument_buffers=true):
//   bindings  -> argument buffer struct at [[buffer(0)]], each at [[id(n)]]
//   constants -> raw bytes at IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX (=3)
//   global id -> thread_position_in_threadgroup + threadgroup_position_in_grid*tg
// Tensors are [n_heads, seq_len, d_head] f32 row-major; L,D are [n_heads, seq_len].
// scale = 1/sqrt(d_head) computed in-kernel. Dispatch is 1D over n_heads*seq_len.

#include <metal_stdlib>
using namespace metal;

#define MAX_DHEAD 64    // d_head (model uses 64). Sized tight: the backward kernels
                        // hold 3-4 per-thread arrays of this length; at 128 they spill
                        // the register file (~0.03 TFLOPS scalar bwd); 64 keeps them in
                        // registers. Requires d_head <= 64.
#define MAX_DHEAD4 16   // MAX_DHEAD / 4 (float4 lanes)
#define TG_SIZE_X 64u   // must match each export's workgroup_size
#define DKV_QTILE 32u   // bwd_dkdv: query rows staged in threadgroup mem per tile

struct PushConstants {
  uint n_heads;
  uint seq_len;
  uint d_head;
};

// ---------------------------------------------------------------------------
// Forward: O = softmax(scale Q Kᵀ + causal) V ; also emits L = logsumexp row.
// ---------------------------------------------------------------------------
struct FwdBindings {
  device const float* Q [[id(0)]];
  device const float* K [[id(1)]];
  device const float* V [[id(2)]];
  device       float* O [[id(3)]];
  device       float* L [[id(4)]];   // [n_heads, seq_len] log-sum-exp per query
};

// One thread per query (online softmax over causal keys). Zero threadgroup
// memory → maximum occupancy; the Apple GPU cache serves the shared K/V reads.
// (A threadgroup-staged "tiled" variant measured ~50% SLOWER here — occupancy and
// caching beat manual staging, same lesson as the GEMM sweep.) The win vs the
// naive scalar loop is float4-vectorized loads + dot/FMA on Q·K and the V accum.
kernel void flash_attention_fwd(
    constant FwdBindings&   args [[buffer(0)]],
    constant PushConstants& p    [[buffer(3)]],
    uint3 wgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    const uint g = lid.x + wgid.x * TG_SIZE_X;
    if (g >= p.n_heads * p.seq_len) return;
    const uint h = g / p.seq_len, q = g % p.seq_len, d = p.d_head;
    const uint dq = d >> 2;                           // float4 lanes (d % 4 == 0)
    const float scale = 1.0f / sqrt((float)d);
    const uint head_off4 = (h * p.seq_len * d) >> 2;
    const uint q_off4 = head_off4 + q * dq;
    device const float4* Q4 = (device const float4*)args.Q;
    device const float4* K4 = (device const float4*)args.K;
    device const float4* V4 = (device const float4*)args.V;

    float4 qreg[MAX_DHEAD4], acc[MAX_DHEAD4];
    for (uint i = 0; i < dq; ++i) { qreg[i] = Q4[q_off4 + i]; acc[i] = float4(0.0f); }

    float m = -INFINITY, l = 0.0f;
    for (uint k = 0; k <= q; ++k) {                  // causal k<=q
        const uint k_off4 = head_off4 + k * dq;
        float s = 0.0f;
        for (uint i = 0; i < dq; ++i) s += dot(qreg[i], K4[k_off4 + i]);
        s *= scale;
        const float m_new = max(m, s);
        const float corr = exp(m - m_new), w = exp(s - m_new);
        l = l * corr + w;
        for (uint i = 0; i < dq; ++i) acc[i] = acc[i] * corr + w * V4[k_off4 + i];
        m = m_new;
    }
    const float inv_l = 1.0f / l;
    device float4* O4 = (device float4*)args.O;
    for (uint i = 0; i < dq; ++i) O4[q_off4 + i] = acc[i] * inv_l;
    args.L[h * p.seq_len + q] = m + log(l);           // logsumexp
}

// ---------------------------------------------------------------------------
// Backward dQ: parallel over query i, loop keys j<=i. Recomputes p from L.
//   p_ij = exp(scale Q_i·K_j - L_i); ds_ij = p_ij (dO_i·V_j - D_i);
//   dQ_i = Σ_j scale ds_ij K_j.   D_i = dO_i·O_i (precomputed in JAX).
// ---------------------------------------------------------------------------
struct DqBindings {
  device const float* Q  [[id(0)]];
  device const float* K  [[id(1)]];
  device const float* V  [[id(2)]];
  device const float* dO [[id(3)]];
  device const float* L  [[id(4)]];
  device const float* D  [[id(5)]];
  device       float* dQ [[id(6)]];
};

kernel void flash_attention_bwd_dq(
    constant DqBindings&    args [[buffer(0)]],
    constant PushConstants& p    [[buffer(3)]],
    uint3 wgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    const uint g = lid.x + wgid.x * TG_SIZE_X;
    if (g >= p.n_heads * p.seq_len) return;
    const uint h = g / p.seq_len, i = g % p.seq_len, d = p.d_head;
    const float scale = 1.0f / sqrt((float)d);
    const uint head_off = h * p.seq_len * d;

    // Tiled (mirror of bwd_dkdv): the 64 query-threads of this head-aligned block all
    // loop over overlapping keys, so stage K/V key-tiles in threadgroup memory and
    // reuse across the 64 queries — ~64× less global traffic. Requires seq_len%64==0.
    threadgroup float4 Kt[DKV_QTILE][MAX_DHEAD4];
    threadgroup float4 Vt[DKV_QTILE][MAX_DHEAD4];

    const uint nq = d >> 2;                          // float4 lanes (d % 4 == 0)
    const uint hoff4 = head_off >> 2;
    const uint i_off4 = hoff4 + i * nq;
    const uint ib = (wgid.x * TG_SIZE_X) % p.seq_len; // first query of this head block
    device const float4* Q4  = (device const float4*)args.Q;
    device const float4* K4  = (device const float4*)args.K;
    device const float4* V4  = (device const float4*)args.V;
    device const float4* dO4 = (device const float4*)args.dO;
    float4 qreg[MAX_DHEAD4], doreg[MAX_DHEAD4], dq[MAX_DHEAD4];
    for (uint e = 0; e < nq; ++e) { qreg[e] = Q4[i_off4 + e]; doreg[e] = dO4[i_off4 + e]; dq[e] = float4(0.0f); }
    const float Li = args.L[h * p.seq_len + i];
    const float Di = args.D[h * p.seq_len + i];

    for (uint kb = 0; kb < ib + TG_SIZE_X; kb += DKV_QTILE) {   // keys j<=max query (ib+63)
        for (uint e = lid.x; e < DKV_QTILE * nq; e += TG_SIZE_X) {
            uint r = e / nq, c = e % nq, kj = kb + r;
            Kt[r][c] = K4[hoff4 + kj * nq + c];
            Vt[r][c] = V4[hoff4 + kj * nq + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DKV_QTILE; ++kk) {
            if (kb + kk > i) break;                 // causal j<=i (keys ascending)
            float s = 0.0f;
            for (uint e = 0; e < nq; ++e) s += dot(qreg[e], Kt[kk][e]);
            const float pij = exp(s * scale - Li);
            float dp = 0.0f;
            for (uint e = 0; e < nq; ++e) dp += dot(doreg[e], Vt[kk][e]);
            const float ds = pij * (dp - Di);
            for (uint e = 0; e < nq; ++e) dq[e] += scale * ds * Kt[kk][e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float4* dQ4 = (device float4*)args.dQ;
    for (uint e = 0; e < nq; ++e) dQ4[i_off4 + e] = dq[e];
}

// ---------------------------------------------------------------------------
// Backward dK/dV: parallel over key j, loop queries i>=j. Recomputes p from L.
//   dV_j = Σ_i p_ij dO_i ;  dK_j = Σ_i scale ds_ij Q_i.
// ---------------------------------------------------------------------------
struct DkvBindings {
  device const float* Q  [[id(0)]];
  device const float* K  [[id(1)]];
  device const float* V  [[id(2)]];
  device const float* dO [[id(3)]];
  device const float* L  [[id(4)]];
  device const float* D  [[id(5)]];
  device       float* dK [[id(6)]];
  device       float* dV [[id(7)]];
};

kernel void flash_attention_bwd_dkdv(
    constant DkvBindings&   args [[buffer(0)]],
    constant PushConstants& p    [[buffer(3)]],
    uint3 wgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    const uint g = lid.x + wgid.x * TG_SIZE_X;
    if (g >= p.n_heads * p.seq_len) return;
    const uint h = g / p.seq_len, j = g % p.seq_len, d = p.d_head;
    const float scale = 1.0f / sqrt((float)d);
    const uint head_off = h * p.seq_len * d;

    // Tiled: the 64 key-threads of this (head-aligned) block all loop over the SAME
    // queries, so stage Q/dO query-tiles in threadgroup memory and reuse across the 64
    // keys — ~64× less global traffic than each thread streaming Q[i]/dO[i] itself.
    // Requires seq_len % 64 == 0 (head-aligned 64-key blocks) — guaranteed by the model
    // (and the dispatch's 1D 64-blocks). float4 throughout.
    threadgroup float4 Qt[DKV_QTILE][MAX_DHEAD4];
    threadgroup float4 dOt[DKV_QTILE][MAX_DHEAD4];
    threadgroup float  Lt[DKV_QTILE], Dt[DKV_QTILE];

    const uint nq = d >> 2;                          // float4 lanes (d % 4 == 0)
    const uint hoff4 = head_off >> 2;
    const uint j_off4 = hoff4 + j * nq;
    const uint jb = (wgid.x * TG_SIZE_X) % p.seq_len; // first key of this head-aligned block
    const uint lrow = h * p.seq_len;
    device const float4* Q4  = (device const float4*)args.Q;
    device const float4* K4  = (device const float4*)args.K;
    device const float4* V4  = (device const float4*)args.V;
    device const float4* dO4 = (device const float4*)args.dO;
    float4 kreg[MAX_DHEAD4], vreg[MAX_DHEAD4], dk[MAX_DHEAD4], dv[MAX_DHEAD4];
    for (uint e = 0; e < nq; ++e) { kreg[e] = K4[j_off4 + e]; vreg[e] = V4[j_off4 + e]; dk[e] = float4(0.0f); dv[e] = float4(0.0f); }

    for (uint qb = jb; qb < p.seq_len; qb += DKV_QTILE) {   // queries i>=jb (block min key)
        for (uint e = lid.x; e < DKV_QTILE * nq; e += TG_SIZE_X) {
            uint r = e / nq, c = e % nq, qi = qb + r;
            Qt[r][c]  = Q4[hoff4 + qi * nq + c];
            dOt[r][c] = dO4[hoff4 + qi * nq + c];
        }
        for (uint r = lid.x; r < DKV_QTILE; r += TG_SIZE_X) { Lt[r] = args.L[lrow + qb + r]; Dt[r] = args.D[lrow + qb + r]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ii = 0; ii < DKV_QTILE; ++ii) {
            if (qb + ii < j) continue;              // causal i>=j
            float s = 0.0f;
            for (uint e = 0; e < nq; ++e) s += dot(Qt[ii][e], kreg[e]);
            const float pij = exp(s * scale - Lt[ii]);
            float dp = 0.0f;
            for (uint e = 0; e < nq; ++e) dp += dot(dOt[ii][e], vreg[e]);
            const float ds = pij * (dp - Dt[ii]);
            for (uint e = 0; e < nq; ++e) {
                dv[e] += pij * dOt[ii][e];
                dk[e] += scale * ds * Qt[ii][e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float4* dK4 = (device float4*)args.dK;
    device float4* dV4 = (device float4*)args.dV;
    for (uint e = 0; e < nq; ++e) { dK4[j_off4 + e] = dk[e]; dV4[j_off4 + e] = dv[e]; }
}
