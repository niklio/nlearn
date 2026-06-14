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

#define MAX_DHEAD 128
#define TG_SIZE_X 64u   // must match each export's workgroup_size

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

kernel void flash_attention_fwd(
    constant FwdBindings&   args [[buffer(0)]],
    constant PushConstants& p    [[buffer(3)]],
    uint3 wgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    const uint g = lid.x + wgid.x * TG_SIZE_X;
    if (g >= p.n_heads * p.seq_len) return;
    const uint h = g / p.seq_len, q = g % p.seq_len, d = p.d_head;
    const float scale = 1.0f / sqrt((float)d);
    const uint head_off = h * p.seq_len * d;
    const uint q_off = head_off + q * d;

    float qreg[MAX_DHEAD];
    for (uint i = 0; i < d; ++i) qreg[i] = args.Q[q_off + i];

    float m = -INFINITY, l = 0.0f;
    float acc[MAX_DHEAD];
    for (uint i = 0; i < d; ++i) acc[i] = 0.0f;

    for (uint k = 0; k <= q; ++k) {                 // causal k<=q
        const uint k_off = head_off + k * d;
        float s = 0.0f;
        for (uint i = 0; i < d; ++i) s += qreg[i] * args.K[k_off + i];
        s *= scale;
        const float m_new = max(m, s);
        const float corr = exp(m - m_new), w = exp(s - m_new);
        l = l * corr + w;
        for (uint i = 0; i < d; ++i) acc[i] = acc[i] * corr + w * args.V[k_off + i];
        m = m_new;
    }
    const float inv_l = 1.0f / l;
    for (uint i = 0; i < d; ++i) args.O[q_off + i] = acc[i] * inv_l;
    args.L[h * p.seq_len + q] = m + log(l);          // logsumexp
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
    const uint i_off = head_off + i * d;

    float qreg[MAX_DHEAD], doreg[MAX_DHEAD], dq[MAX_DHEAD];
    for (uint e = 0; e < d; ++e) { qreg[e] = args.Q[i_off + e]; doreg[e] = args.dO[i_off + e]; dq[e] = 0.0f; }
    const float Li = args.L[h * p.seq_len + i];
    const float Di = args.D[h * p.seq_len + i];

    for (uint j = 0; j <= i; ++j) {                 // causal j<=i
        const uint j_off = head_off + j * d;
        float s = 0.0f;
        for (uint e = 0; e < d; ++e) s += qreg[e] * args.K[j_off + e];
        const float pij = exp(s * scale - Li);
        float dp = 0.0f;
        for (uint e = 0; e < d; ++e) dp += doreg[e] * args.V[j_off + e];
        const float ds = pij * (dp - Di);
        for (uint e = 0; e < d; ++e) dq[e] += scale * ds * args.K[j_off + e];
    }
    for (uint e = 0; e < d; ++e) args.dQ[i_off + e] = dq[e];
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
    const uint j_off = head_off + j * d;

    float kreg[MAX_DHEAD], vreg[MAX_DHEAD], dk[MAX_DHEAD], dv[MAX_DHEAD];
    for (uint e = 0; e < d; ++e) { kreg[e] = args.K[j_off + e]; vreg[e] = args.V[j_off + e]; dk[e] = 0.0f; dv[e] = 0.0f; }

    for (uint i = j; i < p.seq_len; ++i) {          // causal i>=j
        const uint i_off = head_off + i * d;
        const float Li = args.L[h * p.seq_len + i];
        const float Di = args.D[h * p.seq_len + i];
        float s = 0.0f;
        for (uint e = 0; e < d; ++e) s += args.Q[i_off + e] * kreg[e];
        const float pij = exp(s * scale - Li);
        float dp = 0.0f;
        for (uint e = 0; e < d; ++e) dp += args.dO[i_off + e] * vreg[e];
        const float ds = pij * (dp - Di);
        for (uint e = 0; e < d; ++e) {
            dv[e] += pij * args.dO[i_off + e];
            dk[e] += scale * ds * args.Q[i_off + e];
        }
    }
    for (uint e = 0; e < d; ++e) { args.dK[j_off + e] = dk[e]; args.dV[j_off + e] = dv[e]; }
}
