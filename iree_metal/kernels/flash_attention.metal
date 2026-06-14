// flash_attention.metal — causal FlashAttention forward (Metal Shading Language)
//
// Fused single-kernel attention with online (streaming) softmax: never
// materialises the seq x seq score matrix, so memory is O(seq) per query row
// instead of O(seq^2). This is the property jax-metal could never give us (it
// rejects custom kernels at legalization); IREE's Metal HAL can load MSL.
//
// Layout (matches model.py attention_forward, which passes
// (n_heads, seq_len, d_head) tensors):
//   Q, K, V : [n_heads, seq_len, d_head]  row-major, float
//   O       : [n_heads, seq_len, d_head]  output
// One thread computes one output row O[head, q] by streaming over keys k<=q
// (causal), maintaining the running max m and normaliser l per the
// FlashAttention recurrence. d_head is bounded (<= 128) so the accumulator
// lives in registers.
//
// Params (push constants): n_heads, seq_len, d_head, scale (1/sqrt(d_head)).

#include <metal_stdlib>
using namespace metal;

#define MAX_DHEAD 128

struct AttnParams {
    uint  n_heads;
    uint  seq_len;
    uint  d_head;
    float scale;
};

kernel void flash_attention_fwd(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device       float* O       [[buffer(3)]],
    constant     AttnParams& p  [[buffer(4)]],
    uint2 gid                   [[thread_position_in_grid]])
{
    // gid.x = query position within the sequence, gid.y = head index.
    const uint q   = gid.x;
    const uint h   = gid.y;
    if (q >= p.seq_len || h >= p.n_heads) return;

    const uint d        = p.d_head;
    const uint head_off = h * p.seq_len * d;
    const uint q_off    = head_off + q * d;

    // Load this query row into registers.
    float qreg[MAX_DHEAD];
    for (uint i = 0; i < d; ++i) qreg[i] = Q[q_off + i];

    // Online-softmax running state.
    float m = -INFINITY;          // running max of scores
    float l = 0.0f;               // running sum of exp(score - m)
    float acc[MAX_DHEAD];
    for (uint i = 0; i < d; ++i) acc[i] = 0.0f;

    // Stream over keys 0..q (causal mask: key index must be <= query index).
    for (uint k = 0; k <= q; ++k) {
        const uint k_off = head_off + k * d;

        // score = scale * dot(q, k_k)
        float s = 0.0f;
        for (uint i = 0; i < d; ++i) s += qreg[i] * K[k_off + i];
        s *= p.scale;

        // Online softmax update: rescale the accumulator when the max grows.
        const float m_new = max(m, s);
        const float corr  = exp(m - m_new);   // 0 on the first iter (m=-inf -> exp(-inf)=0)
        const float w     = exp(s - m_new);

        l = l * corr + w;
        const uint v_off = head_off + k * d;
        for (uint i = 0; i < d; ++i) acc[i] = acc[i] * corr + w * V[v_off + i];

        m = m_new;
    }

    // Normalise and write out. l > 0 always (k==q term contributes w=1 at s=m).
    const float inv_l = 1.0f / l;
    for (uint i = 0; i < d; ++i) O[q_off + i] = acc[i] * inv_l;
}
