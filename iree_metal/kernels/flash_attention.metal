// flash_attention.metal — causal FlashAttention forward (Metal Shading Language)
//
// Fused single-kernel attention with online (streaming) softmax: never
// materialises the seq x seq score matrix, so memory is O(seq) per query row
// instead of O(seq^2). This is the property jax-metal could never give us (it
// rejects custom kernels at legalization); IREE's Metal HAL loads MSL.
//
// Conforms to IREE's Metal dispatch ABI (spirv-cross argument_buffers=true):
//   bindings   -> argument buffer struct at [[buffer(0)]], each at [[id(n)]]
//   constants  -> raw bytes at IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX (=3)
//   global id  -> thread_position_in_threadgroup + threadgroup_position_in_grid * tg_size
// See kernels/abi_reference/ and kernels/README.md.
//
// Pipeline layout: constants = 3 (n_heads, seq_len, d_head),
//   bindings = [RO Q, RO K, RO V, RW O], all [n_heads, seq_len, d_head] f32.
// Dispatch is 1D over n_heads*seq_len; each thread computes one output row
// O[head, q] by streaming over keys k<=q (causal). scale = 1/sqrt(d_head) is
// computed in-kernel to avoid a float push constant.

#include <metal_stdlib>
using namespace metal;

#define MAX_DHEAD 128
#define TG_SIZE_X 64u   // must match the export's workgroup_size

struct Bindings {
  device const float* Q [[id(0)]];
  device const float* K [[id(1)]];
  device const float* V [[id(2)]];
  device       float* O [[id(3)]];
};

struct PushConstants {
  uint n_heads;
  uint seq_len;
  uint d_head;
};

kernel void flash_attention_fwd(
    constant Bindings&      args [[buffer(0)]],
    constant PushConstants& p    [[buffer(3)]],
    uint3 wgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    const uint g = lid.x + wgid.x * TG_SIZE_X;   // linear row index
    const uint total = p.n_heads * p.seq_len;
    if (g >= total) return;

    const uint h = g / p.seq_len;                // head
    const uint q = g % p.seq_len;                // query position
    const uint d = p.d_head;
    const float scale = 1.0f / sqrt((float)d);

    const uint head_off = h * p.seq_len * d;
    const uint q_off    = head_off + q * d;

    float qreg[MAX_DHEAD];
    for (uint i = 0; i < d; ++i) qreg[i] = args.Q[q_off + i];

    // Online-softmax running state.
    float m = -INFINITY;          // running max score
    float l = 0.0f;               // running sum of exp(score - m)
    float acc[MAX_DHEAD];
    for (uint i = 0; i < d; ++i) acc[i] = 0.0f;

    // Stream over keys 0..q (causal mask).
    for (uint k = 0; k <= q; ++k) {
        const uint k_off = head_off + k * d;
        float s = 0.0f;
        for (uint i = 0; i < d; ++i) s += qreg[i] * args.K[k_off + i];
        s *= scale;

        const float m_new = max(m, s);
        const float corr  = exp(m - m_new);   // 0 on first iter (m=-inf)
        const float w     = exp(s - m_new);
        l = l * corr + w;
        for (uint i = 0; i < d; ++i) acc[i] = acc[i] * corr + w * args.V[k_off + i];
        m = m_new;
    }

    const float inv_l = 1.0f / l;
    for (uint i = 0; i < d; ++i) args.O[q_off + i] = acc[i] * inv_l;
}
