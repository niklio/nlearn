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

// MMA FlashAttention-2 forward — uses Apple's matrix units (simdgroup_matrix 8x8)
// for both Q·Kᵀ and P·V, instead of the old one-thread-per-query scalar dot loop
// (~0.15 TFLOPS, the matrix units sat idle). Same 1D launch as before (TG_SIZE_X=64
// threads, grid = ceildiv(n_heads*seq_len, 64)) so the compiler pass is unchanged:
// each workgroup owns 64 consecutive (head, query) pairs — and since seq_len % 64 == 0
// they're all one head — split as NWARPS=2 simdgroups, each looping over 8-query
// sub-blocks. The streaming softmax + per-row O rescale (which don't map to matrix
// ops) run scalar in threadgroup memory; the two matmuls run on the matrix units.
//   S = scale·Q Kᵀ (+causal) ; online softmax ; O += P V ; L = m + log l.
#define D_TILES 8           // d_head / 8 (model d_head = 64)
#define NWARPS  2           // TG_SIZE_X / 32
#define QSUB    (TG_SIZE_X / 8)   // 8 query sub-blocks of 8 per 64-query workgroup

kernel void flash_attention_fwd(
    constant FwdBindings&   args [[buffer(0)]],
    constant PushConstants& p    [[buffer(3)]],
    uint  wgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint d = p.d_head, S = p.seq_len;
    const float scale = 1.0f / sqrt((float)d);
    const uint gbase = wgid * TG_SIZE_X;          // first (h,q) of this 64-block
    const uint h  = gbase / S;
    const uint qb = gbase % S;                    // first query (64-aligned, one head)
    const uint head_off = h * S * d;
    device const float* Q = args.Q + head_off;
    device const float* K = args.K + head_off;
    device const float* V = args.V + head_off;
    device       float* O = args.O + head_off;

    threadgroup float Smem[NWARPS][8][8];         // S / P tile per simdgroup
    threadgroup float Omem[NWARPS][8][MAX_DHEAD]; // O accumulator (threadgroup so we can
    threadgroup float PVm_s[NWARPS][8][MAX_DHEAD];// per-row rescale) + P·V scratch

    // Each simdgroup handles sub-blocks sb = sgid + NWARPS*r (covers all QSUB sub-blocks).
    for (uint r = 0; r < QSUB / NWARPS; ++r) {
        const uint sb   = sgid + NWARPS * r;
        const uint qrow = qb + sb * 8;            // first query of this 8-query sub-block

        for (uint e = lane; e < 8 * d; e += 32) Omem[sgid][e / d][e % d] = 0.0f;
        float m_row = -INFINITY, l_row = 0.0f;    // per-row state (rows owned by lane<8)

        simdgroup_float8x8 Qm[D_TILES];
        for (uint dt = 0; dt < D_TILES; ++dt)
            simdgroup_load(Qm[dt], Q + qrow * d + dt * 8, d);

        const uint kmax = qrow + 8;               // causal: keys < kmax (8-aligned)
        for (uint kb = 0; kb < kmax; kb += 8) {
            // S[8q,8k] = Σ_dt Q[:,dt]·Kᵀ[dt,:]  (load K transposed → [8d,8k])
            simdgroup_float8x8 Sm = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint dt = 0; dt < D_TILES; ++dt) {
                simdgroup_float8x8 KTm;
                simdgroup_load(KTm, K + kb * d + dt * 8, d, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sm, Qm[dt], KTm, Sm);
            }
            simdgroup_store(Sm, &Smem[sgid][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // streaming softmax for this key-block (row = lane, lane<8)
            if (lane < 8) {
                const uint qi = qrow + lane;
                float rmax = -INFINITY;
                for (uint ki = 0; ki < 8; ++ki) {
                    float s = (kb + ki <= qi) ? Smem[sgid][lane][ki] * scale : -INFINITY;
                    Smem[sgid][lane][ki] = s;
                    rmax = max(rmax, s);
                }
                const float m_new = max(m_row, rmax);
                const float corr  = exp(m_row - m_new);
                float rsum = 0.0f;
                for (uint ki = 0; ki < 8; ++ki) {
                    float pv = exp(Smem[sgid][lane][ki] - m_new);  // masked → exp(-inf)=0
                    Smem[sgid][lane][ki] = pv;
                    rsum += pv;
                }
                for (uint c = 0; c < d; ++c) Omem[sgid][lane][c] *= corr;  // rescale O row
                l_row = l_row * corr + rsum;
                m_row = m_new;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O += P[8q,8k] · V[8k,8d]  per d-tile
            simdgroup_float8x8 Pm;
            simdgroup_load(Pm, &Smem[sgid][0][0], 8);
            for (uint dt = 0; dt < D_TILES; ++dt) {
                simdgroup_float8x8 Vm, PV = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_load(Vm, V + kb * d + dt * 8, d);
                simdgroup_multiply_accumulate(PV, Pm, Vm, PV);
                simdgroup_store(PV, &PVm_s[sgid][0][dt * 8], d);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 8 * d; e += 32) Omem[sgid][e / d][e % d] += PVm_s[sgid][e / d][e % d];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < 8) {
            const uint qi = qrow + lane;
            const float inv = 1.0f / l_row;
            for (uint c = 0; c < d; ++c) O[qi * d + c] = Omem[sgid][lane][c] * inv;
            args.L[h * S + qi] = m_row + log(l_row);
        }
    }
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

// MMA dQ. Same 1D launch as the forward (64 threads = NWARPS simdgroups, 8-query
// sub-blocks). No online softmax (L precomputed), so dQ accumulates straight into
// simdgroup matrices across key-blocks — no per-row rescale. Matmuls on the matrix
// units: S=Q·Kᵀ, dP=dO·Vᵀ (both MMA), then dQ += (scale·dS)·K (MMA). The softmax-
// derivative ds = p(dp−D) runs scalar in threadgroup memory (8 rows / lane<8).
kernel void flash_attention_bwd_dq(
    constant DqBindings&    args [[buffer(0)]],
    constant PushConstants& p    [[buffer(3)]],
    uint  wgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint d = p.d_head, S = p.seq_len;
    const float scale = 1.0f / sqrt((float)d);
    const uint gbase = wgid * TG_SIZE_X;
    const uint h = gbase / S, qb = gbase % S;
    const uint head_off = h * S * d;
    device const float* Q  = args.Q  + head_off;
    device const float* K  = args.K  + head_off;
    device const float* V  = args.V  + head_off;
    device const float* dO = args.dO + head_off;
    device       float* dQ = args.dQ + head_off;

    threadgroup float Smem[NWARPS][8][8];    // S, then dS·scale
    threadgroup float dPmem[NWARPS][8][8];

    for (uint r = 0; r < QSUB / NWARPS; ++r) {
        const uint sb = sgid + NWARPS * r;
        const uint qrow = qb + sb * 8;

        simdgroup_float8x8 Qm[D_TILES], dOm[D_TILES], dQm[D_TILES];
        for (uint dt = 0; dt < D_TILES; ++dt) {
            simdgroup_load(Qm[dt],  Q  + qrow * d + dt * 8, d);
            simdgroup_load(dOm[dt], dO + qrow * d + dt * 8, d);
            dQm[dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        float Li = 0.0f, Di = 0.0f;
        if (lane < 8) { Li = args.L[h * S + qrow + lane]; Di = args.D[h * S + qrow + lane]; }

        const uint kmax = qrow + 8;                 // causal: keys < kmax (8-aligned)
        for (uint kb = 0; kb < kmax; kb += 8) {
            simdgroup_float8x8 Sm  = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 dPm = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint dt = 0; dt < D_TILES; ++dt) {
                simdgroup_float8x8 KT, VT;
                simdgroup_load(KT, K + kb * d + dt * 8, d, ulong2(0, 0), true);
                simdgroup_load(VT, V + kb * d + dt * 8, d, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sm,  Qm[dt],  KT, Sm);
                simdgroup_multiply_accumulate(dPm, dOm[dt], VT, dPm);
            }
            simdgroup_store(Sm,  &Smem[sgid][0][0],  8);
            simdgroup_store(dPm, &dPmem[sgid][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (lane < 8) {
                const uint qi = qrow + lane;
                for (uint ki = 0; ki < 8; ++ki) {
                    float ds = 0.0f;
                    if (kb + ki <= qi) {            // causal j<=i
                        float pij = exp(Smem[sgid][lane][ki] * scale - Li);
                        ds = pij * (dPmem[sgid][lane][ki] - Di);
                    }
                    Smem[sgid][lane][ki] = scale * ds;   // dS·scale
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 dSm;
            simdgroup_load(dSm, &Smem[sgid][0][0], 8);
            for (uint dt = 0; dt < D_TILES; ++dt) {
                simdgroup_float8x8 Km;
                simdgroup_load(Km, K + kb * d + dt * 8, d);     // [8k,8d] (not transposed)
                simdgroup_multiply_accumulate(dQm[dt], dSm, Km, dQm[dt]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint dt = 0; dt < D_TILES; ++dt)
            simdgroup_store(dQm[dt], dQ + qrow * d + dt * 8, d);
    }
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

// MMA dK/dV. Same launch; each simdgroup owns an 8-KEY sub-block and loops query-
// blocks i>=j. dK/dV contract over QUERIES, so the accumulation matmuls need Pᵀ and
// dSᵀ — loaded transposed from threadgroup memory. dV += Pᵀ·dO, dK += (scale·dS)ᵀ·Q,
// both on the matrix units; S=Q·Kᵀ and dP=dO·Vᵀ likewise. dK/dV accumulate straight
// into simdgroup matrices across query-blocks (L precomputed → no rescale).
kernel void flash_attention_bwd_dkdv(
    constant DkvBindings&   args [[buffer(0)]],
    constant PushConstants& p    [[buffer(3)]],
    uint  wgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint d = p.d_head, S = p.seq_len;
    const float scale = 1.0f / sqrt((float)d);
    const uint gbase = wgid * TG_SIZE_X;
    const uint h = gbase / S, kb0 = gbase % S;
    const uint head_off = h * S * d;
    device const float* Q  = args.Q  + head_off;
    device const float* K  = args.K  + head_off;
    device const float* V  = args.V  + head_off;
    device const float* dO = args.dO + head_off;
    device       float* dK = args.dK + head_off;
    device       float* dV = args.dV + head_off;

    threadgroup float Smem[NWARPS][8][8];    // S, then P
    threadgroup float dPmem[NWARPS][8][8];   // dP, then dS·scale

    for (uint r = 0; r < QSUB / NWARPS; ++r) {
        const uint sb = sgid + NWARPS * r;
        const uint jrow = kb0 + sb * 8;            // first key of this 8-key sub-block

        simdgroup_float8x8 dKm[D_TILES], dVm[D_TILES];
        for (uint dt = 0; dt < D_TILES; ++dt) {
            dKm[dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            dVm[dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }

        for (uint qb2 = jrow; qb2 < S; qb2 += 8) {   // queries i>=jrow (causal)
            simdgroup_float8x8 Sm  = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 dPm = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint dt = 0; dt < D_TILES; ++dt) {
                simdgroup_float8x8 Qm, KT, dOm, VT;
                simdgroup_load(Qm, Q + qb2 * d + dt * 8, d);
                simdgroup_load(KT, K + jrow * d + dt * 8, d, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sm, Qm, KT, Sm);
                simdgroup_load(dOm, dO + qb2 * d + dt * 8, d);
                simdgroup_load(VT, V + jrow * d + dt * 8, d, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(dPm, dOm, VT, dPm);
            }
            simdgroup_store(Sm,  &Smem[sgid][0][0],  8);
            simdgroup_store(dPm, &dPmem[sgid][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (lane < 8) {
                const uint qi = qb2 + lane;            // query row
                const float Lq = args.L[h * S + qi];
                const float Dq = args.D[h * S + qi];
                for (uint ki = 0; ki < 8; ++ki) {
                    float pij = 0.0f, ds = 0.0f;
                    if (qi >= jrow + ki) {             // causal i>=j
                        pij = exp(Smem[sgid][lane][ki] * scale - Lq);
                        ds = pij * (dPmem[sgid][lane][ki] - Dq);
                    }
                    Smem[sgid][lane][ki]  = pij;        // P
                    dPmem[sgid][lane][ki] = scale * ds; // dS·scale
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 Pt, dSt;
            simdgroup_load(Pt,  &Smem[sgid][0][0],  8, ulong2(0, 0), true);  // [8k,8q]
            simdgroup_load(dSt, &dPmem[sgid][0][0], 8, ulong2(0, 0), true);  // [8k,8q]
            for (uint dt = 0; dt < D_TILES; ++dt) {
                simdgroup_float8x8 dOm2, Qm2;
                simdgroup_load(dOm2, dO + qb2 * d + dt * 8, d);
                simdgroup_multiply_accumulate(dVm[dt], Pt, dOm2, dVm[dt]);
                simdgroup_load(Qm2, Q + qb2 * d + dt * 8, d);
                simdgroup_multiply_accumulate(dKm[dt], dSt, Qm2, dKm[dt]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint dt = 0; dt < D_TILES; ++dt) {
            simdgroup_store(dKm[dt], dK + jrow * d + dt * 8, d);
            simdgroup_store(dVm[dt], dV + jrow * d + dt * 8, d);
        }
    }
}
