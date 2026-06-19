// gemm.metal — Metal GEMM using Apple's matrix coprocessor (simdgroup_matrix).
//
// IREE's metal-spirv codegen emits naive vectorized-FMA matmuls (~0.8 TFLOPS on
// M4) because (a) its Apple target advertises no MMA ops and (b) spirv-cross
// can't translate cooperative-matrix SPIR-V to MSL simdgroup_matrix. So we hand-
// write the GEMM with simdgroup_matrix and bind it via the custom-dispatch pass.
// See kernels/GEMM.md.
//
// C[M,N] = A[M,K] @ B[K,N], f16 in/out f32 accumulate. Requires M,N,K multiples
// of the block (BM/BN). IREE Metal ABI: bindings in the argument buffer at
// [[buffer(0)]], push constants at [[buffer(3)]].
//
// Multi-simdgroup design: a threadgroup of SG_M*SG_N simdgroups computes a
// BM x BN tile of C, sharing one BM x BK / BK x BN slab staged in threadgroup
// memory each K-step. Each simdgroup owns a WM x WN sub-tile of TM x TN 8x8
// tiles — kept at <=2x2 so its accumulators fit in simdgroup registers (4x4
// silently overflows). Sharing the staged slab across simdgroups gives the reuse
// + occupancy a single simdgroup can't.

#include <metal_stdlib>
using namespace metal;

// bf16 inputs (was fp16): bf16 has f32-like exponent range, which is what actually
// caps training loss on this stack (fp16 overflows as activations grow — verified via
// CPU A/B). bf16 has fewer mantissa bits than fp16 but the matmul accumulates in f32,
// so precision is unaffected. Uses native MSL `bfloat` + simdgroup_bfloat8x8 (bf16 MMA,
// f32 accumulate) — requires the runtime to compile at MSL language version 3.1, which
// IREE's Metal HAL (executable.m) now sets. ~2x the f32-compute fallback.
struct GemmBindings {
  device const bfloat* A [[id(0)]];  // M x K, row-major
  device const bfloat* B [[id(1)]];  // K x N, row-major
  device       float*  C [[id(2)]];  // M x N, row-major (f32 accumulate output)
};

struct GemmParams { uint M; uint N; uint K; };

#define SG_M 2            // simdgroups down M
#define SG_N 2            // simdgroups across N
#define WM   16           // rows per simdgroup
#define WN   16           // cols per simdgroup
#define BM   (SG_M * WM)  // 32
#define BN   (SG_N * WN)  // 32
#define BK   64           // K-slab depth (swept: 64 best; 128+ loses occupancy)
#define TM   (WM / 8)     // 2 — 8x8 tiles per simdgroup down M
#define TN   (WN / 8)     // 2 — across N
#define NTHREADS (SG_M * SG_N * 32)   // 128

kernel void gemm_sg(
    constant GemmBindings& args [[buffer(0)]],
    constant GemmParams&   p    [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup bfloat As[BM][BK];
    threadgroup bfloat Bs[BK][BN];

    const uint row0 = tgid.y * BM;             // block origin in C
    const uint col0 = tgid.x * BN;
    const uint sr = (sg / SG_N) * WM;          // this simdgroup's sub-tile offset
    const uint sc = (sg % SG_N) * WN;

    simdgroup_float8x8 acc[TM][TN];
#pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0; j < TN; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        // Vectorized cooperative staging: each thread moves a bfloat4 (rows are
        // contiguous in both global and the slab), so 1/4 the load instrs and no
        // per-element div/mod. BK and BN are multiples of 4.
        threadgroup bfloat4* As4 = (threadgroup bfloat4*)As;
        threadgroup bfloat4* Bs4 = (threadgroup bfloat4*)Bs;
        device const bfloat4* A4 = (device const bfloat4*)args.A;
        device const bfloat4* B4 = (device const bfloat4*)args.B;
        const uint Kq = p.K >> 2, Nq = p.N >> 2, BKq = BK >> 2, BNq = BN >> 2;
        for (uint e = tid; e < BM * BKq; e += NTHREADS) {
            uint r = e / BKq, c = e % BKq;
            As4[r * BKq + c] = A4[(row0 + r) * Kq + (k0 >> 2) + c];
        }
        for (uint e = tid; e < BK * BNq; e += NTHREADS) {
            uint r = e / BNq, c = e % BNq;
            Bs4[r * BNq + c] = B4[(k0 + r) * Nq + (col0 >> 2) + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Consume the staged slab in 8-deep K steps (BK/8 of them per barrier).
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_bfloat8x8 a[TM], b[TN];
#pragma clang loop unroll(full)
            for (uint i = 0; i < TM; ++i)
                simdgroup_load(a[i], &As[sr + i * 8][kk], BK);
#pragma clang loop unroll(full)
            for (uint j = 0; j < TN; ++j)
                simdgroup_load(b[j], &Bs[kk][sc + j * 8], BN);
#pragma clang loop unroll(full)
            for (uint i = 0; i < TM; ++i)
#pragma clang loop unroll(full)
                for (uint j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0; j < TN; ++j)
            simdgroup_store(acc[i][j],
                args.C + (row0 + sr + i * 8) * p.N + (col0 + sc + j * 8), p.N);
}
