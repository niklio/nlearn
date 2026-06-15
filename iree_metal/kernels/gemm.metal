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

struct GemmBindings {
  device const half*  A [[id(0)]];  // M x K, row-major
  device const half*  B [[id(1)]];  // K x N, row-major
  device       float* C [[id(2)]];  // M x N, row-major (f32 accumulate output)
};

struct GemmParams { uint M; uint N; uint K; };

#define SG_M 2            // simdgroups down M
#define SG_N 2            // simdgroups across N
#define WM   16           // rows per simdgroup
#define WN   16           // cols per simdgroup
#define BM   (SG_M * WM)  // 32
#define BN   (SG_N * WN)  // 32
#define BK   32           // K-slab depth: thicker = fewer barriers, more compute/stage
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
    threadgroup half As[BM][BK];
    threadgroup half Bs[BK][BN];

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
        // All NTHREADS cooperatively stage the shared A/B slabs.
        for (uint e = tid; e < BM * BK; e += NTHREADS) {
            uint r = e / BK, c = e % BK;
            As[r][c] = args.A[(row0 + r) * p.K + (k0 + c)];
        }
        for (uint e = tid; e < BK * BN; e += NTHREADS) {
            uint r = e / BN, c = e % BN;
            Bs[r][c] = args.B[(k0 + r) * p.N + (col0 + c)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Consume the staged slab in 8-deep K steps (BK/8 of them per barrier).
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 a[TM], b[TN];
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
