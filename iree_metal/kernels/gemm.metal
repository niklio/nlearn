// gemm.metal — Metal GEMM using Apple's matrix coprocessor (simdgroup_matrix).
//
// IREE's metal-spirv codegen emits naive vectorized-FMA matmuls (~0.8 TFLOPS on
// M4) because (a) its Apple target advertises no MMA ops and (b) spirv-cross
// can't translate cooperative-matrix SPIR-V to MSL simdgroup_matrix. So we hand-
// write the GEMM with simdgroup_matrix and bind it via the custom-dispatch pass.
//
// C[M,N] = A[M,K] @ B[K,N], f16 in/out, f32 accumulate. Requires M,N,K multiples
// of 8 (the simdgroup_matrix tile size). IREE Metal ABI: bindings in the
// argument buffer at [[buffer(0)]], push constants at [[buffer(3)]].

#include <metal_stdlib>
using namespace metal;

struct GemmBindings {
  device const half*  A [[id(0)]];  // M x K, row-major
  device const half*  B [[id(1)]];  // K x N, row-major
  device       float* C [[id(2)]];  // M x N, row-major (f32; simdgroup_store
                                    // requires the dest type to match the f32
                                    // accumulator — cast to f16 at the JAX edge)
};

struct GemmParams { uint M; uint N; uint K; };

// One simdgroup (32 threads) per threadgroup computes a BM x BN block of C as
// TM x TN 8x8 tiles. Each K-step stages a BM x BK slab of A and a BK x BN slab
// of B into threadgroup memory (loaded cooperatively by all 32 threads), then
// the matrix units consume them with full reuse — so global-memory traffic drops
// by ~TN/TM× and the kernel becomes compute-bound on the matrix coprocessor.
// CORRECTNESS LIMIT (measured on M4): with ONE simdgroup per threadgroup, the
// accumulator array acc[TM][TN] must fit in simdgroup registers. TM=TN=2 (4
// accumulators) is exact; TM=TN=4 (16) silently corrupts (register overflow).
// PERF: at 2×2 this is ~0.94 TFLOPS (vs ~0.8 naive) — correct but not the win.
// To approach the M4's ~teens-TFLOPS peak, the next step is a MULTI-simdgroup
// GEMM: N simdgroups per threadgroup share one large staged BMxBK / BKxBN tile,
// each computing a small sub-tile (≤2×2) within its register budget. See
// kernels/GEMM.md.
#define BM 16
#define BN 16
#define BK 8
#define TM (BM / 8)   // 8x8 output tiles down M
#define TN (BN / 8)   // 8x8 output tiles across N

kernel void gemm_sg(
    constant GemmBindings& args [[buffer(0)]],
    constant GemmParams&   p    [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup half As[BM][BK];
    threadgroup half Bs[BK][BN];

    const uint row0 = tgid.y * BM;
    const uint col0 = tgid.x * BN;

    // NOTE: loops over the simdgroup_matrix arrays MUST be fully unrolled — these
    // matrices live in registers and Metal can't dynamically index them; without
    // unrolling the multi-tile accumulation silently produces wrong results.
    simdgroup_float8x8 acc[TM][TN];
#pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0; j < TN; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        // Cooperatively stage A (BM*BK) and B (BK*BN) into threadgroup memory.
        for (uint e = tid; e < BM * BK; e += 32) {
            uint r = e / BK, c = e % BK;
            As[r][c] = args.A[(row0 + r) * p.K + (k0 + c)];
        }
        for (uint e = tid; e < BK * BN; e += 32) {
            uint r = e / BN, c = e % BN;
            Bs[r][c] = args.B[(k0 + r) * p.N + (col0 + c)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 a[TM], b[TN];
#pragma clang loop unroll(full)
        for (uint i = 0; i < TM; ++i)
            simdgroup_load(a[i], &As[i * 8][0], BK);
#pragma clang loop unroll(full)
        for (uint j = 0; j < TN; ++j)
            simdgroup_load(b[j], &Bs[0][j * 8], BN);
#pragma clang loop unroll(full)
        for (uint i = 0; i < TM; ++i)
#pragma clang loop unroll(full)
            for (uint j = 0; j < TN; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0; j < TN; ++j)
            simdgroup_store(acc[i][j], args.C + (row0 + i * 8) * p.N + (col0 + j * 8), p.N);
}
