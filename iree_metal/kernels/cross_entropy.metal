// cross_entropy.metal — fused softmax cross-entropy (loss + dlogits) in MSL.
//
// Replaces the one-hot CE the model used (logsumexp + one_hot(M,V) + multiply +
// sum), which IREE codegens as ~6 memory-bound passes over the (M,V) logits AND
// materialises a full (M,V) one-hot — ~444ms fwd+bwd at 3.7 GB/s, 2.4GB peak.
// Here each ROW m is owned by one threadgroup that streams the vocab once:
//   fwd: online (max, sumexp) reduction -> logsumexp; loss = lse - logits[target].
//   bwd: dlogits[v] = (softmax(v) - [v==target]) / M     (streamed, no one-hot).
// No (M,V) one-hot is ever materialised -> the long-context memory enabler.
//
// IREE Metal dispatch ABI: bindings in the argument buffer at [[buffer(0)]],
// push constants at [[buffer(3)]]. Grid = M threadgroups (one per row).

#include <metal_stdlib>
using namespace metal;

#define CE_TG 256u   // threads per row; must match each export's workgroup_size

struct CeParams { uint M; uint V; };

// Combine two (max, sumexp-relative-to-max) partials into one.
static inline void combine(thread float& m, thread float& s, float m2, float s2) {
  float mn = max(m, m2);
  s = s * exp(m - mn) + s2 * exp(m2 - mn);
  m = mn;
}

// ---------------------------------------------------------------------------
// Forward: loss[m] = logsumexp(logits[m]) - logits[m, target[m]]; also emit lse.
// ---------------------------------------------------------------------------
struct CeFwdBindings {
  device const float* logits  [[id(0)]];  // M x V, row-major
  device const int*   targets [[id(1)]];  // M
  device       float* loss    [[id(2)]];  // M
  device       float* lse     [[id(3)]];  // M (logsumexp per row, saved for bwd)
};

kernel void ce_fwd(
    constant CeFwdBindings& a [[buffer(0)]],
    constant CeParams&      p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
  if (row >= p.M) return;
  device const float* lrow = a.logits + (ulong)row * p.V;

  // Each thread streams its strided slice of the vocab as an online (max, sumexp).
  float m = -INFINITY, s = 0.0f;
  for (uint v = tid; v < p.V; v += CE_TG) combine(m, s, lrow[v], 1.0f);

  // Threadgroup reduction of the per-thread (max, sumexp) pairs.
  threadgroup float tm[CE_TG], ts[CE_TG];
  tm[tid] = m; ts[tid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = CE_TG >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float mm = tm[tid], ss = ts[tid];
      combine(mm, ss, tm[tid + stride], ts[tid + stride]);
      tm[tid] = mm; ts[tid] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float lse_ = tm[0] + log(ts[0]);
    a.lse[row]  = lse_;
    a.loss[row] = lse_ - lrow[a.targets[row]];
  }
}

// ---------------------------------------------------------------------------
// Backward: dlogits[m,v] = (softmax(logits[m])[v] - [v==target[m]]) / M.
// (mean over rows folded in as 1/M; the scalar cotangent is applied in JAX.)
// ---------------------------------------------------------------------------
struct CeBwdBindings {
  device const float* logits  [[id(0)]];  // M x V
  device const int*   targets [[id(1)]];  // M
  device const float* lse     [[id(2)]];  // M
  device       float* dlogits [[id(3)]];  // M x V
};

kernel void ce_bwd(
    constant CeBwdBindings& a [[buffer(0)]],
    constant CeParams&      p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
  if (row >= p.M) return;
  device const float* lrow = a.logits + (ulong)row * p.V;
  device       float* drow = a.dlogits + (ulong)row * p.V;
  const float lse_ = a.lse[row];
  const uint  tgt  = (uint)a.targets[row];
  const float invM = 1.0f / (float)p.M;
  for (uint v = tid; v < p.V; v += CE_TG) {
    float sm = exp(lrow[v] - lse_);              // softmax(v)
    drow[v] = (sm - (v == tgt ? 1.0f : 0.0f)) * invM;
  }
}
