#include <metal_stdlib>
using namespace metal;

// Conforms to IREE's Metal dispatch ABI (spirv-cross argument_buffers=true):
//   bindings -> argument buffer struct at [[buffer(0)]], each at [[id(n)]]
//   push constants -> raw bytes at buffer index IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX (=3)
//   global id = thread_position_in_threadgroup + threadgroup_position_in_grid * threadgroup_size
//
// Pipeline layout: constants = 1 (dim), bindings = [RO a, RO b, RW out].

struct Bindings {
  device const float* a   [[id(0)]];
  device const float* b   [[id(1)]];
  device       float* out [[id(2)]];
};

struct PushConstants {
  uint dim;
};

kernel void spike_mul(
    constant Bindings&       args [[buffer(0)]],
    constant PushConstants&  pc   [[buffer(3)]],
    uint3 wgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
  uint gid = lid.x + wgid.x * 64u;   // threadgroup size x = 64 (see export workgroup_size)
  if (gid >= pc.dim) return;
  args.out[gid] = args.a[gid] * args.b[gid];
}
