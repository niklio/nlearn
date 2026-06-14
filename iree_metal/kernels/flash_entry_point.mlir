// Entry-point function the JAX binding lowers to: wraps the external
// flash_attention_fwd MSL kernel in a hal.dispatch.extern so the kernel author
// controls the ABI (operand order, push constants, workload) by hand.
//
// A preprocessing pass (see ../BINDING.md) matches `stablehlo.custom_call`
// with call_target_name = "flash_attention" and replaces it with a call to
// this function (cf. IREE's custom_dispatch transform sample, but matched by
// name in C++ since the transform dialect has no custom_call-name matcher).
//
// Shapes here are illustrative (n*s*d flattened f32); the real pass specializes
// per call from the custom_call's operand shapes / backend_config dims.

#metal_target = #hal.executable.target<"metal-spirv", "metal-msl-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32],
    max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
    max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>
}>

module attributes {transform.with_named_sequence} {
  // O = flash_attention(Q, K, V), all flattened (n_heads*seq_len*d_head) f32.
  func.func private @flash_attention_entry(
      %Q: tensor<256xf32>, %K: tensor<256xf32>, %V: tensor<256xf32>) -> tensor<256xf32> {
    %workload = arith.constant 32 : index   // n_heads*seq_len
    %n = arith.constant 2 : i32
    %s = arith.constant 16 : i32
    %d = arith.constant 8 : i32
    %0 = hal.dispatch.extern "flash_attention_fwd"[%workload](%n, %s, %d, %Q, %K, %V)
        : (i32, i32, i32, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      count(%device: !hal.device, %workload_c: index) -> (index, index, index) {
        %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload_c]
        %c1 = arith.constant 1 : index
        hal.return %x, %c1, %c1 : index, index, index
      }
      layout(#hal.pipeline.layout<constants = 3, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>
      ]>)
      objects({
        #metal_target ordinal(0) = [
          #hal.executable.object<{path = "flash_attention.metal"}>
        ]
      })
    return %0 : tensor<256xf32>
  }
}
