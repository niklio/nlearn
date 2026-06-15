// Tiled flash_attention_fwd test: seq_len % 64 == 0 (head-aligned 64-query
// blocks) and d_head = 64, matching the model. 5 bindings (Q,K,V,O,L); returns
// both O and the logsumexp L. n=2 heads, s=128, d=64 -> 16384 elems, L = 256.

#metal_target = #hal.executable.target<"metal-spirv", "metal-msl-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32],
    max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
    max_workgroup_memory_bytes = 32768, max_workgroup_counts = [65535, 65535, 65535]>>
}>
#metal_device = #hal.device.target<"metal", [#metal_target]> : !hal.device

module @flash attributes {hal.device.targets = [#metal_device]} {
  hal.executable.source private @flash_attention attributes {
    objects = #hal.executable.objects<{
      #metal_target = [#hal.executable.object<{path = "flash_attention.metal"}>]
    }>
  } {
    hal.executable.export public @flash_attention_fwd ordinal(0)
        layout(#hal.pipeline.layout<constants = 3, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer>,
          #hal.pipeline.binding<storage_buffer>
        ]>)
        count(%device: !hal.device, %workload: index) -> (index, index, index) {
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [64 : index, 1 : index, 1 : index]}
  }

  func.func @flash(%Q: tensor<16384xf32>, %K: tensor<16384xf32>, %V: tensor<16384xf32>)
      -> (tensor<16384xf32>, tensor<256xf32>) {
    %workload = arith.constant 256 : index
    %n = arith.constant 2 : i32
    %s = arith.constant 128 : i32
    %d = arith.constant 64 : i32
    %O, %L = flow.dispatch @flash_attention::@flash_attention_fwd[%workload](%n, %s, %d, %Q, %K, %V)
        : (i32, i32, i32, tensor<16384xf32>, tensor<16384xf32>, tensor<16384xf32>)
          -> (tensor<16384xf32>, tensor<256xf32>)
    return %O, %L : tensor<16384xf32>, tensor<256xf32>
  }
}
