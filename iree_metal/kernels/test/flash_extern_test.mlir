#metal_target = #hal.executable.target<"metal-spirv", "metal-msl-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32],
    max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
    max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>
}>
#metal_device = #hal.device.target<"metal", [#metal_target]> : !hal.device
module @flashx attributes {hal.device.targets = [#metal_device]} {
  func.func @flash(%Q: tensor<256xf32>, %K: tensor<256xf32>, %V: tensor<256xf32>) -> tensor<256xf32> {
    %workload = arith.constant 32 : index
    %n = arith.constant 2 : i32
    %s = arith.constant 16 : i32
    %d = arith.constant 8 : i32
    %0 = hal.dispatch.extern "flash_attention_fwd"[%workload](%n, %s, %d, %Q, %K, %V)
        : (i32, i32, i32, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      count(%device: !hal.device, %wl: index) -> (index, index, index) {
        %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%wl]
        %c1 = arith.constant 1 : index
        hal.return %x, %c1, %c1 : index, index, index
      }
      layout(#hal.pipeline.layout<constants = 3, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>
      ]>)
      objects({#metal_target ordinal(0) = [#hal.executable.object<{path = "flash_attention.metal"}>]})
    return %0 : tensor<256xf32>
  }
}
