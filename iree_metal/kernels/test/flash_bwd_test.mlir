// Forward (->O,L) + backward (dQ, dKdV) external Metal kernels. n=2,s=16,d=8.
// tensors 256xf32; L,D 32xf32 (n*s).

#mt = #hal.executable.target<"metal-spirv", "metal-msl-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32],
    max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
    max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>
}>
#dev = #hal.device.target<"metal", [#mt]> : !hal.device

module @fa attributes {hal.device.targets = [#dev]} {
  hal.executable.source private @fwd attributes {
    objects = #hal.executable.objects<{#mt = [#hal.executable.object<{path = "flash_attention.metal"}>]}>
  } {
    hal.executable.export public @flash_attention_fwd ordinal(0)
        layout(#hal.pipeline.layout<constants = 3, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>,
          #hal.pipeline.binding<storage_buffer>]>)
        count(%d: !hal.device, %w: index) -> (index, index, index) {
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%w]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [64 : index, 1 : index, 1 : index]}
  }
  hal.executable.source private @dq attributes {
    objects = #hal.executable.objects<{#mt = [#hal.executable.object<{path = "flash_attention.metal"}>]}>
  } {
    hal.executable.export public @flash_attention_bwd_dq ordinal(0)
        layout(#hal.pipeline.layout<constants = 3, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer>]>)
        count(%d: !hal.device, %w: index) -> (index, index, index) {
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%w]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [64 : index, 1 : index, 1 : index]}
  }
  hal.executable.source private @dkdv attributes {
    objects = #hal.executable.objects<{#mt = [#hal.executable.object<{path = "flash_attention.metal"}>]}>
  } {
    hal.executable.export public @flash_attention_bwd_dkdv ordinal(0)
        layout(#hal.pipeline.layout<constants = 3, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
        count(%d: !hal.device, %w: index) -> (index, index, index) {
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%w]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [64 : index, 1 : index, 1 : index]}
  }

  func.func @run_fwd(%Q: tensor<256xf32>, %K: tensor<256xf32>, %V: tensor<256xf32>) -> (tensor<256xf32>, tensor<32xf32>) {
    %w = arith.constant 32 : index
    %n = arith.constant 2 : i32
    %s = arith.constant 16 : i32
    %d = arith.constant 8 : i32
    %O, %L = flow.dispatch @fwd::@flash_attention_fwd[%w](%n, %s, %d, %Q, %K, %V)
        : (i32, i32, i32, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<256xf32>, tensor<32xf32>)
    return %O, %L : tensor<256xf32>, tensor<32xf32>
  }
  func.func @run_dq(%Q: tensor<256xf32>, %K: tensor<256xf32>, %V: tensor<256xf32>,
                %dO: tensor<256xf32>, %L: tensor<32xf32>, %D: tensor<32xf32>) -> tensor<256xf32> {
    %w = arith.constant 32 : index
    %n = arith.constant 2 : i32
    %s = arith.constant 16 : i32
    %d = arith.constant 8 : i32
    %dQ = flow.dispatch @dq::@flash_attention_bwd_dq[%w](%n, %s, %d, %Q, %K, %V, %dO, %L, %D)
        : (i32, i32, i32, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<256xf32>
    return %dQ : tensor<256xf32>
  }
  func.func @run_dkdv(%Q: tensor<256xf32>, %K: tensor<256xf32>, %V: tensor<256xf32>,
                  %dO: tensor<256xf32>, %L: tensor<32xf32>, %D: tensor<32xf32>) -> (tensor<256xf32>, tensor<256xf32>) {
    %w = arith.constant 32 : index
    %n = arith.constant 2 : i32
    %s = arith.constant 16 : i32
    %d = arith.constant 8 : i32
    %dK, %dV = flow.dispatch @dkdv::@flash_attention_bwd_dkdv[%w](%n, %s, %d, %Q, %K, %V, %dO, %L, %D)
        : (i32, i32, i32, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32xf32>, tensor<32xf32>) -> (tensor<256xf32>, tensor<256xf32>)
    return %dK, %dV : tensor<256xf32>, tensor<256xf32>
  }
}
