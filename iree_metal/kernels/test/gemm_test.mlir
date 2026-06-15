#mt = #hal.executable.target<"metal-spirv", "metal-msl-fb">
#dev = #hal.device.target<"metal", [#hal.executable.target<"metal-spirv", "metal-msl-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
    compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic, subgroup_size_choices = [32],
    max_workgroup_sizes = [1024,1024,1024], max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 32768, max_workgroup_counts = [65535,65535,65535]>>}>]> : !hal.device
module @g attributes {hal.device.targets = [#dev]} {
  hal.executable.source private @gemm attributes {
    objects = #hal.executable.objects<{#mt = [#hal.executable.object<{path = "gemm.metal"}>]}>
  } {
    hal.executable.export public @gemm_sg ordinal(0)
        layout(#hal.pipeline.layout<constants = 3, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer>]>)
        count(%d: !hal.device, %wn: index, %wm: index) -> (index, index, index) {
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%wn]
      %y = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%wm]
      %c1 = arith.constant 1 : index
      hal.return %x, %y, %c1 : index, index, index
    } attributes {workgroup_size = [32 : index, 1 : index, 1 : index]}
  }
  func.func @run_gemm(%A: tensor<1024x1024xf16>, %B: tensor<1024x1024xf16>) -> tensor<1024x1024xf32> {
    %wn = arith.constant 1024 : index
    %wm = arith.constant 1024 : index
    %M = arith.constant 1024 : i32
    %N = arith.constant 1024 : i32
    %K = arith.constant 1024 : i32
    %C = flow.dispatch @gemm::@gemm_sg[%wn, %wm](%M, %N, %K, %A, %B)
        : (i32, i32, i32, tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf32>
    return %C : tensor<1024x1024xf32>
  }
}
