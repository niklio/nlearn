// Trivial Metal custom-dispatch spike: out = a * b via a hand-authored MSL
// kernel embedded as an external metal-msl-fb object (exercises patch 04).
//
// Compile: iree-compile --iree-hal-target-device=metal \
//            --iree-hal-executable-object-search-path=/tmp/metalspike spike.mlir -o spike.vmfb
// Run:     iree-run-module --device=metal --module=spike.vmfb \
//            --function=mul --input=128xf32=2 --input=128xf32=4

#metal_target = #hal.executable.target<"metal-spirv", "metal-msl-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32],
    max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
    max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>
}>

#metal_device = #hal.device.target<"metal", [#metal_target]> : !hal.device

module @spike attributes {hal.device.targets = [#metal_device]} {

  hal.executable.source private @spike_mul attributes {
    objects = #hal.executable.objects<{
      #metal_target = [
        #hal.executable.object<{path = "spike_mul.metal"}>
      ]
    }>
  } {
    hal.executable.export public @spike_mul ordinal(0)
        layout(#hal.pipeline.layout<constants = 1, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer>
        ]>)
        count(%device: !hal.device, %workload: index) -> (index, index, index) {
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [64 : index, 1 : index, 1 : index]}
  }

  func.func @mul(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %c128 = arith.constant 128 : index
    %dim_i32 = arith.constant 128 : i32
    %0 = flow.dispatch @spike_mul::@spike_mul[%c128](%dim_i32, %arg0, %arg1)
        : (i32, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
}
