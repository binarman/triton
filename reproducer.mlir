// -----// IR Dump Before ConvertTritonAMDGPUToLLVM (convert-triton-amdgpu-to-llvm) ('builtin.module' operation) //----- //
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [16, 1, 4], warpsPerCTA = [1, 1, 1], order = [2, 0, 1]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.shared = 33792 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @failure_kernel(%arg0: tensor<64x8x32xf16, #blocked>) attributes {noinline = false} {
    %0 = triton_gpu.local_alloc %arg0 {allocation.offset = 0 : i32} : (tensor<64x8x32xf16, #blocked>) -> !tt.memdesc<64x8x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %1 = triton_gpu.local_load %0 : !tt.memdesc<64x8x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<64x8x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
    tt.return
  }
}
