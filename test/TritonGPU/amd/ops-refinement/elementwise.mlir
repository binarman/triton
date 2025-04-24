// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints='variant=refine_ops' -triton-amdgpu-refine-ops='arch=gfx942' | FileCheck %s

#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @mma_mul_kernel
  tt.func public @mma_mul_kernel(%arg0: tensor<128x32xf32, #mma>, %arg1: tensor<128x32xf32, #mma>) -> tensor<128x32xf32, #mma> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = arith.mulf %arg0, %arg1 : tensor<128x32xf32, #mma>
    tt.return %0 : tensor<128x32xf32, #mma>
  }
}

// -----

#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @mma_blocked_convert_kernel
  tt.func public @mma_blocked_convert_kernel(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf32, #mma> -> tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}
