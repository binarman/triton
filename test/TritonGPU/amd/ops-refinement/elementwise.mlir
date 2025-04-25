// RUN: triton-opt %s -split-input-file -triton-amdgpu-refine-ops='arch=gfx942' | FileCheck %s

#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @mma_mul_kernel
  tt.func public @mma_mul_kernel(%arg0: tensor<256x32xf32, #mma>, %arg1: tensor<256x32xf32, #mma>) -> tensor<256x32xf32, #mma> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = arith.mulf %arg0, %arg1 : tensor<256x32xf32, #mma>
    tt.return %0 : tensor<256x32xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @exp_kernel
  tt.func public @exp_kernel(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @multiple_hints_kernel
  tt.func public @multiple_hints_kernel(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<128x32xf32, #blocked>
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}

// -----

// blocked layout cta tile has size of whole tensor, no transformation should happen
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @mma_blocked_convert_kernel_neg
  tt.func public @mma_blocked_convert_kernel_neg(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf32, #mma> -> tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}

// -----

#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#blocked = #ttg.blocked<{sizePerThread = [4, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @mma_blocked_convert_kernel
  tt.func public @mma_blocked_convert_kernel(%arg0: tensor<128x32xf32, #mma>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf32, #mma> -> tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}

// -----

#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @mma_mul_kernel
  tt.func public @multiple_operations_kernel(%arg0: tensor<128x32xf32, #mma>, %arg1: tensor<128x32xf32, #mma>) -> tensor<128x32xf32, #mma> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<128x32xf32, #mma>
    %1 = math.exp2 %0 : tensor<128x32xf32, #mma>
    %2 = math.exp2 %1 : tensor<128x32xf32, #mma>
    tt.return %2 : tensor<128x32xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: @nested_operations_kernel
  tt.func public @nested_operations_kernel(%arg0: tensor<128x32xf32, #blocked>, %arg1: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = arith.mulf %arg0, %arg1 : tensor<128x32xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<128x32xf32, #blocked>) : i32 {
      amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
      %2 = math.exp2 %0 : tensor<128x32xf32, #blocked>
      scf.yield %2 : tensor<128x32xf32, #blocked>
    }
    tt.return %1 : tensor<128x32xf32, #blocked>
  }
}
