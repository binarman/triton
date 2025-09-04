// -----// IR Dump Before AllocateAMDGPUSharedMemory (allocate-amdgpu-shared-memory) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<256x4xi8, #blocked>
    %c1 = arith.constant 1 : i1
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    cf.br ^bb1(%buf: !ttg.memdesc<128x128xf32, #shared, #smem, mutable, 128x128>)

  ^bb1(%arg: !ttg.memdesc<128x128xf32, #shared, #smem, mutable, 128x128>):
    %data = ttg.local_load %arg : !ttg.memdesc<128x128xf32, #shared, #smem, mutable, 128x128> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    ttg.local_dealloc %arg : !ttg.memdesc<128x128xf32, #shared, #smem, mutable, 128x128>
    %1 = ttg.convert_layout %cst : tensor<256x4xi8, #blocked> -> tensor<256x4xi8, #linear1>
    %new_data = arith.constant dense<0.000> : tensor<128x128xf32, #blocked>
    %new_buf = ttg.local_alloc : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    ttg.local_store %new_data, %new_buf : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable, 128x128>
    cf.cond_br %c1, ^bb1(%new_buf: !ttg.memdesc<128x128xf32, #shared, #smem, mutable, 128x128>), ^bb2

  ^bb2:
    tt.return
  }
}
