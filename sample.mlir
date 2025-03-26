#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<f8E4M3FNUZ> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant dense<32> : tensor<16x1xi32, #blocked>

    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %3 = tt.expand_dims %1 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>

    %4 = arith.muli %3, %c32_i32 : tensor<16x1xi32, #blocked>

    %5 = tt.broadcast %2 : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>

    %7 = arith.addi %5, %6 : tensor<16x32xi32, #blocked>

    %8 = tt.splat %arg0 : !tt.ptr<f8E4M3FNUZ> -> tensor<16x32x!tt.ptr<f8E4M3FNUZ>, #blocked>

    %9 = tt.addptr %8, %7 : tensor<16x32x!tt.ptr<f8E4M3FNUZ>, #blocked>, tensor<16x32xi32, #blocked>

    %10 = tt.load %9 : tensor<16x32x!tt.ptr<f8E4M3FNUZ>, #blocked>
    %11 = ttg.local_alloc %10 : (tensor<16x32xf8E4M3FNUZ, #blocked>) -> !ttg.memdesc<16x32xf8E4M3FNUZ, #shared, #smem>
    %12 = ttg.local_load %11 : !ttg.memdesc<16x32xf8E4M3FNUZ, #shared, #smem> -> tensor<16x32xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

    %13 = tt.fp_to_fp %12 : tensor<16x32xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %7 : tensor<16x32x!tt.ptr<f32>, #blocked>, tensor<16x32xi32, #blocked>

    %16 = ttg.convert_layout %15 : tensor<16x32x!tt.ptr<f32>, #blocked> -> tensor<16x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

    tt.store %16, %13 : tensor<16x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    tt.return
  }
}
