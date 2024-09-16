#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @streamk_gemm_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<128> : tensor<128xi32, #blocked1>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %1 = arith.addi %0, %cst_0 : tensor<128xi32, #blocked1>
    %2 = scf.for %arg14 = %arg8 to %c1_i32 step %c1_i32 iter_args(%arg15 = %0) -> (tensor<128xi32, #blocked1>)  : i32 {
      scf.yield %1 : tensor<128xi32, #blocked1>
    }
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = arith.extsi %4 : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
    %6 = triton_gpu.convert_layout %2 : tensor<128xi32, #blocked1> -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %5 : tensor<128x1xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %9 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %10 = arith.extsi %9 : tensor<128x128xi32, #blocked> to tensor<128x128xi64, #blocked>
    %11 = arith.addi %10, %8 : tensor<128x128xi64, #blocked>
    %12 = tt.splat %arg8 : i32 -> tensor<128xi32, #blocked1>
    %13 = arith.cmpi slt, %2, %12 : tensor<128xi32, #blocked1>
    %14 = triton_gpu.convert_layout %13 : tensor<128xi1, #blocked1> -> tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.expand_dims %14 {axis = 0 : i32} : tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi1, #blocked>
    %16 = tt.broadcast %15 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %17 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    %18 = tt.addptr %17, %11 : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi64, #blocked>
    tt.store %18, %cst, %16 cacheModifier = wt : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
