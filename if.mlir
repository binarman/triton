#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @streamk_gemm_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128xi32, #blocked1>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %1 = arith.cmpi slt, %arg8, %c32_i32 : i32
    %2 = scf.if %1 -> (tensor<128xi32, #blocked1>) {
      %24 = arith.addi %0, %cst_0 : tensor<128xi32, #blocked1>
      scf.yield %24 : tensor<128xi32, #blocked1>
    } else {
      scf.yield %0 : tensor<128xi32, #blocked1>
    }
    %3 = triton_gpu.convert_layout %0 : tensor<128xi32, #blocked1> -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %6 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %7 = tt.addptr %6, %4 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
    %8 = triton_gpu.convert_layout %2 : tensor<128xi32, #blocked1> -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %11 = tt.broadcast %7 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    %13 = tt.broadcast %9 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %14 = tt.addptr %11, %13 : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
    %15 = tt.splat %arg8 : i32 -> tensor<128xi32, #blocked1>
    %16 = arith.cmpi slt, %2, %15 : tensor<128xi32, #blocked1>
    %17 = triton_gpu.convert_layout %16 : tensor<128xi1, #blocked1> -> tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %18 = tt.expand_dims %17 {axis = 0 : i32} : tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi1, #blocked>
    %20 = tt.broadcast %18 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    tt.store %14, %cst, %20 cacheModifier = wt : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
