// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

// CHECK-LABEL: store_folded_mfma32
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
 tt.func public @store_folded_mfma32(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride : i32 {tt.divisibility = 16 : i32}, %data: tensor<16x16xf32, #mma>) attributes {noinline = false} {

    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x16xi32, #mma>
    %2 = tt.broadcast %1 : tensor<1x16xi32, #mma> -> tensor<16x16xi32, #mma>

    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<16x1xi32, #mma>
    %str = tt.splat %stride : i32 -> tensor<16x1xi32, #mma>
    %5 = arith.muli %4, %str : tensor<16x1xi32, #mma>
    %6 = tt.broadcast %5 : tensor<16x1xi32, #mma> -> tensor<16x16xi32, #mma>

    %7 = arith.addi %2, %6 : tensor<16x16xi32, #mma>

    %8 = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #mma>
    %9 = tt.addptr %8, %6 : tensor<16x16x!tt.ptr<f32>, #mma>, tensor<16x16xi32, #mma>

    tt.store %9, %data : tensor<16x16x!tt.ptr<f32>, #mma>
    tt.return
  }
}
