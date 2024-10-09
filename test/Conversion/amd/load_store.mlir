// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    tt.func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // Load 8 elements from A with two vectorized load instruction
    // CHECK-COUNT-2: llvm.intr.masked.load {{.*}} : (!llvm.ptr, vector<4xi1>, vector<4xf32>) -> vector<4xf32>
    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #blocked0>
    // Load 8 elements from B with two vectorized load instruction
    // CHECK-COUNT-2: llvm.intr.masked.load {{.*}} : (!llvm.ptr, vector<4xi1>, vector<4xf32>) -> vector<4xf32>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>

// -----

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
