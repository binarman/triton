// RUN: triton-opt %s -split-input-file -tritonamdgpu-simplify-convert-layout | FileCheck %s

// CHECK-LABEL: convert

// CHECK: [[NEW_LAYOUT_PTR:%.*]] = ttg.convert_layout
// CHECK: [[NEW_LAYOUT_DATA:%.*]] = tt.load [[NEW_LAYOUT_PTR]]
// CHECK: [[FINAL_DATA:%.*]] = ttg.convert_layout [[NEW_LAYOUT_DATA]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @convert(%ptr: tensor<256x4x!tt.ptr<i8>, #blocked>) {
    %load = tt.load %ptr : tensor<256x4x!tt.ptr<i8>, #blocked>
    %converted = ttg.convert_layout %load : tensor<256x4xi8, #blocked> -> tensor<256x4xi8, #linear>
    tt.return
  }
}

// -----

// CHECK-LABEL: already_intrawarp

// CHECK-NOT: ttg.convert_layout
// CHECK: tt.load
// CHECK: ttg.convert_layout

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [128, 0], [0, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @already_intrawarp(%ptr: tensor<256x4x!tt.ptr<i8>, #blocked>) {
    %load = tt.load %ptr : tensor<256x4x!tt.ptr<i8>, #blocked>
    %converted = ttg.convert_layout %load : tensor<256x4xi8, #blocked> -> tensor<256x4xi8, #linear>
    tt.return
  }
}
