// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=4}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=4}>
module attributes {"triton_gpu.threads-per-warp" = 64 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  //  CHECK-LABEL: convert_dot
  tt.func @convert_dot(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = triton_gpu.local_alloc %A : (tensor<16x16xf16, #blocked0>) -> !tt.memdesc<16x16xf16, #shared0>
    %BB = triton_gpu.local_alloc %B : (tensor<16x16xf16, #blocked0>) -> !tt.memdesc<16x16xf16, #shared0>
    %AA_DOT = triton_gpu.local_load %AA : !tt.memdesc<16x16xf16, #shared0> -> tensor<16x16xf16, #dot_operand_a>
    %BB_DOT = triton_gpu.local_load %BB : !tt.memdesc<16x16xf16, #shared0> -> tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}
