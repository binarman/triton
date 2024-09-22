// -----// IR Dump Before ConvertTritonAMDGPUToLLVM (convert-triton-amdgpu-to-llvm) ('builtin.module' operation) //----- //
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [8, 1, 8], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [16, 1, 4], warpsPerCTA = [1, 1, 1], order = [2, 0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 64, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [64, 1, 1], warpsPerCTA = [1, 1, 1], order = [2, 0, 1]}>
#loc = loc("/triton/./test_kernel.py":64:0)
#loc1 = loc(unknown)
#loc16 = loc("/triton/./test_kernel.py":88:18)
#loc17 = loc("/triton/./test_kernel.py":81:25)
#loc24 = loc("/triton/./test_kernel.py":89:17)
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0], hasLeadingOffset = false}>
#loc32 = loc(callsite(#loc1 at #loc24))
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.shared = 33792 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @failure_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/triton/./test_kernel.py":64:0), %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/triton/./test_kernel.py":64:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/triton/./test_kernel.py":64:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/triton/./test_kernel.py":64:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/triton/./test_kernel.py":64:0), %arg5: i32 {tt.divisibility = 16 : i32} loc("/triton/./test_kernel.py":64:0), %arg6: i32 {tt.divisibility = 16 : i32} loc("/triton/./test_kernel.py":64:0)) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x1x32xf32, #blocked> loc(#loc1)
    %c511_i32 = arith.constant 511 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c512_i32 = arith.constant 512 : i32 loc(#loc1)
    %cst_0 = arith.constant dense<512> : tensor<64x1x1xi32, #blocked1> loc(#loc1)
    %cst_1 = arith.constant dense<64> : tensor<64x1x1xi32, #blocked1> loc(#loc1)
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc2)
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<512xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x512xi32, #blocked2> loc(#loc3)
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.slice<{dim = 2, parent = #blocked1}>}>> loc(#loc4)
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<64x1xi32, #triton_gpu.slice<{dim = 2, parent = #blocked1}>> loc(#loc4)
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<64x1xi32, #triton_gpu.slice<{dim = 2, parent = #blocked1}>> -> tensor<64x1x1xi32, #blocked1> loc(#loc4)
    %5 = arith.muli %4, %cst_0 : tensor<64x1x1xi32, #blocked1> loc(#loc5)
    %6 = arith.divsi %5, %cst_1 : tensor<64x1x1xi32, #blocked1> loc(#loc6)
    %7 = tt.splat %arg4 : i32 -> tensor<64x1x1xi32, #blocked1> loc(#loc7)
    %8 = arith.muli %6, %7 : tensor<64x1x1xi32, #blocked1> loc(#loc7)
    %9 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked1}>}>> loc(#loc8)
    %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<8xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<1x8xi32, #triton_gpu.slice<{dim = 2, parent = #blocked1}>> loc(#loc7)
    %11 = tt.expand_dims %10 {axis = 2 : i32} : tensor<1x8xi32, #triton_gpu.slice<{dim = 2, parent = #blocked1}>> -> tensor<1x8x1xi32, #blocked1> loc(#loc7)
    %12 = tt.splat %arg4 : i32 -> tensor<1x8x1xi32, #blocked1> loc(#loc7)
    %13 = arith.muli %11, %12 : tensor<1x8x1xi32, #blocked1> loc(#loc7)
    %14 = tt.broadcast %8 : tensor<64x1x1xi32, #blocked1> -> tensor<64x8x1xi32, #blocked1> loc(#loc7)
    %15 = tt.broadcast %13 : tensor<1x8x1xi32, #blocked1> -> tensor<64x8x1xi32, #blocked1> loc(#loc7)
    %16 = arith.addi %14, %15 : tensor<64x8x1xi32, #blocked1> loc(#loc7)
    %17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> loc(#loc9)
    %18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 1, parent = #blocked1}>}>> loc(#loc9)
    %19 = tt.expand_dims %17 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3> loc(#loc10)
    %20 = tt.expand_dims %18 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 1, parent = #blocked1}>}>> -> tensor<1x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc7)
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<1x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<1x1x32xi32, #blocked1> loc(#loc7)
    %22 = tt.broadcast %16 : tensor<64x8x1xi32, #blocked1> -> tensor<64x8x32xi32, #blocked1> loc(#loc7)
    %23 = tt.broadcast %21 : tensor<1x1x32xi32, #blocked1> -> tensor<64x8x32xi32, #blocked1> loc(#loc7)
    %24 = arith.addi %22, %23 : tensor<64x8x32xi32, #blocked1> loc(#loc7)
    %25 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f16>, i32 loc(#loc7)
    %26 = arith.addi %arg6, %c511_i32 : i32 loc(#loc29)
    %27 = arith.divsi %26, %c512_i32 : i32 loc(#loc30)
    %28 = arith.muli %arg4, %c512_i32 : i32 loc(#loc14)
    %29 = tt.splat %25 : !tt.ptr<f16> -> tensor<64x8x32x!tt.ptr<f16>, #blocked1> loc(#loc15)
    %30 = tt.addptr %29, %24 : tensor<64x8x32x!tt.ptr<f16>, #blocked1>, tensor<64x8x32xi32, #blocked1> loc(#loc15)
    %31 = tt.load %30 : tensor<64x8x32x!tt.ptr<f16>, #blocked1> loc(#loc15)
    %32 = triton_gpu.local_alloc %31 {allocation.offset = 0 : i32} : (tensor<64x8x32xf16, #blocked1>) -> !tt.memdesc<64x8x32xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc15)
    %33 = tt.addptr %25, %28 : !tt.ptr<f16>, i32 loc(#loc16)
    %34 = arith.subi %27, %c1_i32 : i32 loc(#loc17)
    cf.br ^bb1(%c0_i32, %cst, %arg0, %33 : i32, tensor<64x1x32xf32, #blocked>, !tt.ptr<f16>, !tt.ptr<f16>) loc(#loc17)
  ^bb1(%35: i32 loc("/triton/./test_kernel.py":81:25), %36: tensor<64x1x32xf32, #blocked> loc(unknown), %37: !tt.ptr<f16> loc("/triton/./test_kernel.py":64:0), %38: !tt.ptr<f16> loc("/triton/./test_kernel.py":88:18)):  // 2 preds: ^bb0, ^bb2
    %39 = arith.cmpi slt, %35, %34 : i32 loc(#loc17)
    cf.cond_br %39, ^bb2, ^bb3 loc(#loc17)
  ^bb2:  // pred: ^bb1
    %40 = tt.splat %38 : !tt.ptr<f16> -> tensor<64x8x32x!tt.ptr<f16>, #blocked1> loc(#loc15)
    %41 = tt.addptr %40, %24 : tensor<64x8x32x!tt.ptr<f16>, #blocked1>, tensor<64x8x32xi32, #blocked1> loc(#loc15)
    %42 = tt.load %41 : tensor<64x8x32x!tt.ptr<f16>, #blocked1> loc(#loc15)
    %43 = tt.splat %37 : !tt.ptr<f16> -> tensor<1x512x!tt.ptr<f16>, #blocked2> loc(#loc18)
    %44 = tt.addptr %43, %1 : tensor<1x512x!tt.ptr<f16>, #blocked2>, tensor<1x512xi32, #blocked2> loc(#loc18)
    %45 = tt.load %44 : tensor<1x512x!tt.ptr<f16>, #blocked2> loc(#loc18)
    %46 = tt.reshape %45 {allow_reorder = false} : tensor<1x512xf16, #blocked2> -> tensor<1x64x8xf16, #blocked4> loc(#loc19)
    %47 = tt.trans %46 {order = array<i32: 1, 0, 2>} : tensor<1x64x8xf16, #blocked4> -> tensor<64x1x8xf16, #blocked5> loc(#loc20)
    %48 = triton_gpu.local_alloc %47 {allocation.offset = 32768 : i32} : (tensor<64x1x8xf16, #blocked5>) -> !tt.memdesc<64x1x8xf16, #shared1, #triton_gpu.shared_memory> loc(#loc21)
    %49 = triton_gpu.local_load %48 : !tt.memdesc<64x1x8xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x1x8xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> loc(#loc21)
    %50 = triton_gpu.local_load %32 : !tt.memdesc<64x8x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<64x8x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> loc(#loc15)
    %51 = tt.dot %49, %50, %36 : tensor<64x1x8xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x8x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x1x32xf32, #blocked> loc(#loc21)
    %52 = tt.addptr %37, %c512_i32 : !tt.ptr<f16>, i32 loc(#loc22)
    %53 = tt.addptr %38, %28 : !tt.ptr<f16>, i32 loc(#loc16)
    triton_gpu.local_store %42, %32 : tensor<64x8x32xf16, #blocked1> -> !tt.memdesc<64x8x32xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc15)
    %54 = arith.addi %35, %c1_i32 : i32 loc(#loc17)
    cf.br ^bb1(%54, %51, %52, %53 : i32, tensor<64x1x32xf32, #blocked>, !tt.ptr<f16>, !tt.ptr<f16>) loc(#loc17)
  ^bb3:  // pred: ^bb1
    %55 = tt.splat %37 : !tt.ptr<f16> -> tensor<1x512x!tt.ptr<f16>, #blocked2> loc(#loc18)
    %56 = tt.addptr %55, %1 : tensor<1x512x!tt.ptr<f16>, #blocked2>, tensor<1x512xi32, #blocked2> loc(#loc18)
    %57 = tt.load %56 : tensor<1x512x!tt.ptr<f16>, #blocked2> loc(#loc18)
    %58 = tt.reshape %57 {allow_reorder = false} : tensor<1x512xf16, #blocked2> -> tensor<1x64x8xf16, #blocked4> loc(#loc19)
    %59 = tt.trans %58 {order = array<i32: 1, 0, 2>} : tensor<1x64x8xf16, #blocked4> -> tensor<64x1x8xf16, #blocked5> loc(#loc20)
    %60 = triton_gpu.local_alloc %59 {allocation.offset = 32768 : i32} : (tensor<64x1x8xf16, #blocked5>) -> !tt.memdesc<64x1x8xf16, #shared1, #triton_gpu.shared_memory> loc(#loc21)
    %61 = triton_gpu.local_load %60 : !tt.memdesc<64x1x8xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x1x8xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> loc(#loc21)
    %62 = triton_gpu.local_load %32 : !tt.memdesc<64x8x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<64x8x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> loc(#loc15)
    %63 = tt.dot %61, %62, %36 : tensor<64x1x8xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x8x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x1x32xf32, #blocked> loc(#loc21)
    %64 = "tt.reduce"(%63) <{axis = 0 : i32}> ({
    ^bb0(%arg7: f32 loc(callsite(#loc1 at #loc24)), %arg8: f32 loc(callsite(#loc1 at #loc24))):
      %69 = arith.addf %arg7, %arg8 : f32 loc(#loc34)
      tt.reduce.return %69 : f32 loc(#loc31)
    }) : (tensor<64x1x32xf32, #blocked>) -> tensor<1x32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc31)
    %65 = tt.addptr %arg2, %c0_i32 : !tt.ptr<f32>, i32 loc(#loc26)
    %66 = triton_gpu.convert_layout %64 {allocation.offset = 0 : i32} : tensor<1x32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf32, #blocked3> loc(#loc27)
    %67 = tt.splat %65 : !tt.ptr<f32> -> tensor<1x32x!tt.ptr<f32>, #blocked3> loc(#loc27)
    %68 = tt.addptr %67, %19 : tensor<1x32x!tt.ptr<f32>, #blocked3>, tensor<1x32xi32, #blocked3> loc(#loc27)
    tt.store %68, %66 : tensor<1x32x!tt.ptr<f32>, #blocked3> loc(#loc27)
    tt.return loc(#loc28)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/triton/./test_kernel.py":77:61)
#loc3 = loc("/triton/./test_kernel.py":77:22)
#loc4 = loc("/triton/./test_kernel.py":78:35)
#loc5 = loc("/triton/./test_kernel.py":78:52)
#loc6 = loc("/triton/./test_kernel.py":78:68)
#loc7 = loc("/triton/./test_kernel.py":78:22)
#loc8 = loc("/triton/./test_kernel.py":79:30)
#loc9 = loc("/triton/./test_kernel.py":79:67)
#loc10 = loc("/triton/./test_kernel.py":92:52)
#loc11 = loc("/triton/python/triton/language/standard.py":40:22)
#loc12 = loc("/triton/./test_kernel.py":81:36)
#loc13 = loc("/triton/python/triton/language/standard.py":40:28)
#loc14 = loc("/triton/./test_kernel.py":88:33)
#loc15 = loc("/triton/./test_kernel.py":83:20)
#loc18 = loc("/triton/./test_kernel.py":82:20)
#loc19 = loc("/triton/./test_kernel.py":84:26)
#loc20 = loc("/triton/./test_kernel.py":85:26)
#loc21 = loc("/triton/./test_kernel.py":86:24)
#loc22 = loc("/triton/./test_kernel.py":87:18)
#loc23 = loc("/triton/python/triton/language/standard.py":267:36)
#loc25 = loc("/triton/python/triton/language/standard.py":256:15)
#loc26 = loc("/triton/./test_kernel.py":92:21)
#loc27 = loc("/triton/./test_kernel.py":93:21)
#loc28 = loc("/triton/./test_kernel.py":93:4)
#loc29 = loc(callsite(#loc11 at #loc12))
#loc30 = loc(callsite(#loc13 at #loc12))
#loc31 = loc(callsite(#loc23 at #loc24))
#loc33 = loc(callsite(#loc25 at #loc23))
#loc34 = loc(callsite(#loc33 at #loc24))
