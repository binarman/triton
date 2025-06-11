#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [4, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [[1, 0], [2, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [128, 0]], lane = [[0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [1, 4], tilesPerWarp = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_gemm_afp4_wfp4_kernel_preshuffled_scales(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg6, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %5 = arith.cmpi sgt, %arg7, %c0_i32 : i32
    scf.if %5 {
      %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
      %8 = arith.muli %3, %c128_i32 : i32
      %9 = tt.splat %8 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = arith.addi %9, %6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %11 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.remsi %10, %11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = arith.muli %4, %c256_i32 : i32
      %14 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %15 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %16 = tt.splat %13 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %17 = arith.addi %16, %14 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %18 = tt.splat %arg6 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %19 = arith.remsi %17, %18 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %21 = tt.expand_dims %12 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %22 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
      %23 = arith.muli %21, %22 : tensor<128x1xi32, #blocked>
      %24 = tt.broadcast %23 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %25 = tt.expand_dims %20 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %26 = tt.broadcast %25 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %27 = arith.addi %24, %26 : tensor<128x128xi32, #blocked>
      %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %30 = tt.broadcast %29 : tensor<128x1xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
      %31 = tt.expand_dims %19 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
      %32 = tt.splat %arg9 : i32 -> tensor<1x256xi32, #blocked1>
      %33 = arith.muli %31, %32 : tensor<1x256xi32, #blocked1>
      %34 = tt.broadcast %33 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
      %35 = arith.addi %30, %34 : tensor<128x256xi32, #blocked1>
      %36 = arith.muli %4, %c8_i32 : i32
      %37 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %38 = tt.splat %36 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %39 = arith.addi %38, %37 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %40 = tt.splat %arg6 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %41 = arith.remsi %39, %40 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<8x1xi32, #blocked2>
      %43 = tt.splat %arg13 : i32 -> tensor<8x1xi32, #blocked2>
      %44 = arith.muli %42, %43 : tensor<8x1xi32, #blocked2>
      %45 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
      %46 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %47 = tt.broadcast %44 : tensor<8x1xi32, #blocked2> -> tensor<8x256xi32, #blocked2>
      %48 = tt.expand_dims %46 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
      %49 = tt.broadcast %48 : tensor<1x256xi32, #blocked2> -> tensor<8x256xi32, #blocked2>
      %50 = arith.addi %49, %47 : tensor<8x256xi32, #blocked2>
      %51 = arith.muli %3, %c4_i32 : i32
      %52 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %53 = tt.splat %51 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %54 = arith.addi %53, %52 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %55 = tt.splat %arg5 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %56 = arith.remsi %54, %55 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %57 = tt.expand_dims %56 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<4x1xi32, #blocked3>
      %58 = tt.splat %arg12 : i32 -> tensor<4x1xi32, #blocked3>
      %59 = arith.muli %57, %58 : tensor<4x1xi32, #blocked3>
      %60 = tt.broadcast %59 : tensor<4x1xi32, #blocked3> -> tensor<4x256xi32, #blocked3>
      %61 = tt.expand_dims %45 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
      %62 = tt.broadcast %61 : tensor<1x256xi32, #blocked3> -> tensor<4x256xi32, #blocked3>
      %63 = arith.addi %62, %60 : tensor<4x256xi32, #blocked3>
      %64 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xi8, #shared, #smem, mutable>
      %65 = ttg.local_alloc : () -> !ttg.memdesc<1x128x256xi8, #shared1, #smem, mutable>
      %66 = ttg.local_alloc : () -> !ttg.memdesc<1x4x256xi8, #shared2, #smem, mutable>
      %67 = ttg.local_alloc : () -> !ttg.memdesc<1x8x256xi8, #shared2, #smem, mutable>
      %68 = amdgpu.buffer_load %arg3[%63] stride = %arg12 : tensor<4x256xi8, #blocked3>
      %69 = amdgpu.buffer_load %arg4[%50] stride = %arg13 : tensor<8x256xi8, #blocked2>
      %70 = amdgpu.buffer_load %arg0[%27] stride = %arg8 : tensor<128x128xi8, #blocked>
      %71 = amdgpu.buffer_load %arg1[%35] cacheModifier = cg stride = %arg9 : tensor<128x256xi8, #blocked1>
      %72 = ttg.memdesc_subview %66[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
      ttg.local_store %68, %72 : tensor<4x256xi8, #blocked3> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
      %73 = ttg.memdesc_subview %67[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
      ttg.local_store %69, %73 : tensor<8x256xi8, #blocked2> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
      %74 = ttg.memdesc_subview %64[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
      ttg.local_store %70, %74 : tensor<128x128xi8, #blocked> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
      %75 = ttg.memdesc_subview %65[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
      ttg.local_store %71, %75 : tensor<128x256xi8, #blocked1> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>

      %dup_62 = arith.muli %4, %c8_i32 : i32
      %dup_63 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_64 = tt.splat %dup_62 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_65 = arith.addi %dup_64, %dup_63 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_66 = tt.splat %arg6 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_67 = arith.remsi %dup_65, %dup_66 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_68 = tt.expand_dims %dup_67 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<8x1xi32, #blocked4>
      %dup_69 = tt.splat %arg13 : i32 -> tensor<8x1xi32, #blocked4>
      %dup_70 = arith.muli %dup_68, %dup_69 : tensor<8x1xi32, #blocked4>
      %dup_71 = arith.muli %3, %c4_i32 : i32
      %dup_72 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_73 = tt.splat %dup_71 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_74 = arith.addi %dup_73, %dup_72 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_75 = tt.splat %arg5 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_76 = arith.remsi %dup_74, %dup_75 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %dup_77 = tt.expand_dims %dup_76 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<4x1xi32, #blocked4>
      %dup_78 = tt.splat %arg12 : i32 -> tensor<4x1xi32, #blocked4>
      %dup_79 = arith.muli %dup_77, %dup_78 : tensor<4x1xi32, #blocked4>
      %dup_80 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %dup_81 = tt.broadcast %dup_79 : tensor<4x1xi32, #blocked4> -> tensor<4x1024xi32, #blocked4>
      %dup_82 = tt.expand_dims %dup_80 {axis = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x1024xi32, #blocked4>
      %dup_83 = tt.broadcast %dup_82 : tensor<1x1024xi32, #blocked4> -> tensor<4x1024xi32, #blocked4>
      %dup_84 = arith.addi %dup_83, %dup_81 : tensor<4x1024xi32, #blocked4>
      %dup_85 = tt.broadcast %dup_70 : tensor<8x1xi32, #blocked4> -> tensor<8x1024xi32, #blocked4>
      %dup_86 = tt.broadcast %dup_82 : tensor<1x1024xi32, #blocked4> -> tensor<8x1024xi32, #blocked4>
      %dup_87 = arith.addi %dup_86, %dup_85 : tensor<8x1024xi32, #blocked4>

      %c60_i32 = arith.constant 60 : i32
      %c4_i32_0 = arith.constant 4 : i32
      %76:10 = scf.for %arg14 = %c0_i32 to %c60_i32 step %c4_i32_0 iter_args(%arg15 = %cst, %arg16 = %arg3, %arg17 = %arg4, %arg18 = %arg0, %arg19 = %arg1, %arg20 = %c0_i32, %arg21 = %72, %arg22 = %73, %arg23 = %74, %arg24 = %75) -> (tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>)  : i32 {
        // insert aggregated loads for scales
        // insert extract_slice
        // remove old non aggregated loads
        %iter_mul = arith.muli %arg14, %c256_i32 : i32
        %base_a = tt.addptr %arg3, %iter_mul : !tt.ptr<i8>, i32
        %base_b = tt.addptr %arg4, %iter_mul : !tt.ptr<i8>, i32
        %a_agg_raw = amdgpu.buffer_load %base_a[%dup_84] stride = %arg12 : tensor<4x1024xi8, #blocked4>
        %b_agg_raw = amdgpu.buffer_load %base_b[%dup_87] stride = %arg13 : tensor<8x1024xi8, #blocked4>
        %a_agg = ttg.convert_layout %a_agg_raw : tensor<4x1024xi8, #blocked4> -> tensor<4x1024xi8, #linear>
        %b_agg = ttg.convert_layout %b_agg_raw : tensor<8x1024xi8, #blocked4> -> tensor<8x1024xi8, #linear1>

        %118 = tt.addptr %arg18, %c128_i32 : !tt.ptr<i8>, i32
        %119 = tt.addptr %arg19, %c128_i32 : !tt.ptr<i8>, i32
        %120 = tt.addptr %arg16, %c256_i32 : !tt.ptr<i8>, i32
        %121 = tt.addptr %arg17, %c256_i32 : !tt.ptr<i8>, i32
        %122 = amdgpu.buffer_load %120[%63] stride = %arg12 : tensor<4x256xi8, #blocked3>
        %123 = ttg.local_load %arg21 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %124 = amdgpu.buffer_load %121[%50] stride = %arg13 : tensor<8x256xi8, #blocked2>


        %125 = ttg.local_load %arg22 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>

        %a_slice0 = amdgpu.extract_slice %a_agg [0, 0] : tensor<4x1024xi8, #linear> to tensor<4x256xi8, #linear>
        %b_slice0 = amdgpu.extract_slice %b_agg [0, 0] : tensor<8x1024xi8, #linear1> to tensor<8x256xi8, #linear1>

        %126 = tt.reshape %a_slice0 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %127 = tt.reshape %b_slice0 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %128 = amdgpu.buffer_load %118[%27] stride = %arg8 : tensor<128x128xi8, #blocked>
        %129 = ttg.local_load %arg23 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %130 = amdgpu.buffer_load %119[%35] cacheModifier = cg stride = %arg9 : tensor<128x256xi8, #blocked1>
        %131 = ttg.local_load %arg24 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %132 = tt.dot_scaled %129 scale %126, %131 scale %127, %arg15 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %133 = arith.addi %arg20, %c1_i32 : i32
        %134 = arith.cmpi slt, %133, %c1_i32 : i32
        %135 = arith.select %134, %133, %c0_i32 : i32
        %136 = ttg.memdesc_subview %66[%135, %c0_i32, %c0_i32] : !ttg.memdesc<1x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        ttg.local_store %122, %136 : tensor<4x256xi8, #blocked3> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %137 = ttg.memdesc_subview %67[%135, %c0_i32, %c0_i32] : !ttg.memdesc<1x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        ttg.local_store %124, %137 : tensor<8x256xi8, #blocked2> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %138 = ttg.memdesc_subview %64[%135, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        ttg.local_store %128, %138 : tensor<128x128xi8, #blocked> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %139 = ttg.memdesc_subview %65[%135, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        ttg.local_store %130, %139 : tensor<128x256xi8, #blocked1> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %140 = tt.addptr %118, %c128_i32 : !tt.ptr<i8>, i32
        %141 = tt.addptr %119, %c128_i32 : !tt.ptr<i8>, i32
        %142 = tt.addptr %120, %c256_i32 : !tt.ptr<i8>, i32
        %143 = tt.addptr %121, %c256_i32 : !tt.ptr<i8>, i32
        %144 = amdgpu.buffer_load %142[%63] stride = %arg12 : tensor<4x256xi8, #blocked3>
        %145 = ttg.local_load %136 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %146 = amdgpu.buffer_load %143[%50] stride = %arg13 : tensor<8x256xi8, #blocked2>
        %147 = ttg.local_load %137 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>

        %a_slice1 = amdgpu.extract_slice %a_agg [0, 0] : tensor<4x1024xi8, #linear> to tensor<4x256xi8, #linear>
        %b_slice1 = amdgpu.extract_slice %b_agg [0, 0] : tensor<8x1024xi8, #linear1> to tensor<8x256xi8, #linear1>
        %148 = tt.reshape %a_slice1 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %149 = tt.reshape %b_slice1 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %150 = amdgpu.buffer_load %140[%27] stride = %arg8 : tensor<128x128xi8, #blocked>
        %151 = ttg.local_load %138 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %152 = amdgpu.buffer_load %141[%35] cacheModifier = cg stride = %arg9 : tensor<128x256xi8, #blocked1>
        %153 = ttg.local_load %139 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %154 = tt.dot_scaled %151 scale %148, %153 scale %149, %132 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %155 = arith.addi %135, %c1_i32 : i32
        %156 = arith.cmpi slt, %155, %c1_i32 : i32
        %157 = arith.select %156, %155, %c0_i32 : i32
        %158 = ttg.memdesc_subview %66[%157, %c0_i32, %c0_i32] : !ttg.memdesc<1x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        ttg.local_store %144, %158 : tensor<4x256xi8, #blocked3> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %159 = ttg.memdesc_subview %67[%157, %c0_i32, %c0_i32] : !ttg.memdesc<1x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        ttg.local_store %146, %159 : tensor<8x256xi8, #blocked2> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %160 = ttg.memdesc_subview %64[%157, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        ttg.local_store %150, %160 : tensor<128x128xi8, #blocked> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %161 = ttg.memdesc_subview %65[%157, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        ttg.local_store %152, %161 : tensor<128x256xi8, #blocked1> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %162 = tt.addptr %140, %c128_i32 : !tt.ptr<i8>, i32
        %163 = tt.addptr %141, %c128_i32 : !tt.ptr<i8>, i32
        %164 = tt.addptr %142, %c256_i32 : !tt.ptr<i8>, i32
        %165 = tt.addptr %143, %c256_i32 : !tt.ptr<i8>, i32
        %166 = amdgpu.buffer_load %164[%63] stride = %arg12 : tensor<4x256xi8, #blocked3>
        %167 = ttg.local_load %158 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %168 = amdgpu.buffer_load %165[%50] stride = %arg13 : tensor<8x256xi8, #blocked2>
        %169 = ttg.local_load %159 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>

        %a_slice2 = amdgpu.extract_slice %a_agg [0, 0] : tensor<4x1024xi8, #linear> to tensor<4x256xi8, #linear>
        %b_slice2 = amdgpu.extract_slice %b_agg [0, 0] : tensor<8x1024xi8, #linear1> to tensor<8x256xi8, #linear1>
        %170 = tt.reshape %a_slice2 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %171 = tt.reshape %b_slice2 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %172 = amdgpu.buffer_load %162[%27] stride = %arg8 : tensor<128x128xi8, #blocked>
        %173 = ttg.local_load %160 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %174 = amdgpu.buffer_load %163[%35] cacheModifier = cg stride = %arg9 : tensor<128x256xi8, #blocked1>
        %175 = ttg.local_load %161 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %176 = tt.dot_scaled %173 scale %170, %175 scale %171, %154 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %177 = arith.addi %157, %c1_i32 : i32
        %178 = arith.cmpi slt, %177, %c1_i32 : i32
        %179 = arith.select %178, %177, %c0_i32 : i32
        %180 = ttg.memdesc_subview %66[%179, %c0_i32, %c0_i32] : !ttg.memdesc<1x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        ttg.local_store %166, %180 : tensor<4x256xi8, #blocked3> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %181 = ttg.memdesc_subview %67[%179, %c0_i32, %c0_i32] : !ttg.memdesc<1x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        ttg.local_store %168, %181 : tensor<8x256xi8, #blocked2> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %182 = ttg.memdesc_subview %64[%179, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        ttg.local_store %172, %182 : tensor<128x128xi8, #blocked> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %183 = ttg.memdesc_subview %65[%179, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        ttg.local_store %174, %183 : tensor<128x256xi8, #blocked1> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %184 = tt.addptr %162, %c128_i32 : !tt.ptr<i8>, i32
        %185 = tt.addptr %163, %c128_i32 : !tt.ptr<i8>, i32
        %186 = tt.addptr %164, %c256_i32 : !tt.ptr<i8>, i32
        %187 = tt.addptr %165, %c256_i32 : !tt.ptr<i8>, i32
        %188 = amdgpu.buffer_load %186[%63] stride = %arg12 : tensor<4x256xi8, #blocked3>
        %189 = ttg.local_load %180 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %190 = amdgpu.buffer_load %187[%50] stride = %arg13 : tensor<8x256xi8, #blocked2>
        %191 = ttg.local_load %181 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>

        %a_slice3 = amdgpu.extract_slice %a_agg [0, 0] : tensor<4x1024xi8, #linear> to tensor<4x256xi8, #linear>
        %b_slice3 = amdgpu.extract_slice %b_agg [0, 0] : tensor<8x1024xi8, #linear1> to tensor<8x256xi8, #linear1>
        %192 = tt.reshape %a_slice3 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %193 = tt.reshape %b_slice3 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %194 = amdgpu.buffer_load %184[%27] stride = %arg8 : tensor<128x128xi8, #blocked>
        %195 = ttg.local_load %182 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %196 = amdgpu.buffer_load %185[%35] cacheModifier = cg stride = %arg9 : tensor<128x256xi8, #blocked1>
        %197 = ttg.local_load %183 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %198 = tt.dot_scaled %195 scale %192, %197 scale %193, %176 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %199 = arith.addi %179, %c1_i32 : i32
        %200 = arith.cmpi slt, %199, %c1_i32 : i32
        %201 = arith.select %200, %199, %c0_i32 : i32
        %202 = ttg.memdesc_subview %66[%201, %c0_i32, %c0_i32] : !ttg.memdesc<1x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        ttg.local_store %188, %202 : tensor<4x256xi8, #blocked3> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %203 = ttg.memdesc_subview %67[%201, %c0_i32, %c0_i32] : !ttg.memdesc<1x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        ttg.local_store %190, %203 : tensor<8x256xi8, #blocked2> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %204 = ttg.memdesc_subview %64[%201, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        ttg.local_store %194, %204 : tensor<128x128xi8, #blocked> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %205 = ttg.memdesc_subview %65[%201, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        ttg.local_store %196, %205 : tensor<128x256xi8, #blocked1> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        scf.yield %198, %186, %187, %184, %185, %201, %202, %203, %204, %205 : tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
      }
      // insert load of scales for second loop
      %77:10 = scf.for %arg14 = %c60_i32 to %c63_i32 step %c1_i32 iter_args(%arg15 = %76#0, %arg16 = %76#1, %arg17 = %76#2, %arg18 = %76#3, %arg19 = %76#4, %arg20 = %76#5, %arg21 = %76#6, %arg22 = %76#7, %arg23 = %76#8, %arg24 = %76#9) -> (tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>)  : i32 {
        %118 = tt.addptr %arg18, %c128_i32 : !tt.ptr<i8>, i32
        %119 = tt.addptr %arg19, %c128_i32 : !tt.ptr<i8>, i32
        %120 = tt.addptr %arg16, %c256_i32 : !tt.ptr<i8>, i32
        %121 = tt.addptr %arg17, %c256_i32 : !tt.ptr<i8>, i32
        %122 = amdgpu.buffer_load %120[%63] stride = %arg12 : tensor<4x256xi8, #blocked3>
        %123 = ttg.local_load %arg21 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %124 = amdgpu.buffer_load %121[%50] stride = %arg13 : tensor<8x256xi8, #blocked2>
        %125 = ttg.local_load %arg22 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
        %126 = tt.reshape %123 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %127 = tt.reshape %125 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %128 = amdgpu.buffer_load %118[%27] stride = %arg8 : tensor<128x128xi8, #blocked>
        %129 = ttg.local_load %arg23 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %130 = amdgpu.buffer_load %119[%35] cacheModifier = cg stride = %arg9 : tensor<128x256xi8, #blocked1>
        %131 = ttg.local_load %arg24 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %132 = tt.dot_scaled %129 scale %126, %131 scale %127, %arg15 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %133 = arith.addi %arg20, %c1_i32 : i32
        %134 = arith.cmpi slt, %133, %c1_i32 : i32
        %135 = arith.select %134, %133, %c0_i32 : i32
        %136 = ttg.memdesc_subview %66[%135, %c0_i32, %c0_i32] : !ttg.memdesc<1x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        ttg.local_store %122, %136 : tensor<4x256xi8, #blocked3> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %137 = ttg.memdesc_subview %67[%135, %c0_i32, %c0_i32] : !ttg.memdesc<1x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        ttg.local_store %124, %137 : tensor<8x256xi8, #blocked2> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %138 = ttg.memdesc_subview %64[%135, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        ttg.local_store %128, %138 : tensor<128x128xi8, #blocked> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %139 = ttg.memdesc_subview %65[%135, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        ttg.local_store %130, %139 : tensor<128x256xi8, #blocked1> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        scf.yield %132, %120, %121, %118, %119, %135, %136, %137, %138, %139 : tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
      }
      %78 = ttg.local_load %77#6 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
      %79 = ttg.local_load %77#7 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
      %80 = tt.reshape %78 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
      %81 = tt.reshape %79 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
      %82 = ttg.local_load %77#8 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %83 = ttg.local_load %77#9 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %84 = tt.dot_scaled %82 scale %80, %83 scale %81, %77#0 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
      ttg.local_dealloc %64 : !ttg.memdesc<1x128x128xi8, #shared, #smem, mutable>
      ttg.local_dealloc %65 : !ttg.memdesc<1x128x256xi8, #shared1, #smem, mutable>
      ttg.local_dealloc %66 : !ttg.memdesc<1x4x256xi8, #shared2, #smem, mutable>
      ttg.local_dealloc %67 : !ttg.memdesc<1x8x256xi8, #shared2, #smem, mutable>
      %85 = arith.truncf %84 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
      %86 = arith.extsi %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %87 = arith.extsi %8 : i32 to i64
      %88 = tt.splat %87 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %89 = arith.addi %88, %86 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %90 = arith.extsi %15 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>> to tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %91 = arith.extsi %13 : i32 to i64
      %92 = tt.splat %91 : i64 -> tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %93 = arith.addi %92, %90 : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %94 = tt.expand_dims %89 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi64, #mma>
      %95 = arith.extsi %arg11 : i32 to i64
      %96 = tt.expand_dims %86 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi64, #mma>
      %97 = arith.muli %95, %87 : i64
      %98 = tt.splat %95 : i64 -> tensor<128x1xi64, #mma>
      %99 = arith.muli %98, %96 : tensor<128x1xi64, #mma>
      %100 = tt.addptr %arg2, %97 : !tt.ptr<bf16>, i64
      %101 = arith.trunci %99 : tensor<128x1xi64, #mma> to tensor<128x1xi32, #mma>
      %102 = tt.expand_dims %93 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi64, #mma>
      %103 = tt.broadcast %101 : tensor<128x1xi32, #mma> -> tensor<128x256xi32, #mma>
      %104 = tt.expand_dims %90 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi64, #mma>
      %105 = tt.broadcast %104 : tensor<1x256xi64, #mma> -> tensor<128x256xi64, #mma>
      %106 = tt.addptr %100, %91 : !tt.ptr<bf16>, i64
      %107 = arith.trunci %105 : tensor<128x256xi64, #mma> to tensor<128x256xi32, #mma>
      %108 = arith.addi %107, %103 : tensor<128x256xi32, #mma>
      %109 = arith.extsi %arg5 : i32 to i64
      %110 = tt.splat %109 : i64 -> tensor<128x1xi64, #mma>
      %111 = arith.cmpi slt, %94, %110 : tensor<128x1xi64, #mma>
      %112 = arith.extsi %arg6 : i32 to i64
      %113 = tt.splat %112 : i64 -> tensor<1x256xi64, #mma>
      %114 = arith.cmpi slt, %102, %113 : tensor<1x256xi64, #mma>
      %115 = tt.broadcast %111 : tensor<128x1xi1, #mma> -> tensor<128x256xi1, #mma>
      %116 = tt.broadcast %114 : tensor<1x256xi1, #mma> -> tensor<128x256xi1, #mma>
      %117 = arith.andi %115, %116 : tensor<128x256xi1, #mma>
      amdgpu.buffer_store %85, %106[%108], %117 : tensor<128x256xbf16, #mma>
    }
    tt.return
  }
}
