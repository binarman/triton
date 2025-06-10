#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
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
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %c255_i32 = arith.constant 255 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<true> : tensor<8x256xi1, #blocked>
    %c63_i32 = arith.constant 63 : i32
    %cst_1 = arith.constant dense<true> : tensor<128x256xi1, #blocked1>
    %cst_2 = arith.constant dense<true> : tensor<128x128xi1, #blocked2>
    %cst_3 = arith.constant dense<true> : tensor<4x256xi1, #blocked>
    %c2_i32 = arith.constant 2 : i32
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
      %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
      %8 = arith.muli %3, %c128_i32 : i32
      %9 = tt.splat %8 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %10 = arith.addi %9, %6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %11 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %12 = arith.remsi %10, %11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %13 = arith.muli %4, %c256_i32 : i32
      %14 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %15 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %16 = tt.splat %13 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %17 = arith.addi %16, %14 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %18 = tt.splat %arg6 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %19 = arith.remsi %17, %18 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %21 = tt.expand_dims %12 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
      %22 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
      %23 = arith.muli %21, %22 : tensor<128x1xi32, #blocked2>
      %24 = tt.broadcast %23 : tensor<128x1xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
      %25 = tt.expand_dims %20 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
      %26 = tt.broadcast %25 : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
      %27 = arith.addi %24, %26 : tensor<128x128xi32, #blocked2>
      %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %30 = tt.broadcast %29 : tensor<128x1xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
      %31 = tt.expand_dims %19 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
      %32 = tt.splat %arg9 : i32 -> tensor<1x256xi32, #blocked1>
      %33 = arith.muli %31, %32 : tensor<1x256xi32, #blocked1>
      %34 = tt.broadcast %33 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
      %35 = arith.addi %30, %34 : tensor<128x256xi32, #blocked1>
      %36 = arith.muli %4, %c8_i32 : i32
      %37 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %38 = tt.splat %36 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %39 = arith.addi %38, %37 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %40 = tt.splat %arg6 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %41 = arith.remsi %39, %40 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
      %43 = tt.splat %arg13 : i32 -> tensor<8x1xi32, #blocked>
      %44 = arith.muli %42, %43 : tensor<8x1xi32, #blocked>
      %45 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %46 = tt.broadcast %44 : tensor<8x1xi32, #blocked> -> tensor<8x256xi32, #blocked>
      %47 = tt.expand_dims %45 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %48 = tt.broadcast %47 : tensor<1x256xi32, #blocked> -> tensor<8x256xi32, #blocked>
      %49 = arith.addi %48, %46 : tensor<8x256xi32, #blocked>
      %50 = arith.muli %3, %c4_i32 : i32
      %51 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %52 = tt.splat %50 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %53 = arith.addi %52, %51 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %54 = tt.splat %arg5 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %55 = arith.remsi %53, %54 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %56 = tt.expand_dims %55 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
      %57 = tt.splat %arg12 : i32 -> tensor<4x1xi32, #blocked>
      %58 = arith.muli %56, %57 : tensor<4x1xi32, #blocked>
      %59 = tt.broadcast %58 : tensor<4x1xi32, #blocked> -> tensor<4x256xi32, #blocked>
      %60 = tt.broadcast %47 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
      %61 = arith.addi %60, %59 : tensor<4x256xi32, #blocked>
      %62 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable>
      %63 = ttg.local_alloc : () -> !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable>
      %64 = ttg.local_alloc : () -> !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable>
      %65 = ttg.local_alloc : () -> !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable>
      %66 = ttg.memdesc_subview %64[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
      %67 = amdgpu.buffer_load_to_local %arg3[%61] mask = %cst_3 stride = %arg12 into %66 : <i8>[tensor<4x256xi32, #blocked>]  -> <4x256xi8, #shared2, #smem, mutable>
      %68 = ttg.async_commit_group %67
      %69 = ttg.memdesc_subview %65[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
      %70 = amdgpu.buffer_load_to_local %arg4[%49] mask = %cst_0 stride = %arg13 into %69 : <i8>[tensor<8x256xi32, #blocked>]  -> <8x256xi8, #shared2, #smem, mutable>
      %71 = ttg.async_commit_group %70
      %72 = ttg.memdesc_subview %62[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
      %73 = amdgpu.buffer_load_to_local %arg0[%27] mask = %cst_2 stride = %arg8 into %72 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
      %74 = ttg.async_commit_group %73
      %75 = ttg.memdesc_subview %63[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
      %76 = amdgpu.buffer_load_to_local %arg1[%35] mask = %cst_1 stride = %arg9 cacheModifier = cg into %75 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
      %77 = ttg.async_commit_group %76
      %c60_i32 = arith.constant 60 : i32
      %c4_i32_4 = arith.constant 4 : i32
      %78:14 = scf.for %arg14 = %c0_i32 to %c60_i32 step %c4_i32_4 iter_args(%arg15 = %cst, %arg16 = %arg3, %arg17 = %arg0, %arg18 = %arg1, %arg19 = %c0_i32, %arg20 = %68, %arg21 = %71, %arg22 = %74, %arg23 = %77, %arg24 = %66, %arg25 = %69, %arg26 = %72, %arg27 = %75, %arg28 = %arg4) -> (tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>, !tt.ptr<i8>)  : i32 {
        %121 = tt.addptr %arg17, %c128_i32 : !tt.ptr<i8>, i32
        %122 = tt.addptr %arg18, %c128_i32 : !tt.ptr<i8>, i32
        %123 = tt.addptr %arg16, %c256_i32 : !tt.ptr<i8>, i32
        %124 = tt.addptr %arg28, %c256_i32 : !tt.ptr<i8>, i32
        %125 = arith.addi %arg19, %c1_i32 : i32
        %126 = arith.cmpi slt, %125, %c2_i32 : i32
        %127 = arith.select %126, %125, %c0_i32 : i32
        %128 = ttg.memdesc_subview %64[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %129 = amdgpu.buffer_load_to_local %123[%61] stride = %arg12 into %128 : <i8>[tensor<4x256xi32, #blocked>]  -> <4x256xi8, #shared2, #smem, mutable>
        %130 = ttg.async_commit_group %129
        %131 = ttg.memdesc_subview %65[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %132 = amdgpu.buffer_load_to_local %124[%49] stride = %arg13 into %131 : <i8>[tensor<8x256xi32, #blocked>]  -> <8x256xi8, #shared2, #smem, mutable>
        %133 = ttg.async_commit_group %132
        %134 = ttg.memdesc_subview %62[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %135 = amdgpu.buffer_load_to_local %121[%27] stride = %arg8 into %134 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %136 = ttg.async_commit_group %135
        %137 = ttg.memdesc_subview %63[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %138 = amdgpu.buffer_load_to_local %122[%35] stride = %arg9 cacheModifier = cg into %137 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %139 = ttg.async_commit_group %138
        %140 = ttg.async_wait %arg20, %arg21, %arg22, %arg23 {num = 15 : i32}
        %141 = ttg.local_load %arg24 token %140 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %142 = ttg.local_load %arg25 token %140 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
        %143 = tt.reshape %141 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %144 = tt.reshape %142 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %145 = ttg.local_load %arg26 token %140 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %146 = ttg.local_load %arg27 token %140 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %147 = tt.dot_scaled %145 scale %143, %146 scale %144, %arg15 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %148 = tt.addptr %121, %c128_i32 : !tt.ptr<i8>, i32
        %149 = tt.addptr %122, %c128_i32 : !tt.ptr<i8>, i32
        %150 = tt.addptr %123, %c256_i32 : !tt.ptr<i8>, i32
        %151 = tt.addptr %124, %c256_i32 : !tt.ptr<i8>, i32
        %152 = arith.addi %127, %c1_i32 : i32
        %153 = arith.cmpi slt, %152, %c2_i32 : i32
        %154 = arith.select %153, %152, %c0_i32 : i32
        %155 = ttg.memdesc_subview %64[%154, %c0_i32, %c0_i32] : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %156 = amdgpu.buffer_load_to_local %150[%61] stride = %arg12 into %155 : <i8>[tensor<4x256xi32, #blocked>]  -> <4x256xi8, #shared2, #smem, mutable>
        %157 = ttg.async_commit_group %156
        %158 = ttg.memdesc_subview %65[%154, %c0_i32, %c0_i32] : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %159 = amdgpu.buffer_load_to_local %151[%49] stride = %arg13 into %158 : <i8>[tensor<8x256xi32, #blocked>]  -> <8x256xi8, #shared2, #smem, mutable>
        %160 = ttg.async_commit_group %159
        %161 = ttg.memdesc_subview %62[%154, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %162 = amdgpu.buffer_load_to_local %148[%27] stride = %arg8 into %161 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %163 = ttg.async_commit_group %162
        %164 = ttg.memdesc_subview %63[%154, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %165 = amdgpu.buffer_load_to_local %149[%35] stride = %arg9 cacheModifier = cg into %164 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %166 = ttg.async_commit_group %165
        %167 = ttg.async_wait %130, %133, %136, %139 {num = 15 : i32}
        %168 = ttg.local_load %128 token %167 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %169 = ttg.local_load %131 token %167 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
        %170 = tt.reshape %168 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %171 = tt.reshape %169 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %172 = ttg.local_load %134 token %167 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %173 = ttg.local_load %137 token %167 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %174 = tt.dot_scaled %172 scale %170, %173 scale %171, %147 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %175 = tt.addptr %148, %c128_i32 : !tt.ptr<i8>, i32
        %176 = tt.addptr %149, %c128_i32 : !tt.ptr<i8>, i32
        %177 = tt.addptr %150, %c256_i32 : !tt.ptr<i8>, i32
        %178 = tt.addptr %151, %c256_i32 : !tt.ptr<i8>, i32
        %179 = arith.addi %154, %c1_i32 : i32
        %180 = arith.cmpi slt, %179, %c2_i32 : i32
        %181 = arith.select %180, %179, %c0_i32 : i32
        %182 = ttg.memdesc_subview %64[%181, %c0_i32, %c0_i32] : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %183 = amdgpu.buffer_load_to_local %177[%61] stride = %arg12 into %182 : <i8>[tensor<4x256xi32, #blocked>]  -> <4x256xi8, #shared2, #smem, mutable>
        %184 = ttg.async_commit_group %183
        %185 = ttg.memdesc_subview %65[%181, %c0_i32, %c0_i32] : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %186 = amdgpu.buffer_load_to_local %178[%49] stride = %arg13 into %185 : <i8>[tensor<8x256xi32, #blocked>]  -> <8x256xi8, #shared2, #smem, mutable>
        %187 = ttg.async_commit_group %186
        %188 = ttg.memdesc_subview %62[%181, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %189 = amdgpu.buffer_load_to_local %175[%27] stride = %arg8 into %188 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %190 = ttg.async_commit_group %189
        %191 = ttg.memdesc_subview %63[%181, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %192 = amdgpu.buffer_load_to_local %176[%35] stride = %arg9 cacheModifier = cg into %191 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %193 = ttg.async_commit_group %192
        %194 = ttg.async_wait %157, %160, %163, %166 {num = 15 : i32}
        %195 = ttg.local_load %155 token %194 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %196 = ttg.local_load %158 token %194 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
        %197 = tt.reshape %195 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %198 = tt.reshape %196 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %199 = ttg.local_load %161 token %194 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %200 = ttg.local_load %164 token %194 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %201 = tt.dot_scaled %199 scale %197, %200 scale %198, %174 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %202 = tt.addptr %175, %c128_i32 : !tt.ptr<i8>, i32
        %203 = tt.addptr %176, %c128_i32 : !tt.ptr<i8>, i32
        %204 = tt.addptr %177, %c256_i32 : !tt.ptr<i8>, i32
        %205 = tt.addptr %178, %c256_i32 : !tt.ptr<i8>, i32
        %206 = arith.addi %181, %c1_i32 : i32
        %207 = arith.cmpi slt, %206, %c2_i32 : i32
        %208 = arith.select %207, %206, %c0_i32 : i32
        %209 = ttg.memdesc_subview %64[%208, %c0_i32, %c0_i32] : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %210 = amdgpu.buffer_load_to_local %204[%61] stride = %arg12 into %209 : <i8>[tensor<4x256xi32, #blocked>]  -> <4x256xi8, #shared2, #smem, mutable>
        %211 = ttg.async_commit_group %210
        %212 = ttg.memdesc_subview %65[%208, %c0_i32, %c0_i32] : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %213 = amdgpu.buffer_load_to_local %205[%49] stride = %arg13 into %212 : <i8>[tensor<8x256xi32, #blocked>]  -> <8x256xi8, #shared2, #smem, mutable>
        %214 = ttg.async_commit_group %213
        %215 = ttg.memdesc_subview %62[%208, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %216 = amdgpu.buffer_load_to_local %202[%27] stride = %arg8 into %215 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %217 = ttg.async_commit_group %216
        %218 = ttg.memdesc_subview %63[%208, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %219 = amdgpu.buffer_load_to_local %203[%35] stride = %arg9 cacheModifier = cg into %218 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %220 = ttg.async_commit_group %219
        %221 = ttg.async_wait %184, %187, %190, %193 {num = 15 : i32}
        %222 = ttg.local_load %182 token %221 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %223 = ttg.local_load %185 token %221 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
        %224 = tt.reshape %222 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %225 = tt.reshape %223 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %226 = ttg.local_load %188 token %221 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %227 = ttg.local_load %191 token %221 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %228 = tt.dot_scaled %226 scale %224, %227 scale %225, %201 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        scf.yield %228, %204, %202, %203, %208, %211, %214, %217, %220, %209, %212, %215, %218, %205 : tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>, !tt.ptr<i8>
      }
      %79:14 = scf.for %arg14 = %c60_i32 to %c63_i32 step %c1_i32 iter_args(%arg15 = %78#0, %arg16 = %78#1, %arg17 = %78#2, %arg18 = %78#3, %arg19 = %78#4, %arg20 = %78#5, %arg21 = %78#6, %arg22 = %78#7, %arg23 = %78#8, %arg24 = %78#9, %arg25 = %78#10, %arg26 = %78#11, %arg27 = %78#12, %arg28 = %78#13) -> (tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>, !tt.ptr<i8>)  : i32 {
        %121 = tt.addptr %arg17, %c128_i32 : !tt.ptr<i8>, i32
        %122 = tt.addptr %arg18, %c128_i32 : !tt.ptr<i8>, i32
        %123 = tt.addptr %arg16, %c256_i32 : !tt.ptr<i8>, i32
        %124 = tt.addptr %arg28, %c256_i32 : !tt.ptr<i8>, i32
        %125 = arith.addi %arg19, %c1_i32 : i32
        %126 = arith.cmpi slt, %125, %c2_i32 : i32
        %127 = arith.select %126, %125, %c0_i32 : i32
        %128 = ttg.memdesc_subview %64[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %129 = amdgpu.buffer_load_to_local %123[%61] stride = %arg12 into %128 : <i8>[tensor<4x256xi32, #blocked>]  -> <4x256xi8, #shared2, #smem, mutable>
        %130 = ttg.async_commit_group %129
        %131 = ttg.memdesc_subview %65[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %132 = amdgpu.buffer_load_to_local %124[%49] stride = %arg13 into %131 : <i8>[tensor<8x256xi32, #blocked>]  -> <8x256xi8, #shared2, #smem, mutable>
        %133 = ttg.async_commit_group %132
        %134 = ttg.memdesc_subview %62[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %135 = amdgpu.buffer_load_to_local %121[%27] stride = %arg8 into %134 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %136 = ttg.async_commit_group %135
        %137 = ttg.memdesc_subview %63[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %138 = amdgpu.buffer_load_to_local %122[%35] stride = %arg9 cacheModifier = cg into %137 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %139 = ttg.async_commit_group %138
        %140 = ttg.async_wait %arg20, %arg21, %arg22, %arg23 {num = 15 : i32}
        %141 = ttg.local_load %arg24 token %140 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %142 = ttg.local_load %arg25 token %140 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
        %143 = tt.reshape %141 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %144 = tt.reshape %142 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %145 = ttg.local_load %arg26 token %140 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %146 = ttg.local_load %arg27 token %140 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %147 = tt.dot_scaled %145 scale %143, %146 scale %144, %arg15 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        scf.yield %147, %123, %121, %122, %127, %130, %133, %136, %139, %128, %131, %134, %137, %124 : tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>, !tt.ptr<i8>
      }
      %80 = ttg.async_wait %79#5, %79#6, %79#7, %79#8 {num = 0 : i32}
      %81 = ttg.local_load %79#9 token %80 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
      %82 = ttg.local_load %79#10 token %80 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
      %83 = tt.reshape %81 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
      %84 = tt.reshape %82 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
      %85 = ttg.local_load %79#11 token %80 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %86 = ttg.local_load %79#12 token %80 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %87 = tt.dot_scaled %85 scale %83, %86 scale %84, %79#0 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
      ttg.local_dealloc %62 : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable>
      ttg.local_dealloc %63 : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable>
      ttg.local_dealloc %64 : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable>
      ttg.local_dealloc %65 : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable>
      %88 = arith.truncf %87 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
      %89 = arith.extsi %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %90 = arith.extsi %8 : i32 to i64
      %91 = tt.splat %90 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %92 = arith.addi %91, %89 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %93 = arith.extsi %15 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>> to tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %94 = arith.extsi %13 : i32 to i64
      %95 = tt.splat %94 : i64 -> tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %96 = arith.addi %95, %93 : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %97 = tt.expand_dims %92 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi64, #mma>
      %98 = arith.extsi %arg11 : i32 to i64
      %99 = tt.expand_dims %89 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi64, #mma>
      %100 = arith.muli %98, %90 : i64
      %101 = tt.splat %98 : i64 -> tensor<128x1xi64, #mma>
      %102 = arith.muli %101, %99 : tensor<128x1xi64, #mma>
      %103 = tt.addptr %arg2, %100 : !tt.ptr<bf16>, i64
      %104 = arith.trunci %102 : tensor<128x1xi64, #mma> to tensor<128x1xi32, #mma>
      %105 = tt.expand_dims %96 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi64, #mma>
      %106 = tt.broadcast %104 : tensor<128x1xi32, #mma> -> tensor<128x256xi32, #mma>
      %107 = tt.expand_dims %93 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi64, #mma>
      %108 = tt.broadcast %107 : tensor<1x256xi64, #mma> -> tensor<128x256xi64, #mma>
      %109 = tt.addptr %103, %94 : !tt.ptr<bf16>, i64
      %110 = arith.trunci %108 : tensor<128x256xi64, #mma> to tensor<128x256xi32, #mma>
      %111 = arith.addi %110, %106 : tensor<128x256xi32, #mma>
      %112 = arith.extsi %arg5 : i32 to i64
      %113 = tt.splat %112 : i64 -> tensor<128x1xi64, #mma>
      %114 = arith.cmpi slt, %97, %113 : tensor<128x1xi64, #mma>
      %115 = arith.extsi %arg6 : i32 to i64
      %116 = tt.splat %115 : i64 -> tensor<1x256xi64, #mma>
      %117 = arith.cmpi slt, %105, %116 : tensor<1x256xi64, #mma>
      %118 = tt.broadcast %114 : tensor<128x1xi1, #mma> -> tensor<128x256xi1, #mma>
      %119 = tt.broadcast %117 : tensor<1x256xi1, #mma> -> tensor<128x256xi1, #mma>
      %120 = arith.andi %118, %119 : tensor<128x256xi1, #mma>
      amdgpu.buffer_store %88, %109[%111], %120 : tensor<128x256xbf16, #mma>
    }
    tt.return
  }
}
