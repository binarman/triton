#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
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
    %c60_i32 = arith.constant 60 : i32
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
      %62 = arith.muli %4, %c8_i32 : i32
      %63 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %64 = tt.splat %62 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %65 = arith.addi %64, %63 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %66 = tt.splat %arg6 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %67 = arith.remsi %65, %66 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %68 = tt.expand_dims %67 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<8x1xi32, #blocked3>
      %69 = tt.splat %arg13 : i32 -> tensor<8x1xi32, #blocked3>
      %70 = arith.muli %68, %69 : tensor<8x1xi32, #blocked3>
      %71 = arith.muli %3, %c4_i32 : i32
      %72 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %73 = tt.splat %71 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %74 = arith.addi %73, %72 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %75 = tt.splat %arg5 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %76 = arith.remsi %74, %75 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %77 = tt.expand_dims %76 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<4x1xi32, #blocked3>
      %78 = tt.splat %arg12 : i32 -> tensor<4x1xi32, #blocked3>
      %79 = arith.muli %77, %78 : tensor<4x1xi32, #blocked3>
      %80 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
      %81 = tt.broadcast %79 : tensor<4x1xi32, #blocked3> -> tensor<4x1024xi32, #blocked3>
      %82 = tt.expand_dims %80 {axis = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x1024xi32, #blocked3>
      %83 = tt.broadcast %82 : tensor<1x1024xi32, #blocked3> -> tensor<4x1024xi32, #blocked3>
      %84 = arith.addi %83, %81 : tensor<4x1024xi32, #blocked3>
      %85 = tt.broadcast %70 : tensor<8x1xi32, #blocked3> -> tensor<8x1024xi32, #blocked3>
      %86 = tt.broadcast %82 : tensor<1x1024xi32, #blocked3> -> tensor<8x1024xi32, #blocked3>
      %87 = arith.addi %86, %85 : tensor<8x1024xi32, #blocked3>
      %88 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable>
      %89 = ttg.local_alloc : () -> !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable>
      %90 = ttg.memdesc_subview %88[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
      %91 = amdgpu.buffer_load_to_local %arg0[%27] mask = %cst_2 stride = %arg8 into %90 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
      %92 = ttg.async_commit_group %91
      %93 = ttg.memdesc_subview %89[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
      %94 = amdgpu.buffer_load_to_local %arg1[%35] mask = %cst_1 stride = %arg9 cacheModifier = cg into %93 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
      %95 = ttg.async_commit_group %94
      %96:10 = scf.for %arg14 = %c0_i32 to %c60_i32 step %c4_i32 iter_args(%arg15 = %cst, %arg16 = %arg3, %arg17 = %arg0, %arg18 = %arg1, %arg19 = %c0_i32, %arg20 = %92, %arg21 = %95, %arg22 = %90, %arg23 = %93, %arg24 = %arg4) -> (tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.async.token, !ttg.async.token, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>, !tt.ptr<i8>)  : i32 {
        %147 = arith.muli %arg14, %c256_i32 : i32
        %148 = tt.addptr %arg3, %147 : !tt.ptr<i8>, i32
        %149 = tt.addptr %arg4, %147 : !tt.ptr<i8>, i32
        %150 = amdgpu.buffer_load %148[%84] stride = %arg12 : tensor<4x1024xi8, #blocked3>
        %151 = amdgpu.buffer_load %149[%87] stride = %arg13 : tensor<8x1024xi8, #blocked3>
        %152 = ttg.convert_layout %150 : tensor<4x1024xi8, #blocked3> -> tensor<4x1024xi8, #linear>
        %153 = ttg.convert_layout %151 : tensor<8x1024xi8, #blocked3> -> tensor<8x1024xi8, #linear1>
        %154 = tt.addptr %arg17, %c128_i32 : !tt.ptr<i8>, i32
        %155 = tt.addptr %arg18, %c128_i32 : !tt.ptr<i8>, i32
        %156 = tt.addptr %arg16, %c256_i32 : !tt.ptr<i8>, i32
        %157 = tt.addptr %arg24, %c256_i32 : !tt.ptr<i8>, i32
        %158 = arith.addi %arg19, %c1_i32 : i32
        %159 = arith.cmpi slt, %158, %c2_i32 : i32
        %160 = arith.select %159, %158, %c0_i32 : i32
        %161 = ttg.memdesc_subview %88[%160, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %162 = amdgpu.buffer_load_to_local %154[%27] stride = %arg8 into %161 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %163 = ttg.async_commit_group %162
        %164 = ttg.memdesc_subview %89[%160, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %165 = amdgpu.buffer_load_to_local %155[%35] stride = %arg9 cacheModifier = cg into %164 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %166 = ttg.async_commit_group %165
        %167 = ttg.async_wait %arg20, %arg21 {num = 12 : i32}
        %168 = amdgpu.extract_slice %152 [0, 0] : tensor<4x1024xi8, #linear> to tensor<4x256xi8, #linear>
        %169 = amdgpu.extract_slice %153 [0, 0] : tensor<8x1024xi8, #linear1> to tensor<8x256xi8, #linear1>
        %170 = tt.reshape %168 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %171 = tt.reshape %169 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %172 = ttg.local_load %arg22 token %167 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %173 = ttg.local_load %arg23 token %167 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %174 = tt.dot_scaled %172 scale %170, %173 scale %171, %arg15 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %175 = tt.addptr %154, %c128_i32 : !tt.ptr<i8>, i32
        %176 = tt.addptr %155, %c128_i32 : !tt.ptr<i8>, i32
        %177 = tt.addptr %156, %c256_i32 : !tt.ptr<i8>, i32
        %178 = tt.addptr %157, %c256_i32 : !tt.ptr<i8>, i32
        %179 = arith.addi %160, %c1_i32 : i32
        %180 = arith.cmpi slt, %179, %c2_i32 : i32
        %181 = arith.select %180, %179, %c0_i32 : i32
        %182 = ttg.memdesc_subview %88[%181, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %183 = amdgpu.buffer_load_to_local %175[%27] stride = %arg8 into %182 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %184 = ttg.async_commit_group %183
        %185 = ttg.memdesc_subview %89[%181, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %186 = amdgpu.buffer_load_to_local %176[%35] stride = %arg9 cacheModifier = cg into %185 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %187 = ttg.async_commit_group %186
        %188 = ttg.async_wait %163, %166 {num = 12 : i32}
        %189 = amdgpu.extract_slice %152 [0, 256] : tensor<4x1024xi8, #linear> to tensor<4x256xi8, #linear>
        %190 = amdgpu.extract_slice %153 [0, 256] : tensor<8x1024xi8, #linear1> to tensor<8x256xi8, #linear1>
        %191 = tt.reshape %189 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %192 = tt.reshape %190 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %193 = ttg.local_load %161 token %188 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %194 = ttg.local_load %164 token %188 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %195 = tt.dot_scaled %193 scale %191, %194 scale %192, %174 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %196 = tt.addptr %175, %c128_i32 : !tt.ptr<i8>, i32
        %197 = tt.addptr %176, %c128_i32 : !tt.ptr<i8>, i32
        %198 = tt.addptr %177, %c256_i32 : !tt.ptr<i8>, i32
        %199 = tt.addptr %178, %c256_i32 : !tt.ptr<i8>, i32
        %200 = arith.addi %181, %c1_i32 : i32
        %201 = arith.cmpi slt, %200, %c2_i32 : i32
        %202 = arith.select %201, %200, %c0_i32 : i32
        %203 = ttg.memdesc_subview %88[%202, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %204 = amdgpu.buffer_load_to_local %196[%27] stride = %arg8 into %203 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %205 = ttg.async_commit_group %204
        %206 = ttg.memdesc_subview %89[%202, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %207 = amdgpu.buffer_load_to_local %197[%35] stride = %arg9 cacheModifier = cg into %206 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %208 = ttg.async_commit_group %207
        %209 = ttg.async_wait %184, %187 {num = 12 : i32}
        %210 = amdgpu.extract_slice %152 [0, 512] : tensor<4x1024xi8, #linear> to tensor<4x256xi8, #linear>
        %211 = amdgpu.extract_slice %153 [0, 512] : tensor<8x1024xi8, #linear1> to tensor<8x256xi8, #linear1>
        %212 = tt.reshape %210 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %213 = tt.reshape %211 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %214 = ttg.local_load %182 token %209 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %215 = ttg.local_load %185 token %209 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %216 = tt.dot_scaled %214 scale %212, %215 scale %213, %195 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        %217 = tt.addptr %196, %c128_i32 : !tt.ptr<i8>, i32
        %218 = tt.addptr %197, %c128_i32 : !tt.ptr<i8>, i32
        %219 = tt.addptr %198, %c256_i32 : !tt.ptr<i8>, i32
        %220 = tt.addptr %199, %c256_i32 : !tt.ptr<i8>, i32
        %221 = arith.addi %202, %c1_i32 : i32
        %222 = arith.cmpi slt, %221, %c2_i32 : i32
        %223 = arith.select %222, %221, %c0_i32 : i32
        %224 = ttg.memdesc_subview %88[%223, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %225 = amdgpu.buffer_load_to_local %217[%27] stride = %arg8 into %224 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %226 = ttg.async_commit_group %225
        %227 = ttg.memdesc_subview %89[%223, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %228 = amdgpu.buffer_load_to_local %218[%35] stride = %arg9 cacheModifier = cg into %227 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %229 = ttg.async_commit_group %228
        %230 = ttg.async_wait %205, %208 {num = 12 : i32}
        %231 = amdgpu.extract_slice %152 [0, 768] : tensor<4x1024xi8, #linear> to tensor<4x256xi8, #linear>
        %232 = amdgpu.extract_slice %153 [0, 768] : tensor<8x1024xi8, #linear1> to tensor<8x256xi8, #linear1>
        %233 = tt.reshape %231 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %234 = tt.reshape %232 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %235 = ttg.local_load %203 token %230 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %236 = ttg.local_load %206 token %230 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %237 = tt.dot_scaled %235 scale %233, %236 scale %234, %216 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        scf.yield %237, %219, %217, %218, %223, %226, %229, %224, %227, %220 : tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.async.token, !ttg.async.token, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>, !tt.ptr<i8>
      }
      %97 = ttg.local_alloc : () -> !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable>
      %98 = ttg.local_alloc : () -> !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable>
      %99 = ttg.memdesc_subview %97[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
      %100 = amdgpu.buffer_load_to_local %arg3[%61] mask = %cst_3 stride = %arg12 into %99 : <i8>[tensor<4x256xi32, #blocked>]  -> <4x256xi8, #shared2, #smem, mutable>
      %101 = ttg.async_commit_group %100
      %102 = ttg.memdesc_subview %98[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
      %103 = amdgpu.buffer_load_to_local %arg4[%49] mask = %cst_0 stride = %arg13 into %102 : <i8>[tensor<8x256xi32, #blocked>]  -> <8x256xi8, #shared2, #smem, mutable>
      %104 = ttg.async_commit_group %103
      %105:14 = scf.for %arg14 = %c60_i32 to %c63_i32 step %c1_i32 iter_args(%arg15 = %96#0, %arg16 = %96#1, %arg17 = %96#2, %arg18 = %96#3, %arg19 = %96#4, %arg20 = %101, %arg21 = %104, %arg22 = %96#5, %arg23 = %96#6, %arg24 = %99, %arg25 = %102, %arg26 = %96#7, %arg27 = %96#8, %arg28 = %96#9) -> (tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>, !tt.ptr<i8>)  : i32 {
        %147 = tt.addptr %arg17, %c128_i32 : !tt.ptr<i8>, i32
        %148 = tt.addptr %arg18, %c128_i32 : !tt.ptr<i8>, i32
        %149 = tt.addptr %arg16, %c256_i32 : !tt.ptr<i8>, i32
        %150 = tt.addptr %arg28, %c256_i32 : !tt.ptr<i8>, i32
        %151 = arith.addi %arg19, %c1_i32 : i32
        %152 = arith.cmpi slt, %151, %c2_i32 : i32
        %153 = arith.select %152, %151, %c0_i32 : i32
        %154 = ttg.memdesc_subview %97[%153, %c0_i32, %c0_i32] : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>
        %155 = amdgpu.buffer_load_to_local %149[%61] stride = %arg12 into %154 : <i8>[tensor<4x256xi32, #blocked>]  -> <4x256xi8, #shared2, #smem, mutable>
        %156 = ttg.async_commit_group %155
        %157 = ttg.memdesc_subview %98[%153, %c0_i32, %c0_i32] : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>
        %158 = amdgpu.buffer_load_to_local %150[%49] stride = %arg13 into %157 : <i8>[tensor<8x256xi32, #blocked>]  -> <8x256xi8, #shared2, #smem, mutable>
        %159 = ttg.async_commit_group %158
        %160 = ttg.memdesc_subview %88[%153, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared, #smem, mutable>
        %161 = amdgpu.buffer_load_to_local %147[%27] stride = %arg8 into %160 : <i8>[tensor<128x128xi32, #blocked2>]  -> <128x128xi8, #shared, #smem, mutable>
        %162 = ttg.async_commit_group %161
        %163 = ttg.memdesc_subview %89[%153, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
        %164 = amdgpu.buffer_load_to_local %148[%35] stride = %arg9 cacheModifier = cg into %163 : <i8>[tensor<128x256xi32, #blocked1>]  -> <128x256xi8, #shared1, #smem, mutable>
        %165 = ttg.async_commit_group %164
        %166 = ttg.async_wait %arg20, %arg21, %arg22, %arg23 {num = 15 : i32}
        %167 = ttg.local_load %arg24 token %166 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
        %168 = ttg.local_load %arg25 token %166 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
        %169 = tt.reshape %167 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
        %170 = tt.reshape %168 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
        %171 = ttg.local_load %arg26 token %166 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
        %172 = ttg.local_load %arg27 token %166 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
        %173 = tt.dot_scaled %171 scale %169, %172 scale %170, %arg15 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
        scf.yield %173, %149, %147, %148, %153, %156, %159, %162, %165, %154, %157, %160, %163, %150 : tensor<128x256xf32, #mma>, !tt.ptr<i8>, !tt.ptr<i8>, !tt.ptr<i8>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.memdesc<4x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<8x256xi8, #shared2, #smem, mutable>, !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>, !tt.ptr<i8>
      }
      %106 = ttg.async_wait %105#5, %105#6, %105#7, %105#8 {num = 0 : i32}
      %107 = ttg.local_load %105#9 token %106 : !ttg.memdesc<4x256xi8, #shared2, #smem, mutable> -> tensor<4x256xi8, #linear>
      %108 = ttg.local_load %105#10 token %106 : !ttg.memdesc<8x256xi8, #shared2, #smem, mutable> -> tensor<8x256xi8, #linear1>
      %109 = tt.reshape %107 : tensor<4x256xi8, #linear> -> tensor<128x8xi8, #linear2>
      %110 = tt.reshape %108 : tensor<8x256xi8, #linear1> -> tensor<256x8xi8, #linear3>
      %111 = ttg.local_load %105#11 token %106 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %112 = ttg.local_load %105#12 token %106 : !ttg.memdesc<128x256xi8, #shared1, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %113 = tt.dot_scaled %111 scale %109, %112 scale %110, %105#0 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear2> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear3> -> tensor<128x256xf32, #mma>
      ttg.local_dealloc %88 : !ttg.memdesc<2x128x128xi8, #shared, #smem, mutable>
      ttg.local_dealloc %89 : !ttg.memdesc<2x128x256xi8, #shared1, #smem, mutable>
      ttg.local_dealloc %97 : !ttg.memdesc<2x4x256xi8, #shared2, #smem, mutable>
      ttg.local_dealloc %98 : !ttg.memdesc<2x8x256xi8, #shared2, #smem, mutable>
      %114 = arith.truncf %113 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
      %115 = arith.extsi %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %116 = arith.extsi %8 : i32 to i64
      %117 = tt.splat %116 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %118 = arith.addi %117, %115 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>>
      %119 = arith.extsi %15 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>> to tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %120 = arith.extsi %13 : i32 to i64
      %121 = tt.splat %120 : i64 -> tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %122 = arith.addi %121, %119 : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>
      %123 = tt.expand_dims %118 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi64, #mma>
      %124 = arith.extsi %arg11 : i32 to i64
      %125 = tt.expand_dims %115 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi64, #mma>
      %126 = arith.muli %124, %116 : i64
      %127 = tt.splat %124 : i64 -> tensor<128x1xi64, #mma>
      %128 = arith.muli %127, %125 : tensor<128x1xi64, #mma>
      %129 = tt.addptr %arg2, %126 : !tt.ptr<bf16>, i64
      %130 = arith.trunci %128 : tensor<128x1xi64, #mma> to tensor<128x1xi32, #mma>
      %131 = tt.expand_dims %122 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi64, #mma>
      %132 = tt.broadcast %130 : tensor<128x1xi32, #mma> -> tensor<128x256xi32, #mma>
      %133 = tt.expand_dims %119 {axis = 0 : i32} : tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi64, #mma>
      %134 = tt.broadcast %133 : tensor<1x256xi64, #mma> -> tensor<128x256xi64, #mma>
      %135 = tt.addptr %129, %120 : !tt.ptr<bf16>, i64
      %136 = arith.trunci %134 : tensor<128x256xi64, #mma> to tensor<128x256xi32, #mma>
      %137 = arith.addi %136, %132 : tensor<128x256xi32, #mma>
      %138 = arith.extsi %arg5 : i32 to i64
      %139 = tt.splat %138 : i64 -> tensor<128x1xi64, #mma>
      %140 = arith.cmpi slt, %123, %139 : tensor<128x1xi64, #mma>
      %141 = arith.extsi %arg6 : i32 to i64
      %142 = tt.splat %141 : i64 -> tensor<1x256xi64, #mma>
      %143 = arith.cmpi slt, %131, %142 : tensor<1x256xi64, #mma>
      %144 = tt.broadcast %140 : tensor<128x1xi1, #mma> -> tensor<128x256xi1, #mma>
      %145 = tt.broadcast %143 : tensor<1x256xi1, #mma> -> tensor<128x256xi1, #mma>
      %146 = arith.andi %144, %145 : tensor<128x256xi1, #mma>
      amdgpu.buffer_store %114, %135[%137], %146 : tensor<128x256xbf16, #mma>
    }
    tt.return
  }
}
