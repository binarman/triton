// -----// IR Dump Before TritonAMDGPUSimplifyConvertLayout (tritonamdgpu-simplify-convert-layout) ('tt.func' operation: @_matmul_ogs_NNT_bf16xbf16xfp8e4nv_128x512x128x1) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 8], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [1, 0, 0], [2, 0, 0], [0, 128, 0], [0, 256, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 8], [0, 0, 16]], warp = [[0, 16, 0], [0, 32, 0], [0, 64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [128, 0], [256, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [1, 0, 0], [2, 0, 0], [0, 0, 128], [0, 0, 256]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 8, 0], [0, 16, 0]], warp = [[0, 0, 16], [0, 0, 32], [0, 0, 64]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 8], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_matmul_ogs_NNT_bf16xbf16xfp8e4nv_128x512x128x1(%Y: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %YPtr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_y_k: i32 {tt.divisibility = 16 : i32}, %stride_y_z: i32 {tt.divisibility = 16 : i32}, %stride_y_m: i32 {tt.divisibility = 16 : i32}, %X: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %XPtr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_x_z: i32 {tt.divisibility = 16 : i32}, %stride_x_m: i32 {tt.divisibility = 16 : i32}, %W: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %WPtr: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_w_e: i32 {tt.divisibility = 16 : i32}, %stride_w_n: i32 {tt.divisibility = 16 : i32}, %WMxScale: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %stride_w_mx_e: i32 {tt.divisibility = 16 : i32}, %stride_w_mx_n: i32, %B: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_b_e: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %K_W: i32 {tt.divisibility = 16 : i32}, %GatherIndx: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ScatterSrcIndx: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %num_idxs: i32 {tt.divisibility = 16 : i32}, %ExptHist: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptOffs: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptTileOffs: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptData: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %grid_m: i32, %reduce_rank: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x512xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x512xf8E4M3FN, #blocked1>
    %cst_2 = arith.constant dense<4> : tensor<512x4xi32, #blocked2>
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant dense<4> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_4 = arith.constant dense<32> : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %cst_5 = arith.constant dense<128> : tensor<128x128xi32, #blocked>
    %cst_6 = arith.constant dense<128> : tensor<128x512xi32, #blocked1>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked3>
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c16_i32 = arith.constant 16 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cst_8 = arith.constant dense<7> : tensor<4x512xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %0 = arith.cmpi sge, %stride_y_k, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    %1 = arith.cmpi sge, %stride_y_z, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sge, %stride_y_m, %c0_i32 : i32
    llvm.intr.assume %2 : i1
    llvm.intr.assume %true : i1
    %3 = arith.cmpi sge, %stride_x_z, %c0_i32 : i32
    llvm.intr.assume %3 : i1
    %4 = arith.cmpi sge, %stride_x_m, %c0_i32 : i32
    llvm.intr.assume %4 : i1
    llvm.intr.assume %true : i1
    %5 = arith.cmpi sge, %stride_w_e, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    llvm.intr.assume %true : i1
    %6 = arith.cmpi sge, %stride_w_n, %c0_i32 : i32
    llvm.intr.assume %6 : i1
    %7 = arith.cmpi sge, %stride_w_mx_e, %c0_i32 : i32
    llvm.intr.assume %7 : i1
    llvm.intr.assume %true : i1
    %8 = arith.cmpi sge, %stride_w_mx_n, %c0_i32 : i32
    llvm.intr.assume %8 : i1
    %9 = arith.cmpi sge, %stride_b_e, %c0_i32 : i32
    llvm.intr.assume %9 : i1
    llvm.intr.assume %true : i1
    %10 = arith.cmpi sge, %grid_m, %c0_i32 : i32
    llvm.intr.assume %10 : i1
    llvm.intr.assume %true : i1
    %pid = tt.get_program_id x : i32
    %padding_m = tt.addptr %ExptTileOffs, %c8_i32 : !tt.ptr<i32>, i32
    %padding_m_9 = tt.load %padding_m : !tt.ptr<i32>
    %padding_m_10 = arith.subi %grid_m, %padding_m_9 : i32
    %unpadded_m = arith.subi %grid_m, %padding_m_10 : i32
    %11 = arith.cmpi sge, %unpadded_m, %c0_i32 : i32
    llvm.intr.assume %11 : i1
    %12 = arith.cmpi sgt, %padding_m_10, %c0_i32 : i32
    %13 = arith.cmpi sge, %pid, %unpadded_m : i32
    %14 = arith.andi %12, %13 : i1
    cf.cond_br %14, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %pids_per_group = arith.divsi %unpadded_m, %c8_i32 : i32
    %extra_pid_groups = arith.remsi %unpadded_m, %c8_i32 : i32
    %group = arith.remsi %pid, %c8_i32 : i32
    %local_pid = arith.divsi %pid, %c8_i32 : i32
    %new_pid = arith.muli %group, %pids_per_group : i32
    %new_pid_11 = arith.minsi %group, %extra_pid_groups : i32
    %new_pid_12 = arith.addi %new_pid, %new_pid_11 : i32
    %new_pid_13 = arith.addi %new_pid_12, %local_pid : i32
    %pid_mnk = arith.remsi %new_pid_13, %unpadded_m : i32
    %group_id = arith.divsi %pid_mnk, %c4_i32 : i32
    %group_size = arith.muli %group_id, %c4_i32 : i32
    %group_size_14 = arith.subi %unpadded_m, %group_size : i32
    %group_size_15 = arith.minsi %group_size_14, %c4_i32 : i32
    %15 = arith.cmpi sge, %group_size_15, %c0_i32 : i32
    llvm.intr.assume %15 : i1
    %pid_m = arith.remsi %pid_mnk, %group_size_15 : i32
    %pid_m_16 = arith.addi %group_size, %pid_m : i32
    %pid_n = arith.remsi %pid_mnk, %c4_i32 : i32
    %pid_n_17 = arith.divsi %pid_n, %group_size_15 : i32
    %k_tiles = arith.addi %K, %c127_i32 : i32
    %k_tiles_18 = arith.divsi %k_tiles, %c128_i32 : i32
    %expt_data = tt.addptr %ExptData, %pid_m_16 : !tt.ptr<i32>, i32
    %expt_data_19 = tt.load %expt_data : !tt.ptr<i32>
    %expt_id = arith.andi %expt_data_19, %c65535_i32 : i32
    %block_id = arith.shrsi %expt_data_19, %c16_i32 : i32
    %eM = tt.addptr %ExptHist, %expt_id : !tt.ptr<i32>, i32
    %eM_20 = tt.load %eM : !tt.ptr<i32>
    %start_m = tt.addptr %ExptOffs, %expt_id : !tt.ptr<i32>, i32
    %start_m_21 = tt.load %start_m : !tt.ptr<i32>
    %off_m = arith.muli %block_id, %c128_i32 : i32
    %offs_x_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_x_m_22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_x_m_24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_x_m_25 = tt.splat %off_m : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_x_m_26 = tt.splat %off_m : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_x_m_27 = arith.addi %offs_x_m_25, %offs_x_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_x_m_28 = arith.addi %offs_x_m_26, %offs_x_m_23 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_x_m_29 = tt.splat %eM_20 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_x_m_30 = tt.splat %eM_20 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_x_m_31 = arith.remsi %offs_x_m_27, %offs_x_m_29 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %GatherIndx_32 = tt.addptr %GatherIndx, %start_m_21 : !tt.ptr<i32>, i32
    %offs_x_m_33 = tt.splat %GatherIndx_32 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_x_m_34 = tt.addptr %offs_x_m_33, %offs_x_m_31 : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_x_m_35 = tt.load %offs_x_m_34 : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_x_m_36 = arith.divsi %offs_x_m_35, %cst_3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %XPtrs = tt.expand_dims %offs_x_m_36 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %XPtrs_37 = tt.splat %stride_x_m : i32 -> tensor<128x1xi32, #blocked>
    %XPtrs_38 = arith.muli %XPtrs, %XPtrs_37 : tensor<128x1xi32, #blocked>
    %XPtrs_39 = tt.splat %X : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>, #blocked>
    %XPtrs_40 = tt.addptr %XPtrs_39, %XPtrs_38 : tensor<128x1x!tt.ptr<bf16>, #blocked>, tensor<128x1xi32, #blocked>
    %XPtrs_41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %XPtrs_42 = tt.expand_dims %XPtrs_41 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %XPtrs_43 = tt.broadcast %XPtrs_40 : tensor<128x1x!tt.ptr<bf16>, #blocked> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %XPtrs_44 = tt.broadcast %XPtrs_42 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %XPtrs_45 = tt.addptr %XPtrs_43, %XPtrs_44 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
    %WMxScale_46 = arith.muli %expt_id, %stride_w_mx_e : i32
    %WMxScale_47 = tt.addptr %WMxScale, %WMxScale_46 : !tt.ptr<i8>, i32
    %offs_n_scale = arith.muli %pid_n_17, %c512_i32 : i32
    %offs_n_scale_48 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_49 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_n_scale_50 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %offs_n_scale_51 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked3>
    %offs_n_scale_52 = tt.splat %offs_n_scale : i32 -> tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_53 = tt.splat %offs_n_scale : i32 -> tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_n_scale_54 = tt.splat %offs_n_scale : i32 -> tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %offs_n_scale_55 = tt.splat %offs_n_scale : i32 -> tensor<512xi32, #blocked3>
    %offs_n_scale_56 = arith.addi %offs_n_scale_52, %offs_n_scale_48 : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_57 = arith.addi %offs_n_scale_53, %offs_n_scale_49 : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_n_scale_58 = arith.addi %offs_n_scale_54, %offs_n_scale_50 : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %offs_n_scale_59 = arith.addi %offs_n_scale_55, %offs_n_scale_51 : tensor<512xi32, #blocked3>
    %offs_n_scale_60 = tt.splat %N : i32 -> tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_61 = tt.splat %N : i32 -> tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_n_scale_62 = tt.splat %N : i32 -> tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %offs_n_scale_63 = tt.splat %N : i32 -> tensor<512xi32, #blocked3>
    %offs_n_scale_64 = arith.remsi %offs_n_scale_56, %offs_n_scale_60 {tt.contiguity = dense<512> : tensor<1xi32>, tt.divisibility = dense<512> : tensor<1xi32>} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_65 = arith.remsi %offs_n_scale_57, %offs_n_scale_61 {tt.contiguity = dense<512> : tensor<1xi32>, tt.divisibility = dense<512> : tensor<1xi32>} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_k_scale = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %WMxScalePtrs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %WMxScalePtrs_66 = tt.expand_dims %WMxScalePtrs {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x4xi32, #blocked2>
    %WMxScalePtrs_67 = tt.splat %WMxScale_47 : !tt.ptr<i8> -> tensor<1x4x!tt.ptr<i8>, #blocked2>
    %WMxScalePtrs_68 = tt.addptr %WMxScalePtrs_67, %WMxScalePtrs_66 : tensor<1x4x!tt.ptr<i8>, #blocked2>, tensor<1x4xi32, #blocked2>
    %WMxScalePtrs_69 = tt.expand_dims %offs_n_scale_64 {axis = 1 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<512x1xi32, #blocked2>
    %WMxScalePtrs_70 = tt.splat %stride_w_mx_n : i32 -> tensor<512x1xi32, #blocked2>
    %WMxScalePtrs_71 = arith.muli %WMxScalePtrs_69, %WMxScalePtrs_70 : tensor<512x1xi32, #blocked2>
    %WMxScalePtrs_72 = tt.broadcast %WMxScalePtrs_68 : tensor<1x4x!tt.ptr<i8>, #blocked2> -> tensor<512x4x!tt.ptr<i8>, #blocked2>
    %WMxScalePtrs_73 = tt.broadcast %WMxScalePtrs_71 : tensor<512x1xi32, #blocked2> -> tensor<512x4xi32, #blocked2>
    %WMxScalePtrs_74 = tt.addptr %WMxScalePtrs_72, %WMxScalePtrs_73 : tensor<512x4x!tt.ptr<i8>, #blocked2>, tensor<512x4xi32, #blocked2>
    %W_75 = arith.muli %expt_id, %stride_w_e : i32
    %W_76 = tt.addptr %W, %W_75 : !tt.ptr<f8E4M3FN>, i32
    %WPtrs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %WPtrs_77 = tt.expand_dims %WPtrs {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %WPtrs_78 = tt.expand_dims %offs_n_scale_65 {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x512xi32, #blocked1>
    %WPtrs_79 = tt.splat %stride_w_n : i32 -> tensor<1x512xi32, #blocked1>
    %WPtrs_80 = arith.muli %WPtrs_78, %WPtrs_79 : tensor<1x512xi32, #blocked1>
    %WPtrs_81 = tt.broadcast %WPtrs_77 : tensor<128x1xi32, #blocked1> -> tensor<128x512xi32, #blocked1>
    %WPtrs_82 = tt.broadcast %WPtrs_80 : tensor<1x512xi32, #blocked1> -> tensor<128x512xi32, #blocked1>
    %WPtrs_83 = arith.addi %WPtrs_81, %WPtrs_82 : tensor<128x512xi32, #blocked1>
    %WPtrs_84 = tt.splat %W_76 : !tt.ptr<f8E4M3FN> -> tensor<128x512x!tt.ptr<f8E4M3FN>, #blocked1>
    %WPtrs_85 = tt.addptr %WPtrs_84, %WPtrs_83 : tensor<128x512x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<128x512xi32, #blocked1>
    %x_k_limit = arith.addi %K, %c128_i32 : i32
    %mask_k_scale = arith.muli %offs_k_scale, %cst_4 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %w_k_limit:6 = scf.for %w_k_limit_110 = %c0_i32 to %k_tiles_18 step %c1_i32 iter_args(%WMxScalePtrs_111 = %WMxScalePtrs_74, %arg32 = %cst, %x_k_limit_112 = %x_k_limit, %x_k_limit_113 = %x_k_limit, %XPtrs_114 = %XPtrs_45, %WPtrs_115 = %WPtrs_85) -> (tensor<512x4x!tt.ptr<i8>, #blocked2>, tensor<128x512xf32, #mma>, i32, i32, tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x512x!tt.ptr<f8E4M3FN>, #blocked1>)  : i32 {
      %x_k_limit_116 = arith.subi %x_k_limit_112, %c128_i32 : i32
      %w_k_limit_117 = arith.subi %x_k_limit_113, %c128_i32 : i32
      %mask_k = tt.splat %x_k_limit_116 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %mask_k_118 = arith.cmpi slt, %offs_x_m_24, %mask_k : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %mask_k_w = tt.splat %w_k_limit_117 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %mask_k_w_119 = arith.cmpi slt, %offs_x_m_22, %mask_k_w : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %mask_k_scale_120 = tt.splat %x_k_limit_116 : i32 -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %mask_k_scale_121 = arith.cmpi slt, %mask_k_scale, %mask_k_scale_120 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %x = tt.expand_dims %mask_k_118 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi1, #blocked>
      %x_122 = tt.broadcast %x : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %x_123 = tt.load %XPtrs_114, %x_122, %cst_0 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %w = tt.expand_dims %mask_k_w_119 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi1, #blocked1>
      %w_124 = tt.broadcast %w : tensor<128x1xi1, #blocked1> -> tensor<128x512xi1, #blocked1>
      %w_125 = tt.load %WPtrs_115, %w_124, %cst_1 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x512x!tt.ptr<f8E4M3FN>, #blocked1>
      %w_126 = ttg.convert_layout %w_125 : tensor<128x512xf8E4M3FN, #blocked1> -> tensor<128x512xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %w_scales = tt.expand_dims %mask_k_scale_121 {axis = 0 : i32} : tensor<4xi1, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x4xi1, #blocked2>
      %w_scales_127 = tt.broadcast %w_scales : tensor<1x4xi1, #blocked2> -> tensor<512x4xi1, #blocked2>
      %w_scales_128 = tt.load %WMxScalePtrs_111, %w_scales_127 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<512x4x!tt.ptr<i8>, #blocked2>
      %w_scales_129 = ttg.convert_layout %w_scales_128 : tensor<512x4xi8, #blocked2> -> tensor<512x4xi8, #linear1>
      %w_scales_130 = tt.trans %w_scales_129 {order = array<i32: 1, 0>} : tensor<512x4xi8, #linear1> -> tensor<4x512xi8, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_scales_131 = arith.extui %w_scales_130 : tensor<4x512xi8, #ttg.slice<{dim = 2, parent = #linear}>> to tensor<4x512xi16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_scales_132 = arith.shli %w_scales_131, %cst_8 : tensor<4x512xi16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_scales_133 = tt.bitcast %w_scales_132 : tensor<4x512xi16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x512xbf16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_scales_134 = tt.expand_dims %w_scales_133 {axis = 2 : i32} : tensor<4x512xbf16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x512x1xbf16, #linear>
      %w_scales_135 = tt.broadcast %w_scales_134 : tensor<4x512x1xbf16, #linear> -> tensor<4x512x32xbf16, #linear>
      %w_scales_136 = tt.trans %w_scales_135 {order = array<i32: 0, 2, 1>} : tensor<4x512x32xbf16, #linear> -> tensor<4x32x512xbf16, #linear2>
      %w_scales_137 = tt.reshape %w_scales_136 : tensor<4x32x512xbf16, #linear2> -> tensor<128x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %acc_138 = amdgpu.scaled_upcast_fp8 %w_126 scale %w_scales_137 : tensor<128x512xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<128x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %x_139 = ttg.convert_layout %x_123 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %acc_140 = tt.dot %x_139, %acc_138, %arg32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x512xf32, #mma>
      %WMxScalePtrs_141 = tt.addptr %WMxScalePtrs_111, %cst_2 : tensor<512x4x!tt.ptr<i8>, #blocked2>, tensor<512x4xi32, #blocked2>
      %XPtrs_142 = tt.addptr %XPtrs_114, %cst_5 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %WPtrs_143 = tt.addptr %WPtrs_115, %cst_6 : tensor<128x512x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<128x512xi32, #blocked1>
      scf.yield %WMxScalePtrs_141, %acc_140, %x_k_limit_116, %w_k_limit_117, %XPtrs_142, %WPtrs_143 : tensor<512x4x!tt.ptr<i8>, #blocked2>, tensor<128x512xf32, #mma>, i32, i32, tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x512x!tt.ptr<f8E4M3FN>, #blocked1>
    } {tt.scheduled_max_stage = 1 : i32}
    %mask_m = arith.cmpi slt, %offs_x_m_28, %offs_x_m_30 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %mask_n = arith.cmpi slt, %offs_n_scale_58, %offs_n_scale_62 : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %mask_n_86 = arith.cmpi slt, %offs_n_scale_59, %offs_n_scale_63 : tensor<512xi32, #blocked3>
    %BPtrs = arith.muli %expt_id, %stride_b_e : i32
    %BPtrs_87 = tt.addptr %B, %BPtrs : !tt.ptr<f32>, i32
    %BPtrs_88 = tt.splat %BPtrs_87 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked3>
    %BPtrs_89 = tt.addptr %BPtrs_88, %offs_n_scale_59 : tensor<512x!tt.ptr<f32>, #blocked3>, tensor<512xi32, #blocked3>
    %bias = tt.load %BPtrs_89, %mask_n_86, %cst_7 : tensor<512x!tt.ptr<f32>, #blocked3>
    %acc = ttg.convert_layout %bias : tensor<512xf32, #blocked3> -> tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %acc_90 = tt.expand_dims %acc {axis = 0 : i32} : tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x512xf32, #blocked5>
    %acc_91 = ttg.convert_layout %acc_90 : tensor<1x512xf32, #blocked5> -> tensor<1x512xf32, #mma>
    %acc_92 = tt.broadcast %acc_91 : tensor<1x512xf32, #mma> -> tensor<128x512xf32, #mma>
    %acc_93 = arith.addf %w_k_limit#1, %acc_92 : tensor<128x512xf32, #mma>
    %Y_94 = arith.muli %start_m_21, %stride_y_m : i32
    %Y_95 = tt.addptr %Y, %Y_94 : !tt.ptr<bf16>, i32
    %YPtrs = tt.expand_dims %offs_x_m_28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xi32, #blocked4>
    %YPtrs_96 = tt.splat %stride_y_m : i32 -> tensor<128x1xi32, #blocked4>
    %YPtrs_97 = arith.muli %YPtrs, %YPtrs_96 : tensor<128x1xi32, #blocked4>
    %YPtrs_98 = tt.splat %Y_95 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>, #blocked4>
    %YPtrs_99 = tt.addptr %YPtrs_98, %YPtrs_97 : tensor<128x1x!tt.ptr<bf16>, #blocked4>, tensor<128x1xi32, #blocked4>
    %YPtrs_100 = tt.expand_dims %offs_n_scale_58 {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x512xi32, #blocked4>
    %YPtrs_101 = tt.broadcast %YPtrs_99 : tensor<128x1x!tt.ptr<bf16>, #blocked4> -> tensor<128x512x!tt.ptr<bf16>, #blocked4>
    %YPtrs_102 = tt.broadcast %YPtrs_100 : tensor<1x512xi32, #blocked4> -> tensor<128x512xi32, #blocked4>
    %YPtrs_103 = tt.addptr %YPtrs_101, %YPtrs_102 : tensor<128x512x!tt.ptr<bf16>, #blocked4>, tensor<128x512xi32, #blocked4>
    %mask = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xi1, #blocked4>
    %mask_104 = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<512xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x512xi1, #blocked4>
    %mask_105 = tt.broadcast %mask : tensor<128x1xi1, #blocked4> -> tensor<128x512xi1, #blocked4>
    %mask_106 = tt.broadcast %mask_104 : tensor<1x512xi1, #blocked4> -> tensor<128x512xi1, #blocked4>
    %mask_107 = arith.andi %mask_105, %mask_106 : tensor<128x512xi1, #blocked4>
    %16 = arith.truncf %acc_93 : tensor<128x512xf32, #mma> to tensor<128x512xbf16, #mma>
    %YPtrs_108 = ttg.convert_layout %YPtrs_103 : tensor<128x512x!tt.ptr<bf16>, #blocked4> -> tensor<128x512x!tt.ptr<bf16>, #mma>
    %mask_109 = ttg.convert_layout %mask_107 : tensor<128x512xi1, #blocked4> -> tensor<128x512xi1, #mma>
    tt.store %YPtrs_108, %16, %mask_109 : tensor<128x512x!tt.ptr<bf16>, #mma>
    tt.return
  }
}
