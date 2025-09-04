// -----// IR Dump Before AllocateAMDGPUSharedMemory (allocate-amdgpu-shared-memory) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 16], [1, 0, 0], [2, 0, 0], [0, 64, 0], [0, 128, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 4], [0, 0, 8]], warp = [[0, 16, 0], [0, 32, 0], [0, 0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0], [0, 0, 64], [0, 0, 128]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 4, 0], [0, 8, 0]], warp = [[0, 0, 16], [0, 0, 32], [0, 0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_matmul_ogs_NNT_bf16xbf16xfp8e4nv_128x256x128x1(%Y: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %YPtr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_y_k: i32 {tt.divisibility = 16 : i32}, %stride_y_z: i32 {tt.divisibility = 16 : i32}, %stride_y_m: i32 {tt.divisibility = 16 : i32}, %X: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %XPtr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_x_z: i32 {tt.divisibility = 16 : i32}, %stride_x_m: i32 {tt.divisibility = 16 : i32}, %W: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %WPtr: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_w_e: i32 {tt.divisibility = 16 : i32}, %stride_w_n: i32 {tt.divisibility = 16 : i32}, %WMxScale: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %stride_w_mx_e: i32 {tt.divisibility = 16 : i32}, %stride_w_mx_n: i32, %B: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_b_e: i32 {tt.divisibility = 16 : i32}, %NRows: i32, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %GatherIndx: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ScatterSrcIndx: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %num_idxs: i32 {tt.divisibility = 16 : i32}, %ExptHist: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptOffs: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptOffsSum: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptData: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %grid_m: i32, %grid_n: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<32> : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<4> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked1>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %cst_3 = arith.constant dense<7> : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
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
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %pid = tt.get_program_id x : i32
    %padding_m = tt.load %ExptOffsSum : !tt.ptr<i32>
    %padding_m_4 = arith.subi %grid_m, %padding_m : i32
    %unpadded_m = arith.subi %grid_m, %padding_m_4 : i32
    llvm.intr.assume %true : i1
    %total_actual_tiles = arith.muli %unpadded_m, %grid_n : i32
    %0 = arith.cmpi sgt, %padding_m_4, %c0_i32 : i32
    %1 = arith.cmpi sge, %pid, %total_actual_tiles : i32
    %2 = arith.andi %0, %1 : i1
    cf.cond_br %2, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    %pid_mn = arith.subi %pid, %total_actual_tiles : i32
    %3 = arith.muli %padding_m_4, %grid_n : i32
    %4 = arith.cmpi slt, %pid_mn, %3 : i32
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.intr.assume %true : i1
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    tt.return
  ^bb4:  // pred: ^bb0
    %pids_per_group = arith.divsi %total_actual_tiles, %c8_i32 : i32
    %extra_pid_groups = arith.remsi %total_actual_tiles, %c8_i32 : i32
    %group = arith.remsi %pid, %c8_i32 : i32
    %local_pid = arith.divsi %pid, %c8_i32 : i32
    %new_pid = arith.muli %group, %pids_per_group : i32
    %new_pid_5 = arith.minsi %group, %extra_pid_groups : i32
    %new_pid_6 = arith.addi %new_pid, %new_pid_5 : i32
    %new_pid_7 = arith.addi %new_pid_6, %local_pid : i32
    %pid_mnk = arith.remsi %new_pid_7, %total_actual_tiles : i32
    %width = arith.muli %grid_n, %c4_i32 : i32
    %group_id = arith.divsi %pid_mnk, %width : i32
    %group_size = arith.muli %group_id, %c4_i32 : i32
    %group_size_8 = arith.subi %unpadded_m, %group_size : i32
    %group_size_9 = arith.minsi %group_size_8, %c4_i32 : i32
    llvm.intr.assume %true : i1
    %pid_m = arith.remsi %pid_mnk, %group_size_9 : i32
    %pid_m_10 = arith.addi %group_size, %pid_m : i32
    %pid_n = arith.remsi %pid_mnk, %width : i32
    %pid_n_11 = arith.divsi %pid_n, %group_size_9 : i32
    %expt_data = tt.addptr %ExptData, %pid_m_10 : !tt.ptr<i32>, i32
    %expt_data_12 = tt.load %expt_data : !tt.ptr<i32>
    %5 = arith.cmpi eq, %expt_data_12, %c-1_i32 : i32
    cf.cond_br %5, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    tt.return
  ^bb6:  // pred: ^bb4
    %expt_id = arith.andi %expt_data_12, %c65535_i32 : i32
    %block_id = arith.shrsi %expt_data_12, %c16_i32 : i32
    %M = tt.addptr %ExptHist, %expt_id : !tt.ptr<i32>, i32
    %M_13 = tt.load %M : !tt.ptr<i32>
    %start_m = tt.addptr %ExptOffs, %expt_id : !tt.ptr<i32>, i32
    %start_m_14 = tt.load %start_m : !tt.ptr<i32>
    %offs_x_m = arith.muli %block_id, %c128_i32 : i32
    %offs_x_m_15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_x_m_17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_x_m_18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_x_m_19 = tt.splat %offs_x_m : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_20 = tt.splat %offs_x_m : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_x_m_21 = arith.addi %offs_x_m_19, %offs_x_m_15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_22 = arith.addi %offs_x_m_20, %offs_x_m_17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_x_m_23 = tt.splat %M_13 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_24 = tt.splat %M_13 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_x_m_25 = arith.remsi %offs_x_m_21, %offs_x_m_23 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %GatherIndx_26 = tt.addptr %GatherIndx, %start_m_14 : !tt.ptr<i32>, i32
    %offs_x_m_27 = tt.splat %GatherIndx_26 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_28 = tt.addptr %offs_x_m_27, %offs_x_m_25 : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_29 = tt.load %offs_x_m_28 : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_30 = arith.divsi %offs_x_m_29, %cst_0 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %XPtrs = tt.expand_dims %offs_x_m_30 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %XPtrs_31 = tt.splat %stride_x_m : i32 -> tensor<128x1xi32, #blocked1>
    %XPtrs_32 = arith.muli %XPtrs, %XPtrs_31 : tensor<128x1xi32, #blocked1>
    %XPtrs_33 = tt.broadcast %XPtrs_32 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %XPtrs_34 = tt.expand_dims %offs_x_m_18 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %XPtrs_35 = tt.broadcast %XPtrs_34 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %XPtrs_36 = arith.addi %XPtrs_35, %XPtrs_33 : tensor<128x128xi32, #blocked1>
    %acc = arith.cmpi sgt, %K, %c0_i32 : i32
    %acc_37 = tt.splat %acc : i1 -> tensor<128x128xi1, #blocked1>
    %mask_k = tt.splat %K : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %mask_k_38 = arith.cmpi slt, %offs_x_m_18, %mask_k : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %x = tt.expand_dims %mask_k_38 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi1, #blocked1>
    %x_39 = tt.broadcast %x : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %acc_40 = arith.andi %acc_37, %x_39 : tensor<128x128xi1, #blocked1>
    %x_41 = tt.splat %X : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
    %x_42 = tt.addptr %x_41, %XPtrs_36 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %x_43 = tt.load %x_42, %acc_40, %cst_2 {amd.pipeliner_part = "prologue"} : tensor<128x128x!tt.ptr<bf16>, #blocked1>
    %WMxScale_44 = arith.muli %expt_id, %stride_w_mx_e : i32
    %WMxScale_45 = tt.addptr %WMxScale, %WMxScale_44 : !tt.ptr<i8>, i32
    %offs_n_scale = arith.muli %pid_n_11, %c256_i32 : i32
    %offs_n_scale_46 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_scale_47 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_n_scale_48 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %offs_n_scale_49 = tt.splat %offs_n_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_scale_50 = tt.splat %offs_n_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_n_scale_51 = tt.splat %offs_n_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %offs_n_scale_52 = arith.addi %offs_n_scale_49, %offs_n_scale_46 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_scale_53 = arith.addi %offs_n_scale_50, %offs_n_scale_47 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_n_scale_54 = arith.addi %offs_n_scale_51, %offs_n_scale_48 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %offs_n_scale_55 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_scale_56 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_n_scale_57 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %offs_n_scale_58 = arith.remsi %offs_n_scale_52, %offs_n_scale_55 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_scale_59 = arith.remsi %offs_n_scale_53, %offs_n_scale_56 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_k_scale = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %WMxScalePtrs = tt.expand_dims %offs_k_scale {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi32, #blocked>
    %WMxScalePtrs_60 = tt.broadcast %WMxScalePtrs : tensor<1x4xi32, #blocked> -> tensor<256x4xi32, #blocked>
    %WMxScalePtrs_61 = tt.expand_dims %offs_n_scale_58 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %WMxScalePtrs_62 = tt.splat %stride_w_mx_n : i32 -> tensor<256x1xi32, #blocked>
    %WMxScalePtrs_63 = arith.muli %WMxScalePtrs_61, %WMxScalePtrs_62 : tensor<256x1xi32, #blocked>
    %WMxScalePtrs_64 = tt.broadcast %WMxScalePtrs_63 : tensor<256x1xi32, #blocked> -> tensor<256x4xi32, #blocked>
    %WMxScalePtrs_65 = arith.addi %WMxScalePtrs_64, %WMxScalePtrs_60 : tensor<256x4xi32, #blocked>
    %W_66 = arith.muli %expt_id, %stride_w_e : i32
    %W_67 = tt.addptr %W, %W_66 : !tt.ptr<f8E4M3FN>, i32
    %WPtrs = tt.expand_dims %offs_x_m_16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %WPtrs_68 = tt.broadcast %WPtrs : tensor<128x1xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
    %WPtrs_69 = tt.expand_dims %offs_n_scale_59 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
    %WPtrs_70 = tt.splat %stride_w_n : i32 -> tensor<1x256xi32, #blocked2>
    %WPtrs_71 = arith.muli %WPtrs_69, %WPtrs_70 : tensor<1x256xi32, #blocked2>
    %WPtrs_72 = tt.broadcast %WPtrs_71 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
    %WPtrs_73 = arith.addi %WPtrs_68, %WPtrs_72 : tensor<128x256xi32, #blocked2>
    %acc_74 = tt.splat %acc : i1 -> tensor<128x256xi1, #blocked2>
    %mask_k_75 = tt.splat %K : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %mask_k_76 = arith.cmpi slt, %offs_x_m_16, %mask_k_75 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %w = tt.expand_dims %mask_k_76 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi1, #blocked2>
    %w_77 = tt.broadcast %w : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
    %acc_78 = arith.andi %acc_74, %w_77 : tensor<128x256xi1, #blocked2>
    %w_79 = amdgpu.buffer_load %W_67[%WPtrs_73], %acc_78 stride = %stride_w_n : tensor<128x256xf8E4M3FN, #blocked2>
    %mask_k_scale = arith.muli %offs_k_scale, %cst : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %x_80 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xbf16, #shared, #smem, mutable>
    %w_81 = ttg.local_alloc : () -> !ttg.memdesc<1x128x256xf8E4M3FN, #shared1, #smem, mutable>
    %x_82 = ttg.memdesc_index %x_80[%c0_i32] : !ttg.memdesc<1x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>
    ttg.local_store %x_43, %x_82 : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>
    %w_83 = ttg.memdesc_index %w_81[%c0_i32] : !ttg.memdesc<1x128x256xf8E4M3FN, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>
    ttg.local_store %w_79, %w_83 : tensor<128x256xf8E4M3FN, #blocked2> -> !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>
    %acc_84 = arith.subi %K, %c128_i32 : i32
    cf.br ^bb7(%c0_i32, %WMxScale_45, %cst_1, %X, %W_67, %c0_i32, %K, %x_82, %w_83 : i32, !tt.ptr<i8>, tensor<128x256xf32, #mma>, !tt.ptr<bf16>, !tt.ptr<f8E4M3FN>, i32, i32, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>)
  ^bb7(%acc_85: i32, %WMxScalePtrs_86: !tt.ptr<i8>, %6: tensor<128x256xf32, #mma>, %XPtrs_87: !tt.ptr<bf16>, %WPtrs_88: !tt.ptr<f8E4M3FN>, %7: i32, %K_89: i32, %x_90: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>, %w_91: !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>):  // 2 preds: ^bb6, ^bb8
    %acc_92 = arith.cmpi slt, %acc_85, %acc_84 : i32
    cf.cond_br %acc_92, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %XPtrs_93 = tt.addptr %XPtrs_87, %c128_i32 : !tt.ptr<bf16>, i32
    %WPtrs_94 = tt.addptr %WPtrs_88, %c128_i32 : !tt.ptr<f8E4M3FN>, i32
    %acc_95 = arith.addi %acc_85, %c128_i32 : i32
    %mask_k_96 = arith.subi %K, %acc_95 : i32
    %mask_k_97 = tt.splat %mask_k_96 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %mask_k_98 = tt.splat %mask_k_96 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %mask_k_99 = arith.cmpi slt, %offs_x_m_18, %mask_k_97 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %mask_k_100 = arith.cmpi slt, %offs_x_m_16, %mask_k_98 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %mask_k_scale_101 = tt.splat %K_89 : i32 -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_k_scale_102 = arith.cmpi slt, %mask_k_scale, %mask_k_scale_101 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %x_103 = tt.expand_dims %mask_k_99 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi1, #blocked1>
    %x_104 = tt.broadcast %x_103 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %x_105 = tt.splat %XPtrs_93 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
    %x_106 = tt.addptr %x_105, %XPtrs_36 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %x_107 = tt.load %x_106, %x_104, %cst_2 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
    %acc_108 = ttg.local_load %x_90 : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %w_109 = tt.expand_dims %mask_k_100 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi1, #blocked2>
    %w_110 = tt.broadcast %w_109 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
    %w_111 = amdgpu.buffer_load %WPtrs_94[%WPtrs_73], %w_110 stride = %stride_w_n : tensor<128x256xf8E4M3FN, #blocked2>
    %acc_112 = ttg.local_load %w_91 : !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256> -> tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_scales = tt.expand_dims %mask_k_scale_102 {axis = 0 : i32} : tensor<4xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi1, #blocked>
    %w_scales_113 = tt.broadcast %w_scales : tensor<1x4xi1, #blocked> -> tensor<256x4xi1, #blocked>
    %w_scales_114 = amdgpu.buffer_load %WMxScalePtrs_86[%WMxScalePtrs_65], %w_scales_113 stride = %stride_w_mx_n : tensor<256x4xi8, #blocked>
    %w_115 = ttg.convert_layout %w_scales_114 : tensor<256x4xi8, #blocked> -> tensor<256x4xi8, #linear1>
    %w_116 = tt.trans %w_115 {order = array<i32: 1, 0>} : tensor<256x4xi8, #linear1> -> tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_117 = tt.fp_to_fp %acc_112 : tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_118 = arith.extui %w_116 : tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>> to tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_119 = arith.shli %w_118, %cst_3 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_120 = tt.bitcast %w_119 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_121 = tt.expand_dims %w_120 {axis = 2 : i32} : tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256x1xbf16, #linear>
    %w_122 = tt.broadcast %w_121 : tensor<4x256x1xbf16, #linear> -> tensor<4x256x32xbf16, #linear>
    %w_123 = tt.trans %w_122 {order = array<i32: 0, 2, 1>} : tensor<4x256x32xbf16, #linear> -> tensor<4x32x256xbf16, #linear2>
    %w_124 = tt.reshape %w_123 : tensor<4x32x256xbf16, #linear2> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_125 = arith.mulf %w_117, %w_124 : tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %acc_126 = tt.dot %acc_108, %w_125, %6 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xf32, #mma>
    %WMxScalePtrs_127 = tt.addptr %WMxScalePtrs_86, %c4_i32 : !tt.ptr<i8>, i32
    %acc_128 = arith.addi %7, %c1_i32 : i32
    %acc_129 = arith.cmpi slt, %acc_128, %c1_i32 : i32
    %acc_130 = arith.select %acc_129, %acc_128, %c0_i32 : i32
    %x_131 = ttg.memdesc_index %x_80[%acc_130] : !ttg.memdesc<1x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>
    ttg.local_store %x_107, %x_131 : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>
    %w_132 = ttg.memdesc_index %w_81[%acc_130] : !ttg.memdesc<1x128x256xf8E4M3FN, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>
    ttg.local_store %w_111, %w_132 : tensor<128x256xf8E4M3FN, #blocked2> -> !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>
    %acc_133 = arith.addi %acc_85, %c128_i32 : i32
    cf.br ^bb7(%acc_133, %WMxScalePtrs_127, %acc_126, %XPtrs_93, %WPtrs_94, %acc_130, %mask_k_96, %x_131, %w_132 : i32, !tt.ptr<i8>, tensor<128x256xf32, #mma>, !tt.ptr<bf16>, !tt.ptr<f8E4M3FN>, i32, i32, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>)
  ^bb9:  // pred: ^bb7
    %acc_134 = arith.addi %K, %c127_i32 : i32
    %acc_135 = arith.divsi %acc_134, %c128_i32 : i32
    %acc_136 = arith.cmpi sge, %acc_135, %c1_i32 : i32
    %mask_k_scale_137 = tt.splat %K_89 : i32 -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_k_scale_138 = arith.cmpi slt, %mask_k_scale, %mask_k_scale_137 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %acc_139 = ttg.local_load %x_90 : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %acc_140 = ttg.local_load %w_91 : !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256> -> tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_scales_141 = tt.expand_dims %mask_k_scale_138 {axis = 0 : i32} : tensor<4xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi1, #blocked>
    %w_scales_142 = tt.broadcast %w_scales_141 : tensor<1x4xi1, #blocked> -> tensor<256x4xi1, #blocked>
    %acc_143 = tt.splat %acc_136 : i1 -> tensor<256x4xi1, #blocked>
    %acc_144 = arith.andi %acc_143, %w_scales_142 : tensor<256x4xi1, #blocked>
    %w_scales_145 = amdgpu.buffer_load %WMxScalePtrs_86[%WMxScalePtrs_65], %acc_144 stride = %stride_w_mx_n : tensor<256x4xi8, #blocked>
    %w_146 = tt.trans %w_scales_145 {order = array<i32: 1, 0>} : tensor<256x4xi8, #blocked> -> tensor<4x256xi8, #blocked3>
    %w_147 = tt.fp_to_fp %acc_140 : tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_148 = ttg.convert_layout %w_146 : tensor<4x256xi8, #blocked3> -> tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_149 = arith.extui %w_148 : tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>> to tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_150 = arith.shli %w_149, %cst_3 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_151 = tt.bitcast %w_150 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_152 = tt.expand_dims %w_151 {axis = 2 : i32} : tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256x1xbf16, #linear>
    %w_153 = tt.broadcast %w_152 : tensor<4x256x1xbf16, #linear> -> tensor<4x256x32xbf16, #linear>
    %w_154 = tt.trans %w_153 {order = array<i32: 0, 2, 1>} : tensor<4x256x32xbf16, #linear> -> tensor<4x32x256xbf16, #linear2>
    %w_155 = tt.reshape %w_154 : tensor<4x32x256xbf16, #linear2> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_156 = arith.mulf %w_147, %w_155 : tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    cf.cond_br %acc_136, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %acc_157 = tt.dot %acc_139, %w_156, %6 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xf32, #mma>
    cf.br ^bb12(%acc_157 : tensor<128x256xf32, #mma>)
  ^bb11:  // pred: ^bb9
    cf.br ^bb12(%6 : tensor<128x256xf32, #mma>)
  ^bb12(%acc_158: tensor<128x256xf32, #mma>):  // 2 preds: ^bb10, ^bb11
    cf.br ^bb13
  ^bb13:  // pred: ^bb12
    %acc_159 = arith.select %acc_136, %acc_158, %6 : tensor<128x256xf32, #mma>
    ttg.local_dealloc %w_81 : !ttg.memdesc<1x128x256xf8E4M3FN, #shared1, #smem, mutable>
    ttg.local_dealloc %x_80 : !ttg.memdesc<1x128x128xbf16, #shared, #smem, mutable>
    %mask_m = arith.cmpi slt, %offs_x_m_22, %offs_x_m_24 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %mask_n = arith.cmpi slt, %offs_n_scale_54, %offs_n_scale_57 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %BPtrs = arith.muli %expt_id, %stride_b_e : i32
    %BPtrs_160 = tt.addptr %B, %BPtrs : !tt.ptr<f32>, i32
    %BPtrs_161 = tt.addptr %BPtrs_160, %offs_n_scale : !tt.ptr<f32>, i32
    %bias = amdgpu.buffer_load %BPtrs_161[%offs_n_scale_48], %mask_n : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
    %acc_162 = tt.expand_dims %bias {axis = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xf32, #mma>
    %acc_163 = tt.broadcast %acc_162 : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
    %acc_164 = arith.addf %acc_159, %acc_163 : tensor<128x256xf32, #mma>
    %Y_165 = arith.muli %start_m_14, %stride_y_m : i32
    %Y_166 = tt.addptr %Y, %Y_165 : !tt.ptr<bf16>, i32
    %YPtrs = tt.expand_dims %offs_x_m_17 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %YPtrs_167 = arith.muli %offs_x_m, %stride_y_m : i32
    %YPtrs_168 = tt.splat %stride_y_m : i32 -> tensor<128x1xi32, #mma>
    %YPtrs_169 = arith.muli %YPtrs, %YPtrs_168 : tensor<128x1xi32, #mma>
    %YPtrs_170 = tt.addptr %Y_166, %YPtrs_167 : !tt.ptr<bf16>, i32
    %YPtrs_171 = tt.broadcast %YPtrs_169 : tensor<128x1xi32, #mma> -> tensor<128x256xi32, #mma>
    %YPtrs_172 = tt.expand_dims %offs_n_scale_48 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi32, #mma>
    %YPtrs_173 = tt.broadcast %YPtrs_172 : tensor<1x256xi32, #mma> -> tensor<128x256xi32, #mma>
    %YPtrs_174 = tt.addptr %YPtrs_170, %offs_n_scale : !tt.ptr<bf16>, i32
    %YPtrs_175 = arith.addi %YPtrs_173, %YPtrs_171 : tensor<128x256xi32, #mma>
    %mask = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi1, #mma>
    %mask_176 = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi1, #mma>
    %mask_177 = tt.broadcast %mask : tensor<128x1xi1, #mma> -> tensor<128x256xi1, #mma>
    %mask_178 = tt.broadcast %mask_176 : tensor<1x256xi1, #mma> -> tensor<128x256xi1, #mma>
    %mask_179 = arith.andi %mask_177, %mask_178 : tensor<128x256xi1, #mma>
    %8 = arith.truncf %acc_164 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
    amdgpu.buffer_store %8, %YPtrs_174[%YPtrs_175], %mask_179 stride = %stride_y_m : tensor<128x256xbf16, #mma>
    tt.return
  }
}
