// -----// IR Dump Before Canonicalizer (canonicalize) ('tt.func' operation: @triton.language.standard.zeros____(0, 0)cconstexpr_128__(0, 1)cconstexpr_256__(1,)cconstexpr_fp32_) //----- //
module {
  tt.func public @_matmul_ogs_NNT_bf16xbf16xfp8e4nv_128x256x128x1(%Y: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %YPtr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_y_k: i32 {tt.divisibility = 16 : i32}, %stride_y_z: i32 {tt.divisibility = 16 : i32}, %stride_y_m: i32 {tt.divisibility = 16 : i32}, %X: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %XPtr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_x_z: i32 {tt.divisibility = 16 : i32}, %stride_x_m: i32 {tt.divisibility = 16 : i32}, %W: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %WPtr: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_w_e: i32 {tt.divisibility = 16 : i32}, %stride_w_n: i32 {tt.divisibility = 16 : i32}, %WMxScale: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %stride_w_mx_e: i32 {tt.divisibility = 16 : i32}, %stride_w_mx_n: i32, %B: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_b_e: i32 {tt.divisibility = 16 : i32}, %NRows: i32, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %GatherIndx: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ScatterSrcIndx: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %num_idxs: i32 {tt.divisibility = 16 : i32}, %ExptHist: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptOffs: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptOffsSum: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptData: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %grid_m: i32, %grid_n: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sge, %stride_y_k, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    %c0_i32_0 = arith.constant 0 : i32
    %1 = arith.cmpi sge, %stride_y_z, %c0_i32_0 : i32
    llvm.intr.assume %1 : i1
    %c0_i32_1 = arith.constant 0 : i32
    %2 = arith.cmpi sge, %stride_y_m, %c0_i32_1 : i32
    llvm.intr.assume %2 : i1
    %true = arith.constant true
    llvm.intr.assume %true : i1
    %c0_i32_2 = arith.constant 0 : i32
    %3 = arith.cmpi sge, %stride_x_z, %c0_i32_2 : i32
    llvm.intr.assume %3 : i1
    %c0_i32_3 = arith.constant 0 : i32
    %4 = arith.cmpi sge, %stride_x_m, %c0_i32_3 : i32
    llvm.intr.assume %4 : i1
    %true_4 = arith.constant true
    llvm.intr.assume %true_4 : i1
    %c0_i32_5 = arith.constant 0 : i32
    %5 = arith.cmpi sge, %stride_w_e, %c0_i32_5 : i32
    llvm.intr.assume %5 : i1
    %true_6 = arith.constant true
    llvm.intr.assume %true_6 : i1
    %c0_i32_7 = arith.constant 0 : i32
    %6 = arith.cmpi sge, %stride_w_n, %c0_i32_7 : i32
    llvm.intr.assume %6 : i1
    %c0_i32_8 = arith.constant 0 : i32
    %7 = arith.cmpi sge, %stride_w_mx_e, %c0_i32_8 : i32
    llvm.intr.assume %7 : i1
    %true_9 = arith.constant true
    llvm.intr.assume %true_9 : i1
    %c0_i32_10 = arith.constant 0 : i32
    %8 = arith.cmpi sge, %stride_w_mx_n, %c0_i32_10 : i32
    llvm.intr.assume %8 : i1
    %c0_i32_11 = arith.constant 0 : i32
    %9 = arith.cmpi sge, %stride_b_e, %c0_i32_11 : i32
    llvm.intr.assume %9 : i1
    %true_12 = arith.constant true
    llvm.intr.assume %true_12 : i1
    %c0_i32_13 = arith.constant 0 : i32
    %10 = arith.cmpi sge, %grid_m, %c0_i32_13 : i32
    llvm.intr.assume %10 : i1
    %c0_i32_14 = arith.constant 0 : i32
    %11 = arith.cmpi sge, %grid_n, %c0_i32_14 : i32
    llvm.intr.assume %11 : i1
    %yN = arith.constant 1 : i32
    %yN_15 = arith.constant 1 : i32
    %yN_16 = arith.divsi %N, %yN_15 : i32
    %pid = tt.get_program_id x : i32
    %padding_m = tt.load %ExptOffsSum : !tt.ptr<i32>
    %padding_m_17 = arith.extsi %grid_m : i32 to i64
    %padding_m_18 = arith.extsi %padding_m : i32 to i64
    %padding_m_19 = arith.subi %padding_m_17, %padding_m_18 : i64
    %padding_m_20 = arith.constant 2147483647 : i64
    %padding_m_21 = arith.constant -2147483648 : i64
    %padding_m_22 = arith.cmpi sle, %padding_m_19, %padding_m_20 : i64
    %padding_m_23 = arith.cmpi sge, %padding_m_19, %padding_m_21 : i64
    %padding_m_24 = arith.andi %padding_m_22, %padding_m_23 : i1
    %padding_m_25 = arith.subi %grid_m, %padding_m : i32
    %unpadded_m = arith.extsi %grid_m : i32 to i64
    %unpadded_m_26 = arith.extsi %padding_m_25 : i32 to i64
    %unpadded_m_27 = arith.subi %unpadded_m, %unpadded_m_26 : i64
    %unpadded_m_28 = arith.constant 2147483647 : i64
    %unpadded_m_29 = arith.constant -2147483648 : i64
    %unpadded_m_30 = arith.cmpi sle, %unpadded_m_27, %unpadded_m_28 : i64
    %unpadded_m_31 = arith.cmpi sge, %unpadded_m_27, %unpadded_m_29 : i64
    %unpadded_m_32 = arith.andi %unpadded_m_30, %unpadded_m_31 : i1
    %unpadded_m_33 = arith.subi %grid_m, %padding_m_25 : i32
    %c0_i32_34 = arith.constant 0 : i32
    %12 = arith.cmpi sge, %unpadded_m_33, %c0_i32_34 : i32
    llvm.intr.assume %12 : i1
    %total_actual_tiles = arith.constant 1 : i32
    %total_actual_tiles_35 = arith.constant 1 : i32
    %total_actual_tiles_36 = arith.extsi %total_actual_tiles_35 : i32 to i64
    %total_actual_tiles_37 = arith.extsi %unpadded_m_33 : i32 to i64
    %total_actual_tiles_38 = arith.muli %total_actual_tiles_36, %total_actual_tiles_37 : i64
    %total_actual_tiles_39 = arith.constant 2147483647 : i64
    %total_actual_tiles_40 = arith.constant -2147483648 : i64
    %total_actual_tiles_41 = arith.cmpi sle, %total_actual_tiles_38, %total_actual_tiles_39 : i64
    %total_actual_tiles_42 = arith.cmpi sge, %total_actual_tiles_38, %total_actual_tiles_40 : i64
    %total_actual_tiles_43 = arith.andi %total_actual_tiles_41, %total_actual_tiles_42 : i1
    %total_actual_tiles_44 = arith.muli %total_actual_tiles_35, %unpadded_m_33 : i32
    %total_actual_tiles_45 = arith.extsi %total_actual_tiles_44 : i32 to i64
    %total_actual_tiles_46 = arith.extsi %grid_n : i32 to i64
    %total_actual_tiles_47 = arith.muli %total_actual_tiles_45, %total_actual_tiles_46 : i64
    %total_actual_tiles_48 = arith.constant 2147483647 : i64
    %total_actual_tiles_49 = arith.constant -2147483648 : i64
    %total_actual_tiles_50 = arith.cmpi sle, %total_actual_tiles_47, %total_actual_tiles_48 : i64
    %total_actual_tiles_51 = arith.cmpi sge, %total_actual_tiles_47, %total_actual_tiles_49 : i64
    %total_actual_tiles_52 = arith.andi %total_actual_tiles_50, %total_actual_tiles_51 : i1
    %total_actual_tiles_53 = arith.muli %total_actual_tiles_44, %grid_n : i32
    %total_actual_tiles_54 = arith.constant 1 : i32
    %total_actual_tiles_55 = arith.constant 1 : i32
    %total_actual_tiles_56 = arith.extsi %total_actual_tiles_53 : i32 to i64
    %total_actual_tiles_57 = arith.extsi %total_actual_tiles_55 : i32 to i64
    %total_actual_tiles_58 = arith.muli %total_actual_tiles_56, %total_actual_tiles_57 : i64
    %total_actual_tiles_59 = arith.constant 2147483647 : i64
    %total_actual_tiles_60 = arith.constant -2147483648 : i64
    %total_actual_tiles_61 = arith.cmpi sle, %total_actual_tiles_58, %total_actual_tiles_59 : i64
    %total_actual_tiles_62 = arith.cmpi sge, %total_actual_tiles_58, %total_actual_tiles_60 : i64
    %total_actual_tiles_63 = arith.andi %total_actual_tiles_61, %total_actual_tiles_62 : i1
    %total_actual_tiles_64 = arith.muli %total_actual_tiles_53, %total_actual_tiles_55 : i32
    %c0_i32_65 = arith.constant 0 : i32
    %13 = arith.cmpi sgt, %padding_m_25, %c0_i32_65 : i32
    %14 = arith.cmpi sge, %pid, %total_actual_tiles_64 : i32
    %15 = arith.andi %13, %14 : i1
    cf.cond_br %15, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %false = arith.constant false
    %pid_mn = arith.extsi %pid : i32 to i64
    %pid_mn_66 = arith.extsi %total_actual_tiles_64 : i32 to i64
    %pid_mn_67 = arith.subi %pid_mn, %pid_mn_66 : i64
    %pid_mn_68 = arith.constant 2147483647 : i64
    %pid_mn_69 = arith.constant -2147483648 : i64
    %pid_mn_70 = arith.cmpi sle, %pid_mn_67, %pid_mn_68 : i64
    %pid_mn_71 = arith.cmpi sge, %pid_mn_67, %pid_mn_69 : i64
    %pid_mn_72 = arith.andi %pid_mn_70, %pid_mn_71 : i1
    %pid_mn_73 = arith.subi %pid, %total_actual_tiles_64 : i32
    %16 = arith.extsi %padding_m_25 : i32 to i64
    %17 = arith.extsi %grid_n : i32 to i64
    %18 = arith.muli %16, %17 : i64
    %c2147483647_i64 = arith.constant 2147483647 : i64
    %c-2147483648_i64 = arith.constant -2147483648 : i64
    %19 = arith.cmpi sle, %18, %c2147483647_i64 : i64
    %20 = arith.cmpi sge, %18, %c-2147483648_i64 : i64
    %21 = arith.andi %19, %20 : i1
    %22 = arith.muli %padding_m_25, %grid_n : i32
    %23 = arith.cmpi slt, %pid_mn_73, %22 : i32
    scf.if %23 {
      %40:2 = tt.call @"triton_kernels.matmul_ogs_details._common.swizzle2d__i32_i32_i32__(3,)cconstexpr_4_"(%pid_mn_73, %padding_m_25, %grid_n) : (i32, i32, i32) -> (i32, i32)
    } else {
    }
    tt.return
  ^bb2:  // pred: ^bb0
    cf.br ^bb4
  ^bb3:  // no predecessors
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %pid_emnk = tt.call @"triton_kernels.matmul_ogs_details._common.xcd_swizzle__i32_i32__(2,)cconstexpr_8_"(%pid, %total_actual_tiles_64) : (i32, i32) -> i32
    %pid_e = arith.extsi %unpadded_m_33 : i32 to i64
    %pid_e_74 = arith.extsi %grid_n : i32 to i64
    %pid_e_75 = arith.muli %pid_e, %pid_e_74 : i64
    %pid_e_76 = arith.constant 2147483647 : i64
    %pid_e_77 = arith.constant -2147483648 : i64
    %pid_e_78 = arith.cmpi sle, %pid_e_75, %pid_e_76 : i64
    %pid_e_79 = arith.cmpi sge, %pid_e_75, %pid_e_77 : i64
    %pid_e_80 = arith.andi %pid_e_78, %pid_e_79 : i1
    %pid_e_81 = arith.muli %unpadded_m_33, %grid_n : i32
    %pid_e_82 = arith.constant 1 : i32
    %pid_e_83 = arith.constant 1 : i32
    %pid_e_84 = arith.extsi %pid_e_81 : i32 to i64
    %pid_e_85 = arith.extsi %pid_e_83 : i32 to i64
    %pid_e_86 = arith.muli %pid_e_84, %pid_e_85 : i64
    %pid_e_87 = arith.constant 2147483647 : i64
    %pid_e_88 = arith.constant -2147483648 : i64
    %pid_e_89 = arith.cmpi sle, %pid_e_86, %pid_e_87 : i64
    %pid_e_90 = arith.cmpi sge, %pid_e_86, %pid_e_88 : i64
    %pid_e_91 = arith.andi %pid_e_89, %pid_e_90 : i1
    %pid_e_92 = arith.muli %pid_e_81, %pid_e_83 : i32
    %pid_e_93 = arith.divsi %pid_emnk, %pid_e_92 : i32
    %pid_mnk = arith.extsi %unpadded_m_33 : i32 to i64
    %pid_mnk_94 = arith.extsi %grid_n : i32 to i64
    %pid_mnk_95 = arith.muli %pid_mnk, %pid_mnk_94 : i64
    %pid_mnk_96 = arith.constant 2147483647 : i64
    %pid_mnk_97 = arith.constant -2147483648 : i64
    %pid_mnk_98 = arith.cmpi sle, %pid_mnk_95, %pid_mnk_96 : i64
    %pid_mnk_99 = arith.cmpi sge, %pid_mnk_95, %pid_mnk_97 : i64
    %pid_mnk_100 = arith.andi %pid_mnk_98, %pid_mnk_99 : i1
    %pid_mnk_101 = arith.muli %unpadded_m_33, %grid_n : i32
    %pid_mnk_102 = arith.constant 1 : i32
    %pid_mnk_103 = arith.constant 1 : i32
    %pid_mnk_104 = arith.extsi %pid_mnk_101 : i32 to i64
    %pid_mnk_105 = arith.extsi %pid_mnk_103 : i32 to i64
    %pid_mnk_106 = arith.muli %pid_mnk_104, %pid_mnk_105 : i64
    %pid_mnk_107 = arith.constant 2147483647 : i64
    %pid_mnk_108 = arith.constant -2147483648 : i64
    %pid_mnk_109 = arith.cmpi sle, %pid_mnk_106, %pid_mnk_107 : i64
    %pid_mnk_110 = arith.cmpi sge, %pid_mnk_106, %pid_mnk_108 : i64
    %pid_mnk_111 = arith.andi %pid_mnk_109, %pid_mnk_110 : i1
    %pid_mnk_112 = arith.muli %pid_mnk_101, %pid_mnk_103 : i32
    %pid_mnk_113 = arith.remsi %pid_emnk, %pid_mnk_112 : i32
    %pid_k = arith.constant 1 : i32
    %pid_k_114 = arith.constant 1 : i32
    %pid_k_115 = arith.remsi %pid_mnk_113, %pid_k_114 : i32
    %pid_mn_116 = arith.constant 1 : i32
    %pid_mn_117 = arith.constant 1 : i32
    %pid_mn_118 = arith.divsi %pid_mnk_113, %pid_mn_117 : i32
    %24:2 = tt.call @"triton_kernels.matmul_ogs_details._common.swizzle2d__i32_i32_i32__(3,)cconstexpr_4_"(%pid_mn_118, %unpadded_m_33, %grid_n) : (i32, i32, i32) -> (i32, i32)
    %expt_data = tt.addptr %ExptData, %24#0 : !tt.ptr<i32>, i32
    %expt_data_119 = tt.load %expt_data : !tt.ptr<i32>
    %c-1_i32 = arith.constant -1 : i32
    %25 = arith.cmpi eq, %expt_data_119, %c-1_i32 : i32
    cf.cond_br %25, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    tt.return
  ^bb6:  // pred: ^bb4
    cf.br ^bb8
  ^bb7:  // no predecessors
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %expt_id = arith.constant 65535 : i32
    %expt_id_120 = arith.constant 65535 : i32
    %expt_id_121 = arith.andi %expt_data_119, %expt_id_120 : i32
    %block_id = arith.constant 16 : i32
    %block_id_122 = arith.constant 16 : i32
    %block_id_123 = arith.shrsi %expt_data_119, %block_id_122 : i32
    %M = tt.addptr %ExptHist, %expt_id_121 : !tt.ptr<i32>, i32
    %M_124 = tt.load %M : !tt.ptr<i32>
    %start_m = tt.addptr %ExptOffs, %expt_id_121 : !tt.ptr<i32>, i32
    %start_m_125 = tt.load %start_m : !tt.ptr<i32>
    %start_z = arith.constant 0 : i32
    %offs_x_m = arith.constant 128 : i32
    %offs_x_m_126 = arith.constant 128 : i32
    %offs_x_m_127 = arith.extsi %offs_x_m_126 : i32 to i64
    %offs_x_m_128 = arith.extsi %block_id_123 : i32 to i64
    %offs_x_m_129 = arith.muli %offs_x_m_127, %offs_x_m_128 : i64
    %offs_x_m_130 = arith.constant 2147483647 : i64
    %offs_x_m_131 = arith.constant -2147483648 : i64
    %offs_x_m_132 = arith.cmpi sle, %offs_x_m_129, %offs_x_m_130 : i64
    %offs_x_m_133 = arith.cmpi sge, %offs_x_m_129, %offs_x_m_131 : i64
    %offs_x_m_134 = arith.andi %offs_x_m_132, %offs_x_m_133 : i1
    %offs_x_m_135 = arith.muli %offs_x_m_126, %block_id_123 : i32
    %offs_x_m_136 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %offs_x_m_137 = tt.splat %offs_x_m_135 : i32 -> tensor<128xi32>
    %offs_x_m_138 = arith.extsi %offs_x_m_137 : tensor<128xi32> to tensor<128xi64>
    %offs_x_m_139 = arith.extsi %offs_x_m_136 : tensor<128xi32> to tensor<128xi64>
    %offs_x_m_140 = arith.addi %offs_x_m_138, %offs_x_m_139 : tensor<128xi64>
    %offs_x_m_141 = arith.constant 2147483647 : i64
    %offs_x_m_142 = arith.constant -2147483648 : i64
    %offs_x_m_143 = arith.constant dense<2147483647> : tensor<128xi64>
    %offs_x_m_144 = arith.cmpi sle, %offs_x_m_140, %offs_x_m_143 : tensor<128xi64>
    %offs_x_m_145 = arith.constant dense<-2147483648> : tensor<128xi64>
    %offs_x_m_146 = arith.cmpi sge, %offs_x_m_140, %offs_x_m_145 : tensor<128xi64>
    %offs_x_m_147 = arith.andi %offs_x_m_144, %offs_x_m_146 : tensor<128xi1>
    %offs_x_m_148 = arith.addi %offs_x_m_137, %offs_x_m_136 : tensor<128xi32>
    %offs_x_m_149 = tt.splat %M_124 : i32 -> tensor<128xi32>
    %offs_x_m_150 = arith.remsi %offs_x_m_148, %offs_x_m_149 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32>
    %X_151 = arith.extsi %start_z : i32 to i64
    %X_152 = arith.extsi %stride_x_z : i32 to i64
    %X_153 = arith.muli %X_151, %X_152 : i64
    %X_154 = arith.constant 2147483647 : i64
    %X_155 = arith.constant -2147483648 : i64
    %X_156 = arith.cmpi sle, %X_153, %X_154 : i64
    %X_157 = arith.cmpi sge, %X_153, %X_155 : i64
    %X_158 = arith.andi %X_156, %X_157 : i1
    %X_159 = arith.muli %start_z, %stride_x_z : i32
    %X_160 = tt.addptr %X, %X_159 : !tt.ptr<bf16>, i32
    %GatherIndx_161 = tt.addptr %GatherIndx, %start_m_125 : !tt.ptr<i32>, i32
    %offs_x_m_162 = tt.splat %GatherIndx_161 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %offs_x_m_163 = tt.addptr %offs_x_m_162, %offs_x_m_150 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %offs_x_m_164 = tt.load %offs_x_m_163 : tensor<128x!tt.ptr<i32>>
    %offs_x_m_165 = arith.constant 4 : i32
    %offs_x_m_166 = arith.constant 4 : i32
    %offs_x_m_167 = arith.constant dense<4> : tensor<128xi32>
    %offs_x_m_168 = arith.divsi %offs_x_m_164, %offs_x_m_167 : tensor<128xi32>
    %offs_k = arith.constant 128 : i32
    %offs_k_169 = arith.constant 128 : i32
    %offs_k_170 = arith.extsi %offs_k_169 : i32 to i64
    %offs_k_171 = arith.extsi %pid_k_115 : i32 to i64
    %offs_k_172 = arith.muli %offs_k_170, %offs_k_171 : i64
    %offs_k_173 = arith.constant 2147483647 : i64
    %offs_k_174 = arith.constant -2147483648 : i64
    %offs_k_175 = arith.cmpi sle, %offs_k_172, %offs_k_173 : i64
    %offs_k_176 = arith.cmpi sge, %offs_k_172, %offs_k_174 : i64
    %offs_k_177 = arith.andi %offs_k_175, %offs_k_176 : i1
    %offs_k_178 = arith.muli %offs_k_169, %pid_k_115 : i32
    %offs_k_179 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %offs_k_180 = tt.splat %offs_k_178 : i32 -> tensor<128xi32>
    %offs_k_181 = arith.extsi %offs_k_180 : tensor<128xi32> to tensor<128xi64>
    %offs_k_182 = arith.extsi %offs_k_179 : tensor<128xi32> to tensor<128xi64>
    %offs_k_183 = arith.addi %offs_k_181, %offs_k_182 : tensor<128xi64>
    %offs_k_184 = arith.constant 2147483647 : i64
    %offs_k_185 = arith.constant -2147483648 : i64
    %offs_k_186 = arith.constant dense<2147483647> : tensor<128xi64>
    %offs_k_187 = arith.cmpi sle, %offs_k_183, %offs_k_186 : tensor<128xi64>
    %offs_k_188 = arith.constant dense<-2147483648> : tensor<128xi64>
    %offs_k_189 = arith.cmpi sge, %offs_k_183, %offs_k_188 : tensor<128xi64>
    %offs_k_190 = arith.andi %offs_k_187, %offs_k_189 : tensor<128xi1>
    %offs_k_191 = arith.addi %offs_k_180, %offs_k_179 : tensor<128xi32>
    %XPtrs = tt.expand_dims %offs_x_m_168 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %XPtrs_192 = tt.splat %stride_x_m : i32 -> tensor<128x1xi32>
    %XPtrs_193 = arith.extsi %XPtrs : tensor<128x1xi32> to tensor<128x1xi64>
    %XPtrs_194 = arith.extsi %XPtrs_192 : tensor<128x1xi32> to tensor<128x1xi64>
    %XPtrs_195 = arith.muli %XPtrs_193, %XPtrs_194 : tensor<128x1xi64>
    %XPtrs_196 = arith.constant 2147483647 : i64
    %XPtrs_197 = arith.constant -2147483648 : i64
    %XPtrs_198 = arith.constant dense<2147483647> : tensor<128x1xi64>
    %XPtrs_199 = arith.cmpi sle, %XPtrs_195, %XPtrs_198 : tensor<128x1xi64>
    %XPtrs_200 = arith.constant dense<-2147483648> : tensor<128x1xi64>
    %XPtrs_201 = arith.cmpi sge, %XPtrs_195, %XPtrs_200 : tensor<128x1xi64>
    %XPtrs_202 = arith.andi %XPtrs_199, %XPtrs_201 : tensor<128x1xi1>
    %XPtrs_203 = arith.muli %XPtrs, %XPtrs_192 : tensor<128x1xi32>
    %XPtrs_204 = tt.splat %X_160 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
    %XPtrs_205 = tt.addptr %XPtrs_204, %XPtrs_203 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
    %XPtrs_206 = tt.expand_dims %offs_k_191 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %XPtrs_207 = arith.constant 1 : i32
    %XPtrs_208 = arith.constant 1 : i32
    %XPtrs_209 = arith.constant dense<1> : tensor<1x128xi32>
    %XPtrs_210 = arith.extsi %XPtrs_206 : tensor<1x128xi32> to tensor<1x128xi64>
    %XPtrs_211 = arith.extsi %XPtrs_209 : tensor<1x128xi32> to tensor<1x128xi64>
    %XPtrs_212 = arith.muli %XPtrs_210, %XPtrs_211 : tensor<1x128xi64>
    %XPtrs_213 = arith.constant 2147483647 : i64
    %XPtrs_214 = arith.constant -2147483648 : i64
    %XPtrs_215 = arith.constant dense<2147483647> : tensor<1x128xi64>
    %XPtrs_216 = arith.cmpi sle, %XPtrs_212, %XPtrs_215 : tensor<1x128xi64>
    %XPtrs_217 = arith.constant dense<-2147483648> : tensor<1x128xi64>
    %XPtrs_218 = arith.cmpi sge, %XPtrs_212, %XPtrs_217 : tensor<1x128xi64>
    %XPtrs_219 = arith.andi %XPtrs_216, %XPtrs_218 : tensor<1x128xi1>
    %XPtrs_220 = arith.muli %XPtrs_206, %XPtrs_209 : tensor<1x128xi32>
    %XPtrs_221 = tt.broadcast %XPtrs_205 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
    %XPtrs_222 = tt.broadcast %XPtrs_220 : tensor<1x128xi32> -> tensor<128x128xi32>
    %XPtrs_223 = tt.addptr %XPtrs_221, %XPtrs_222 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
    %WMxScale_224 = arith.extsi %expt_id_121 : i32 to i64
    %WMxScale_225 = arith.extsi %stride_w_mx_e : i32 to i64
    %WMxScale_226 = arith.muli %WMxScale_224, %WMxScale_225 : i64
    %WMxScale_227 = arith.constant 2147483647 : i64
    %WMxScale_228 = arith.constant -2147483648 : i64
    %WMxScale_229 = arith.cmpi sle, %WMxScale_226, %WMxScale_227 : i64
    %WMxScale_230 = arith.cmpi sge, %WMxScale_226, %WMxScale_228 : i64
    %WMxScale_231 = arith.andi %WMxScale_229, %WMxScale_230 : i1
    %WMxScale_232 = arith.muli %expt_id_121, %stride_w_mx_e : i32
    %WMxScale_233 = tt.addptr %WMxScale, %WMxScale_232 : !tt.ptr<i8>, i32
    %stride_scale_k = arith.constant 1 : i32
    %offs_n_scale = arith.constant 256 : i32
    %offs_n_scale_234 = arith.constant 256 : i32
    %offs_n_scale_235 = arith.extsi %24#1 : i32 to i64
    %offs_n_scale_236 = arith.extsi %offs_n_scale_234 : i32 to i64
    %offs_n_scale_237 = arith.muli %offs_n_scale_235, %offs_n_scale_236 : i64
    %offs_n_scale_238 = arith.constant 2147483647 : i64
    %offs_n_scale_239 = arith.constant -2147483648 : i64
    %offs_n_scale_240 = arith.cmpi sle, %offs_n_scale_237, %offs_n_scale_238 : i64
    %offs_n_scale_241 = arith.cmpi sge, %offs_n_scale_237, %offs_n_scale_239 : i64
    %offs_n_scale_242 = arith.andi %offs_n_scale_240, %offs_n_scale_241 : i1
    %offs_n_scale_243 = arith.muli %24#1, %offs_n_scale_234 : i32
    %offs_n_scale_244 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %offs_n_scale_245 = tt.splat %offs_n_scale_243 : i32 -> tensor<256xi32>
    %offs_n_scale_246 = arith.extsi %offs_n_scale_245 : tensor<256xi32> to tensor<256xi64>
    %offs_n_scale_247 = arith.extsi %offs_n_scale_244 : tensor<256xi32> to tensor<256xi64>
    %offs_n_scale_248 = arith.addi %offs_n_scale_246, %offs_n_scale_247 : tensor<256xi64>
    %offs_n_scale_249 = arith.constant 2147483647 : i64
    %offs_n_scale_250 = arith.constant -2147483648 : i64
    %offs_n_scale_251 = arith.constant dense<2147483647> : tensor<256xi64>
    %offs_n_scale_252 = arith.cmpi sle, %offs_n_scale_248, %offs_n_scale_251 : tensor<256xi64>
    %offs_n_scale_253 = arith.constant dense<-2147483648> : tensor<256xi64>
    %offs_n_scale_254 = arith.cmpi sge, %offs_n_scale_248, %offs_n_scale_253 : tensor<256xi64>
    %offs_n_scale_255 = arith.andi %offs_n_scale_252, %offs_n_scale_254 : tensor<256xi1>
    %offs_n_scale_256 = arith.addi %offs_n_scale_245, %offs_n_scale_244 : tensor<256xi32>
    %offs_n_scale_257 = tt.splat %N : i32 -> tensor<256xi32>
    %offs_n_scale_258 = arith.remsi %offs_n_scale_256, %offs_n_scale_257 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi32>
    %offs_k_scale = arith.constant 4 : i32
    %offs_k_scale_259 = arith.constant 4 : i32
    %offs_k_scale_260 = arith.extsi %offs_k_scale_259 : i32 to i64
    %offs_k_scale_261 = arith.extsi %pid_k_115 : i32 to i64
    %offs_k_scale_262 = arith.muli %offs_k_scale_260, %offs_k_scale_261 : i64
    %offs_k_scale_263 = arith.constant 2147483647 : i64
    %offs_k_scale_264 = arith.constant -2147483648 : i64
    %offs_k_scale_265 = arith.cmpi sle, %offs_k_scale_262, %offs_k_scale_263 : i64
    %offs_k_scale_266 = arith.cmpi sge, %offs_k_scale_262, %offs_k_scale_264 : i64
    %offs_k_scale_267 = arith.andi %offs_k_scale_265, %offs_k_scale_266 : i1
    %offs_k_scale_268 = arith.muli %offs_k_scale_259, %pid_k_115 : i32
    %offs_k_scale_269 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %offs_k_scale_270 = tt.splat %offs_k_scale_268 : i32 -> tensor<4xi32>
    %offs_k_scale_271 = arith.extsi %offs_k_scale_270 : tensor<4xi32> to tensor<4xi64>
    %offs_k_scale_272 = arith.extsi %offs_k_scale_269 : tensor<4xi32> to tensor<4xi64>
    %offs_k_scale_273 = arith.addi %offs_k_scale_271, %offs_k_scale_272 : tensor<4xi64>
    %offs_k_scale_274 = arith.constant 2147483647 : i64
    %offs_k_scale_275 = arith.constant -2147483648 : i64
    %offs_k_scale_276 = arith.constant dense<2147483647> : tensor<4xi64>
    %offs_k_scale_277 = arith.cmpi sle, %offs_k_scale_273, %offs_k_scale_276 : tensor<4xi64>
    %offs_k_scale_278 = arith.constant dense<-2147483648> : tensor<4xi64>
    %offs_k_scale_279 = arith.cmpi sge, %offs_k_scale_273, %offs_k_scale_278 : tensor<4xi64>
    %offs_k_scale_280 = arith.andi %offs_k_scale_277, %offs_k_scale_279 : tensor<4xi1>
    %offs_k_scale_281 = arith.addi %offs_k_scale_270, %offs_k_scale_269 : tensor<4xi32>
    %WMxScalePtrs = tt.expand_dims %offs_k_scale_281 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %WMxScalePtrs_282 = arith.constant dense<1> : tensor<1x4xi32>
    %WMxScalePtrs_283 = arith.extsi %WMxScalePtrs : tensor<1x4xi32> to tensor<1x4xi64>
    %WMxScalePtrs_284 = arith.extsi %WMxScalePtrs_282 : tensor<1x4xi32> to tensor<1x4xi64>
    %WMxScalePtrs_285 = arith.muli %WMxScalePtrs_283, %WMxScalePtrs_284 : tensor<1x4xi64>
    %WMxScalePtrs_286 = arith.constant 2147483647 : i64
    %WMxScalePtrs_287 = arith.constant -2147483648 : i64
    %WMxScalePtrs_288 = arith.constant dense<2147483647> : tensor<1x4xi64>
    %WMxScalePtrs_289 = arith.cmpi sle, %WMxScalePtrs_285, %WMxScalePtrs_288 : tensor<1x4xi64>
    %WMxScalePtrs_290 = arith.constant dense<-2147483648> : tensor<1x4xi64>
    %WMxScalePtrs_291 = arith.cmpi sge, %WMxScalePtrs_285, %WMxScalePtrs_290 : tensor<1x4xi64>
    %WMxScalePtrs_292 = arith.andi %WMxScalePtrs_289, %WMxScalePtrs_291 : tensor<1x4xi1>
    %WMxScalePtrs_293 = arith.muli %WMxScalePtrs, %WMxScalePtrs_282 : tensor<1x4xi32>
    %WMxScalePtrs_294 = tt.splat %WMxScale_233 : !tt.ptr<i8> -> tensor<1x4x!tt.ptr<i8>>
    %WMxScalePtrs_295 = tt.addptr %WMxScalePtrs_294, %WMxScalePtrs_293 : tensor<1x4x!tt.ptr<i8>>, tensor<1x4xi32>
    %WMxScalePtrs_296 = tt.expand_dims %offs_n_scale_258 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
    %WMxScalePtrs_297 = tt.splat %stride_w_mx_n : i32 -> tensor<256x1xi32>
    %WMxScalePtrs_298 = arith.extsi %WMxScalePtrs_296 : tensor<256x1xi32> to tensor<256x1xi64>
    %WMxScalePtrs_299 = arith.extsi %WMxScalePtrs_297 : tensor<256x1xi32> to tensor<256x1xi64>
    %WMxScalePtrs_300 = arith.muli %WMxScalePtrs_298, %WMxScalePtrs_299 : tensor<256x1xi64>
    %WMxScalePtrs_301 = arith.constant 2147483647 : i64
    %WMxScalePtrs_302 = arith.constant -2147483648 : i64
    %WMxScalePtrs_303 = arith.constant dense<2147483647> : tensor<256x1xi64>
    %WMxScalePtrs_304 = arith.cmpi sle, %WMxScalePtrs_300, %WMxScalePtrs_303 : tensor<256x1xi64>
    %WMxScalePtrs_305 = arith.constant dense<-2147483648> : tensor<256x1xi64>
    %WMxScalePtrs_306 = arith.cmpi sge, %WMxScalePtrs_300, %WMxScalePtrs_305 : tensor<256x1xi64>
    %WMxScalePtrs_307 = arith.andi %WMxScalePtrs_304, %WMxScalePtrs_306 : tensor<256x1xi1>
    %WMxScalePtrs_308 = arith.muli %WMxScalePtrs_296, %WMxScalePtrs_297 : tensor<256x1xi32>
    %WMxScalePtrs_309 = tt.broadcast %WMxScalePtrs_295 : tensor<1x4x!tt.ptr<i8>> -> tensor<256x4x!tt.ptr<i8>>
    %WMxScalePtrs_310 = tt.broadcast %WMxScalePtrs_308 : tensor<256x1xi32> -> tensor<256x4xi32>
    %WMxScalePtrs_311 = tt.addptr %WMxScalePtrs_309, %WMxScalePtrs_310 : tensor<256x4x!tt.ptr<i8>>, tensor<256x4xi32>
    %offs_w_n = arith.constant 256 : i32
    %offs_w_n_312 = arith.constant 256 : i32
    %offs_w_n_313 = arith.extsi %24#1 : i32 to i64
    %offs_w_n_314 = arith.extsi %offs_w_n_312 : i32 to i64
    %offs_w_n_315 = arith.muli %offs_w_n_313, %offs_w_n_314 : i64
    %offs_w_n_316 = arith.constant 2147483647 : i64
    %offs_w_n_317 = arith.constant -2147483648 : i64
    %offs_w_n_318 = arith.cmpi sle, %offs_w_n_315, %offs_w_n_316 : i64
    %offs_w_n_319 = arith.cmpi sge, %offs_w_n_315, %offs_w_n_317 : i64
    %offs_w_n_320 = arith.andi %offs_w_n_318, %offs_w_n_319 : i1
    %offs_w_n_321 = arith.muli %24#1, %offs_w_n_312 : i32
    %offs_w_n_322 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %offs_w_n_323 = tt.splat %offs_w_n_321 : i32 -> tensor<256xi32>
    %offs_w_n_324 = arith.extsi %offs_w_n_323 : tensor<256xi32> to tensor<256xi64>
    %offs_w_n_325 = arith.extsi %offs_w_n_322 : tensor<256xi32> to tensor<256xi64>
    %offs_w_n_326 = arith.addi %offs_w_n_324, %offs_w_n_325 : tensor<256xi64>
    %offs_w_n_327 = arith.constant 2147483647 : i64
    %offs_w_n_328 = arith.constant -2147483648 : i64
    %offs_w_n_329 = arith.constant dense<2147483647> : tensor<256xi64>
    %offs_w_n_330 = arith.cmpi sle, %offs_w_n_326, %offs_w_n_329 : tensor<256xi64>
    %offs_w_n_331 = arith.constant dense<-2147483648> : tensor<256xi64>
    %offs_w_n_332 = arith.cmpi sge, %offs_w_n_326, %offs_w_n_331 : tensor<256xi64>
    %offs_w_n_333 = arith.andi %offs_w_n_330, %offs_w_n_332 : tensor<256xi1>
    %offs_w_n_334 = arith.addi %offs_w_n_323, %offs_w_n_322 : tensor<256xi32>
    %offs_w_n_335 = arith.constant 1 : i32
    %offs_w_n_336 = arith.constant 1 : i32
    %offs_w_n_337 = arith.divsi %N, %offs_w_n_336 : i32
    %offs_w_n_338 = tt.splat %offs_w_n_337 : i32 -> tensor<256xi32>
    %offs_w_n_339 = arith.remsi %offs_w_n_334, %offs_w_n_338 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi32>
    %offs_w_k = arith.constant 128 : i32
    %offs_w_k_340 = arith.constant 128 : i32
    %offs_w_k_341 = arith.extsi %offs_w_k_340 : i32 to i64
    %offs_w_k_342 = arith.extsi %pid_k_115 : i32 to i64
    %offs_w_k_343 = arith.muli %offs_w_k_341, %offs_w_k_342 : i64
    %offs_w_k_344 = arith.constant 2147483647 : i64
    %offs_w_k_345 = arith.constant -2147483648 : i64
    %offs_w_k_346 = arith.cmpi sle, %offs_w_k_343, %offs_w_k_344 : i64
    %offs_w_k_347 = arith.cmpi sge, %offs_w_k_343, %offs_w_k_345 : i64
    %offs_w_k_348 = arith.andi %offs_w_k_346, %offs_w_k_347 : i1
    %offs_w_k_349 = arith.muli %offs_w_k_340, %pid_k_115 : i32
    %offs_w_k_350 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %offs_w_k_351 = tt.splat %offs_w_k_349 : i32 -> tensor<128xi32>
    %offs_w_k_352 = arith.extsi %offs_w_k_351 : tensor<128xi32> to tensor<128xi64>
    %offs_w_k_353 = arith.extsi %offs_w_k_350 : tensor<128xi32> to tensor<128xi64>
    %offs_w_k_354 = arith.addi %offs_w_k_352, %offs_w_k_353 : tensor<128xi64>
    %offs_w_k_355 = arith.constant 2147483647 : i64
    %offs_w_k_356 = arith.constant -2147483648 : i64
    %offs_w_k_357 = arith.constant dense<2147483647> : tensor<128xi64>
    %offs_w_k_358 = arith.cmpi sle, %offs_w_k_354, %offs_w_k_357 : tensor<128xi64>
    %offs_w_k_359 = arith.constant dense<-2147483648> : tensor<128xi64>
    %offs_w_k_360 = arith.cmpi sge, %offs_w_k_354, %offs_w_k_359 : tensor<128xi64>
    %offs_w_k_361 = arith.andi %offs_w_k_358, %offs_w_k_360 : tensor<128xi1>
    %offs_w_k_362 = arith.addi %offs_w_k_351, %offs_w_k_350 : tensor<128xi32>
    %W_363 = arith.extsi %expt_id_121 : i32 to i64
    %W_364 = arith.extsi %stride_w_e : i32 to i64
    %W_365 = arith.muli %W_363, %W_364 : i64
    %W_366 = arith.constant 2147483647 : i64
    %W_367 = arith.constant -2147483648 : i64
    %W_368 = arith.cmpi sle, %W_365, %W_366 : i64
    %W_369 = arith.cmpi sge, %W_365, %W_367 : i64
    %W_370 = arith.andi %W_368, %W_369 : i1
    %W_371 = arith.muli %expt_id_121, %stride_w_e : i32
    %W_372 = tt.addptr %W, %W_371 : !tt.ptr<f8E4M3FN>, i32
    %WPtrs = tt.expand_dims %offs_w_k_362 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %WPtrs_373 = arith.constant 1 : i32
    %WPtrs_374 = arith.constant 1 : i32
    %WPtrs_375 = arith.constant dense<1> : tensor<128x1xi32>
    %WPtrs_376 = arith.extsi %WPtrs : tensor<128x1xi32> to tensor<128x1xi64>
    %WPtrs_377 = arith.extsi %WPtrs_375 : tensor<128x1xi32> to tensor<128x1xi64>
    %WPtrs_378 = arith.muli %WPtrs_376, %WPtrs_377 : tensor<128x1xi64>
    %WPtrs_379 = arith.constant 2147483647 : i64
    %WPtrs_380 = arith.constant -2147483648 : i64
    %WPtrs_381 = arith.constant dense<2147483647> : tensor<128x1xi64>
    %WPtrs_382 = arith.cmpi sle, %WPtrs_378, %WPtrs_381 : tensor<128x1xi64>
    %WPtrs_383 = arith.constant dense<-2147483648> : tensor<128x1xi64>
    %WPtrs_384 = arith.cmpi sge, %WPtrs_378, %WPtrs_383 : tensor<128x1xi64>
    %WPtrs_385 = arith.andi %WPtrs_382, %WPtrs_384 : tensor<128x1xi1>
    %WPtrs_386 = arith.muli %WPtrs, %WPtrs_375 : tensor<128x1xi32>
    %WPtrs_387 = tt.expand_dims %offs_w_n_339 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %WPtrs_388 = tt.splat %stride_w_n : i32 -> tensor<1x256xi32>
    %WPtrs_389 = arith.extsi %WPtrs_387 : tensor<1x256xi32> to tensor<1x256xi64>
    %WPtrs_390 = arith.extsi %WPtrs_388 : tensor<1x256xi32> to tensor<1x256xi64>
    %WPtrs_391 = arith.muli %WPtrs_389, %WPtrs_390 : tensor<1x256xi64>
    %WPtrs_392 = arith.constant 2147483647 : i64
    %WPtrs_393 = arith.constant -2147483648 : i64
    %WPtrs_394 = arith.constant dense<2147483647> : tensor<1x256xi64>
    %WPtrs_395 = arith.cmpi sle, %WPtrs_391, %WPtrs_394 : tensor<1x256xi64>
    %WPtrs_396 = arith.constant dense<-2147483648> : tensor<1x256xi64>
    %WPtrs_397 = arith.cmpi sge, %WPtrs_391, %WPtrs_396 : tensor<1x256xi64>
    %WPtrs_398 = arith.andi %WPtrs_395, %WPtrs_397 : tensor<1x256xi1>
    %WPtrs_399 = arith.muli %WPtrs_387, %WPtrs_388 : tensor<1x256xi32>
    %WPtrs_400 = tt.broadcast %WPtrs_386 : tensor<128x1xi32> -> tensor<128x256xi32>
    %WPtrs_401 = tt.broadcast %WPtrs_399 : tensor<1x256xi32> -> tensor<128x256xi32>
    %WPtrs_402 = arith.extsi %WPtrs_400 : tensor<128x256xi32> to tensor<128x256xi64>
    %WPtrs_403 = arith.extsi %WPtrs_401 : tensor<128x256xi32> to tensor<128x256xi64>
    %WPtrs_404 = arith.addi %WPtrs_402, %WPtrs_403 : tensor<128x256xi64>
    %WPtrs_405 = arith.constant 2147483647 : i64
    %WPtrs_406 = arith.constant -2147483648 : i64
    %WPtrs_407 = arith.constant dense<2147483647> : tensor<128x256xi64>
    %WPtrs_408 = arith.cmpi sle, %WPtrs_404, %WPtrs_407 : tensor<128x256xi64>
    %WPtrs_409 = arith.constant dense<-2147483648> : tensor<128x256xi64>
    %WPtrs_410 = arith.cmpi sge, %WPtrs_404, %WPtrs_409 : tensor<128x256xi64>
    %WPtrs_411 = arith.andi %WPtrs_408, %WPtrs_410 : tensor<128x256xi1>
    %WPtrs_412 = arith.addi %WPtrs_400, %WPtrs_401 : tensor<128x256xi32>
    %WPtrs_413 = tt.splat %W_372 : !tt.ptr<f8E4M3FN> -> tensor<128x256x!tt.ptr<f8E4M3FN>>
    %WPtrs_414 = tt.addptr %WPtrs_413, %WPtrs_412 : tensor<128x256x!tt.ptr<f8E4M3FN>>, tensor<128x256xi32>
    %acc = tt.call @"triton.language.standard.zeros____(0, 0)cconstexpr_128__(0, 1)cconstexpr_256__(1,)cconstexpr_fp32_"() : () -> tensor<128x256xf32>
    %c128_i32 = arith.constant 128 : i32
    %c128_i32_415 = arith.constant 128 : i32
    %26 = arith.extsi %c128_i32_415 : i32 to i64
    %27 = arith.extsi %pid_k_115 : i32 to i64
    %28 = arith.muli %26, %27 : i64
    %c2147483647_i64_416 = arith.constant 2147483647 : i64
    %c-2147483648_i64_417 = arith.constant -2147483648 : i64
    %29 = arith.cmpi sle, %28, %c2147483647_i64_416 : i64
    %30 = arith.cmpi sge, %28, %c-2147483648_i64_417 : i64
    %31 = arith.andi %29, %30 : i1
    %32 = arith.muli %c128_i32_415, %pid_k_115 : i32
    %c128_i32_418 = arith.constant 128 : i32
    %33 = arith.bitcast %32 : i32 to i32
    %34 = arith.bitcast %K : i32 to i32
    %35 = arith.bitcast %c128_i32_418 : i32 to i32
    %36 = ub.poison : i32
    %acc_419:4 = scf.for %arg30 = %33 to %34 step %35 iter_args(%XPtrs_550 = %XPtrs_223, %WMxScalePtrs_551 = %WMxScalePtrs_311, %WPtrs_552 = %WPtrs_414, %acc_553 = %acc) -> (tensor<128x128x!tt.ptr<bf16>>, tensor<256x4x!tt.ptr<i8>>, tensor<128x256x!tt.ptr<f8E4M3FN>>, tensor<128x256xf32>)  : i32 {
      %mask_k = arith.subi %34, %arg30 : i32
      %k = arith.addi %mask_k, %33 : i32
      %mask_k_554 = tt.splat %k : i32 -> tensor<128xi32>
      %mask_k_555 = arith.cmpi slt, %offs_k_191, %mask_k_554 : tensor<128xi32>
      %mask_k_w = arith.constant 1 : i32
      %mask_k_w_556 = arith.constant 1 : i32
      %mask_k_w_557 = arith.divsi %k, %mask_k_w_556 : i32
      %mask_k_w_558 = arith.constant 1 : i32
      %mask_k_w_559 = arith.constant 1 : i32
      %mask_k_w_560 = arith.extsi %mask_k_w_557 : i32 to i64
      %mask_k_w_561 = arith.extsi %mask_k_w_559 : i32 to i64
      %mask_k_w_562 = arith.muli %mask_k_w_560, %mask_k_w_561 : i64
      %mask_k_w_563 = arith.constant 2147483647 : i64
      %mask_k_w_564 = arith.constant -2147483648 : i64
      %mask_k_w_565 = arith.cmpi sle, %mask_k_w_562, %mask_k_w_563 : i64
      %mask_k_w_566 = arith.cmpi sge, %mask_k_w_562, %mask_k_w_564 : i64
      %mask_k_w_567 = arith.andi %mask_k_w_565, %mask_k_w_566 : i1
      %mask_k_w_568 = arith.muli %mask_k_w_557, %mask_k_w_559 : i32
      %mask_k_w_569 = tt.splat %mask_k_w_568 : i32 -> tensor<128xi32>
      %mask_k_w_570 = arith.cmpi slt, %offs_w_k_362, %mask_k_w_569 : tensor<128xi32>
      %mask_k_scale = arith.constant 32 : i32
      %mask_k_scale_571 = arith.constant 32 : i32
      %mask_k_scale_572 = arith.constant dense<32> : tensor<4xi32>
      %mask_k_scale_573 = arith.extsi %offs_k_scale_281 : tensor<4xi32> to tensor<4xi64>
      %mask_k_scale_574 = arith.extsi %mask_k_scale_572 : tensor<4xi32> to tensor<4xi64>
      %mask_k_scale_575 = arith.muli %mask_k_scale_573, %mask_k_scale_574 : tensor<4xi64>
      %mask_k_scale_576 = arith.constant 2147483647 : i64
      %mask_k_scale_577 = arith.constant -2147483648 : i64
      %mask_k_scale_578 = arith.constant dense<2147483647> : tensor<4xi64>
      %mask_k_scale_579 = arith.cmpi sle, %mask_k_scale_575, %mask_k_scale_578 : tensor<4xi64>
      %mask_k_scale_580 = arith.constant dense<-2147483648> : tensor<4xi64>
      %mask_k_scale_581 = arith.cmpi sge, %mask_k_scale_575, %mask_k_scale_580 : tensor<4xi64>
      %mask_k_scale_582 = arith.andi %mask_k_scale_579, %mask_k_scale_581 : tensor<4xi1>
      %mask_k_scale_583 = arith.muli %offs_k_scale_281, %mask_k_scale_572 : tensor<4xi32>
      %mask_k_scale_584 = tt.splat %k : i32 -> tensor<4xi32>
      %mask_k_scale_585 = arith.cmpi slt, %mask_k_scale_583, %mask_k_scale_584 : tensor<4xi32>
      %x = tt.expand_dims %mask_k_555 {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
      %x_586 = arith.constant 0.000000e+00 : f32
      %x_587 = tt.broadcast %x : tensor<1x128xi1> -> tensor<128x128xi1>
      %x_588 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
      %x_589 = arith.truncf %x_588 : tensor<128x128xf32> to tensor<128x128xbf16>
      %x_590 = tt.load %XPtrs_550, %x_587, %x_589 : tensor<128x128x!tt.ptr<bf16>>
      %w = tt.expand_dims %mask_k_w_570 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
      %w_591 = arith.constant 0.000000e+00 : f32
      %w_592 = tt.broadcast %w : tensor<128x1xi1> -> tensor<128x256xi1>
      %w_593 = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
      %w_594 = tt.fp_to_fp %w_593, rounding = rtne : tensor<128x256xf32> -> tensor<128x256xf8E4M3FN>
      %w_595 = tt.load %WPtrs_552, %w_592, %w_594 : tensor<128x256x!tt.ptr<f8E4M3FN>>
      %w_scales = tt.expand_dims %mask_k_scale_585 {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
      %w_scales_596 = tt.broadcast %w_scales : tensor<1x4xi1> -> tensor<256x4xi1>
      %w_scales_597 = tt.load %WMxScalePtrs_551, %w_scales_596 : tensor<256x4x!tt.ptr<i8>>
      %acc_598 = arith.constant 0.000000e+00 : f32
      %acc_599 = tt.dot_scaled %x_590, %w_595 scale %w_scales_597, %acc_553 lhs = bf16 rhs = e4m3 {fastMath = true} : tensor<128x128xbf16> * tensor<128x256xf8E4M3FN>, tensor<256x4xi8> -> tensor<128x256xf32>
      %WMxScalePtrs_600 = arith.constant 4 : i32
      %WMxScalePtrs_601 = arith.constant dense<4> : tensor<256x4xi32>
      %WMxScalePtrs_602 = tt.addptr %WMxScalePtrs_551, %WMxScalePtrs_601 : tensor<256x4x!tt.ptr<i8>>, tensor<256x4xi32>
      %XPtrs_603 = arith.constant 128 : i32
      %XPtrs_604 = arith.constant dense<128> : tensor<128x128xi32>
      %XPtrs_605 = tt.addptr %XPtrs_550, %XPtrs_604 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %WPtrs_606 = arith.constant 128 : i32
      %WPtrs_607 = arith.constant dense<128> : tensor<128x256xi32>
      %WPtrs_608 = tt.addptr %WPtrs_552, %WPtrs_607 : tensor<128x256x!tt.ptr<f8E4M3FN>>, tensor<128x256xi32>
      scf.yield %XPtrs_605, %WMxScalePtrs_602, %WPtrs_608, %acc_599 : tensor<128x128x!tt.ptr<bf16>>, tensor<256x4x!tt.ptr<i8>>, tensor<128x256x!tt.ptr<f8E4M3FN>>, tensor<128x256xf32>
    }
    %offs_m = arith.constant 128 : i32
    %offs_m_420 = arith.constant 128 : i32
    %offs_m_421 = arith.extsi %offs_m_420 : i32 to i64
    %offs_m_422 = arith.extsi %block_id_123 : i32 to i64
    %offs_m_423 = arith.muli %offs_m_421, %offs_m_422 : i64
    %offs_m_424 = arith.constant 2147483647 : i64
    %offs_m_425 = arith.constant -2147483648 : i64
    %offs_m_426 = arith.cmpi sle, %offs_m_423, %offs_m_424 : i64
    %offs_m_427 = arith.cmpi sge, %offs_m_423, %offs_m_425 : i64
    %offs_m_428 = arith.andi %offs_m_426, %offs_m_427 : i1
    %offs_m_429 = arith.muli %offs_m_420, %block_id_123 : i32
    %offs_m_430 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %offs_m_431 = tt.splat %offs_m_429 : i32 -> tensor<128xi32>
    %offs_m_432 = arith.extsi %offs_m_431 : tensor<128xi32> to tensor<128xi64>
    %offs_m_433 = arith.extsi %offs_m_430 : tensor<128xi32> to tensor<128xi64>
    %offs_m_434 = arith.addi %offs_m_432, %offs_m_433 : tensor<128xi64>
    %offs_m_435 = arith.constant 2147483647 : i64
    %offs_m_436 = arith.constant -2147483648 : i64
    %offs_m_437 = arith.constant dense<2147483647> : tensor<128xi64>
    %offs_m_438 = arith.cmpi sle, %offs_m_434, %offs_m_437 : tensor<128xi64>
    %offs_m_439 = arith.constant dense<-2147483648> : tensor<128xi64>
    %offs_m_440 = arith.cmpi sge, %offs_m_434, %offs_m_439 : tensor<128xi64>
    %offs_m_441 = arith.andi %offs_m_438, %offs_m_440 : tensor<128xi1>
    %offs_m_442 = arith.addi %offs_m_431, %offs_m_430 : tensor<128xi32>
    %offs_y_n = arith.constant 256 : i32
    %offs_y_n_443 = arith.constant 256 : i32
    %offs_y_n_444 = arith.extsi %offs_y_n_443 : i32 to i64
    %offs_y_n_445 = arith.extsi %24#1 : i32 to i64
    %offs_y_n_446 = arith.muli %offs_y_n_444, %offs_y_n_445 : i64
    %offs_y_n_447 = arith.constant 2147483647 : i64
    %offs_y_n_448 = arith.constant -2147483648 : i64
    %offs_y_n_449 = arith.cmpi sle, %offs_y_n_446, %offs_y_n_447 : i64
    %offs_y_n_450 = arith.cmpi sge, %offs_y_n_446, %offs_y_n_448 : i64
    %offs_y_n_451 = arith.andi %offs_y_n_449, %offs_y_n_450 : i1
    %offs_y_n_452 = arith.muli %offs_y_n_443, %24#1 : i32
    %offs_y_n_453 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %offs_y_n_454 = tt.splat %offs_y_n_452 : i32 -> tensor<256xi32>
    %offs_y_n_455 = arith.extsi %offs_y_n_454 : tensor<256xi32> to tensor<256xi64>
    %offs_y_n_456 = arith.extsi %offs_y_n_453 : tensor<256xi32> to tensor<256xi64>
    %offs_y_n_457 = arith.addi %offs_y_n_455, %offs_y_n_456 : tensor<256xi64>
    %offs_y_n_458 = arith.constant 2147483647 : i64
    %offs_y_n_459 = arith.constant -2147483648 : i64
    %offs_y_n_460 = arith.constant dense<2147483647> : tensor<256xi64>
    %offs_y_n_461 = arith.cmpi sle, %offs_y_n_457, %offs_y_n_460 : tensor<256xi64>
    %offs_y_n_462 = arith.constant dense<-2147483648> : tensor<256xi64>
    %offs_y_n_463 = arith.cmpi sge, %offs_y_n_457, %offs_y_n_462 : tensor<256xi64>
    %offs_y_n_464 = arith.andi %offs_y_n_461, %offs_y_n_463 : tensor<256xi1>
    %offs_y_n_465 = arith.addi %offs_y_n_454, %offs_y_n_453 : tensor<256xi32>
    %mask_m = tt.splat %M_124 : i32 -> tensor<128xi32>
    %mask_m_466 = arith.cmpi slt, %offs_m_442, %mask_m : tensor<128xi32>
    %mask_n = tt.splat %N : i32 -> tensor<256xi32>
    %mask_n_467 = arith.cmpi slt, %offs_y_n_465, %mask_n : tensor<256xi32>
    %BPtrs = arith.extsi %expt_id_121 : i32 to i64
    %BPtrs_468 = arith.extsi %stride_b_e : i32 to i64
    %BPtrs_469 = arith.muli %BPtrs, %BPtrs_468 : i64
    %BPtrs_470 = arith.constant 2147483647 : i64
    %BPtrs_471 = arith.constant -2147483648 : i64
    %BPtrs_472 = arith.cmpi sle, %BPtrs_469, %BPtrs_470 : i64
    %BPtrs_473 = arith.cmpi sge, %BPtrs_469, %BPtrs_471 : i64
    %BPtrs_474 = arith.andi %BPtrs_472, %BPtrs_473 : i1
    %BPtrs_475 = arith.muli %expt_id_121, %stride_b_e : i32
    %BPtrs_476 = tt.addptr %B, %BPtrs_475 : !tt.ptr<f32>, i32
    %BPtrs_477 = tt.splat %BPtrs_476 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %BPtrs_478 = tt.addptr %BPtrs_477, %offs_y_n_465 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %c0_i32_479 = arith.constant 0 : i32
    %37 = arith.cmpi eq, %pid_k_115, %c0_i32_479 : i32
    %38 = scf.if %37 -> (tensor<256xf32>) {
      %bias = arith.constant 0 : i32
      %bias_550 = arith.constant dense<0> : tensor<256xi32>
      %bias_551 = arith.sitofp %bias_550 : tensor<256xi32> to tensor<256xf32>
      %bias_552 = tt.load %BPtrs_478, %mask_n_467, %bias_551 : tensor<256x!tt.ptr<f32>>
      scf.yield %bias_552 : tensor<256xf32>
    } else {
      %bias = arith.constant 0.000000e+00 : f32
      %bias_550 = arith.constant dense<0.000000e+00> : tensor<256xf32>
      scf.yield %bias_550 : tensor<256xf32>
    }
    %betas = arith.constant 1.000000e+00 : f32
    %betas_480 = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %gammas = arith.constant 1.000000e+00 : f32
    %gammas_481 = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %x_scale = tt.call @"triton_kernels.numerics_details.flexpoint.load_scale____(0,)cconstexpr_None_"() : () -> f32
    %w_scale = tt.call @"triton_kernels.numerics_details.flexpoint.load_scale____(0,)cconstexpr_None_"() : () -> f32
    %acc_482 = arith.mulf %x_scale, %w_scale : f32
    %acc_483 = tt.splat %acc_482 : f32 -> tensor<128x256xf32>
    %acc_484 = arith.mulf %acc_419#3, %acc_483 : tensor<128x256xf32>
    %acc_485 = tt.expand_dims %38 {axis = 0 : i32} : tensor<256xf32> -> tensor<1x256xf32>
    %acc_486 = tt.expand_dims %betas_480 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    %acc_487 = tt.broadcast %acc_485 : tensor<1x256xf32> -> tensor<128x256xf32>
    %acc_488 = tt.broadcast %acc_486 : tensor<128x1xf32> -> tensor<128x256xf32>
    %acc_489 = arith.mulf %acc_487, %acc_488 : tensor<128x256xf32>
    %acc_490 = arith.addf %acc_484, %acc_489 : tensor<128x256xf32>
    %out = tt.expand_dims %gammas_481 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    %out_491 = tt.broadcast %out : tensor<128x1xf32> -> tensor<128x256xf32>
    %out_492 = arith.mulf %acc_490, %out_491 : tensor<128x256xf32>
    %Y_493 = arith.extsi %start_z : i32 to i64
    %Y_494 = arith.extsi %stride_y_z : i32 to i64
    %Y_495 = arith.muli %Y_493, %Y_494 : i64
    %Y_496 = arith.constant 2147483647 : i64
    %Y_497 = arith.constant -2147483648 : i64
    %Y_498 = arith.cmpi sle, %Y_495, %Y_496 : i64
    %Y_499 = arith.cmpi sge, %Y_495, %Y_497 : i64
    %Y_500 = arith.andi %Y_498, %Y_499 : i1
    %Y_501 = arith.muli %start_z, %stride_y_z : i32
    %Y_502 = tt.addptr %Y, %Y_501 : !tt.ptr<bf16>, i32
    %Y_503 = arith.extsi %start_m_125 : i32 to i64
    %Y_504 = arith.extsi %stride_y_m : i32 to i64
    %Y_505 = arith.muli %Y_503, %Y_504 : i64
    %Y_506 = arith.constant 2147483647 : i64
    %Y_507 = arith.constant -2147483648 : i64
    %Y_508 = arith.cmpi sle, %Y_505, %Y_506 : i64
    %Y_509 = arith.cmpi sge, %Y_505, %Y_507 : i64
    %Y_510 = arith.andi %Y_508, %Y_509 : i1
    %Y_511 = arith.muli %start_m_125, %stride_y_m : i32
    %Y_512 = tt.addptr %Y_502, %Y_511 : !tt.ptr<bf16>, i32
    %YPtrs = tt.expand_dims %offs_m_442 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %YPtrs_513 = tt.splat %stride_y_m : i32 -> tensor<128x1xi32>
    %YPtrs_514 = arith.extsi %YPtrs : tensor<128x1xi32> to tensor<128x1xi64>
    %YPtrs_515 = arith.extsi %YPtrs_513 : tensor<128x1xi32> to tensor<128x1xi64>
    %YPtrs_516 = arith.muli %YPtrs_514, %YPtrs_515 : tensor<128x1xi64>
    %YPtrs_517 = arith.constant 2147483647 : i64
    %YPtrs_518 = arith.constant -2147483648 : i64
    %YPtrs_519 = arith.constant dense<2147483647> : tensor<128x1xi64>
    %YPtrs_520 = arith.cmpi sle, %YPtrs_516, %YPtrs_519 : tensor<128x1xi64>
    %YPtrs_521 = arith.constant dense<-2147483648> : tensor<128x1xi64>
    %YPtrs_522 = arith.cmpi sge, %YPtrs_516, %YPtrs_521 : tensor<128x1xi64>
    %YPtrs_523 = arith.andi %YPtrs_520, %YPtrs_522 : tensor<128x1xi1>
    %YPtrs_524 = arith.muli %YPtrs, %YPtrs_513 : tensor<128x1xi32>
    %YPtrs_525 = tt.splat %Y_512 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
    %YPtrs_526 = tt.addptr %YPtrs_525, %YPtrs_524 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
    %YPtrs_527 = tt.expand_dims %offs_y_n_465 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %YPtrs_528 = arith.constant 1 : i32
    %YPtrs_529 = arith.constant 1 : i32
    %YPtrs_530 = arith.constant dense<1> : tensor<1x256xi32>
    %YPtrs_531 = arith.extsi %YPtrs_527 : tensor<1x256xi32> to tensor<1x256xi64>
    %YPtrs_532 = arith.extsi %YPtrs_530 : tensor<1x256xi32> to tensor<1x256xi64>
    %YPtrs_533 = arith.muli %YPtrs_531, %YPtrs_532 : tensor<1x256xi64>
    %YPtrs_534 = arith.constant 2147483647 : i64
    %YPtrs_535 = arith.constant -2147483648 : i64
    %YPtrs_536 = arith.constant dense<2147483647> : tensor<1x256xi64>
    %YPtrs_537 = arith.cmpi sle, %YPtrs_533, %YPtrs_536 : tensor<1x256xi64>
    %YPtrs_538 = arith.constant dense<-2147483648> : tensor<1x256xi64>
    %YPtrs_539 = arith.cmpi sge, %YPtrs_533, %YPtrs_538 : tensor<1x256xi64>
    %YPtrs_540 = arith.andi %YPtrs_537, %YPtrs_539 : tensor<1x256xi1>
    %YPtrs_541 = arith.muli %YPtrs_527, %YPtrs_530 : tensor<1x256xi32>
    %YPtrs_542 = tt.broadcast %YPtrs_526 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x256x!tt.ptr<bf16>>
    %YPtrs_543 = tt.broadcast %YPtrs_541 : tensor<1x256xi32> -> tensor<128x256xi32>
    %YPtrs_544 = tt.addptr %YPtrs_542, %YPtrs_543 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %mask = tt.expand_dims %mask_m_466 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
    %mask_545 = tt.expand_dims %mask_n_467 {axis = 0 : i32} : tensor<256xi1> -> tensor<1x256xi1>
    %mask_546 = tt.broadcast %mask : tensor<128x1xi1> -> tensor<128x256xi1>
    %mask_547 = tt.broadcast %mask_545 : tensor<1x256xi1> -> tensor<128x256xi1>
    %mask_548 = arith.andi %mask_546, %mask_547 : tensor<128x256xi1>
    %out_549 = tt.call @"triton_kernels.numerics_details.flexpoint.float_to_flex__fp32S128_256S_u1S128_256S_Pbf16__(1,)cconstexpr_None__(2,)cconstexpr_None__(3,)cconstexpr_None__(6,)cconstexpr_False_"(%out_492, %mask_548, %Y_512) : (tensor<128x256xf32>, tensor<128x256xi1>, !tt.ptr<bf16>) -> tensor<128x256xf32>
    %39 = arith.truncf %out_549 : tensor<128x256xf32> to tensor<128x256xbf16>
    tt.store %YPtrs_544, %39, %mask_548 : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
  tt.func private @"triton_kernels.matmul_ogs_details._common.swizzle2d__i32_i32_i32__(3,)cconstexpr_4_"(%pid: i32, %grid_m: i32, %grid_n: i32) -> (i32, i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %width = arith.muli %grid_n, %c4_i32 : i32
    %group_id = arith.divsi %pid, %width : i32
    %group_size = arith.muli %group_id, %c4_i32 : i32
    %group_size_0 = arith.subi %grid_m, %group_size : i32
    %group_size_1 = arith.minsi %group_size_0, %c4_i32 : i32
    %0 = arith.cmpi sge, %group_size_1, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    %pid_m = arith.muli %group_id, %c4_i32 : i32
    %pid_m_2 = arith.remsi %pid, %group_size_1 : i32
    %pid_m_3 = arith.addi %pid_m, %pid_m_2 : i32
    %pid_n = arith.remsi %pid, %width : i32
    %pid_n_4 = arith.divsi %pid_n, %group_size_1 : i32
    tt.return %pid_m_3, %pid_n_4 : i32, i32
  }
  tt.func private @"triton_kernels.matmul_ogs_details._common.xcd_swizzle__i32_i32__(2,)cconstexpr_8_"(%pid: i32, %domain_size: i32) -> i32 attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %pids_per_group = arith.divsi %domain_size, %c8_i32 : i32
    %extra_pid_groups = arith.remsi %domain_size, %c8_i32 : i32
    %group = arith.remsi %pid, %c8_i32 : i32
    %local_pid = arith.divsi %pid, %c8_i32 : i32
    %new_pid = arith.muli %group, %pids_per_group : i32
    %new_pid_0 = arith.minsi %group, %extra_pid_groups : i32
    %new_pid_1 = arith.addi %new_pid, %new_pid_0 : i32
    %new_pid_2 = arith.addi %new_pid_1, %local_pid : i32
    tt.return %new_pid_2 : i32
  }
  tt.func private @"triton.language.standard.zeros____(0, 0)cconstexpr_128__(0, 1)cconstexpr_256__(1,)cconstexpr_fp32_"() -> tensor<128x256xf32> attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
    tt.return %cst_0 : tensor<128x256xf32>
  ^bb1:  // no predecessors
    %0 = ub.poison : tensor<128x256xf32>
    tt.return %0 : tensor<128x256xf32>
  }
  tt.func private @"triton_kernels.numerics_details.flexpoint.load_scale____(0,)cconstexpr_None_"() -> f32 attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    tt.return %cst : f32
  ^bb1:  // no predecessors
    %0 = ub.poison : f32
    tt.return %0 : f32
  }
  tt.func private @"triton_kernels.numerics_details.flexpoint.float_to_flex__fp32S128_256S_u1S128_256S_Pbf16__(1,)cconstexpr_None__(2,)cconstexpr_None__(3,)cconstexpr_None__(6,)cconstexpr_False_"(%x: tensor<128x256xf32>, %mask: tensor<128x256xi1>, %Out: !tt.ptr<bf16>) -> tensor<128x256xf32> attributes {noinline = false} {
    %invscale = arith.constant 1.000000e+00 : f32
    tt.call @"triton_kernels.numerics_details.flexpoint.update_scale__fp32S128_256S_Pbf16__(1,)cconstexpr_None_"(%x, %Out) : (tensor<128x256xf32>, !tt.ptr<bf16>) -> ()
    %x_0 = arith.constant dense<1.000000e+00> : tensor<128x256xf32>
    %x_1 = arith.mulf %x, %x_0 : tensor<128x256xf32>
    tt.return %x_1 : tensor<128x256xf32>
  ^bb1:  // no predecessors
    %0 = ub.poison : tensor<128x256xf32>
    tt.return %0 : tensor<128x256xf32>
  }
  tt.func private @"triton_kernels.numerics_details.flexpoint.update_scale__fp32S128_256S_Pbf16__(1,)cconstexpr_None_"(%x: tensor<128x256xf32>, %Out: !tt.ptr<bf16>) attributes {noinline = false} {
    tt.return
  }
}

// -----// IR Dump Before TritonAMDGPUStreamPipeline (tritonamdgpu-stream-pipeline) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 16], [1, 0, 0], [2, 0, 0], [0, 64, 0], [0, 128, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 4], [0, 0, 8]], warp = [[0, 16, 0], [0, 32, 0], [0, 0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0], [0, 0, 64], [0, 0, 128]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 4, 0], [0, 8, 0]], warp = [[0, 0, 16], [0, 0, 32], [0, 0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_matmul_ogs_NNT_bf16xbf16xfp8e4nv_128x256x128x1(%Y: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %YPtr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_y_k: i32 {tt.divisibility = 16 : i32}, %stride_y_z: i32 {tt.divisibility = 16 : i32}, %stride_y_m: i32 {tt.divisibility = 16 : i32}, %X: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %XPtr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_x_z: i32 {tt.divisibility = 16 : i32}, %stride_x_m: i32 {tt.divisibility = 16 : i32}, %W: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %WPtr: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_w_e: i32 {tt.divisibility = 16 : i32}, %stride_w_n: i32 {tt.divisibility = 16 : i32}, %WMxScale: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %stride_w_mx_e: i32 {tt.divisibility = 16 : i32}, %stride_w_mx_n: i32, %B: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_b_e: i32 {tt.divisibility = 16 : i32}, %NRows: i32, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %GatherIndx: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ScatterSrcIndx: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %num_idxs: i32 {tt.divisibility = 16 : i32}, %ExptHist: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptOffs: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptOffsSum: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ExptData: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %grid_m: i32, %grid_n: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
    %cst_0 = arith.constant dense<128> : tensor<128x256xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<128x128xi32, #blocked1>
    %cst_2 = arith.constant dense<32> : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %cst_3 = arith.constant dense<4> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked1>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x256xf8E4M3FN, #blocked>
    %cst_7 = arith.constant dense<4> : tensor<256x4xi32, #blocked2>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %cst_8 = arith.constant dense<7> : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
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
    %11 = arith.cmpi sge, %grid_n, %c0_i32 : i32
    llvm.intr.assume %11 : i1
    %pid = tt.get_program_id x : i32
    %padding_m = tt.load %ExptOffsSum : !tt.ptr<i32>
    %padding_m_9 = arith.subi %grid_m, %padding_m : i32
    %unpadded_m = arith.subi %grid_m, %padding_m_9 : i32
    %12 = arith.cmpi sge, %unpadded_m, %c0_i32 : i32
    llvm.intr.assume %12 : i1
    %total_actual_tiles = arith.muli %unpadded_m, %grid_n : i32
    %13 = arith.cmpi sgt, %padding_m_9, %c0_i32 : i32
    %14 = arith.cmpi sge, %pid, %total_actual_tiles : i32
    %15 = arith.andi %13, %14 : i1
    cf.cond_br %15, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %pid_mn = arith.subi %pid, %total_actual_tiles : i32
    %16 = arith.muli %padding_m_9, %grid_n : i32
    %17 = arith.cmpi slt, %pid_mn, %16 : i32
    scf.if %17 {
      %width_108 = arith.muli %grid_n, %c4_i32 : i32
      %group_id_109 = arith.divsi %pid_mn, %width_108 : i32
      %group_size_110 = arith.muli %group_id_109, %c4_i32 : i32
      %group_size_111 = arith.subi %padding_m_9, %group_size_110 : i32
      %group_size_112 = arith.minsi %group_size_111, %c4_i32 : i32
      %21 = arith.cmpi sge, %group_size_112, %c0_i32 : i32
      llvm.intr.assume %21 : i1
    }
    tt.return
  ^bb2:  // pred: ^bb0
    %pids_per_group = arith.divsi %total_actual_tiles, %c8_i32 : i32
    %extra_pid_groups = arith.remsi %total_actual_tiles, %c8_i32 : i32
    %group = arith.remsi %pid, %c8_i32 : i32
    %local_pid = arith.divsi %pid, %c8_i32 : i32
    %new_pid = arith.muli %group, %pids_per_group : i32
    %new_pid_10 = arith.minsi %group, %extra_pid_groups : i32
    %new_pid_11 = arith.addi %new_pid, %new_pid_10 : i32
    %new_pid_12 = arith.addi %new_pid_11, %local_pid : i32
    %pid_mnk = arith.remsi %new_pid_12, %total_actual_tiles : i32
    %width = arith.muli %grid_n, %c4_i32 : i32
    %group_id = arith.divsi %pid_mnk, %width : i32
    %group_size = arith.muli %group_id, %c4_i32 : i32
    %group_size_13 = arith.subi %unpadded_m, %group_size : i32
    %group_size_14 = arith.minsi %group_size_13, %c4_i32 : i32
    %18 = arith.cmpi sge, %group_size_14, %c0_i32 : i32
    llvm.intr.assume %18 : i1
    %pid_m = arith.remsi %pid_mnk, %group_size_14 : i32
    %pid_m_15 = arith.addi %group_size, %pid_m : i32
    %pid_n = arith.remsi %pid_mnk, %width : i32
    %pid_n_16 = arith.divsi %pid_n, %group_size_14 : i32
    %expt_data = tt.addptr %ExptData, %pid_m_15 : !tt.ptr<i32>, i32
    %expt_data_17 = tt.load %expt_data : !tt.ptr<i32>
    %19 = arith.cmpi eq, %expt_data_17, %c-1_i32 : i32
    cf.cond_br %19, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    tt.return
  ^bb4:  // pred: ^bb2
    %expt_id = arith.andi %expt_data_17, %c65535_i32 : i32
    %block_id = arith.shrsi %expt_data_17, %c16_i32 : i32
    %M = tt.addptr %ExptHist, %expt_id : !tt.ptr<i32>, i32
    %M_18 = tt.load %M : !tt.ptr<i32>
    %start_m = tt.addptr %ExptOffs, %expt_id : !tt.ptr<i32>, i32
    %start_m_19 = tt.load %start_m : !tt.ptr<i32>
    %offs_x_m = arith.muli %block_id, %c128_i32 : i32
    %offs_x_m_20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_x_m_22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %offs_x_m_23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_x_m_24 = tt.splat %offs_x_m : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_25 = tt.splat %offs_x_m : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %offs_x_m_26 = arith.addi %offs_x_m_24, %offs_x_m_20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_27 = arith.addi %offs_x_m_25, %offs_x_m_22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %offs_x_m_28 = tt.splat %M_18 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_29 = tt.splat %M_18 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %offs_x_m_30 = arith.remsi %offs_x_m_26, %offs_x_m_28 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %GatherIndx_31 = tt.addptr %GatherIndx, %start_m_19 : !tt.ptr<i32>, i32
    %offs_x_m_32 = tt.splat %GatherIndx_31 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_33 = tt.addptr %offs_x_m_32, %offs_x_m_30 : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_34 = tt.load %offs_x_m_33 : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_x_m_35 = arith.divsi %offs_x_m_34, %cst_3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %XPtrs = tt.expand_dims %offs_x_m_35 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %XPtrs_36 = tt.splat %stride_x_m : i32 -> tensor<128x1xi32, #blocked1>
    %XPtrs_37 = arith.muli %XPtrs, %XPtrs_36 : tensor<128x1xi32, #blocked1>
    %XPtrs_38 = tt.splat %X : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>, #blocked1>
    %XPtrs_39 = tt.addptr %XPtrs_38, %XPtrs_37 : tensor<128x1x!tt.ptr<bf16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %XPtrs_40 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %XPtrs_41 = tt.expand_dims %XPtrs_40 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %XPtrs_42 = tt.broadcast %XPtrs_39 : tensor<128x1x!tt.ptr<bf16>, #blocked1> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
    %XPtrs_43 = tt.broadcast %XPtrs_41 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %XPtrs_44 = tt.addptr %XPtrs_42, %XPtrs_43 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %WMxScale_45 = arith.muli %expt_id, %stride_w_mx_e : i32
    %WMxScale_46 = tt.addptr %WMxScale, %WMxScale_45 : !tt.ptr<i8>, i32
    %offs_n_scale = arith.muli %pid_n_16, %c256_i32 : i32
    %offs_n_scale_47 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_48 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_scale_49 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %offs_n_scale_50 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %offs_n_scale_51 = tt.splat %offs_n_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_52 = tt.splat %offs_n_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_scale_53 = tt.splat %offs_n_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %offs_n_scale_54 = tt.splat %offs_n_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %offs_n_scale_55 = arith.addi %offs_n_scale_51, %offs_n_scale_47 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_56 = arith.addi %offs_n_scale_52, %offs_n_scale_48 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_scale_57 = arith.addi %offs_n_scale_53, %offs_n_scale_49 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %offs_n_scale_58 = arith.addi %offs_n_scale_54, %offs_n_scale_50 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %offs_n_scale_59 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_60 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_scale_61 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %offs_n_scale_62 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %offs_n_scale_63 = arith.remsi %offs_n_scale_55, %offs_n_scale_59 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_scale_64 = arith.remsi %offs_n_scale_56, %offs_n_scale_60 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_k_scale = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %WMxScalePtrs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %WMxScalePtrs_65 = tt.expand_dims %WMxScalePtrs {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x4xi32, #blocked2>
    %WMxScalePtrs_66 = tt.splat %WMxScale_46 : !tt.ptr<i8> -> tensor<1x4x!tt.ptr<i8>, #blocked2>
    %WMxScalePtrs_67 = tt.addptr %WMxScalePtrs_66, %WMxScalePtrs_65 : tensor<1x4x!tt.ptr<i8>, #blocked2>, tensor<1x4xi32, #blocked2>
    %WMxScalePtrs_68 = tt.expand_dims %offs_n_scale_63 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<256x1xi32, #blocked2>
    %WMxScalePtrs_69 = tt.splat %stride_w_mx_n : i32 -> tensor<256x1xi32, #blocked2>
    %WMxScalePtrs_70 = arith.muli %WMxScalePtrs_68, %WMxScalePtrs_69 : tensor<256x1xi32, #blocked2>
    %WMxScalePtrs_71 = tt.broadcast %WMxScalePtrs_67 : tensor<1x4x!tt.ptr<i8>, #blocked2> -> tensor<256x4x!tt.ptr<i8>, #blocked2>
    %WMxScalePtrs_72 = tt.broadcast %WMxScalePtrs_70 : tensor<256x1xi32, #blocked2> -> tensor<256x4xi32, #blocked2>
    %WMxScalePtrs_73 = tt.addptr %WMxScalePtrs_71, %WMxScalePtrs_72 : tensor<256x4x!tt.ptr<i8>, #blocked2>, tensor<256x4xi32, #blocked2>
    %W_74 = arith.muli %expt_id, %stride_w_e : i32
    %W_75 = tt.addptr %W, %W_74 : !tt.ptr<f8E4M3FN>, i32
    %WPtrs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %WPtrs_76 = tt.expand_dims %WPtrs {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %WPtrs_77 = tt.expand_dims %offs_n_scale_64 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %WPtrs_78 = tt.splat %stride_w_n : i32 -> tensor<1x256xi32, #blocked>
    %WPtrs_79 = arith.muli %WPtrs_77, %WPtrs_78 : tensor<1x256xi32, #blocked>
    %WPtrs_80 = tt.broadcast %WPtrs_76 : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %WPtrs_81 = tt.broadcast %WPtrs_79 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %WPtrs_82 = arith.addi %WPtrs_80, %WPtrs_81 : tensor<128x256xi32, #blocked>
    %WPtrs_83 = tt.splat %W_75 : !tt.ptr<f8E4M3FN> -> tensor<128x256x!tt.ptr<f8E4M3FN>, #blocked>
    %WPtrs_84 = tt.addptr %WPtrs_83, %WPtrs_82 : tensor<128x256x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x256xi32, #blocked>
    %mask_k_scale = arith.muli %offs_k_scale, %cst_2 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %acc:4 = scf.for %acc_108 = %c0_i32 to %K step %c128_i32 iter_args(%WMxScalePtrs_109 = %WMxScalePtrs_73, %arg32 = %cst_4, %XPtrs_110 = %XPtrs_44, %WPtrs_111 = %WPtrs_84) -> (tensor<256x4x!tt.ptr<i8>, #blocked2>, tensor<128x256xf32, #mma>, tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x256x!tt.ptr<f8E4M3FN>, #blocked>)  : i32 {
      %mask_k = arith.subi %K, %acc_108 : i32
      %mask_k_112 = tt.splat %mask_k : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %mask_k_113 = tt.splat %mask_k : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %mask_k_114 = arith.cmpi slt, %offs_x_m_23, %mask_k_112 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %mask_k_115 = arith.cmpi slt, %offs_x_m_21, %mask_k_113 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %mask_k_scale_116 = tt.splat %mask_k : i32 -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %mask_k_scale_117 = arith.cmpi slt, %mask_k_scale, %mask_k_scale_116 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %x = tt.expand_dims %mask_k_114 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi1, #blocked1>
      %x_118 = tt.broadcast %x : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
      %x_119 = tt.load %XPtrs_110, %x_118, %cst_5 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
      %w = tt.expand_dims %mask_k_115 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
      %w_120 = tt.broadcast %w : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
      %w_121 = tt.load %WPtrs_111, %w_120, %cst_6 : tensor<128x256x!tt.ptr<f8E4M3FN>, #blocked>
      %acc_122 = ttg.convert_layout %w_121 : tensor<128x256xf8E4M3FN, #blocked> -> tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %w_scales = tt.expand_dims %mask_k_scale_117 {axis = 0 : i32} : tensor<4xi1, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x4xi1, #blocked2>
      %w_scales_123 = tt.broadcast %w_scales : tensor<1x4xi1, #blocked2> -> tensor<256x4xi1, #blocked2>
      %w_scales_124 = tt.load %WMxScalePtrs_109, %w_scales_123 : tensor<256x4x!tt.ptr<i8>, #blocked2>
      %w_125 = ttg.convert_layout %w_scales_124 : tensor<256x4xi8, #blocked2> -> tensor<256x4xi8, #linear1>
      %w_126 = tt.fp_to_fp %acc_122 : tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %w_127 = tt.trans %w_125 {order = array<i32: 1, 0>} : tensor<256x4xi8, #linear1> -> tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_128 = arith.extui %w_127 : tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>> to tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_129 = arith.shli %w_128, %cst_8 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_130 = tt.bitcast %w_129 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_131 = tt.expand_dims %w_130 {axis = 2 : i32} : tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256x1xbf16, #linear>
      %w_132 = tt.broadcast %w_131 : tensor<4x256x1xbf16, #linear> -> tensor<4x256x32xbf16, #linear>
      %w_133 = tt.trans %w_132 {order = array<i32: 0, 2, 1>} : tensor<4x256x32xbf16, #linear> -> tensor<4x32x256xbf16, #linear2>
      %w_134 = tt.reshape %w_133 : tensor<4x32x256xbf16, #linear2> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %w_135 = arith.mulf %w_126, %w_134 : tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %acc_136 = ttg.convert_layout %x_119 : tensor<128x128xbf16, #blocked1> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %acc_137 = tt.dot %acc_136, %w_135, %arg32 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xf32, #mma>
      %WMxScalePtrs_138 = tt.addptr %WMxScalePtrs_109, %cst_7 : tensor<256x4x!tt.ptr<i8>, #blocked2>, tensor<256x4xi32, #blocked2>
      %XPtrs_139 = tt.addptr %XPtrs_110, %cst_1 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
      %WPtrs_140 = tt.addptr %WPtrs_111, %cst_0 : tensor<128x256x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x256xi32, #blocked>
      scf.yield %WMxScalePtrs_138, %acc_137, %XPtrs_139, %WPtrs_140 : tensor<256x4x!tt.ptr<i8>, #blocked2>, tensor<128x256xf32, #mma>, tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x256x!tt.ptr<f8E4M3FN>, #blocked>
    }
    %mask_m = arith.cmpi slt, %offs_x_m_27, %offs_x_m_29 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %mask_n = arith.cmpi slt, %offs_n_scale_57, %offs_n_scale_61 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %mask_n_85 = arith.cmpi slt, %offs_n_scale_58, %offs_n_scale_62 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %BPtrs = arith.muli %expt_id, %stride_b_e : i32
    %BPtrs_86 = tt.addptr %B, %BPtrs : !tt.ptr<f32>, i32
    %BPtrs_87 = tt.splat %BPtrs_86 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
    %BPtrs_88 = tt.addptr %BPtrs_87, %offs_n_scale_58 : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %bias = tt.load %BPtrs_88, %mask_n_85, %cst : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
    %acc_89 = tt.expand_dims %bias {axis = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xf32, #mma>
    %acc_90 = tt.broadcast %acc_89 : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
    %acc_91 = arith.addf %acc#1, %acc_90 : tensor<128x256xf32, #mma>
    %Y_92 = arith.muli %start_m_19, %stride_y_m : i32
    %Y_93 = tt.addptr %Y, %Y_92 : !tt.ptr<bf16>, i32
    %YPtrs = tt.expand_dims %offs_x_m_27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi32, #blocked3>
    %YPtrs_94 = tt.splat %stride_y_m : i32 -> tensor<128x1xi32, #blocked3>
    %YPtrs_95 = arith.muli %YPtrs, %YPtrs_94 : tensor<128x1xi32, #blocked3>
    %YPtrs_96 = tt.splat %Y_93 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>, #blocked3>
    %YPtrs_97 = tt.addptr %YPtrs_96, %YPtrs_95 : tensor<128x1x!tt.ptr<bf16>, #blocked3>, tensor<128x1xi32, #blocked3>
    %YPtrs_98 = tt.expand_dims %offs_n_scale_57 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
    %YPtrs_99 = tt.broadcast %YPtrs_97 : tensor<128x1x!tt.ptr<bf16>, #blocked3> -> tensor<128x256x!tt.ptr<bf16>, #blocked3>
    %YPtrs_100 = tt.broadcast %YPtrs_98 : tensor<1x256xi32, #blocked3> -> tensor<128x256xi32, #blocked3>
    %YPtrs_101 = tt.addptr %YPtrs_99, %YPtrs_100 : tensor<128x256x!tt.ptr<bf16>, #blocked3>, tensor<128x256xi32, #blocked3>
    %mask = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi1, #blocked3>
    %mask_102 = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi1, #blocked3>
    %mask_103 = tt.broadcast %mask : tensor<128x1xi1, #blocked3> -> tensor<128x256xi1, #blocked3>
    %mask_104 = tt.broadcast %mask_102 : tensor<1x256xi1, #blocked3> -> tensor<128x256xi1, #blocked3>
    %mask_105 = arith.andi %mask_103, %mask_104 : tensor<128x256xi1, #blocked3>
    %20 = arith.truncf %acc_91 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
    %YPtrs_106 = ttg.convert_layout %YPtrs_101 : tensor<128x256x!tt.ptr<bf16>, #blocked3> -> tensor<128x256x!tt.ptr<bf16>, #mma>
    %mask_107 = ttg.convert_layout %mask_105 : tensor<128x256xi1, #blocked3> -> tensor<128x256xi1, #mma>
    tt.store %YPtrs_106, %20, %mask_107 : tensor<128x256x!tt.ptr<bf16>, #mma>
    tt.return
  }
}

// -----// IR Dump Before OptimizeAMDLDSUsage (optimize-amd-lds-usage) ('builtin.module' operation) //----- //
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
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %pid_mn = arith.subi %pid, %total_actual_tiles : i32
    %3 = arith.muli %padding_m_4, %grid_n : i32
    %4 = arith.cmpi slt, %pid_mn, %3 : i32
    scf.if %4 {
      llvm.intr.assume %true : i1
    }
    tt.return
  ^bb2:  // pred: ^bb0
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
    cf.cond_br %5, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    tt.return
  ^bb4:  // pred: ^bb2
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
    %acc_85:8 = scf.for %acc_130 = %c0_i32 to %acc_84 step %c128_i32 iter_args(%WMxScalePtrs_131 = %WMxScale_45, %arg32 = %cst_1, %XPtrs_132 = %X, %WPtrs_133 = %W_67, %arg35 = %c0_i32, %K_134 = %K, %x_135 = %x_82, %w_136 = %w_83) -> (!tt.ptr<i8>, tensor<128x256xf32, #mma>, !tt.ptr<bf16>, !tt.ptr<f8E4M3FN>, i32, i32, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>)  : i32 {
      %XPtrs_137 = tt.addptr %XPtrs_132, %c128_i32 : !tt.ptr<bf16>, i32
      %WPtrs_138 = tt.addptr %WPtrs_133, %c128_i32 : !tt.ptr<f8E4M3FN>, i32
      %acc_139 = arith.addi %acc_130, %c128_i32 : i32
      %mask_k_140 = arith.subi %K, %acc_139 : i32
      %mask_k_141 = tt.splat %mask_k_140 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %mask_k_142 = tt.splat %mask_k_140 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %mask_k_143 = arith.cmpi slt, %offs_x_m_18, %mask_k_141 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %mask_k_144 = arith.cmpi slt, %offs_x_m_16, %mask_k_142 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %mask_k_scale_145 = tt.splat %K_134 : i32 -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %mask_k_scale_146 = arith.cmpi slt, %mask_k_scale, %mask_k_scale_145 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %x_147 = tt.expand_dims %mask_k_143 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi1, #blocked1>
      %x_148 = tt.broadcast %x_147 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
      %x_149 = tt.splat %XPtrs_137 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
      %x_150 = tt.addptr %x_149, %XPtrs_36 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
      %x_151 = tt.load %x_150, %x_148, %cst_2 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
      %acc_152 = ttg.local_load %x_135 : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %w_153 = tt.expand_dims %mask_k_144 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi1, #blocked2>
      %w_154 = tt.broadcast %w_153 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
      %w_155 = amdgpu.buffer_load %WPtrs_138[%WPtrs_73], %w_154 stride = %stride_w_n : tensor<128x256xf8E4M3FN, #blocked2>
      %acc_156 = ttg.local_load %w_136 : !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256> -> tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %w_scales_157 = tt.expand_dims %mask_k_scale_146 {axis = 0 : i32} : tensor<4xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi1, #blocked>
      %w_scales_158 = tt.broadcast %w_scales_157 : tensor<1x4xi1, #blocked> -> tensor<256x4xi1, #blocked>
      %w_scales_159 = amdgpu.buffer_load %WMxScalePtrs_131[%WMxScalePtrs_65], %w_scales_158 stride = %stride_w_mx_n : tensor<256x4xi8, #blocked>
      %w_160 = ttg.convert_layout %w_scales_159 : tensor<256x4xi8, #blocked> -> tensor<256x4xi8, #linear1>
      %w_161 = tt.trans %w_160 {order = array<i32: 1, 0>} : tensor<256x4xi8, #linear1> -> tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_162 = tt.fp_to_fp %acc_156 : tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %w_163 = arith.extui %w_161 : tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>> to tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_164 = arith.shli %w_163, %cst_3 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_165 = tt.bitcast %w_164 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>>
      %w_166 = tt.expand_dims %w_165 {axis = 2 : i32} : tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256x1xbf16, #linear>
      %w_167 = tt.broadcast %w_166 : tensor<4x256x1xbf16, #linear> -> tensor<4x256x32xbf16, #linear>
      %w_168 = tt.trans %w_167 {order = array<i32: 0, 2, 1>} : tensor<4x256x32xbf16, #linear> -> tensor<4x32x256xbf16, #linear2>
      %w_169 = tt.reshape %w_168 : tensor<4x32x256xbf16, #linear2> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %w_170 = arith.mulf %w_162, %w_169 : tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %acc_171 = tt.dot %acc_152, %w_170, %arg32 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xf32, #mma>
      %WMxScalePtrs_172 = tt.addptr %WMxScalePtrs_131, %c4_i32 : !tt.ptr<i8>, i32
      %acc_173 = arith.addi %arg35, %c1_i32 : i32
      %acc_174 = arith.cmpi slt, %acc_173, %c1_i32 : i32
      %acc_175 = arith.select %acc_174, %acc_173, %c0_i32 : i32
      %x_176 = ttg.memdesc_index %x_80[%acc_175] : !ttg.memdesc<1x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>
      ttg.local_store %x_151, %x_176 : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>
      %w_177 = ttg.memdesc_index %w_81[%acc_175] : !ttg.memdesc<1x128x256xf8E4M3FN, #shared1, #smem, mutable> -> !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>
      ttg.local_store %w_155, %w_177 : tensor<128x256xf8E4M3FN, #blocked2> -> !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>
      scf.yield %WMxScalePtrs_172, %acc_171, %XPtrs_137, %WPtrs_138, %acc_175, %mask_k_140, %x_176, %w_177 : !tt.ptr<i8>, tensor<128x256xf32, #mma>, !tt.ptr<bf16>, !tt.ptr<f8E4M3FN>, i32, i32, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256>
    }
    %acc_86 = arith.addi %K, %c127_i32 : i32
    %acc_87 = arith.divsi %acc_86, %c128_i32 : i32
    %acc_88 = arith.cmpi sge, %acc_87, %c1_i32 : i32
    %mask_k_scale_89 = tt.splat %acc_85#5 : i32 -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_k_scale_90 = arith.cmpi slt, %mask_k_scale, %mask_k_scale_89 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %acc_91 = ttg.local_load %acc_85#6 : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable, 1x128x128> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %acc_92 = ttg.local_load %acc_85#7 : !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem, mutable, 1x128x256> -> tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_scales = tt.expand_dims %mask_k_scale_90 {axis = 0 : i32} : tensor<4xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi1, #blocked>
    %w_scales_93 = tt.broadcast %w_scales : tensor<1x4xi1, #blocked> -> tensor<256x4xi1, #blocked>
    %acc_94 = tt.splat %acc_88 : i1 -> tensor<256x4xi1, #blocked>
    %acc_95 = arith.andi %acc_94, %w_scales_93 : tensor<256x4xi1, #blocked>
    %w_scales_96 = amdgpu.buffer_load %acc_85#0[%WMxScalePtrs_65], %acc_95 stride = %stride_w_mx_n : tensor<256x4xi8, #blocked>
    %w_97 = tt.trans %w_scales_96 {order = array<i32: 1, 0>} : tensor<256x4xi8, #blocked> -> tensor<4x256xi8, #blocked3>
    %w_98 = tt.fp_to_fp %acc_92 : tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_99 = ttg.convert_layout %w_97 : tensor<4x256xi8, #blocked3> -> tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_100 = arith.extui %w_99 : tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>> to tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_101 = arith.shli %w_100, %cst_3 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_102 = tt.bitcast %w_101 : tensor<4x256xi16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>>
    %w_103 = tt.expand_dims %w_102 {axis = 2 : i32} : tensor<4x256xbf16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x256x1xbf16, #linear>
    %w_104 = tt.broadcast %w_103 : tensor<4x256x1xbf16, #linear> -> tensor<4x256x32xbf16, #linear>
    %w_105 = tt.trans %w_104 {order = array<i32: 0, 2, 1>} : tensor<4x256x32xbf16, #linear> -> tensor<4x32x256xbf16, #linear2>
    %w_106 = tt.reshape %w_105 : tensor<4x32x256xbf16, #linear2> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %w_107 = arith.mulf %w_98, %w_106 : tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %acc_108 = scf.if %acc_88 -> (tensor<128x256xf32, #mma>) {
      %acc_130 = tt.dot %acc_91, %w_107, %acc_85#1 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xf32, #mma>
      scf.yield %acc_130 : tensor<128x256xf32, #mma>
    } else {
      scf.yield %acc_85#1 : tensor<128x256xf32, #mma>
    }
    %acc_109 = arith.select %acc_88, %acc_108, %acc_85#1 : tensor<128x256xf32, #mma>
    ttg.local_dealloc %w_81 : !ttg.memdesc<1x128x256xf8E4M3FN, #shared1, #smem, mutable>
    ttg.local_dealloc %x_80 : !ttg.memdesc<1x128x128xbf16, #shared, #smem, mutable>
    %mask_m = arith.cmpi slt, %offs_x_m_22, %offs_x_m_24 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %mask_n = arith.cmpi slt, %offs_n_scale_54, %offs_n_scale_57 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %BPtrs = arith.muli %expt_id, %stride_b_e : i32
    %BPtrs_110 = tt.addptr %B, %BPtrs : !tt.ptr<f32>, i32
    %BPtrs_111 = tt.addptr %BPtrs_110, %offs_n_scale : !tt.ptr<f32>, i32
    %bias = amdgpu.buffer_load %BPtrs_111[%offs_n_scale_48], %mask_n : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
    %acc_112 = tt.expand_dims %bias {axis = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xf32, #mma>
    %acc_113 = tt.broadcast %acc_112 : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
    %acc_114 = arith.addf %acc_109, %acc_113 : tensor<128x256xf32, #mma>
    %Y_115 = arith.muli %start_m_14, %stride_y_m : i32
    %Y_116 = tt.addptr %Y, %Y_115 : !tt.ptr<bf16>, i32
    %YPtrs = tt.expand_dims %offs_x_m_17 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %YPtrs_117 = arith.muli %offs_x_m, %stride_y_m : i32
    %YPtrs_118 = tt.splat %stride_y_m : i32 -> tensor<128x1xi32, #mma>
    %YPtrs_119 = arith.muli %YPtrs, %YPtrs_118 : tensor<128x1xi32, #mma>
    %YPtrs_120 = tt.addptr %Y_116, %YPtrs_117 : !tt.ptr<bf16>, i32
    %YPtrs_121 = tt.broadcast %YPtrs_119 : tensor<128x1xi32, #mma> -> tensor<128x256xi32, #mma>
    %YPtrs_122 = tt.expand_dims %offs_n_scale_48 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi32, #mma>
    %YPtrs_123 = tt.broadcast %YPtrs_122 : tensor<1x256xi32, #mma> -> tensor<128x256xi32, #mma>
    %YPtrs_124 = tt.addptr %YPtrs_120, %offs_n_scale : !tt.ptr<bf16>, i32
    %YPtrs_125 = arith.addi %YPtrs_123, %YPtrs_121 : tensor<128x256xi32, #mma>
    %mask = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi1, #mma>
    %mask_126 = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi1, #mma>
    %mask_127 = tt.broadcast %mask : tensor<128x1xi1, #mma> -> tensor<128x256xi1, #mma>
    %mask_128 = tt.broadcast %mask_126 : tensor<1x256xi1, #mma> -> tensor<128x256xi1, #mma>
    %mask_129 = arith.andi %mask_127, %mask_128 : tensor<128x256xi1, #mma>
    %6 = arith.truncf %acc_114 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
    amdgpu.buffer_store %6, %YPtrs_124[%YPtrs_125], %mask_129 stride = %stride_y_m : tensor<128x256xbf16, #mma>
    tt.return
  }
}

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


// -----// IR Dump Before ConvertTritonAMDGPUToLLVM (convert-triton-amdgpu-to-llvm) ('builtin.module' operation) //----- //
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
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 66560 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
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
    %x_80 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xbf16, #shared, #smem, mutable>
    %w_81 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1x128x256xf8E4M3FN, #shared1, #smem, mutable>
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
    %w_115 = ttg.convert_layout %w_scales_114 {allocation.offset = 65536 : i32} : tensor<256x4xi8, #blocked> -> tensor<256x4xi8, #linear1>
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
    %w_148 = ttg.convert_layout %w_146 {allocation.offset = 65536 : i32} : tensor<4x256xi8, #blocked3> -> tensor<4x256xi8, #ttg.slice<{dim = 2, parent = #linear}>>
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
