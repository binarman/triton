// -----// IR Dump Before ConvertTritonToTritonGPU (convert-triton-to-tritongpu) ('builtin.module' operation) //----- //
module {
  tt.func public @_topk_forward(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %stride_xm: i32, %Yv: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Yi: !tt.ptr<i16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_ym: i32, %Bits: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %n_rows: i32, %n_expts_tot: i32, %S: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<32x32xi32>
    %cst_0 = arith.constant dense<16> : tensor<32x4xi32>
    %cst_1 = arith.constant dense<0xFC00> : tensor<32x32xf16>
    %cst_2 = arith.constant dense<0> : tensor<32x32xi32>
    %cst_3 = arith.constant dense<-32768> : tensor<32x32xi16>
    %cst_4 = arith.constant dense<-1> : tensor<32x32xi16>
    %cst_5 = arith.constant dense<32> : tensor<32xi32>
    %cst_6 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32>
    %cst_7 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32>
    %cst_8 = arith.constant dense<0> : tensor<32x4xi32>
    %cst_9 = arith.constant dense<-32768> : tensor<32x4xi16>
    %cst_10 = arith.constant dense<-1> : tensor<32x4xi16>
    %cst_11 = arith.constant dense<0> : tensor<128xi32>
    %cst_12 = arith.constant dense<0> : tensor<32x4x1xi32>
    %cst_13 = arith.constant dense<1> : tensor<32x4xi32>
    %cst_14 = arith.constant dense<32> : tensor<32x4xi32>
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %pid = tt.get_program_id x : i32
    %0 = arith.cmpi slt, %pid, %c1_i32 : i32
    scf.if %0 {
      %5 = arith.muli %pid, %c128_i32 : i32
      %6 = tt.addptr %S, %5 : !tt.ptr<i32>, i32
      %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %8 = tt.splat %6 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
      %9 = tt.addptr %8, %7 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      tt.store %9, %cst_11 : tensor<128x!tt.ptr<i32>>
    }
    %1 = arith.muli %pid, %c32_i32 : i32
    %2 = arith.cmpi sge, %1, %n_rows : i32
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %offs_m = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %offs_m_15 = tt.splat %1 : i32 -> tensor<32xi32>
    %offs_m_16 = arith.addi %offs_m_15, %offs_m : tensor<32xi32>
    %offs_y_n = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %mask_m = tt.expand_dims %offs_m_16 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %mask_m_17 = tt.splat %n_rows : i32 -> tensor<32x1xi32>
    %mask_m_18 = arith.cmpi slt, %mask_m, %mask_m_17 : tensor<32x1xi32>
    %mask_n = tt.expand_dims %offs_m {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %mask_n_19 = tt.splat %n_expts_tot : i32 -> tensor<1x32xi32>
    %mask_n_20 = arith.cmpi slt, %mask_n, %mask_n_19 : tensor<1x32xi32>
    %X_ptrs = tt.splat %stride_xm : i32 -> tensor<32x1xi32>
    %X_ptrs_21 = arith.muli %mask_m, %X_ptrs : tensor<32x1xi32>
    %X_ptrs_22 = tt.splat %X : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
    %X_ptrs_23 = tt.addptr %X_ptrs_22, %X_ptrs_21 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
    %X_ptrs_24 = tt.broadcast %X_ptrs_23 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x32x!tt.ptr<f16>>
    %X_ptrs_25 = tt.broadcast %mask_n : tensor<1x32xi32> -> tensor<32x32xi32>
    %X_ptrs_26 = tt.addptr %X_ptrs_24, %X_ptrs_25 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
    %x = tt.broadcast %mask_m_18 : tensor<32x1xi1> -> tensor<32x32xi1>
    %x_27 = tt.broadcast %mask_n_20 : tensor<1x32xi1> -> tensor<32x32xi1>
    %x_28 = arith.andi %x, %x_27 : tensor<32x32xi1>
    %x_29 = tt.load %X_ptrs_26, %x_28, %cst_1 : tensor<32x32x!tt.ptr<f16>>
    %x_30 = tt.bitcast %x_29 : tensor<32x32xf16> -> tensor<32x32xi16>
    %x_31 = arith.andi %x_30, %cst_3 : tensor<32x32xi16>
    %x_32 = arith.extui %x_31 : tensor<32x32xi16> to tensor<32x32xi32>
    %x_33 = arith.cmpi ne, %x_32, %cst_2 : tensor<32x32xi32>
    %x_34 = arith.select %x_33, %cst_4, %cst_3 : tensor<32x32xi1>, tensor<32x32xi16>
    %x_35 = arith.xori %x_30, %x_34 : tensor<32x32xi16>
    %x_36 = arith.extui %x_35 : tensor<32x32xi16> to tensor<32x32xi32>
    %x_37 = arith.shli %x_36, %cst : tensor<32x32xi32>
    %x_38 = arith.subi %cst_5, %offs_m : tensor<32xi32>
    %x_39 = tt.expand_dims %x_38 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %x_40 = tt.broadcast %x_39 : tensor<1x32xi32> -> tensor<32x32xi32>
    %x_41 = arith.ori %x_37, %x_40 : tensor<32x32xi32>
    %h = tt.reshape %x_41 : tensor<32x32xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ar = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %ar_42 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x1x1x2x1xi32>
    %ix = tt.bitcast %h : tensor<2x2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %iy = "tt.reduce"(%ix) <{axis = 9 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %iy_43 = tt.expand_dims %iy {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x1xi32>
    %iy_44 = tt.broadcast %iy_43 : tensor<2x2x2x2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %iy_45 = arith.xori %ix, %iy_44 : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %y = tt.bitcast %iy_45 : tensor<2x2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ar_46 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x1x1x1x2xi32>
    %ret = arith.cmpi ugt, %h, %y : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_47 = tt.broadcast %ar_42 : tensor<1x1x1x1x1x1x1x1x2x1xi32> -> tensor<1x1x1x1x1x1x1x1x2x2xi32>
    %ret_48 = tt.broadcast %ar_46 : tensor<1x1x1x1x1x1x1x1x1x2xi32> -> tensor<1x1x1x1x1x1x1x1x2x2xi32>
    %ret_49 = arith.xori %ret_47, %ret_48 : tensor<1x1x1x1x1x1x1x1x2x2xi32>
    %ret_50 = arith.extui %ret : tensor<2x2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_51 = tt.broadcast %ret_49 : tensor<1x1x1x1x1x1x1x1x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_52 = arith.cmpi ne, %ret_50, %ret_51 : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_53 = arith.select %ret_52, %y, %h : tensor<2x2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ar_54 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x1x2x1x1xi32>
    %ix_55 = tt.bitcast %ret_53 : tensor<2x2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %iy_56 = "tt.reduce"(%ix_55) <{axis = 8 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %iy_57 = tt.expand_dims %iy_56 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x1x2xi32>
    %iy_58 = tt.broadcast %iy_57 : tensor<2x2x2x2x2x2x2x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %iy_59 = arith.xori %ix_55, %iy_58 : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %y_60 = tt.bitcast %iy_59 : tensor<2x2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_61 = arith.cmpi ugt, %ret_53, %y_60 : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_62 = tt.broadcast %ar_54 : tensor<1x1x1x1x1x1x1x2x1x1xi32> -> tensor<1x1x1x1x1x1x1x2x2x1xi32>
    %ret_63 = tt.broadcast %ar_42 : tensor<1x1x1x1x1x1x1x1x2x1xi32> -> tensor<1x1x1x1x1x1x1x2x2x1xi32>
    %ret_64 = arith.xori %ret_62, %ret_63 : tensor<1x1x1x1x1x1x1x2x2x1xi32>
    %ret_65 = arith.extui %ret_61 : tensor<2x2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_66 = tt.broadcast %ret_64 : tensor<1x1x1x1x1x1x1x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_67 = arith.cmpi ne, %ret_65, %ret_66 : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_68 = arith.select %ret_67, %y_60, %ret_53 : tensor<2x2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ix_69 = tt.bitcast %ret_68 : tensor<2x2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %iy_70 = "tt.reduce"(%ix_69) <{axis = 9 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %iy_71 = tt.expand_dims %iy_70 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x1xi32>
    %iy_72 = tt.broadcast %iy_71 : tensor<2x2x2x2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %iy_73 = arith.xori %ix_69, %iy_72 : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %y_74 = tt.bitcast %iy_73 : tensor<2x2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_75 = arith.cmpi ugt, %ret_68, %y_74 : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_76 = tt.broadcast %ar_54 : tensor<1x1x1x1x1x1x1x2x1x1xi32> -> tensor<1x1x1x1x1x1x1x2x1x2xi32>
    %ret_77 = tt.broadcast %ar_46 : tensor<1x1x1x1x1x1x1x1x1x2xi32> -> tensor<1x1x1x1x1x1x1x2x1x2xi32>
    %ret_78 = arith.xori %ret_76, %ret_77 : tensor<1x1x1x1x1x1x1x2x1x2xi32>
    %ret_79 = arith.extui %ret_75 : tensor<2x2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_80 = tt.broadcast %ret_78 : tensor<1x1x1x1x1x1x1x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_81 = arith.cmpi ne, %ret_79, %ret_80 : tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %ret_82 = arith.select %ret_81, %y_74, %ret_68 : tensor<2x2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2x2xi32>
    %h_83 = "tt.reduce"(%ret_82) <{axis = 7 : i32}> ({
    ^bb0(%h_241: i32, %h_242: i32):
      %h_243 = arith.maxui %h_241, %h_242 : i32
      tt.reduce.return %h_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %ar_84 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x2x1x1xi32>
    %ix_85 = tt.bitcast %h_83 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %iy_86 = "tt.reduce"(%ix_85) <{axis = 7 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %iy_87 = tt.expand_dims %iy_86 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x1x2xi32>
    %iy_88 = tt.broadcast %iy_87 : tensor<2x2x2x2x2x2x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %iy_89 = arith.xori %ix_85, %iy_88 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %y_90 = tt.bitcast %iy_89 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %ar_91 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x1x2x1xi32>
    %ret_92 = arith.cmpi ugt, %h_83, %y_90 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %ret_93 = tt.broadcast %ar_84 : tensor<1x1x1x1x1x1x2x1x1xi32> -> tensor<1x1x1x1x1x1x2x2x1xi32>
    %ret_94 = tt.broadcast %ar_91 : tensor<1x1x1x1x1x1x1x2x1xi32> -> tensor<1x1x1x1x1x1x2x2x1xi32>
    %ret_95 = arith.xori %ret_93, %ret_94 : tensor<1x1x1x1x1x1x2x2x1xi32>
    %ret_96 = arith.extui %ret_92 : tensor<2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2xi32>
    %ret_97 = tt.broadcast %ret_95 : tensor<1x1x1x1x1x1x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %ret_98 = arith.cmpi ne, %ret_96, %ret_97 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %ret_99 = arith.select %ret_98, %y_90, %h_83 : tensor<2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2xi32>
    %ix_100 = tt.bitcast %ret_99 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %iy_101 = "tt.reduce"(%ix_100) <{axis = 8 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %iy_102 = tt.expand_dims %iy_101 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x1xi32>
    %iy_103 = tt.broadcast %iy_102 : tensor<2x2x2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %iy_104 = arith.xori %ix_100, %iy_103 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %y_105 = tt.bitcast %iy_104 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %ar_106 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x1x1x2xi32>
    %ret_107 = arith.cmpi ugt, %ret_99, %y_105 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %ret_108 = tt.broadcast %ar_84 : tensor<1x1x1x1x1x1x2x1x1xi32> -> tensor<1x1x1x1x1x1x2x1x2xi32>
    %ret_109 = tt.broadcast %ar_106 : tensor<1x1x1x1x1x1x1x1x2xi32> -> tensor<1x1x1x1x1x1x2x1x2xi32>
    %ret_110 = arith.xori %ret_108, %ret_109 : tensor<1x1x1x1x1x1x2x1x2xi32>
    %ret_111 = arith.extui %ret_107 : tensor<2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2xi32>
    %ret_112 = tt.broadcast %ret_110 : tensor<1x1x1x1x1x1x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %ret_113 = arith.cmpi ne, %ret_111, %ret_112 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %ret_114 = arith.select %ret_113, %y_105, %ret_99 : tensor<2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2xi32>
    %h_115 = "tt.reduce"(%ret_114) <{axis = 6 : i32}> ({
    ^bb0(%h_241: i32, %h_242: i32):
      %h_243 = arith.maxui %h_241, %h_242 : i32
      tt.reduce.return %h_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %ar_116 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x2x1x1xi32>
    %ix_117 = tt.bitcast %h_115 : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2xi32>
    %iy_118 = "tt.reduce"(%ix_117) <{axis = 6 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2xi32>
    %iy_119 = tt.expand_dims %iy_118 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x1x2xi32>
    %iy_120 = tt.broadcast %iy_119 : tensor<2x2x2x2x2x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2xi32>
    %iy_121 = arith.xori %ix_117, %iy_120 : tensor<2x2x2x2x2x2x2x2xi32>
    %y_122 = tt.bitcast %iy_121 : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2xi32>
    %ar_123 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x2x1xi32>
    %ret_124 = arith.cmpi ugt, %h_115, %y_122 : tensor<2x2x2x2x2x2x2x2xi32>
    %ret_125 = tt.broadcast %ar_116 : tensor<1x1x1x1x1x2x1x1xi32> -> tensor<1x1x1x1x1x2x2x1xi32>
    %ret_126 = tt.broadcast %ar_123 : tensor<1x1x1x1x1x1x2x1xi32> -> tensor<1x1x1x1x1x2x2x1xi32>
    %ret_127 = arith.xori %ret_125, %ret_126 : tensor<1x1x1x1x1x2x2x1xi32>
    %ret_128 = arith.extui %ret_124 : tensor<2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2xi32>
    %ret_129 = tt.broadcast %ret_127 : tensor<1x1x1x1x1x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2xi32>
    %ret_130 = arith.cmpi ne, %ret_128, %ret_129 : tensor<2x2x2x2x2x2x2x2xi32>
    %ret_131 = arith.select %ret_130, %y_122, %h_115 : tensor<2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2xi32>
    %ix_132 = tt.bitcast %ret_131 : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2xi32>
    %iy_133 = "tt.reduce"(%ix_132) <{axis = 7 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2xi32>
    %iy_134 = tt.expand_dims %iy_133 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x1xi32>
    %iy_135 = tt.broadcast %iy_134 : tensor<2x2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2xi32>
    %iy_136 = arith.xori %ix_132, %iy_135 : tensor<2x2x2x2x2x2x2x2xi32>
    %y_137 = tt.bitcast %iy_136 : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2xi32>
    %ar_138 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x1x2xi32>
    %ret_139 = arith.cmpi ugt, %ret_131, %y_137 : tensor<2x2x2x2x2x2x2x2xi32>
    %ret_140 = tt.broadcast %ar_116 : tensor<1x1x1x1x1x2x1x1xi32> -> tensor<1x1x1x1x1x2x1x2xi32>
    %ret_141 = tt.broadcast %ar_138 : tensor<1x1x1x1x1x1x1x2xi32> -> tensor<1x1x1x1x1x2x1x2xi32>
    %ret_142 = arith.xori %ret_140, %ret_141 : tensor<1x1x1x1x1x2x1x2xi32>
    %ret_143 = arith.extui %ret_139 : tensor<2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2xi32>
    %ret_144 = tt.broadcast %ret_142 : tensor<1x1x1x1x1x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2xi32>
    %ret_145 = arith.cmpi ne, %ret_143, %ret_144 : tensor<2x2x2x2x2x2x2x2xi32>
    %ret_146 = arith.select %ret_145, %y_137, %ret_131 : tensor<2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2xi32>
    %h_147 = "tt.reduce"(%ret_146) <{axis = 5 : i32}> ({
    ^bb0(%h_241: i32, %h_242: i32):
      %h_243 = arith.maxui %h_241, %h_242 : i32
      tt.reduce.return %h_243 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2xi32>
    %ix_148 = tt.bitcast %h_147 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_149 = "tt.reduce"(%ix_148) <{axis = 5 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2xi32>
    %iy_150 = tt.expand_dims %iy_149 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x1x2xi32>
    %iy_151 = tt.broadcast %iy_150 : tensor<2x2x2x2x2x1x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_152 = arith.xori %ix_148, %iy_151 : tensor<2x2x2x2x2x2x2xi32>
    %y_153 = tt.bitcast %iy_152 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ar_154 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x2x1xi32>
    %ret_155 = arith.cmpi ugt, %h_147, %y_153 : tensor<2x2x2x2x2x2x2xi32>
    %ret_156 = arith.xori %ar_154, %cst_6 : tensor<1x1x1x1x1x2x1xi32>
    %ret_157 = arith.extui %ret_155 : tensor<2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2xi32>
    %ret_158 = tt.broadcast %ret_156 : tensor<1x1x1x1x1x2x1xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ret_159 = arith.cmpi ne, %ret_157, %ret_158 : tensor<2x2x2x2x2x2x2xi32>
    %ret_160 = arith.select %ret_159, %y_153, %h_147 : tensor<2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2xi32>
    %ix_161 = tt.bitcast %ret_160 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_162 = "tt.reduce"(%ix_161) <{axis = 6 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2xi32>
    %iy_163 = tt.expand_dims %iy_162 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x1xi32>
    %iy_164 = tt.broadcast %iy_163 : tensor<2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_165 = arith.xori %ix_161, %iy_164 : tensor<2x2x2x2x2x2x2xi32>
    %y_166 = tt.bitcast %iy_165 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ar_167 = tt.reshape %ar : tensor<2xi32> -> tensor<1x1x1x1x1x1x2xi32>
    %ret_168 = arith.cmpi ugt, %ret_160, %y_166 : tensor<2x2x2x2x2x2x2xi32>
    %ret_169 = arith.xori %ar_167, %cst_7 : tensor<1x1x1x1x1x1x2xi32>
    %ret_170 = arith.extui %ret_168 : tensor<2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2xi32>
    %ret_171 = tt.broadcast %ret_169 : tensor<1x1x1x1x1x1x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ret_172 = arith.cmpi ne, %ret_170, %ret_171 : tensor<2x2x2x2x2x2x2xi32>
    %ret_173 = arith.select %ret_172, %y_166, %ret_160 : tensor<2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2xi32>
    %x_174 = tt.reshape %ret_173 : tensor<2x2x2x2x2x2x2xi32> -> tensor<32x4xi32>
    %acc = arith.shli %x_174, %cst_0 : tensor<32x4xi32>
    %acc_175 = arith.shrui %x_174, %cst_0 : tensor<32x4xi32>
    %acc_176 = arith.ori %acc, %acc_175 : tensor<32x4xi32>
    %h_177 = tt.reshape %acc_176 : tensor<32x4xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ix_178 = tt.bitcast %h_177 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_179 = "tt.reduce"(%ix_178) <{axis = 6 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2xi32>
    %iy_180 = tt.expand_dims %iy_179 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x1xi32>
    %iy_181 = tt.broadcast %iy_180 : tensor<2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_182 = arith.xori %ix_178, %iy_181 : tensor<2x2x2x2x2x2x2xi32>
    %y_183 = tt.bitcast %iy_182 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ret_184 = arith.cmpi ugt, %h_177, %y_183 : tensor<2x2x2x2x2x2x2xi32>
    %ret_185 = tt.broadcast %ar_154 : tensor<1x1x1x1x1x2x1xi32> -> tensor<1x1x1x1x1x2x2xi32>
    %ret_186 = tt.broadcast %ar_167 : tensor<1x1x1x1x1x1x2xi32> -> tensor<1x1x1x1x1x2x2xi32>
    %ret_187 = arith.xori %ret_185, %ret_186 : tensor<1x1x1x1x1x2x2xi32>
    %ret_188 = arith.extui %ret_184 : tensor<2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2xi32>
    %ret_189 = tt.broadcast %ret_187 : tensor<1x1x1x1x1x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ret_190 = arith.cmpi ne, %ret_188, %ret_189 : tensor<2x2x2x2x2x2x2xi32>
    %ret_191 = arith.select %ret_190, %y_183, %h_177 : tensor<2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2xi32>
    %ix_192 = tt.bitcast %ret_191 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_193 = "tt.reduce"(%ix_192) <{axis = 5 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2xi32>
    %iy_194 = tt.expand_dims %iy_193 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x1x2xi32>
    %iy_195 = tt.broadcast %iy_194 : tensor<2x2x2x2x2x1x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_196 = arith.xori %ix_192, %iy_195 : tensor<2x2x2x2x2x2x2xi32>
    %y_197 = tt.bitcast %iy_196 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ret_198 = arith.cmpi ugt, %ret_191, %y_197 : tensor<2x2x2x2x2x2x2xi32>
    %ret_199 = arith.extui %ret_198 : tensor<2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2xi32>
    %ret_200 = arith.cmpi ne, %ret_199, %ret_158 : tensor<2x2x2x2x2x2x2xi32>
    %ret_201 = arith.select %ret_200, %y_197, %ret_191 : tensor<2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2xi32>
    %ix_202 = tt.bitcast %ret_201 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_203 = "tt.reduce"(%ix_202) <{axis = 6 : i32}> ({
    ^bb0(%iy_241: i32, %iy_242: i32):
      %iy_243 = arith.xori %iy_241, %iy_242 : i32
      tt.reduce.return %iy_243 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2xi32>
    %iy_204 = tt.expand_dims %iy_203 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x1xi32>
    %iy_205 = tt.broadcast %iy_204 : tensor<2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %iy_206 = arith.xori %ix_202, %iy_205 : tensor<2x2x2x2x2x2x2xi32>
    %y_207 = tt.bitcast %iy_206 : tensor<2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2xi32>
    %ret_208 = arith.cmpi ugt, %ret_201, %y_207 : tensor<2x2x2x2x2x2x2xi32>
    %ret_209 = arith.extui %ret_208 : tensor<2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2xi32>
    %ret_210 = arith.cmpi ne, %ret_209, %ret_171 : tensor<2x2x2x2x2x2x2xi32>
    %ret_211 = arith.select %ret_210, %y_207, %ret_201 : tensor<2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2xi32>
    %x_212 = tt.reshape %ret_211 : tensor<2x2x2x2x2x2x2xi32> -> tensor<32x4xi32>
    %y_indices_raw = arith.shrui %x_212, %cst_0 : tensor<32x4xi32>
    %y_indices = arith.subi %cst_14, %y_indices_raw : tensor<32x4xi32>
    %y_values_raw = arith.trunci %x_212 : tensor<32x4xi32> to tensor<32x4xi16>
    %y_values = arith.andi %y_values_raw, %cst_9 : tensor<32x4xi16>
    %y_values_213 = arith.extui %y_values : tensor<32x4xi16> to tensor<32x4xi32>
    %y_values_214 = arith.cmpi eq, %y_values_213, %cst_8 : tensor<32x4xi32>
    %y_values_215 = arith.select %y_values_214, %cst_10, %cst_9 : tensor<32x4xi1>, tensor<32x4xi16>
    %y_values_216 = arith.xori %y_values_raw, %y_values_215 : tensor<32x4xi16>
    %y_values_217 = tt.bitcast %y_values_216 : tensor<32x4xi16> -> tensor<32x4xf16>
    %y_values_218 = arith.extf %y_values_217 : tensor<32x4xf16> to tensor<32x4xf32>
    %z = "tt.reduce"(%y_values_218) <{axis = 1 : i32}> ({
    ^bb0(%z_241: f32, %z_242: f32):
      %z_243 = arith.maxnumf %z_241, %z_242 : f32
      tt.reduce.return %z_243 : f32
    }) : (tensor<32x4xf32>) -> tensor<32xf32>
    %z_219 = tt.expand_dims %z {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
    %z_220 = tt.broadcast %z_219 : tensor<32x1xf32> -> tensor<32x4xf32>
    %z_221 = arith.subf %y_values_218, %z_220 : tensor<32x4xf32>
    %num = math.exp %z_221 : tensor<32x4xf32>
    %den = "tt.reduce"(%num) <{axis = 1 : i32}> ({
    ^bb0(%den_241: f32, %den_242: f32):
      %den_243 = arith.addf %den_241, %den_242 : f32
      tt.reduce.return %den_243 : f32
    }) : (tensor<32x4xf32>) -> tensor<32xf32>
    %den_222 = tt.expand_dims %den {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
    %y_values_223 = tt.broadcast %den_222 : tensor<32x1xf32> -> tensor<32x4xf32>
    %y_values_224 = arith.divf %num, %y_values_223 : tensor<32x4xf32>
    %y_values_225 = arith.truncf %y_values_224 : tensor<32x4xf32> to tensor<32x4xf16>
    %Yv_ptrs = tt.splat %stride_ym : i32 -> tensor<32x1xi32>
    %Yv_ptrs_226 = arith.muli %mask_m, %Yv_ptrs : tensor<32x1xi32>
    %Yv_ptrs_227 = tt.splat %Yv : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
    %Yv_ptrs_228 = tt.addptr %Yv_ptrs_227, %Yv_ptrs_226 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
    %Yv_ptrs_229 = tt.expand_dims %offs_y_n {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %Yv_ptrs_230 = tt.broadcast %Yv_ptrs_228 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x4x!tt.ptr<f16>>
    %Yv_ptrs_231 = tt.broadcast %Yv_ptrs_229 : tensor<1x4xi32> -> tensor<32x4xi32>
    %Yv_ptrs_232 = tt.addptr %Yv_ptrs_230, %Yv_ptrs_231 : tensor<32x4x!tt.ptr<f16>>, tensor<32x4xi32>
    %3 = tt.broadcast %mask_m_18 : tensor<32x1xi1> -> tensor<32x4xi1>
    tt.store %Yv_ptrs_232, %y_values_225, %3 : tensor<32x4x!tt.ptr<f16>>
    %Yi_ptrs = tt.splat %Yi : !tt.ptr<i16> -> tensor<32x1x!tt.ptr<i16>>
    %Yi_ptrs_233 = tt.addptr %Yi_ptrs, %Yv_ptrs_226 : tensor<32x1x!tt.ptr<i16>>, tensor<32x1xi32>
    %Yi_ptrs_234 = tt.broadcast %Yi_ptrs_233 : tensor<32x1x!tt.ptr<i16>> -> tensor<32x4x!tt.ptr<i16>>
    %Yi_ptrs_235 = tt.addptr %Yi_ptrs_234, %Yv_ptrs_231 : tensor<32x4x!tt.ptr<i16>>, tensor<32x4xi32>
    %4 = arith.trunci %y_indices : tensor<32x4xi32> to tensor<32x4xi16>
    tt.store %Yi_ptrs_235, %4, %3 : tensor<32x4x!tt.ptr<i16>>
    %y_div = arith.divui %y_indices, %cst_14 : tensor<32x4xi32>
    %y_rem = arith.remui %y_indices, %cst_14 : tensor<32x4xi32>
    %y2 = tt.expand_dims %y_div {axis = 2 : i32} : tensor<32x4xi32> -> tensor<32x4x1xi32>
    %y2_236 = arith.cmpi eq, %y2, %cst_12 : tensor<32x4x1xi32>
    %y2_237 = arith.shli %cst_13, %y_rem : tensor<32x4xi32>
    %y2_238 = tt.expand_dims %y2_237 {axis = 2 : i32} : tensor<32x4xi32> -> tensor<32x4x1xi32>
    %y2_239 = arith.select %y2_236, %y2_238, %cst_12 : tensor<32x4x1xi1>, tensor<32x4x1xi32>
    %r = "tt.reduce"(%y2_239) <{axis = 1 : i32}> ({
    ^bb0(%r_241: i32, %r_242: i32):
      %r_243 = arith.ori %r_241, %r_242 : i32
      tt.reduce.return %r_243 : i32
    }) : (tensor<32x4x1xi32>) -> tensor<32x1xi32>
    %BitsPtrs = tt.splat %Bits : !tt.ptr<i32> -> tensor<32x1x!tt.ptr<i32>>
    %BitsPtrs_240 = tt.addptr %BitsPtrs, %mask_m : tensor<32x1x!tt.ptr<i32>>, tensor<32x1xi32>
    tt.store %BitsPtrs_240, %r, %mask_m_18 : tensor<32x1x!tt.ptr<i32>>
    tt.return
  }
}

// -----// IR Dump Before TritonAMDGPUStreamPipeline (tritonamdgpu-stream-pipeline) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [16, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2], warpsPerCTA = [1, 1, 2, 2, 1, 1, 1, 1, 1, 1], order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 0], [0, 0], [0, 0], [1, 0]], warp = [[2, 0], [4, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], lane = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]], warp = [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [], lane = [[0], [0], [1], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear3 = #ttg.linear<{register = [], lane = [[0], [0], [0], [1], [0], [0]], warp = [[0], [0]], block = []}>
#linear4 = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [1], [0]], warp = [[0], [0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[0], [1], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [0], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_topk_forward(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %stride_xm: i32, %Yv: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Yi: !tt.ptr<i16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_ym: i32, %Bits: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %n_rows: i32, %n_expts_tot: i32, %S: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x4xi32, #linear>
    %cst_0 = arith.constant dense<1> : tensor<32x4xi32, #linear>
    %cst_1 = arith.constant dense<-1> : tensor<32x4xi16, #linear>
    %cst_2 = arith.constant dense<-32768> : tensor<32x4xi16, #linear>
    %cst_3 = arith.constant dense<0> : tensor<32x4xi32, #linear>
    %cst_4 = arith.constant dense<16> : tensor<32x4xi32, #linear>
    %cst_5 = arith.constant dense<16> : tensor<32x32xi32, #blocked>
    %cst_6 = arith.constant dense<0xFC00> : tensor<32x32xf16, #blocked>
    %cst_7 = arith.constant dense<0> : tensor<32x32xi32, #blocked>
    %cst_8 = arith.constant dense<-32768> : tensor<32x32xi16, #blocked>
    %cst_9 = arith.constant dense<-1> : tensor<32x32xi16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_10 = arith.constant dense<0> : tensor<32x4x1xi32, #blocked1>
    %cst_11 = arith.constant dense<0> : tensor<128xi32, #blocked2>
    %cst_12 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32, #linear1>
    %cst_13 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %cst_14 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32, #linear1>
    %cst_15 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %cst_16 = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %pid = tt.get_program_id x : i32
    %0 = arith.cmpi slt, %pid, %c1_i32 : i32
    scf.if %0 {
      %8 = arith.muli %pid, %c128_i32 : i32
      %9 = tt.addptr %S, %8 : !tt.ptr<i32>, i32
      %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
      %11 = tt.splat %9 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked2>
      %12 = tt.addptr %11, %10 : tensor<128x!tt.ptr<i32>, #blocked2>, tensor<128xi32, #blocked2>
      tt.store %12, %cst_11 : tensor<128x!tt.ptr<i32>, #blocked2>
    }
    %1 = arith.muli %pid, %c32_i32 : i32
    %2 = arith.cmpi sge, %1, %n_rows : i32
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %offs_m = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %offs_m_19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_20 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_21 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_22 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %offs_m_23 = arith.addi %offs_m_20, %offs_m : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_24 = arith.addi %offs_m_21, %offs_m_17 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_25 = arith.addi %offs_m_22, %offs_m_18 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %mask_m = tt.expand_dims %offs_m_23 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %mask_m_26 = tt.expand_dims %offs_m_24 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %mask_m_27 = tt.expand_dims %offs_m_25 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<32x1xi32, #blocked5>
    %mask_m_28 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked>
    %mask_m_29 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked4>
    %mask_m_30 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked5>
    %mask_m_31 = arith.cmpi slt, %mask_m, %mask_m_28 : tensor<32x1xi32, #blocked>
    %mask_m_32 = arith.cmpi slt, %mask_m_26, %mask_m_29 : tensor<32x1xi32, #blocked4>
    %mask_m_33 = arith.cmpi slt, %mask_m_27, %mask_m_30 : tensor<32x1xi32, #blocked5>
    %mask_n = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_n_34 = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %mask_n_35 = tt.splat %n_expts_tot : i32 -> tensor<1x32xi32, #blocked>
    %mask_n_36 = arith.cmpi slt, %mask_n_34, %mask_n_35 : tensor<1x32xi32, #blocked>
    %X_ptrs = tt.splat %stride_xm : i32 -> tensor<32x1xi32, #blocked>
    %X_ptrs_37 = arith.muli %mask_m, %X_ptrs : tensor<32x1xi32, #blocked>
    %X_ptrs_38 = tt.splat %X : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %X_ptrs_39 = tt.addptr %X_ptrs_38, %X_ptrs_37 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %X_ptrs_40 = tt.broadcast %X_ptrs_39 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %X_ptrs_41 = tt.broadcast %mask_n_34 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %X_ptrs_42 = tt.addptr %X_ptrs_40, %X_ptrs_41 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %x = tt.broadcast %mask_m_31 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %x_43 = tt.broadcast %mask_n_36 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %x_44 = arith.andi %x, %x_43 : tensor<32x32xi1, #blocked>
    %x_45 = tt.load %X_ptrs_42, %x_44, %cst_6 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %x_46 = tt.bitcast %x_45 : tensor<32x32xf16, #blocked> -> tensor<32x32xi16, #blocked>
    %x_47 = arith.andi %x_46, %cst_8 : tensor<32x32xi16, #blocked>
    %x_48 = arith.extui %x_47 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %x_49 = arith.cmpi ne, %x_48, %cst_7 : tensor<32x32xi32, #blocked>
    %x_50 = arith.select %x_49, %cst_9, %cst_8 : tensor<32x32xi1, #blocked>, tensor<32x32xi16, #blocked>
    %x_51 = arith.xori %x_46, %x_50 : tensor<32x32xi16, #blocked>
    %x_52 = arith.extui %x_51 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %x_53 = arith.shli %x_52, %cst_5 : tensor<32x32xi32, #blocked>
    %x_54 = arith.subi %cst_16, %offs_m_19 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %x_55 = tt.expand_dims %x_54 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %x_56 = tt.broadcast %x_55 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %x_57 = arith.ori %x_53, %x_56 : tensor<32x32xi32, #blocked>
    %h = tt.reshape %x_57 : tensor<32x32xi32, #blocked> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear2>
    %ar_58 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear3>
    %ar_59 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear4>
    %ar_60 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear5>
    %ar_61 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear5>
    %ar_62 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear6>
    %ar_63 = tt.reshape %ar_60 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3>
    %ar_64 = tt.reshape %ar_61 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3>
    %ix = tt.bitcast %h : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy = "tt.reduce"(%ix) <{axis = 9 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>>
    %iy_65 = tt.expand_dims %iy {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3>
    %iy_66 = tt.broadcast %iy_65 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_67 = arith.xori %ix, %iy_66 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y = tt.bitcast %iy_67 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar_68 = tt.reshape %ar_62 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3>
    %ar_69 = tt.reshape %ar_62 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3>
    %ret = arith.cmpi ugt, %h, %y : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_70 = tt.broadcast %ar_63 : tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_71 = tt.broadcast %ar_68 : tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_72 = arith.xori %ret_70, %ret_71 : tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_73 = arith.extui %ret : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_74 = tt.broadcast %ret_72 : tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_75 = arith.cmpi ne, %ret_73, %ret_74 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_76 = arith.select %ret_75, %y, %h : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar_77 = tt.reshape %ar : tensor<2xi32, #linear2> -> tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3>
    %ar_78 = tt.reshape %ar : tensor<2xi32, #linear2> -> tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3>
    %ix_79 = tt.bitcast %ret_76 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_80 = "tt.reduce"(%ix_79) <{axis = 8 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked3}>>
    %iy_81 = tt.expand_dims %iy_80 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked3>
    %iy_82 = tt.broadcast %iy_81 : tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_83 = arith.xori %ix_79, %iy_82 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y_84 = tt.bitcast %iy_83 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_85 = arith.cmpi ugt, %ret_76, %y_84 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_86 = tt.broadcast %ar_77 : tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_87 = tt.broadcast %ar_64 : tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_88 = arith.xori %ret_86, %ret_87 : tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_89 = arith.extui %ret_85 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_90 = tt.broadcast %ret_88 : tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_91 = arith.cmpi ne, %ret_89, %ret_90 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_92 = arith.select %ret_91, %y_84, %ret_76 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ix_93 = tt.bitcast %ret_92 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_94 = "tt.reduce"(%ix_93) <{axis = 9 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>>
    %iy_95 = tt.expand_dims %iy_94 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3>
    %iy_96 = tt.broadcast %iy_95 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_97 = arith.xori %ix_93, %iy_96 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y_98 = tt.bitcast %iy_97 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_99 = arith.cmpi ugt, %ret_92, %y_98 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_100 = tt.broadcast %ar_78 : tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_101 = tt.broadcast %ar_69 : tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_102 = arith.xori %ret_100, %ret_101 : tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_103 = arith.extui %ret_99 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_104 = tt.broadcast %ret_102 : tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_105 = arith.cmpi ne, %ret_103, %ret_104 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_106 = arith.select %ret_105, %y_98, %ret_92 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %h_107 = "tt.reduce"(%ret_106) <{axis = 7 : i32}> ({
    ^bb0(%h_278: i32, %h_279: i32):
      %h_280 = arith.maxui %h_278, %h_279 : i32
      tt.reduce.return %h_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_108 = tt.reshape %ar_58 : tensor<2xi32, #linear3> -> tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_109 = tt.reshape %ar_58 : tensor<2xi32, #linear3> -> tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ix_110 = tt.bitcast %h_107 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_111 = "tt.reduce"(%ix_110) <{axis = 7 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_112 = tt.expand_dims %iy_111 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_113 = tt.broadcast %iy_112 : tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_114 = arith.xori %ix_110, %iy_113 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %y_115 = tt.bitcast %iy_114 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_116 = tt.reshape %ar_61 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_117 = arith.cmpi ugt, %h_107, %y_115 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_118 = tt.broadcast %ar_108 : tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_119 = tt.broadcast %ar_116 : tensor<1x1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_120 = arith.xori %ret_118, %ret_119 : tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_121 = arith.extui %ret_117 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_122 = tt.broadcast %ret_120 : tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_123 = arith.cmpi ne, %ret_121, %ret_122 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_124 = arith.select %ret_123, %y_115, %h_107 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ix_125 = tt.bitcast %ret_124 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_126 = "tt.reduce"(%ix_125) <{axis = 8 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_127 = tt.expand_dims %iy_126 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_128 = tt.broadcast %iy_127 : tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_129 = arith.xori %ix_125, %iy_128 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %y_130 = tt.bitcast %iy_129 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_131 = tt.reshape %ar_62 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_132 = arith.cmpi ugt, %ret_124, %y_130 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_133 = tt.broadcast %ar_109 : tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_134 = tt.broadcast %ar_131 : tensor<1x1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_135 = arith.xori %ret_133, %ret_134 : tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_136 = arith.extui %ret_132 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_137 = tt.broadcast %ret_135 : tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_138 = arith.cmpi ne, %ret_136, %ret_137 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_139 = arith.select %ret_138, %y_130, %ret_124 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %h_140 = "tt.reduce"(%ret_139) <{axis = 6 : i32}> ({
    ^bb0(%h_278: i32, %h_279: i32):
      %h_280 = arith.maxui %h_278, %h_279 : i32
      tt.reduce.return %h_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_141 = tt.reshape %ar_59 : tensor<2xi32, #linear4> -> tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_142 = tt.reshape %ar_59 : tensor<2xi32, #linear4> -> tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ix_143 = tt.bitcast %h_140 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_144 = "tt.reduce"(%ix_143) <{axis = 6 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_145 = tt.expand_dims %iy_144 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_146 = tt.broadcast %iy_145 : tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_147 = arith.xori %ix_143, %iy_146 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %y_148 = tt.bitcast %iy_147 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_149 = tt.reshape %ar_61 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_150 = arith.cmpi ugt, %h_140, %y_148 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_151 = tt.broadcast %ar_141 : tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_152 = tt.broadcast %ar_149 : tensor<1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_153 = arith.xori %ret_151, %ret_152 : tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_154 = arith.extui %ret_150 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_155 = tt.broadcast %ret_153 : tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_156 = arith.cmpi ne, %ret_154, %ret_155 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_157 = arith.select %ret_156, %y_148, %h_140 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ix_158 = tt.bitcast %ret_157 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_159 = "tt.reduce"(%ix_158) <{axis = 7 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_160 = tt.expand_dims %iy_159 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_161 = tt.broadcast %iy_160 : tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_162 = arith.xori %ix_158, %iy_161 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %y_163 = tt.bitcast %iy_162 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_164 = tt.reshape %ar_62 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_165 = arith.cmpi ugt, %ret_157, %y_163 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_166 = tt.broadcast %ar_142 : tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_167 = tt.broadcast %ar_164 : tensor<1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_168 = arith.xori %ret_166, %ret_167 : tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_169 = arith.extui %ret_165 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_170 = tt.broadcast %ret_168 : tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_171 = arith.cmpi ne, %ret_169, %ret_170 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_172 = arith.select %ret_171, %y_163, %ret_157 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %h_173 = "tt.reduce"(%ret_172) <{axis = 5 : i32}> ({
    ^bb0(%h_278: i32, %h_279: i32):
      %h_280 = arith.maxui %h_278, %h_279 : i32
      tt.reduce.return %h_280 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ix_174 = tt.bitcast %h_173 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_175 = "tt.reduce"(%ix_174) <{axis = 5 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>>
    %iy_176 = tt.expand_dims %iy_175 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>> -> tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_177 = tt.broadcast %iy_176 : tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_178 = arith.xori %ix_174, %iy_177 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %y_179 = tt.bitcast %iy_178 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ar_180 = tt.reshape %ar_60 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #linear1>
    %ar_181 = tt.reshape %ar_60 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ar_182 = tt.reshape %ar_60 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #linear1>
    %ret_183 = arith.cmpi ugt, %h_173, %y_179 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_184 = arith.xori %ar_180, %cst_14 : tensor<1x1x1x1x1x2x1xi32, #linear1>
    %ret_185 = arith.xori %ar_181, %cst_15 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_186 = arith.extui %ret_183 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_187 = tt.broadcast %ret_184 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_188 = tt.broadcast %ret_185 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_189 = arith.cmpi ne, %ret_186, %ret_188 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_190 = arith.select %ret_189, %y_179, %h_173 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ix_191 = tt.bitcast %ret_190 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_192 = "tt.reduce"(%ix_191) <{axis = 6 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>>
    %iy_193 = tt.expand_dims %iy_192 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>> -> tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_194 = tt.broadcast %iy_193 : tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_195 = arith.xori %ix_191, %iy_194 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %y_196 = tt.bitcast %iy_195 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ar_197 = tt.reshape %ar_62 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x2xi32, #linear1>
    %ar_198 = tt.reshape %ar_62 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ar_199 = tt.reshape %ar_62 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x2xi32, #linear1>
    %ret_200 = arith.cmpi ugt, %ret_190, %y_196 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_201 = arith.xori %ar_197, %cst_12 : tensor<1x1x1x1x1x1x2xi32, #linear1>
    %ret_202 = arith.xori %ar_198, %cst_13 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_203 = arith.extui %ret_200 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_204 = tt.broadcast %ret_201 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_205 = tt.broadcast %ret_202 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_206 = arith.cmpi ne, %ret_203, %ret_205 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_207 = arith.select %ret_206, %y_196, %ret_190 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %x_208 = tt.reshape %ret_207 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<32x4xi32, #linear>
    %acc = arith.shli %x_208, %cst_4 : tensor<32x4xi32, #linear>
    %acc_209 = arith.shrui %x_208, %cst_4 : tensor<32x4xi32, #linear>
    %acc_210 = arith.ori %acc, %acc_209 : tensor<32x4xi32, #linear>
    %h_211 = tt.reshape %acc_210 : tensor<32x4xi32, #linear> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_212 = tt.bitcast %h_211 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_213 = "tt.reduce"(%ix_212) <{axis = 6 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %iy_214 = tt.expand_dims %iy_213 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %iy_215 = tt.broadcast %iy_214 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_216 = arith.xori %ix_212, %iy_215 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_217 = tt.bitcast %iy_216 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_218 = arith.cmpi ugt, %h_211, %y_217 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_219 = tt.broadcast %ar_182 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_220 = tt.broadcast %ar_199 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_221 = arith.xori %ret_219, %ret_220 : tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_222 = arith.extui %ret_218 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_223 = tt.broadcast %ret_221 : tensor<1x1x1x1x1x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_224 = arith.cmpi ne, %ret_222, %ret_223 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_225 = arith.select %ret_224, %y_217, %h_211 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_226 = tt.bitcast %ret_225 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_227 = "tt.reduce"(%ix_226) <{axis = 5 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>>
    %iy_228 = tt.expand_dims %iy_227 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>> -> tensor<2x2x2x2x2x1x2xi32, #linear1>
    %iy_229 = tt.broadcast %iy_228 : tensor<2x2x2x2x2x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_230 = arith.xori %ix_226, %iy_229 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_231 = tt.bitcast %iy_230 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_232 = arith.cmpi ugt, %ret_225, %y_231 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_233 = arith.extui %ret_232 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_234 = arith.cmpi ne, %ret_233, %ret_187 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_235 = arith.select %ret_234, %y_231, %ret_225 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_236 = tt.bitcast %ret_235 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_237 = "tt.reduce"(%ix_236) <{axis = 6 : i32}> ({
    ^bb0(%iy_278: i32, %iy_279: i32):
      %iy_280 = arith.xori %iy_278, %iy_279 : i32
      tt.reduce.return %iy_280 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %iy_238 = tt.expand_dims %iy_237 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %iy_239 = tt.broadcast %iy_238 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_240 = arith.xori %ix_236, %iy_239 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_241 = tt.bitcast %iy_240 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_242 = arith.cmpi ugt, %ret_235, %y_241 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_243 = arith.extui %ret_242 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_244 = arith.cmpi ne, %ret_243, %ret_204 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_245 = arith.select %ret_244, %y_241, %ret_235 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %x_246 = tt.reshape %ret_245 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<32x4xi32, #linear>
    %y_indices_raw = arith.shrui %x_246, %cst_4 : tensor<32x4xi32, #linear>
    %y_indices = arith.subi %cst, %y_indices_raw : tensor<32x4xi32, #linear>
    %y_values_raw = arith.trunci %x_246 : tensor<32x4xi32, #linear> to tensor<32x4xi16, #linear>
    %y_values = arith.andi %y_values_raw, %cst_2 : tensor<32x4xi16, #linear>
    %y_values_247 = arith.extui %y_values : tensor<32x4xi16, #linear> to tensor<32x4xi32, #linear>
    %y_values_248 = arith.cmpi eq, %y_values_247, %cst_3 : tensor<32x4xi32, #linear>
    %y_values_249 = arith.select %y_values_248, %cst_1, %cst_2 : tensor<32x4xi1, #linear>, tensor<32x4xi16, #linear>
    %y_values_250 = arith.xori %y_values_raw, %y_values_249 : tensor<32x4xi16, #linear>
    %y_values_251 = tt.bitcast %y_values_250 : tensor<32x4xi16, #linear> -> tensor<32x4xf16, #linear>
    %y_values_252 = arith.extf %y_values_251 : tensor<32x4xf16, #linear> to tensor<32x4xf32, #linear>
    %z = "tt.reduce"(%y_values_252) <{axis = 1 : i32}> ({
    ^bb0(%z_278: f32, %z_279: f32):
      %z_280 = arith.maxnumf %z_278, %z_279 : f32
      tt.reduce.return %z_280 : f32
    }) : (tensor<32x4xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %z_253 = tt.expand_dims %z {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xf32, #linear>
    %z_254 = tt.broadcast %z_253 : tensor<32x1xf32, #linear> -> tensor<32x4xf32, #linear>
    %z_255 = arith.subf %y_values_252, %z_254 : tensor<32x4xf32, #linear>
    %num = math.exp %z_255 : tensor<32x4xf32, #linear>
    %den = "tt.reduce"(%num) <{axis = 1 : i32}> ({
    ^bb0(%den_278: f32, %den_279: f32):
      %den_280 = arith.addf %den_278, %den_279 : f32
      tt.reduce.return %den_280 : f32
    }) : (tensor<32x4xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %den_256 = tt.expand_dims %den {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xf32, #linear>
    %y_values_257 = tt.broadcast %den_256 : tensor<32x1xf32, #linear> -> tensor<32x4xf32, #linear>
    %y_values_258 = arith.divf %num, %y_values_257 : tensor<32x4xf32, #linear>
    %y_values_259 = arith.truncf %y_values_258 : tensor<32x4xf32, #linear> to tensor<32x4xf16, #linear>
    %Yv_ptrs = tt.splat %stride_ym : i32 -> tensor<32x1xi32, #blocked4>
    %Yv_ptrs_260 = arith.muli %mask_m_26, %Yv_ptrs : tensor<32x1xi32, #blocked4>
    %Yv_ptrs_261 = tt.splat %Yv : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked4>
    %Yv_ptrs_262 = tt.addptr %Yv_ptrs_261, %Yv_ptrs_260 : tensor<32x1x!tt.ptr<f16>, #blocked4>, tensor<32x1xi32, #blocked4>
    %Yv_ptrs_263 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %Yv_ptrs_264 = tt.expand_dims %Yv_ptrs_263 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x4xi32, #blocked4>
    %Yv_ptrs_265 = tt.broadcast %Yv_ptrs_262 : tensor<32x1x!tt.ptr<f16>, #blocked4> -> tensor<32x4x!tt.ptr<f16>, #blocked4>
    %Yv_ptrs_266 = tt.broadcast %Yv_ptrs_264 : tensor<1x4xi32, #blocked4> -> tensor<32x4xi32, #blocked4>
    %Yv_ptrs_267 = tt.addptr %Yv_ptrs_265, %Yv_ptrs_266 : tensor<32x4x!tt.ptr<f16>, #blocked4>, tensor<32x4xi32, #blocked4>
    %3 = tt.broadcast %mask_m_32 : tensor<32x1xi1, #blocked4> -> tensor<32x4xi1, #blocked4>
    %4 = ttg.convert_layout %y_values_259 : tensor<32x4xf16, #linear> -> tensor<32x4xf16, #blocked4>
    tt.store %Yv_ptrs_267, %4, %3 : tensor<32x4x!tt.ptr<f16>, #blocked4>
    %Yi_ptrs = tt.splat %Yi : !tt.ptr<i16> -> tensor<32x1x!tt.ptr<i16>, #blocked4>
    %Yi_ptrs_268 = tt.addptr %Yi_ptrs, %Yv_ptrs_260 : tensor<32x1x!tt.ptr<i16>, #blocked4>, tensor<32x1xi32, #blocked4>
    %Yi_ptrs_269 = tt.broadcast %Yi_ptrs_268 : tensor<32x1x!tt.ptr<i16>, #blocked4> -> tensor<32x4x!tt.ptr<i16>, #blocked4>
    %Yi_ptrs_270 = tt.addptr %Yi_ptrs_269, %Yv_ptrs_266 : tensor<32x4x!tt.ptr<i16>, #blocked4>, tensor<32x4xi32, #blocked4>
    %5 = arith.trunci %y_indices : tensor<32x4xi32, #linear> to tensor<32x4xi16, #linear>
    %6 = ttg.convert_layout %5 : tensor<32x4xi16, #linear> -> tensor<32x4xi16, #blocked4>
    tt.store %Yi_ptrs_270, %6, %3 : tensor<32x4x!tt.ptr<i16>, #blocked4>
    %y_div = arith.divui %y_indices, %cst : tensor<32x4xi32, #linear>
    %y_rem = arith.remui %y_indices, %cst : tensor<32x4xi32, #linear>
    %y2 = ttg.convert_layout %y_div : tensor<32x4xi32, #linear> -> tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %y2_271 = tt.expand_dims %y2 {axis = 2 : i32} : tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<32x4x1xi32, #blocked1>
    %y2_272 = arith.cmpi eq, %y2_271, %cst_10 : tensor<32x4x1xi32, #blocked1>
    %y2_273 = arith.shli %cst_0, %y_rem : tensor<32x4xi32, #linear>
    %y2_274 = ttg.convert_layout %y2_273 : tensor<32x4xi32, #linear> -> tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %y2_275 = tt.expand_dims %y2_274 {axis = 2 : i32} : tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<32x4x1xi32, #blocked1>
    %y2_276 = arith.select %y2_272, %y2_275, %cst_10 : tensor<32x4x1xi1, #blocked1>, tensor<32x4x1xi32, #blocked1>
    %r = "tt.reduce"(%y2_276) <{axis = 1 : i32}> ({
    ^bb0(%r_278: i32, %r_279: i32):
      %r_280 = arith.ori %r_278, %r_279 : i32
      tt.reduce.return %r_280 : i32
    }) : (tensor<32x4x1xi32, #blocked1>) -> tensor<32x1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %BitsPtrs = tt.splat %Bits : !tt.ptr<i32> -> tensor<32x1x!tt.ptr<i32>, #blocked5>
    %BitsPtrs_277 = tt.addptr %BitsPtrs, %mask_m_27 : tensor<32x1x!tt.ptr<i32>, #blocked5>, tensor<32x1xi32, #blocked5>
    %7 = ttg.convert_layout %r : tensor<32x1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked5>
    tt.store %BitsPtrs_277, %7, %mask_m_33 : tensor<32x1x!tt.ptr<i32>, #blocked5>
    tt.return
  }
}

// -----// IR Dump Before OptimizeAMDLDSUsage (optimize-amd-lds-usage) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [16, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2], warpsPerCTA = [1, 1, 2, 2, 1, 1, 1, 1, 1, 1], order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 0], [0, 0], [0, 0], [1, 0]], warp = [[2, 0], [4, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], lane = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]], warp = [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [], lane = [[0], [0], [1], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear3 = #ttg.linear<{register = [], lane = [[0], [0], [0], [1], [0], [0]], warp = [[0], [0]], block = []}>
#linear4 = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [1], [0]], warp = [[0], [0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[0], [1], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [0], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_topk_forward(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %stride_xm: i32, %Yv: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Yi: !tt.ptr<i16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_ym: i32, %Bits: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %n_rows: i32, %n_expts_tot: i32, %S: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x4xi32, #linear>
    %cst_0 = arith.constant dense<1> : tensor<32x4xi32, #linear>
    %cst_1 = arith.constant dense<-1> : tensor<32x4xi16, #linear>
    %cst_2 = arith.constant dense<-32768> : tensor<32x4xi16, #linear>
    %cst_3 = arith.constant dense<0> : tensor<32x4xi32, #linear>
    %cst_4 = arith.constant dense<16> : tensor<32x4xi32, #linear>
    %cst_5 = arith.constant dense<16> : tensor<32x32xi32, #blocked>
    %cst_6 = arith.constant dense<0xFC00> : tensor<32x32xf16, #blocked>
    %cst_7 = arith.constant dense<0> : tensor<32x32xi32, #blocked>
    %cst_8 = arith.constant dense<-32768> : tensor<32x32xi16, #blocked>
    %cst_9 = arith.constant dense<-1> : tensor<32x32xi16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_10 = arith.constant dense<0> : tensor<32x4x1xi32, #blocked1>
    %cst_11 = arith.constant dense<0> : tensor<128xi32, #blocked2>
    %cst_12 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32, #linear1>
    %cst_13 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %cst_14 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32, #linear1>
    %cst_15 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %cst_16 = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %pid = tt.get_program_id x : i32
    %0 = arith.cmpi slt, %pid, %c1_i32 : i32
    scf.if %0 {
      %12 = arith.muli %pid, %c128_i32 : i32
      %13 = tt.addptr %S, %12 : !tt.ptr<i32>, i32
      %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
      amdgpu.buffer_store %cst_11, %13[%14] : tensor<128xi32, #blocked2>
    }
    %1 = arith.muli %pid, %c32_i32 : i32
    %2 = arith.cmpi sge, %1, %n_rows : i32
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %offs_m = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %offs_m_19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_20 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_21 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_22 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %offs_m_23 = arith.addi %offs_m_20, %offs_m : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_24 = arith.addi %offs_m_21, %offs_m_17 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_25 = arith.addi %offs_m_22, %offs_m_18 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %mask_m = tt.expand_dims %offs_m_23 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %mask_m_26 = tt.expand_dims %offs_m_24 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %mask_m_27 = tt.expand_dims %offs_m_25 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<32x1xi32, #blocked5>
    %mask_m_28 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked>
    %mask_m_29 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked4>
    %mask_m_30 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked5>
    %mask_m_31 = arith.cmpi slt, %mask_m, %mask_m_28 : tensor<32x1xi32, #blocked>
    %mask_m_32 = arith.cmpi slt, %mask_m_26, %mask_m_29 : tensor<32x1xi32, #blocked4>
    %mask_m_33 = arith.cmpi slt, %mask_m_27, %mask_m_30 : tensor<32x1xi32, #blocked5>
    %mask_n = tt.expand_dims %offs_m_19 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %mask_n_34 = tt.splat %n_expts_tot : i32 -> tensor<1x32xi32, #blocked>
    %mask_n_35 = arith.cmpi slt, %mask_n, %mask_n_34 : tensor<1x32xi32, #blocked>
    %X_ptrs = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %X_ptrs_36 = arith.muli %1, %stride_xm : i32
    %X_ptrs_37 = tt.splat %stride_xm : i32 -> tensor<32x1xi32, #blocked>
    %X_ptrs_38 = arith.muli %X_ptrs, %X_ptrs_37 : tensor<32x1xi32, #blocked>
    %X_ptrs_39 = tt.addptr %X, %X_ptrs_36 : !tt.ptr<f16>, i32
    %X_ptrs_40 = arith.extsi %X_ptrs_38 : tensor<32x1xi32, #blocked> to tensor<32x1xi64, #blocked>
    %X_ptrs_41 = tt.broadcast %X_ptrs_40 : tensor<32x1xi64, #blocked> -> tensor<32x32xi64, #blocked>
    %X_ptrs_42 = tt.broadcast %mask_n : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %X_ptrs_43 = arith.extsi %X_ptrs_42 : tensor<32x32xi32, #blocked> to tensor<32x32xi64, #blocked>
    %X_ptrs_44 = arith.addi %X_ptrs_43, %X_ptrs_41 : tensor<32x32xi64, #blocked>
    %x = tt.broadcast %mask_m_31 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %x_45 = tt.broadcast %mask_n_35 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %x_46 = arith.andi %x, %x_45 : tensor<32x32xi1, #blocked>
    %x_47 = tt.splat %X_ptrs_39 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %x_48 = tt.addptr %x_47, %X_ptrs_44 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi64, #blocked>
    %x_49 = tt.load %x_48, %x_46, %cst_6 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %x_50 = tt.bitcast %x_49 : tensor<32x32xf16, #blocked> -> tensor<32x32xi16, #blocked>
    %x_51 = arith.andi %x_50, %cst_8 : tensor<32x32xi16, #blocked>
    %x_52 = arith.extui %x_51 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %x_53 = arith.cmpi ne, %x_52, %cst_7 : tensor<32x32xi32, #blocked>
    %x_54 = arith.select %x_53, %cst_9, %cst_8 : tensor<32x32xi1, #blocked>, tensor<32x32xi16, #blocked>
    %x_55 = arith.xori %x_50, %x_54 : tensor<32x32xi16, #blocked>
    %x_56 = arith.extui %x_55 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %x_57 = arith.shli %x_56, %cst_5 : tensor<32x32xi32, #blocked>
    %x_58 = arith.subi %cst_16, %offs_m_19 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %x_59 = tt.expand_dims %x_58 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %x_60 = tt.broadcast %x_59 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %x_61 = arith.ori %x_57, %x_60 : tensor<32x32xi32, #blocked>
    %h = tt.reshape %x_61 : tensor<32x32xi32, #blocked> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear2>
    %ar_62 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear3>
    %ar_63 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear4>
    %ar_64 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear5>
    %ar_65 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear6>
    %ar_66 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3>
    %ix = tt.bitcast %h : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy = "tt.reduce"(%ix) <{axis = 9 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>>
    %iy_67 = tt.expand_dims %iy {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3>
    %iy_68 = tt.broadcast %iy_67 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_69 = arith.xori %ix, %iy_68 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y = tt.bitcast %iy_69 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar_70 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3>
    %ret = arith.cmpi ugt, %h, %y : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_71 = tt.broadcast %ar_66 : tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_72 = tt.broadcast %ar_70 : tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_73 = arith.xori %ret_71, %ret_72 : tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_74 = arith.extui %ret : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_75 = tt.broadcast %ret_73 : tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_76 = arith.cmpi ne, %ret_74, %ret_75 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_77 = arith.select %ret_76, %y, %h : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar_78 = tt.reshape %ar : tensor<2xi32, #linear2> -> tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3>
    %ix_79 = tt.bitcast %ret_77 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_80 = "tt.reduce"(%ix_79) <{axis = 8 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked3}>>
    %iy_81 = tt.expand_dims %iy_80 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked3>
    %iy_82 = tt.broadcast %iy_81 : tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_83 = arith.xori %ix_79, %iy_82 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y_84 = tt.bitcast %iy_83 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_85 = arith.cmpi ugt, %ret_77, %y_84 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_86 = tt.broadcast %ar_78 : tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_87 = tt.broadcast %ar_66 : tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_88 = arith.xori %ret_86, %ret_87 : tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_89 = arith.extui %ret_85 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_90 = tt.broadcast %ret_88 : tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_91 = arith.cmpi ne, %ret_89, %ret_90 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_92 = arith.select %ret_91, %y_84, %ret_77 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ix_93 = tt.bitcast %ret_92 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_94 = "tt.reduce"(%ix_93) <{axis = 9 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>>
    %iy_95 = tt.expand_dims %iy_94 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3>
    %iy_96 = tt.broadcast %iy_95 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_97 = arith.xori %ix_93, %iy_96 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y_98 = tt.bitcast %iy_97 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_99 = arith.cmpi ugt, %ret_92, %y_98 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_100 = tt.broadcast %ar_78 : tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_101 = tt.broadcast %ar_70 : tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_102 = arith.xori %ret_100, %ret_101 : tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_103 = arith.extui %ret_99 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_104 = tt.broadcast %ret_102 : tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_105 = arith.cmpi ne, %ret_103, %ret_104 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_106 = arith.select %ret_105, %y_98, %ret_92 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %h_107 = "tt.reduce"(%ret_106) <{axis = 7 : i32}> ({
    ^bb0(%h_272: i32, %h_273: i32):
      %h_274 = arith.maxui %h_272, %h_273 : i32
      tt.reduce.return %h_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_108 = tt.reshape %ar_62 : tensor<2xi32, #linear3> -> tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ix_109 = tt.bitcast %h_107 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_110 = "tt.reduce"(%ix_109) <{axis = 7 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_111 = tt.expand_dims %iy_110 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_112 = tt.broadcast %iy_111 : tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_113 = arith.xori %ix_109, %iy_112 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %y_114 = tt.bitcast %iy_113 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_115 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_116 = arith.cmpi ugt, %h_107, %y_114 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_117 = tt.broadcast %ar_108 : tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_118 = tt.broadcast %ar_115 : tensor<1x1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_119 = arith.xori %ret_117, %ret_118 : tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_120 = arith.extui %ret_116 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_121 = tt.broadcast %ret_119 : tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_122 = arith.cmpi ne, %ret_120, %ret_121 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_123 = arith.select %ret_122, %y_114, %h_107 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ix_124 = tt.bitcast %ret_123 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_125 = "tt.reduce"(%ix_124) <{axis = 8 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_126 = tt.expand_dims %iy_125 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_127 = tt.broadcast %iy_126 : tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_128 = arith.xori %ix_124, %iy_127 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %y_129 = tt.bitcast %iy_128 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_130 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_131 = arith.cmpi ugt, %ret_123, %y_129 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_132 = tt.broadcast %ar_108 : tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_133 = tt.broadcast %ar_130 : tensor<1x1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_134 = arith.xori %ret_132, %ret_133 : tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_135 = arith.extui %ret_131 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_136 = tt.broadcast %ret_134 : tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_137 = arith.cmpi ne, %ret_135, %ret_136 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_138 = arith.select %ret_137, %y_129, %ret_123 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %h_139 = "tt.reduce"(%ret_138) <{axis = 6 : i32}> ({
    ^bb0(%h_272: i32, %h_273: i32):
      %h_274 = arith.maxui %h_272, %h_273 : i32
      tt.reduce.return %h_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_140 = tt.reshape %ar_63 : tensor<2xi32, #linear4> -> tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ix_141 = tt.bitcast %h_139 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_142 = "tt.reduce"(%ix_141) <{axis = 6 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_143 = tt.expand_dims %iy_142 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_144 = tt.broadcast %iy_143 : tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_145 = arith.xori %ix_141, %iy_144 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %y_146 = tt.bitcast %iy_145 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_147 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_148 = arith.cmpi ugt, %h_139, %y_146 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_149 = tt.broadcast %ar_140 : tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_150 = tt.broadcast %ar_147 : tensor<1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_151 = arith.xori %ret_149, %ret_150 : tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_152 = arith.extui %ret_148 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_153 = tt.broadcast %ret_151 : tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_154 = arith.cmpi ne, %ret_152, %ret_153 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_155 = arith.select %ret_154, %y_146, %h_139 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ix_156 = tt.bitcast %ret_155 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_157 = "tt.reduce"(%ix_156) <{axis = 7 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_158 = tt.expand_dims %iy_157 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_159 = tt.broadcast %iy_158 : tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_160 = arith.xori %ix_156, %iy_159 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %y_161 = tt.bitcast %iy_160 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_162 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_163 = arith.cmpi ugt, %ret_155, %y_161 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_164 = tt.broadcast %ar_140 : tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_165 = tt.broadcast %ar_162 : tensor<1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_166 = arith.xori %ret_164, %ret_165 : tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_167 = arith.extui %ret_163 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_168 = tt.broadcast %ret_166 : tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_169 = arith.cmpi ne, %ret_167, %ret_168 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_170 = arith.select %ret_169, %y_161, %ret_155 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %h_171 = "tt.reduce"(%ret_170) <{axis = 5 : i32}> ({
    ^bb0(%h_272: i32, %h_273: i32):
      %h_274 = arith.maxui %h_272, %h_273 : i32
      tt.reduce.return %h_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ix_172 = tt.bitcast %h_171 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_173 = "tt.reduce"(%ix_172) <{axis = 5 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>>
    %iy_174 = tt.expand_dims %iy_173 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>> -> tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_175 = tt.broadcast %iy_174 : tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_176 = arith.xori %ix_172, %iy_175 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %y_177 = tt.bitcast %iy_176 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ar_178 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #linear1>
    %ar_179 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_180 = arith.cmpi ugt, %h_171, %y_177 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_181 = arith.xori %ar_178, %cst_14 : tensor<1x1x1x1x1x2x1xi32, #linear1>
    %ret_182 = arith.xori %ar_179, %cst_15 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_183 = arith.extui %ret_180 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_184 = tt.broadcast %ret_181 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_185 = tt.broadcast %ret_182 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_186 = arith.cmpi ne, %ret_183, %ret_185 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_187 = arith.select %ret_186, %y_177, %h_171 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ix_188 = tt.bitcast %ret_187 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_189 = "tt.reduce"(%ix_188) <{axis = 6 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>>
    %iy_190 = tt.expand_dims %iy_189 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>> -> tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_191 = tt.broadcast %iy_190 : tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_192 = arith.xori %ix_188, %iy_191 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %y_193 = tt.bitcast %iy_192 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ar_194 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x2xi32, #linear1>
    %ar_195 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_196 = arith.cmpi ugt, %ret_187, %y_193 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_197 = arith.xori %ar_194, %cst_12 : tensor<1x1x1x1x1x1x2xi32, #linear1>
    %ret_198 = arith.xori %ar_195, %cst_13 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_199 = arith.extui %ret_196 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_200 = tt.broadcast %ret_197 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_201 = tt.broadcast %ret_198 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_202 = arith.cmpi ne, %ret_199, %ret_201 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_203 = arith.select %ret_202, %y_193, %ret_187 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %x_204 = tt.reshape %ret_203 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<32x4xi32, #linear>
    %acc = arith.shli %x_204, %cst_4 : tensor<32x4xi32, #linear>
    %acc_205 = arith.shrui %x_204, %cst_4 : tensor<32x4xi32, #linear>
    %acc_206 = arith.ori %acc, %acc_205 : tensor<32x4xi32, #linear>
    %h_207 = tt.reshape %acc_206 : tensor<32x4xi32, #linear> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_208 = tt.bitcast %h_207 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_209 = "tt.reduce"(%ix_208) <{axis = 6 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %iy_210 = tt.expand_dims %iy_209 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %iy_211 = tt.broadcast %iy_210 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_212 = arith.xori %ix_208, %iy_211 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_213 = tt.bitcast %iy_212 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_214 = arith.cmpi ugt, %h_207, %y_213 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_215 = tt.broadcast %ar_178 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_216 = tt.broadcast %ar_194 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_217 = arith.xori %ret_215, %ret_216 : tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_218 = arith.extui %ret_214 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_219 = tt.broadcast %ret_217 : tensor<1x1x1x1x1x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_220 = arith.cmpi ne, %ret_218, %ret_219 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_221 = arith.select %ret_220, %y_213, %h_207 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_222 = tt.bitcast %ret_221 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_223 = "tt.reduce"(%ix_222) <{axis = 5 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>>
    %iy_224 = tt.expand_dims %iy_223 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>> -> tensor<2x2x2x2x2x1x2xi32, #linear1>
    %iy_225 = tt.broadcast %iy_224 : tensor<2x2x2x2x2x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_226 = arith.xori %ix_222, %iy_225 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_227 = tt.bitcast %iy_226 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_228 = arith.cmpi ugt, %ret_221, %y_227 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_229 = arith.extui %ret_228 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_230 = arith.cmpi ne, %ret_229, %ret_184 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_231 = arith.select %ret_230, %y_227, %ret_221 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_232 = tt.bitcast %ret_231 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_233 = "tt.reduce"(%ix_232) <{axis = 6 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %iy_234 = tt.expand_dims %iy_233 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %iy_235 = tt.broadcast %iy_234 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_236 = arith.xori %ix_232, %iy_235 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_237 = tt.bitcast %iy_236 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_238 = arith.cmpi ugt, %ret_231, %y_237 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_239 = arith.extui %ret_238 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_240 = arith.cmpi ne, %ret_239, %ret_200 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_241 = arith.select %ret_240, %y_237, %ret_231 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %x_242 = tt.reshape %ret_241 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<32x4xi32, #linear>
    %y_indices_raw = arith.shrui %x_242, %cst_4 : tensor<32x4xi32, #linear>
    %y_indices = arith.subi %cst, %y_indices_raw : tensor<32x4xi32, #linear>
    %y_values_raw = arith.trunci %x_242 : tensor<32x4xi32, #linear> to tensor<32x4xi16, #linear>
    %y_values = arith.andi %y_values_raw, %cst_2 : tensor<32x4xi16, #linear>
    %y_values_243 = arith.extui %y_values : tensor<32x4xi16, #linear> to tensor<32x4xi32, #linear>
    %y_values_244 = arith.cmpi eq, %y_values_243, %cst_3 : tensor<32x4xi32, #linear>
    %y_values_245 = arith.select %y_values_244, %cst_1, %cst_2 : tensor<32x4xi1, #linear>, tensor<32x4xi16, #linear>
    %y_values_246 = arith.xori %y_values_raw, %y_values_245 : tensor<32x4xi16, #linear>
    %y_values_247 = tt.bitcast %y_values_246 : tensor<32x4xi16, #linear> -> tensor<32x4xf16, #linear>
    %y_values_248 = arith.extf %y_values_247 : tensor<32x4xf16, #linear> to tensor<32x4xf32, #linear>
    %z = "tt.reduce"(%y_values_248) <{axis = 1 : i32}> ({
    ^bb0(%z_272: f32, %z_273: f32):
      %z_274 = arith.maxnumf %z_272, %z_273 : f32
      tt.reduce.return %z_274 : f32
    }) : (tensor<32x4xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %z_249 = tt.expand_dims %z {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xf32, #linear>
    %z_250 = tt.broadcast %z_249 : tensor<32x1xf32, #linear> -> tensor<32x4xf32, #linear>
    %z_251 = arith.subf %y_values_248, %z_250 : tensor<32x4xf32, #linear>
    %num = math.exp %z_251 : tensor<32x4xf32, #linear>
    %den = "tt.reduce"(%num) <{axis = 1 : i32}> ({
    ^bb0(%den_272: f32, %den_273: f32):
      %den_274 = arith.addf %den_272, %den_273 : f32
      tt.reduce.return %den_274 : f32
    }) : (tensor<32x4xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %den_252 = tt.expand_dims %den {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xf32, #linear>
    %y_values_253 = tt.broadcast %den_252 : tensor<32x1xf32, #linear> -> tensor<32x4xf32, #linear>
    %y_values_254 = arith.divf %num, %y_values_253 : tensor<32x4xf32, #linear>
    %y_values_255 = arith.truncf %y_values_254 : tensor<32x4xf32, #linear> to tensor<32x4xf16, #linear>
    %Yv_ptrs = tt.expand_dims %offs_m_17 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %Yv_ptrs_256 = arith.muli %1, %stride_ym : i32
    %Yv_ptrs_257 = tt.splat %stride_ym : i32 -> tensor<32x1xi32, #blocked4>
    %Yv_ptrs_258 = arith.muli %Yv_ptrs, %Yv_ptrs_257 : tensor<32x1xi32, #blocked4>
    %Yv_ptrs_259 = tt.addptr %Yv, %Yv_ptrs_256 : !tt.ptr<f16>, i32
    %Yv_ptrs_260 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %Yv_ptrs_261 = tt.broadcast %Yv_ptrs_258 : tensor<32x1xi32, #blocked4> -> tensor<32x4xi32, #blocked4>
    %Yv_ptrs_262 = tt.expand_dims %Yv_ptrs_260 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x4xi32, #blocked4>
    %Yv_ptrs_263 = tt.broadcast %Yv_ptrs_262 : tensor<1x4xi32, #blocked4> -> tensor<32x4xi32, #blocked4>
    %Yv_ptrs_264 = arith.addi %Yv_ptrs_263, %Yv_ptrs_261 : tensor<32x4xi32, #blocked4>
    %3 = tt.broadcast %mask_m_32 : tensor<32x1xi1, #blocked4> -> tensor<32x4xi1, #blocked4>
    %4 = ttg.convert_layout %y_values_255 : tensor<32x4xf16, #linear> -> tensor<32x4xf16, #blocked4>
    %5 = tt.splat %Yv_ptrs_259 : !tt.ptr<f16> -> tensor<32x4x!tt.ptr<f16>, #blocked4>
    %6 = tt.addptr %5, %Yv_ptrs_264 : tensor<32x4x!tt.ptr<f16>, #blocked4>, tensor<32x4xi32, #blocked4>
    tt.store %6, %4, %3 : tensor<32x4x!tt.ptr<f16>, #blocked4>
    %Yi_ptrs = tt.addptr %Yi, %Yv_ptrs_256 : !tt.ptr<i16>, i32
    %7 = arith.trunci %y_indices : tensor<32x4xi32, #linear> to tensor<32x4xi16, #linear>
    %8 = ttg.convert_layout %7 : tensor<32x4xi16, #linear> -> tensor<32x4xi16, #blocked4>
    %9 = tt.splat %Yi_ptrs : !tt.ptr<i16> -> tensor<32x4x!tt.ptr<i16>, #blocked4>
    %10 = tt.addptr %9, %Yv_ptrs_264 : tensor<32x4x!tt.ptr<i16>, #blocked4>, tensor<32x4xi32, #blocked4>
    tt.store %10, %8, %3 : tensor<32x4x!tt.ptr<i16>, #blocked4>
    %y_div = arith.divui %y_indices, %cst : tensor<32x4xi32, #linear>
    %y_rem = arith.remui %y_indices, %cst : tensor<32x4xi32, #linear>
    %y2 = ttg.convert_layout %y_div : tensor<32x4xi32, #linear> -> tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %y2_265 = tt.expand_dims %y2 {axis = 2 : i32} : tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<32x4x1xi32, #blocked1>
    %y2_266 = arith.cmpi eq, %y2_265, %cst_10 : tensor<32x4x1xi32, #blocked1>
    %y2_267 = arith.shli %cst_0, %y_rem : tensor<32x4xi32, #linear>
    %y2_268 = ttg.convert_layout %y2_267 : tensor<32x4xi32, #linear> -> tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %y2_269 = tt.expand_dims %y2_268 {axis = 2 : i32} : tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<32x4x1xi32, #blocked1>
    %y2_270 = arith.select %y2_266, %y2_269, %cst_10 : tensor<32x4x1xi1, #blocked1>, tensor<32x4x1xi32, #blocked1>
    %r = "tt.reduce"(%y2_270) <{axis = 1 : i32}> ({
    ^bb0(%r_272: i32, %r_273: i32):
      %r_274 = arith.ori %r_272, %r_273 : i32
      tt.reduce.return %r_274 : i32
    }) : (tensor<32x4x1xi32, #blocked1>) -> tensor<32x1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %BitsPtrs = tt.expand_dims %offs_m_18 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<32x1xi32, #blocked5>
    %BitsPtrs_271 = tt.addptr %Bits, %1 : !tt.ptr<i32>, i32
    %11 = ttg.convert_layout %r : tensor<32x1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked5>
    amdgpu.buffer_store %11, %BitsPtrs_271[%BitsPtrs], %mask_m_33 : tensor<32x1xi32, #blocked5>
    tt.return
  }
}

// -----// IR Dump Before AllocateAMDGPUSharedMemory (allocate-amdgpu-shared-memory) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [16, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2], warpsPerCTA = [1, 1, 2, 2, 1, 1, 1, 1, 1, 1], order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 0], [0, 0], [0, 0], [1, 0]], warp = [[2, 0], [4, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], lane = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]], warp = [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [], lane = [[0], [0], [1], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear3 = #ttg.linear<{register = [], lane = [[0], [0], [0], [1], [0], [0]], warp = [[0], [0]], block = []}>
#linear4 = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [1], [0]], warp = [[0], [0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[0], [1], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [0], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_topk_forward(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %stride_xm: i32, %Yv: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Yi: !tt.ptr<i16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_ym: i32, %Bits: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %n_rows: i32, %n_expts_tot: i32, %S: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x4xi32, #linear>
    %cst_0 = arith.constant dense<1> : tensor<32x4xi32, #linear>
    %cst_1 = arith.constant dense<-1> : tensor<32x4xi16, #linear>
    %cst_2 = arith.constant dense<-32768> : tensor<32x4xi16, #linear>
    %cst_3 = arith.constant dense<0> : tensor<32x4xi32, #linear>
    %cst_4 = arith.constant dense<16> : tensor<32x4xi32, #linear>
    %cst_5 = arith.constant dense<16> : tensor<32x32xi32, #blocked>
    %cst_6 = arith.constant dense<0xFC00> : tensor<32x32xf16, #blocked>
    %cst_7 = arith.constant dense<0> : tensor<32x32xi32, #blocked>
    %cst_8 = arith.constant dense<-32768> : tensor<32x32xi16, #blocked>
    %cst_9 = arith.constant dense<-1> : tensor<32x32xi16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_10 = arith.constant dense<0> : tensor<32x4x1xi32, #blocked1>
    %cst_11 = arith.constant dense<0> : tensor<128xi32, #blocked2>
    %cst_12 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32, #linear1>
    %cst_13 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %cst_14 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32, #linear1>
    %cst_15 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %cst_16 = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %pid = tt.get_program_id x : i32
    %0 = arith.cmpi slt, %pid, %c1_i32 : i32
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %1 = arith.muli %pid, %c128_i32 : i32
    %2 = tt.addptr %S, %1 : !tt.ptr<i32>, i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    amdgpu.buffer_store %cst_11, %2[%3] : tensor<128xi32, #blocked2>
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %4 = arith.muli %pid, %c32_i32 : i32
    %5 = arith.cmpi sge, %4, %n_rows : i32
    cf.cond_br %5, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    tt.return
  ^bb4:  // pred: ^bb2
    %offs_m = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %offs_m_19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_20 = tt.splat %4 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_21 = tt.splat %4 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_22 = tt.splat %4 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %offs_m_23 = arith.addi %offs_m_20, %offs_m : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_24 = arith.addi %offs_m_21, %offs_m_17 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %offs_m_25 = arith.addi %offs_m_22, %offs_m_18 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %mask_m = tt.expand_dims %offs_m_23 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %mask_m_26 = tt.expand_dims %offs_m_24 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %mask_m_27 = tt.expand_dims %offs_m_25 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<32x1xi32, #blocked5>
    %mask_m_28 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked>
    %mask_m_29 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked4>
    %mask_m_30 = tt.splat %n_rows : i32 -> tensor<32x1xi32, #blocked5>
    %mask_m_31 = arith.cmpi slt, %mask_m, %mask_m_28 : tensor<32x1xi32, #blocked>
    %mask_m_32 = arith.cmpi slt, %mask_m_26, %mask_m_29 : tensor<32x1xi32, #blocked4>
    %mask_m_33 = arith.cmpi slt, %mask_m_27, %mask_m_30 : tensor<32x1xi32, #blocked5>
    %mask_n = tt.expand_dims %offs_m_19 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %mask_n_34 = tt.splat %n_expts_tot : i32 -> tensor<1x32xi32, #blocked>
    %mask_n_35 = arith.cmpi slt, %mask_n, %mask_n_34 : tensor<1x32xi32, #blocked>
    %X_ptrs = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %X_ptrs_36 = arith.muli %4, %stride_xm : i32
    %X_ptrs_37 = tt.splat %stride_xm : i32 -> tensor<32x1xi32, #blocked>
    %X_ptrs_38 = arith.muli %X_ptrs, %X_ptrs_37 : tensor<32x1xi32, #blocked>
    %X_ptrs_39 = tt.addptr %X, %X_ptrs_36 : !tt.ptr<f16>, i32
    %X_ptrs_40 = arith.extsi %X_ptrs_38 : tensor<32x1xi32, #blocked> to tensor<32x1xi64, #blocked>
    %X_ptrs_41 = tt.broadcast %X_ptrs_40 : tensor<32x1xi64, #blocked> -> tensor<32x32xi64, #blocked>
    %X_ptrs_42 = tt.broadcast %mask_n : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %X_ptrs_43 = arith.extsi %X_ptrs_42 : tensor<32x32xi32, #blocked> to tensor<32x32xi64, #blocked>
    %X_ptrs_44 = arith.addi %X_ptrs_43, %X_ptrs_41 : tensor<32x32xi64, #blocked>
    %x = tt.broadcast %mask_m_31 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %x_45 = tt.broadcast %mask_n_35 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %x_46 = arith.andi %x, %x_45 : tensor<32x32xi1, #blocked>
    %x_47 = tt.splat %X_ptrs_39 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %x_48 = tt.addptr %x_47, %X_ptrs_44 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi64, #blocked>
    %x_49 = tt.load %x_48, %x_46, %cst_6 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %x_50 = tt.bitcast %x_49 : tensor<32x32xf16, #blocked> -> tensor<32x32xi16, #blocked>
    %x_51 = arith.andi %x_50, %cst_8 : tensor<32x32xi16, #blocked>
    %x_52 = arith.extui %x_51 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %x_53 = arith.cmpi ne, %x_52, %cst_7 : tensor<32x32xi32, #blocked>
    %x_54 = arith.select %x_53, %cst_9, %cst_8 : tensor<32x32xi1, #blocked>, tensor<32x32xi16, #blocked>
    %x_55 = arith.xori %x_50, %x_54 : tensor<32x32xi16, #blocked>
    %x_56 = arith.extui %x_55 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %x_57 = arith.shli %x_56, %cst_5 : tensor<32x32xi32, #blocked>
    %x_58 = arith.subi %cst_16, %offs_m_19 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %x_59 = tt.expand_dims %x_58 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %x_60 = tt.broadcast %x_59 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %x_61 = arith.ori %x_57, %x_60 : tensor<32x32xi32, #blocked>
    %h = tt.reshape %x_61 : tensor<32x32xi32, #blocked> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear2>
    %ar_62 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear3>
    %ar_63 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear4>
    %ar_64 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear5>
    %ar_65 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear6>
    %ar_66 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3>
    %ix = tt.bitcast %h : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy = "tt.reduce"(%ix) <{axis = 9 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>>
    %iy_67 = tt.expand_dims %iy {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3>
    %iy_68 = tt.broadcast %iy_67 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_69 = arith.xori %ix, %iy_68 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y = tt.bitcast %iy_69 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar_70 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3>
    %ret = arith.cmpi ugt, %h, %y : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_71 = tt.broadcast %ar_66 : tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_72 = tt.broadcast %ar_70 : tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_73 = arith.xori %ret_71, %ret_72 : tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3>
    %ret_74 = arith.extui %ret : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_75 = tt.broadcast %ret_73 : tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_76 = arith.cmpi ne, %ret_74, %ret_75 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_77 = arith.select %ret_76, %y, %h : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ar_78 = tt.reshape %ar : tensor<2xi32, #linear2> -> tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3>
    %ix_79 = tt.bitcast %ret_77 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_80 = "tt.reduce"(%ix_79) <{axis = 8 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked3}>>
    %iy_81 = tt.expand_dims %iy_80 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked3>
    %iy_82 = tt.broadcast %iy_81 : tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_83 = arith.xori %ix_79, %iy_82 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y_84 = tt.bitcast %iy_83 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_85 = arith.cmpi ugt, %ret_77, %y_84 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_86 = tt.broadcast %ar_78 : tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_87 = tt.broadcast %ar_66 : tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_88 = arith.xori %ret_86, %ret_87 : tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3>
    %ret_89 = arith.extui %ret_85 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_90 = tt.broadcast %ret_88 : tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_91 = arith.cmpi ne, %ret_89, %ret_90 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_92 = arith.select %ret_91, %y_84, %ret_77 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ix_93 = tt.bitcast %ret_92 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_94 = "tt.reduce"(%ix_93) <{axis = 9 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>>
    %iy_95 = tt.expand_dims %iy_94 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3>
    %iy_96 = tt.broadcast %iy_95 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %iy_97 = arith.xori %ix_93, %iy_96 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %y_98 = tt.bitcast %iy_97 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_99 = arith.cmpi ugt, %ret_92, %y_98 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_100 = tt.broadcast %ar_78 : tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_101 = tt.broadcast %ar_70 : tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked3> -> tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_102 = arith.xori %ret_100, %ret_101 : tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3>
    %ret_103 = arith.extui %ret_99 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_104 = tt.broadcast %ret_102 : tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked3> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_105 = arith.cmpi ne, %ret_103, %ret_104 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %ret_106 = arith.select %ret_105, %y_98, %ret_92 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked3>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>
    %h_107 = "tt.reduce"(%ret_106) <{axis = 7 : i32}> ({
    ^bb0(%h_272: i32, %h_273: i32):
      %h_274 = arith.maxui %h_272, %h_273 : i32
      tt.reduce.return %h_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked3>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_108 = tt.reshape %ar_62 : tensor<2xi32, #linear3> -> tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ix_109 = tt.bitcast %h_107 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_110 = "tt.reduce"(%ix_109) <{axis = 7 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_111 = tt.expand_dims %iy_110 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_112 = tt.broadcast %iy_111 : tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_113 = arith.xori %ix_109, %iy_112 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %y_114 = tt.bitcast %iy_113 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_115 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_116 = arith.cmpi ugt, %h_107, %y_114 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_117 = tt.broadcast %ar_108 : tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_118 = tt.broadcast %ar_115 : tensor<1x1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_119 = arith.xori %ret_117, %ret_118 : tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_120 = arith.extui %ret_116 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_121 = tt.broadcast %ret_119 : tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_122 = arith.cmpi ne, %ret_120, %ret_121 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_123 = arith.select %ret_122, %y_114, %h_107 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ix_124 = tt.bitcast %ret_123 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_125 = "tt.reduce"(%ix_124) <{axis = 8 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_126 = tt.expand_dims %iy_125 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_127 = tt.broadcast %iy_126 : tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %iy_128 = arith.xori %ix_124, %iy_127 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %y_129 = tt.bitcast %iy_128 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ar_130 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_131 = arith.cmpi ugt, %ret_123, %y_129 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_132 = tt.broadcast %ar_108 : tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_133 = tt.broadcast %ar_130 : tensor<1x1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_134 = arith.xori %ret_132, %ret_133 : tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_135 = arith.extui %ret_131 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_136 = tt.broadcast %ret_134 : tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_137 = arith.cmpi ne, %ret_135, %ret_136 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %ret_138 = arith.select %ret_137, %y_129, %ret_123 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked3}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>
    %h_139 = "tt.reduce"(%ret_138) <{axis = 6 : i32}> ({
    ^bb0(%h_272: i32, %h_273: i32):
      %h_274 = arith.maxui %h_272, %h_273 : i32
      tt.reduce.return %h_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked3}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_140 = tt.reshape %ar_63 : tensor<2xi32, #linear4> -> tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ix_141 = tt.bitcast %h_139 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_142 = "tt.reduce"(%ix_141) <{axis = 6 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_143 = tt.expand_dims %iy_142 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_144 = tt.broadcast %iy_143 : tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_145 = arith.xori %ix_141, %iy_144 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %y_146 = tt.bitcast %iy_145 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_147 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_148 = arith.cmpi ugt, %h_139, %y_146 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_149 = tt.broadcast %ar_140 : tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_150 = tt.broadcast %ar_147 : tensor<1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_151 = arith.xori %ret_149, %ret_150 : tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_152 = arith.extui %ret_148 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_153 = tt.broadcast %ret_151 : tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_154 = arith.cmpi ne, %ret_152, %ret_153 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_155 = arith.select %ret_154, %y_146, %h_139 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ix_156 = tt.bitcast %ret_155 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_157 = "tt.reduce"(%ix_156) <{axis = 7 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_158 = tt.expand_dims %iy_157 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_159 = tt.broadcast %iy_158 : tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %iy_160 = arith.xori %ix_156, %iy_159 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %y_161 = tt.bitcast %iy_160 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ar_162 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_163 = arith.cmpi ugt, %ret_155, %y_161 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_164 = tt.broadcast %ar_140 : tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_165 = tt.broadcast %ar_162 : tensor<1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_166 = arith.xori %ret_164, %ret_165 : tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_167 = arith.extui %ret_163 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_168 = tt.broadcast %ret_166 : tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_169 = arith.cmpi ne, %ret_167, %ret_168 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %ret_170 = arith.select %ret_169, %y_161, %ret_155 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>
    %h_171 = "tt.reduce"(%ret_170) <{axis = 5 : i32}> ({
    ^bb0(%h_272: i32, %h_273: i32):
      %h_274 = arith.maxui %h_272, %h_273 : i32
      tt.reduce.return %h_274 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ix_172 = tt.bitcast %h_171 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_173 = "tt.reduce"(%ix_172) <{axis = 5 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>>
    %iy_174 = tt.expand_dims %iy_173 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>> -> tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_175 = tt.broadcast %iy_174 : tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_176 = arith.xori %ix_172, %iy_175 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %y_177 = tt.bitcast %iy_176 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ar_178 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #linear1>
    %ar_179 = tt.reshape %ar_64 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_180 = arith.cmpi ugt, %h_171, %y_177 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_181 = arith.xori %ar_178, %cst_14 : tensor<1x1x1x1x1x2x1xi32, #linear1>
    %ret_182 = arith.xori %ar_179, %cst_15 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_183 = arith.extui %ret_180 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_184 = tt.broadcast %ret_181 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_185 = tt.broadcast %ret_182 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_186 = arith.cmpi ne, %ret_183, %ret_185 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_187 = arith.select %ret_186, %y_177, %h_171 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ix_188 = tt.bitcast %ret_187 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_189 = "tt.reduce"(%ix_188) <{axis = 6 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>>
    %iy_190 = tt.expand_dims %iy_189 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>}>> -> tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_191 = tt.broadcast %iy_190 : tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %iy_192 = arith.xori %ix_188, %iy_191 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %y_193 = tt.bitcast %iy_192 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ar_194 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x2xi32, #linear1>
    %ar_195 = tt.reshape %ar_65 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_196 = arith.cmpi ugt, %ret_187, %y_193 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_197 = arith.xori %ar_194, %cst_12 : tensor<1x1x1x1x1x1x2xi32, #linear1>
    %ret_198 = arith.xori %ar_195, %cst_13 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_199 = arith.extui %ret_196 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_200 = tt.broadcast %ret_197 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_201 = tt.broadcast %ret_198 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_202 = arith.cmpi ne, %ret_199, %ret_201 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %ret_203 = arith.select %ret_202, %y_193, %ret_187 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>>
    %x_204 = tt.reshape %ret_203 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked3}>}>}>> -> tensor<32x4xi32, #linear>
    %acc = arith.shli %x_204, %cst_4 : tensor<32x4xi32, #linear>
    %acc_205 = arith.shrui %x_204, %cst_4 : tensor<32x4xi32, #linear>
    %acc_206 = arith.ori %acc, %acc_205 : tensor<32x4xi32, #linear>
    %h_207 = tt.reshape %acc_206 : tensor<32x4xi32, #linear> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_208 = tt.bitcast %h_207 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_209 = "tt.reduce"(%ix_208) <{axis = 6 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %iy_210 = tt.expand_dims %iy_209 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %iy_211 = tt.broadcast %iy_210 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_212 = arith.xori %ix_208, %iy_211 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_213 = tt.bitcast %iy_212 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_214 = arith.cmpi ugt, %h_207, %y_213 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_215 = tt.broadcast %ar_178 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_216 = tt.broadcast %ar_194 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_217 = arith.xori %ret_215, %ret_216 : tensor<1x1x1x1x1x2x2xi32, #linear1>
    %ret_218 = arith.extui %ret_214 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_219 = tt.broadcast %ret_217 : tensor<1x1x1x1x1x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_220 = arith.cmpi ne, %ret_218, %ret_219 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_221 = arith.select %ret_220, %y_213, %h_207 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_222 = tt.bitcast %ret_221 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_223 = "tt.reduce"(%ix_222) <{axis = 5 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>>
    %iy_224 = tt.expand_dims %iy_223 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>> -> tensor<2x2x2x2x2x1x2xi32, #linear1>
    %iy_225 = tt.broadcast %iy_224 : tensor<2x2x2x2x2x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_226 = arith.xori %ix_222, %iy_225 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_227 = tt.bitcast %iy_226 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_228 = arith.cmpi ugt, %ret_221, %y_227 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_229 = arith.extui %ret_228 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_230 = arith.cmpi ne, %ret_229, %ret_184 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_231 = arith.select %ret_230, %y_227, %ret_221 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ix_232 = tt.bitcast %ret_231 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_233 = "tt.reduce"(%ix_232) <{axis = 6 : i32}> ({
    ^bb0(%iy_272: i32, %iy_273: i32):
      %iy_274 = arith.xori %iy_272, %iy_273 : i32
      tt.reduce.return %iy_274 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %iy_234 = tt.expand_dims %iy_233 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %iy_235 = tt.broadcast %iy_234 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %iy_236 = arith.xori %ix_232, %iy_235 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %y_237 = tt.bitcast %iy_236 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_238 = arith.cmpi ugt, %ret_231, %y_237 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_239 = arith.extui %ret_238 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_240 = arith.cmpi ne, %ret_239, %ret_200 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %ret_241 = arith.select %ret_240, %y_237, %ret_231 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %x_242 = tt.reshape %ret_241 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<32x4xi32, #linear>
    %y_indices_raw = arith.shrui %x_242, %cst_4 : tensor<32x4xi32, #linear>
    %y_indices = arith.subi %cst, %y_indices_raw : tensor<32x4xi32, #linear>
    %y_values_raw = arith.trunci %x_242 : tensor<32x4xi32, #linear> to tensor<32x4xi16, #linear>
    %y_values = arith.andi %y_values_raw, %cst_2 : tensor<32x4xi16, #linear>
    %y_values_243 = arith.extui %y_values : tensor<32x4xi16, #linear> to tensor<32x4xi32, #linear>
    %y_values_244 = arith.cmpi eq, %y_values_243, %cst_3 : tensor<32x4xi32, #linear>
    %y_values_245 = arith.select %y_values_244, %cst_1, %cst_2 : tensor<32x4xi1, #linear>, tensor<32x4xi16, #linear>
    %y_values_246 = arith.xori %y_values_raw, %y_values_245 : tensor<32x4xi16, #linear>
    %y_values_247 = tt.bitcast %y_values_246 : tensor<32x4xi16, #linear> -> tensor<32x4xf16, #linear>
    %y_values_248 = arith.extf %y_values_247 : tensor<32x4xf16, #linear> to tensor<32x4xf32, #linear>
    %z = "tt.reduce"(%y_values_248) <{axis = 1 : i32}> ({
    ^bb0(%z_272: f32, %z_273: f32):
      %z_274 = arith.maxnumf %z_272, %z_273 : f32
      tt.reduce.return %z_274 : f32
    }) : (tensor<32x4xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %z_249 = tt.expand_dims %z {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xf32, #linear>
    %z_250 = tt.broadcast %z_249 : tensor<32x1xf32, #linear> -> tensor<32x4xf32, #linear>
    %z_251 = arith.subf %y_values_248, %z_250 : tensor<32x4xf32, #linear>
    %num = math.exp %z_251 : tensor<32x4xf32, #linear>
    %den = "tt.reduce"(%num) <{axis = 1 : i32}> ({
    ^bb0(%den_272: f32, %den_273: f32):
      %den_274 = arith.addf %den_272, %den_273 : f32
      tt.reduce.return %den_274 : f32
    }) : (tensor<32x4xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %den_252 = tt.expand_dims %den {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xf32, #linear>
    %y_values_253 = tt.broadcast %den_252 : tensor<32x1xf32, #linear> -> tensor<32x4xf32, #linear>
    %y_values_254 = arith.divf %num, %y_values_253 : tensor<32x4xf32, #linear>
    %y_values_255 = arith.truncf %y_values_254 : tensor<32x4xf32, #linear> to tensor<32x4xf16, #linear>
    %Yv_ptrs = tt.expand_dims %offs_m_17 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %Yv_ptrs_256 = arith.muli %4, %stride_ym : i32
    %Yv_ptrs_257 = tt.splat %stride_ym : i32 -> tensor<32x1xi32, #blocked4>
    %Yv_ptrs_258 = arith.muli %Yv_ptrs, %Yv_ptrs_257 : tensor<32x1xi32, #blocked4>
    %Yv_ptrs_259 = tt.addptr %Yv, %Yv_ptrs_256 : !tt.ptr<f16>, i32
    %Yv_ptrs_260 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %Yv_ptrs_261 = tt.broadcast %Yv_ptrs_258 : tensor<32x1xi32, #blocked4> -> tensor<32x4xi32, #blocked4>
    %Yv_ptrs_262 = tt.expand_dims %Yv_ptrs_260 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x4xi32, #blocked4>
    %Yv_ptrs_263 = tt.broadcast %Yv_ptrs_262 : tensor<1x4xi32, #blocked4> -> tensor<32x4xi32, #blocked4>
    %Yv_ptrs_264 = arith.addi %Yv_ptrs_263, %Yv_ptrs_261 : tensor<32x4xi32, #blocked4>
    %6 = tt.broadcast %mask_m_32 : tensor<32x1xi1, #blocked4> -> tensor<32x4xi1, #blocked4>
    %7 = ttg.convert_layout %y_values_255 : tensor<32x4xf16, #linear> -> tensor<32x4xf16, #blocked4>
    %8 = tt.splat %Yv_ptrs_259 : !tt.ptr<f16> -> tensor<32x4x!tt.ptr<f16>, #blocked4>
    %9 = tt.addptr %8, %Yv_ptrs_264 : tensor<32x4x!tt.ptr<f16>, #blocked4>, tensor<32x4xi32, #blocked4>
    tt.store %9, %7, %6 : tensor<32x4x!tt.ptr<f16>, #blocked4>
    %Yi_ptrs = tt.addptr %Yi, %Yv_ptrs_256 : !tt.ptr<i16>, i32
    %10 = arith.trunci %y_indices : tensor<32x4xi32, #linear> to tensor<32x4xi16, #linear>
    %11 = ttg.convert_layout %10 : tensor<32x4xi16, #linear> -> tensor<32x4xi16, #blocked4>
    %12 = tt.splat %Yi_ptrs : !tt.ptr<i16> -> tensor<32x4x!tt.ptr<i16>, #blocked4>
    %13 = tt.addptr %12, %Yv_ptrs_264 : tensor<32x4x!tt.ptr<i16>, #blocked4>, tensor<32x4xi32, #blocked4>
    tt.store %13, %11, %6 : tensor<32x4x!tt.ptr<i16>, #blocked4>
    %y_div = arith.divui %y_indices, %cst : tensor<32x4xi32, #linear>
    %y_rem = arith.remui %y_indices, %cst : tensor<32x4xi32, #linear>
    %y2 = ttg.convert_layout %y_div : tensor<32x4xi32, #linear> -> tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %y2_265 = tt.expand_dims %y2 {axis = 2 : i32} : tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<32x4x1xi32, #blocked1>
    %y2_266 = arith.cmpi eq, %y2_265, %cst_10 : tensor<32x4x1xi32, #blocked1>
    %y2_267 = arith.shli %cst_0, %y_rem : tensor<32x4xi32, #linear>
    %y2_268 = ttg.convert_layout %y2_267 : tensor<32x4xi32, #linear> -> tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %y2_269 = tt.expand_dims %y2_268 {axis = 2 : i32} : tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<32x4x1xi32, #blocked1>
    %y2_270 = arith.select %y2_266, %y2_269, %cst_10 : tensor<32x4x1xi1, #blocked1>, tensor<32x4x1xi32, #blocked1>
    %r = "tt.reduce"(%y2_270) <{axis = 1 : i32}> ({
    ^bb0(%r_272: i32, %r_273: i32):
      %r_274 = arith.ori %r_272, %r_273 : i32
      tt.reduce.return %r_274 : i32
    }) : (tensor<32x4x1xi32, #blocked1>) -> tensor<32x1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %BitsPtrs = tt.expand_dims %offs_m_18 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<32x1xi32, #blocked5>
    %BitsPtrs_271 = tt.addptr %Bits, %4 : !tt.ptr<i32>, i32
    %14 = ttg.convert_layout %r : tensor<32x1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked5>
    amdgpu.buffer_store %14, %BitsPtrs_271[%BitsPtrs], %mask_m_33 : tensor<32x1xi32, #blocked5>
    tt.return
  }
}
