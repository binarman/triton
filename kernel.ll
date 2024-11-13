; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%0 = type { i64, i64, i32, i32 }
%1 = type { [64 x [8 x i64]] }

@printfFormat_29 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_28 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_27 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_26 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_25 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_24 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_23 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_22 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_21 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_20 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_19 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_18 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_17 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_16 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_15 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_14 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_13 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_12 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_11 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_10 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_9 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_8 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_7 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_6 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_5 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_4 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_3 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_2 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_1 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@printfFormat_0 = internal constant [150 x i8] c"device assertion failed: 'overflow detected', in unknown at /tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0/test_scan_layouts.ttgir:28\0A"
@global_smem = external local_unnamed_addr addrspace(3) global [0 x i8], align 16

; Function Attrs: nounwind
define amdgpu_kernel void @kernel_0d1d(ptr addrspace(1) inreg nocapture readonly %0, ptr addrspace(1) inreg nocapture writeonly %1) local_unnamed_addr #0 !dbg !6 {
  %3 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !9
  %4 = and i32 %3, 63, !dbg !9
  %5 = lshr i32 %3, 6, !dbg !9
  %6 = lshr i32 %3, 2, !dbg !9
  %7 = and i32 %6, 14, !dbg !9
  %8 = lshr i32 %3, 3, !dbg !9
  %9 = and i32 %8, 16, !dbg !9
  %10 = or disjoint i32 %7, %9, !dbg !9
  %11 = shl nuw nsw i32 %10, 6, !dbg !10
  %12 = or disjoint i32 %11, 64, !dbg !10
  %13 = zext nneg i32 %11 to i64, !dbg !11
  %14 = getelementptr i32, ptr addrspace(1) %0, i64 %13, !dbg !11
  %15 = zext nneg i32 %12 to i64, !dbg !11
  %16 = getelementptr i32, ptr addrspace(1) %0, i64 %15, !dbg !11
  %17 = shl i32 %3, 1, !dbg !12
  %18 = and i32 %17, 14, !dbg !12
  %19 = and i32 %6, 16, !dbg !12
  %20 = or disjoint i32 %18, %19, !dbg !12
  %21 = or disjoint i32 %20, 32, !dbg !12
  %22 = zext nneg i32 %20 to i64, !dbg !13
  %23 = getelementptr i32, ptr addrspace(1) %14, i64 %22, !dbg !13
  %24 = getelementptr i32, ptr addrspace(1) %16, i64 %22, !dbg !13
  %25 = zext nneg i32 %21 to i64, !dbg !13
  %26 = getelementptr i32, ptr addrspace(1) %14, i64 %25, !dbg !13
  %27 = getelementptr i32, ptr addrspace(1) %16, i64 %25, !dbg !13
  %unmaskedload = load <2 x i32>, ptr addrspace(1) %23, align 16, !dbg !14
  %28 = extractelement <2 x i32> %unmaskedload, i64 0, !dbg !14
  %29 = extractelement <2 x i32> %unmaskedload, i64 1, !dbg !14
  %unmaskedload1 = load <2 x i32>, ptr addrspace(1) %24, align 16, !dbg !14
  %30 = extractelement <2 x i32> %unmaskedload1, i64 0, !dbg !14
  %unmaskedload2 = load <2 x i32>, ptr addrspace(1) %26, align 16, !dbg !14
  %31 = extractelement <2 x i32> %unmaskedload2, i64 0, !dbg !14
  %32 = extractelement <2 x i32> %unmaskedload2, i64 1, !dbg !14
  %unmaskedload3 = load <2 x i32>, ptr addrspace(1) %27, align 16, !dbg !14
  %33 = extractelement <2 x i32> %unmaskedload3, i64 0, !dbg !14
  %34 = extractelement <2 x i32> %unmaskedload3, i64 1, !dbg !14
  %35 = and i32 %3, 7, !dbg !15
  %36 = lshr i32 %4, 3, !dbg !15
  %37 = and i32 %5, 1, !dbg !15
  %38 = lshr i32 %3, 4, !dbg !15
  %39 = and i32 %38, 8, !dbg !15
  %40 = or disjoint i32 %36, %39, !dbg !15
  %41 = add i32 %28, %29, !dbg !16
  %42 = sext i32 %28 to i64, !dbg !17
  %43 = sext i32 %29 to i64, !dbg !18
  %44 = add nsw i64 %42, -2147483647, !dbg !19
  %45 = add nsw i64 %44, %43, !dbg !20
  %46 = icmp ult i64 %45, -4294967295, !dbg !20
  br i1 %46, label %47, label %49, !dbg !21

47:                                               ; preds = %2
  %48 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %48, ptr nonnull @printfFormat_0)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

49:                                               ; preds = %2
  %50 = extractelement <2 x i32> %unmaskedload1, i64 1, !dbg !14
  %51 = add i32 %30, %50, !dbg !16
  %52 = sext i32 %30 to i64, !dbg !17
  %53 = sext i32 %50 to i64, !dbg !18
  %54 = add nsw i64 %52, -2147483647, !dbg !19
  %55 = add nsw i64 %54, %53, !dbg !20
  %56 = icmp ult i64 %55, -4294967295, !dbg !20
  br i1 %56, label %57, label %59, !dbg !21

57:                                               ; preds = %49
  %58 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %58, ptr nonnull @printfFormat_1)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

59:                                               ; preds = %49
  %60 = add i32 %31, %32, !dbg !16
  %61 = sext i32 %31 to i64, !dbg !17
  %62 = sext i32 %32 to i64, !dbg !18
  %63 = add nsw i64 %61, -2147483647, !dbg !19
  %64 = add nsw i64 %63, %62, !dbg !20
  %65 = icmp ult i64 %64, -4294967295, !dbg !20
  br i1 %65, label %66, label %68, !dbg !21

66:                                               ; preds = %59
  %67 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %67, ptr nonnull @printfFormat_2)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

68:                                               ; preds = %59
  %69 = add i32 %33, %34, !dbg !16
  %70 = sext i32 %33 to i64, !dbg !17
  %71 = sext i32 %34 to i64, !dbg !18
  %72 = add nsw i64 %70, -2147483647, !dbg !19
  %73 = add nsw i64 %72, %71, !dbg !20
  %74 = icmp ult i64 %73, -4294967295, !dbg !20
  br i1 %74, label %75, label %77, !dbg !21

75:                                               ; preds = %68
  %76 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %76, ptr nonnull @printfFormat_3)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

77:                                               ; preds = %68
  %78 = tail call i32 @llvm.usub.sat.i32(i32 %4, i32 1), !dbg !15
  %79 = shl nuw nsw i32 %78, 2, !dbg !15
  %80 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %79, i32 %41), !dbg !15
  %.not = icmp eq i32 %35, 0, !dbg !15
  br i1 %.not, label %81, label %404, !dbg !15

81:                                               ; preds = %404, %77
  %82 = phi i32 [ %41, %77 ], [ %405, %404 ], !dbg !15
  %83 = icmp samesign ult i32 %4, 2, !dbg !15
  %84 = shl nuw nsw i32 %4, 2, !dbg !15
  %85 = add nsw i32 %84, -8, !dbg !15
  %86 = select i1 %83, i32 %84, i32 %85, !dbg !15
  %87 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %86, i32 %82), !dbg !15
  %88 = icmp samesign ugt i32 %35, 1, !dbg !15
  br i1 %88, label %395, label %89, !dbg !15

89:                                               ; preds = %395, %81
  %90 = phi i32 [ %82, %81 ], [ %396, %395 ], !dbg !15
  %91 = icmp samesign ult i32 %4, 4, !dbg !15
  %92 = add nsw i32 %84, -16, !dbg !15
  %93 = select i1 %91, i32 %84, i32 %92, !dbg !15
  %94 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %93, i32 %90), !dbg !15
  %95 = icmp samesign ugt i32 %35, 3, !dbg !15
  br i1 %95, label %386, label %96, !dbg !15

96:                                               ; preds = %386, %89
  %97 = phi i32 [ %90, %89 ], [ %387, %386 ], !dbg !15
  %98 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %79, i32 %51), !dbg !15
  br i1 %.not, label %99, label %377, !dbg !15

99:                                               ; preds = %377, %96
  %100 = phi i32 [ %51, %96 ], [ %378, %377 ], !dbg !15
  %101 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %86, i32 %100), !dbg !15
  br i1 %88, label %368, label %102, !dbg !15

102:                                              ; preds = %368, %99
  %103 = phi i32 [ %100, %99 ], [ %369, %368 ], !dbg !15
  %104 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %93, i32 %103), !dbg !15
  br i1 %95, label %359, label %105, !dbg !15

105:                                              ; preds = %359, %102
  %106 = phi i32 [ %103, %102 ], [ %360, %359 ], !dbg !15
  %107 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %79, i32 %60), !dbg !15
  br i1 %.not, label %108, label %350, !dbg !15

108:                                              ; preds = %350, %105
  %109 = phi i32 [ %60, %105 ], [ %351, %350 ], !dbg !15
  %110 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %86, i32 %109), !dbg !15
  br i1 %88, label %341, label %111, !dbg !15

111:                                              ; preds = %341, %108
  %112 = phi i32 [ %109, %108 ], [ %342, %341 ], !dbg !15
  %113 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %93, i32 %112), !dbg !15
  br i1 %95, label %332, label %114, !dbg !15

114:                                              ; preds = %332, %111
  %115 = phi i32 [ %112, %111 ], [ %333, %332 ], !dbg !15
  %116 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %79, i32 %69), !dbg !15
  br i1 %.not, label %117, label %323, !dbg !15

117:                                              ; preds = %323, %114
  %118 = phi i32 [ %69, %114 ], [ %324, %323 ], !dbg !15
  %119 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %86, i32 %118), !dbg !15
  br i1 %88, label %314, label %120, !dbg !15

120:                                              ; preds = %314, %117
  %121 = phi i32 [ %118, %117 ], [ %315, %314 ], !dbg !15
  %122 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %93, i32 %121), !dbg !15
  br i1 %95, label %305, label %123, !dbg !15

123:                                              ; preds = %305, %120
  %124 = phi i32 [ %121, %120 ], [ %306, %305 ], !dbg !15
  %125 = icmp eq i32 %35, 7, !dbg !15
  br i1 %125, label %.critedge, label %.critedge9, !dbg !15

.critedge:                                        ; preds = %123
  %126 = shl nuw nsw i32 %37, 4, !dbg !15
  %127 = or disjoint i32 %40, %126, !dbg !15
  %128 = or disjoint i32 %127, 96
  %129 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %128
  %130 = or disjoint i32 %127, 64
  %131 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %130
  %132 = or disjoint i32 %127, 32
  %133 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %132
  %134 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %127
  store i32 %97, ptr addrspace(3) %134, align 4, !dbg !15
  store i32 %106, ptr addrspace(3) %133, align 4, !dbg !15
  store i32 %115, ptr addrspace(3) %131, align 4, !dbg !15
  store i32 %124, ptr addrspace(3) %129, align 4, !dbg !15
  br label %.critedge9, !dbg !15

.critedge9:                                       ; preds = %123, %.critedge
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %.not5 = icmp eq i32 %37, 0, !dbg !15
  %135 = or i32 %37, %35, !dbg !15
  %.not4 = icmp eq i32 %135, 0, !dbg !15
  %136 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %40, !dbg !15
  %137 = load i32, ptr addrspace(3) %136, align 4, !dbg !15
  %138 = or disjoint i32 %40, 16, !dbg !15
  %139 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %138, !dbg !15
  %140 = load i32, ptr addrspace(3) %139, align 4, !dbg !15
  %141 = add i32 %140, %137, !dbg !16
  %142 = sext i32 %137 to i64, !dbg !17
  %143 = sext i32 %140 to i64, !dbg !18
  %144 = add nsw i64 %142, -2147483647, !dbg !19
  %145 = add nsw i64 %144, %143, !dbg !20
  %146 = icmp ult i64 %145, -4294967295, !dbg !20
  br i1 %146, label %147, label %149, !dbg !21

147:                                              ; preds = %.critedge9
  %148 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %148, ptr nonnull @printfFormat_16)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

149:                                              ; preds = %.critedge9
  br i1 %.not5, label %150, label %297, !dbg !15

150:                                              ; preds = %297, %149
  %151 = phi i32 [ %97, %149 ], [ %298, %297 ], !dbg !15
  %152 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %79, i32 %151), !dbg !15
  br i1 %.not4, label %153, label %289, !dbg !15

153:                                              ; preds = %289, %150
  %154 = phi i32 [ %28, %150 ], [ %291, %289 ], !dbg !15
  %155 = or disjoint i32 %40, 32, !dbg !15
  %156 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %155, !dbg !15
  %157 = load i32, ptr addrspace(3) %156, align 4, !dbg !15
  %158 = or disjoint i32 %40, 48, !dbg !15
  %159 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %158, !dbg !15
  %160 = load i32, ptr addrspace(3) %159, align 4, !dbg !15
  %161 = add i32 %160, %157, !dbg !16
  %162 = sext i32 %157 to i64, !dbg !17
  %163 = sext i32 %160 to i64, !dbg !18
  %164 = add nsw i64 %162, -2147483647, !dbg !19
  %165 = add nsw i64 %164, %163, !dbg !20
  %166 = icmp ult i64 %165, -4294967295, !dbg !20
  br i1 %166, label %167, label %169, !dbg !21

167:                                              ; preds = %153
  %168 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %168, ptr nonnull @printfFormat_19)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

169:                                              ; preds = %153
  br i1 %.not5, label %170, label %281, !dbg !15

170:                                              ; preds = %281, %169
  %171 = phi i32 [ %106, %169 ], [ %282, %281 ], !dbg !15
  %172 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %79, i32 %171), !dbg !15
  br i1 %.not4, label %173, label %273, !dbg !15

173:                                              ; preds = %273, %170
  %174 = phi i32 [ %30, %170 ], [ %275, %273 ], !dbg !15
  %175 = or disjoint i32 %40, 64, !dbg !15
  %176 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %175, !dbg !15
  %177 = load i32, ptr addrspace(3) %176, align 4, !dbg !15
  %178 = sext i32 %141 to i64, !dbg !17
  %179 = sext i32 %177 to i64, !dbg !18
  %180 = add nsw i64 %178, -2147483647, !dbg !19
  %181 = add nsw i64 %180, %179, !dbg !20
  %182 = icmp ult i64 %181, -4294967295, !dbg !20
  br i1 %182, label %183, label %185, !dbg !21

183:                                              ; preds = %173
  %184 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %184, ptr nonnull @printfFormat_22)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

185:                                              ; preds = %173
  %186 = add i32 %177, %141, !dbg !16
  %187 = select i1 %.not5, i32 %141, i32 %186, !dbg !15
  %188 = or disjoint i32 %40, 80, !dbg !15
  %189 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %188, !dbg !15
  %190 = load i32, ptr addrspace(3) %189, align 4, !dbg !15
  %191 = sext i32 %186 to i64, !dbg !17
  %192 = sext i32 %190 to i64, !dbg !18
  %193 = add nsw i64 %191, -2147483647, !dbg !19
  %194 = add nsw i64 %193, %192, !dbg !20
  %195 = icmp ult i64 %194, -4294967295, !dbg !20
  br i1 %195, label %196, label %198, !dbg !21

196:                                              ; preds = %185
  %197 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %197, ptr nonnull @printfFormat_23)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

198:                                              ; preds = %185
  %199 = add i32 %187, %115, !dbg !16
  %200 = sext i32 %187 to i64, !dbg !17
  %201 = sext i32 %115 to i64, !dbg !18
  %202 = add nsw i64 %201, -2147483647, !dbg !19
  %203 = add nsw i64 %202, %200, !dbg !20
  %204 = icmp ult i64 %203, -4294967295, !dbg !20
  br i1 %204, label %205, label %207, !dbg !21

205:                                              ; preds = %198
  %206 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %206, ptr nonnull @printfFormat_24)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

207:                                              ; preds = %198
  %208 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %79, i32 %199), !dbg !15
  %209 = select i1 %.not, i32 %187, i32 %208, !dbg !15
  %210 = sext i32 %209 to i64, !dbg !17
  %211 = add nsw i64 %63, %210, !dbg !20
  %212 = icmp ult i64 %211, -4294967295, !dbg !20
  br i1 %212, label %213, label %215, !dbg !21

213:                                              ; preds = %207
  %214 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %214, ptr nonnull @printfFormat_25)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

215:                                              ; preds = %207
  %216 = or disjoint i32 %40, 96, !dbg !15
  %217 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %216, !dbg !15
  %218 = load i32, ptr addrspace(3) %217, align 4, !dbg !15
  %219 = sext i32 %161 to i64, !dbg !17
  %220 = sext i32 %218 to i64, !dbg !18
  %221 = add nsw i64 %219, -2147483647, !dbg !19
  %222 = add nsw i64 %221, %220, !dbg !20
  %223 = icmp ult i64 %222, -4294967295, !dbg !20
  br i1 %223, label %224, label %226, !dbg !21

224:                                              ; preds = %215
  %225 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %225, ptr nonnull @printfFormat_26)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

226:                                              ; preds = %215
  %227 = add i32 %218, %161, !dbg !16
  %228 = select i1 %.not5, i32 %161, i32 %227, !dbg !15
  %229 = or disjoint i32 %40, 112, !dbg !15
  %230 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %229, !dbg !15
  %231 = load i32, ptr addrspace(3) %230, align 4, !dbg !15
  %232 = sext i32 %227 to i64, !dbg !17
  %233 = sext i32 %231 to i64, !dbg !18
  %234 = add nsw i64 %232, -2147483647, !dbg !19
  %235 = add nsw i64 %234, %233, !dbg !20
  %236 = icmp ult i64 %235, -4294967295, !dbg !20
  br i1 %236, label %237, label %239, !dbg !21

237:                                              ; preds = %226
  %238 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %238, ptr nonnull @printfFormat_27)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

239:                                              ; preds = %226
  %240 = add i32 %228, %124, !dbg !16
  %241 = sext i32 %228 to i64, !dbg !17
  %242 = sext i32 %124 to i64, !dbg !18
  %243 = add nsw i64 %242, -2147483647, !dbg !19
  %244 = add nsw i64 %243, %241, !dbg !20
  %245 = icmp ult i64 %244, -4294967295, !dbg !20
  br i1 %245, label %246, label %248, !dbg !21

246:                                              ; preds = %239
  %247 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %247, ptr nonnull @printfFormat_28)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

248:                                              ; preds = %239
  %249 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %79, i32 %240), !dbg !15
  %250 = select i1 %.not, i32 %228, i32 %249, !dbg !15
  %251 = sext i32 %250 to i64, !dbg !17
  %252 = add nsw i64 %72, %251, !dbg !20
  %253 = icmp ult i64 %252, -4294967295, !dbg !20
  br i1 %253, label %254, label %256, !dbg !21

254:                                              ; preds = %248
  %255 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %255, ptr nonnull @printfFormat_29)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

256:                                              ; preds = %248
  %257 = add i32 %250, %33, !dbg !16
  %258 = add i32 %209, %31, !dbg !16
  %259 = getelementptr i32, ptr addrspace(1) %1, i64 %13, !dbg !22
  %260 = getelementptr i32, ptr addrspace(1) %1, i64 %15, !dbg !22
  %261 = getelementptr i32, ptr addrspace(1) %259, i64 %22, !dbg !23
  %262 = getelementptr i32, ptr addrspace(1) %260, i64 %22, !dbg !23
  %263 = getelementptr i32, ptr addrspace(1) %259, i64 %25, !dbg !23
  %264 = getelementptr i32, ptr addrspace(1) %260, i64 %25, !dbg !23
  %265 = insertelement <2 x i32> poison, i32 %154, i64 0, !dbg !24
  %266 = insertelement <2 x i32> %265, i32 %151, i64 1, !dbg !24
  store <2 x i32> %266, ptr addrspace(1) %261, align 16, !dbg !24
  %267 = insertelement <2 x i32> poison, i32 %174, i64 0, !dbg !24
  %268 = insertelement <2 x i32> %267, i32 %171, i64 1, !dbg !24
  store <2 x i32> %268, ptr addrspace(1) %262, align 16, !dbg !24
  %269 = insertelement <2 x i32> poison, i32 %258, i64 0, !dbg !24
  %270 = insertelement <2 x i32> %269, i32 %199, i64 1, !dbg !24
  store <2 x i32> %270, ptr addrspace(1) %263, align 16, !dbg !24
  %271 = insertelement <2 x i32> poison, i32 %257, i64 0, !dbg !24
  %272 = insertelement <2 x i32> %271, i32 %240, i64 1, !dbg !24
  store <2 x i32> %272, ptr addrspace(1) %264, align 16, !dbg !24
  ret void, !dbg !25

273:                                              ; preds = %170
  %274 = select i1 %.not, i32 %157, i32 %172, !dbg !15
  %275 = add i32 %274, %30, !dbg !16
  %276 = sext i32 %274 to i64, !dbg !17
  %277 = add nsw i64 %54, %276, !dbg !20
  %278 = icmp ult i64 %277, -4294967295, !dbg !20
  br i1 %278, label %279, label %173, !dbg !21

279:                                              ; preds = %273
  %280 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %280, ptr nonnull @printfFormat_21)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

281:                                              ; preds = %169
  %282 = add i32 %157, %106, !dbg !16
  %283 = sext i32 %106 to i64, !dbg !18
  %284 = add nsw i64 %283, -2147483647, !dbg !19
  %285 = add nsw i64 %284, %162, !dbg !20
  %286 = icmp ult i64 %285, -4294967295, !dbg !20
  br i1 %286, label %287, label %170, !dbg !21

287:                                              ; preds = %281
  %288 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %288, ptr nonnull @printfFormat_20)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

289:                                              ; preds = %150
  %290 = select i1 %.not, i32 %137, i32 %152, !dbg !15
  %291 = add i32 %290, %28, !dbg !16
  %292 = sext i32 %290 to i64, !dbg !17
  %293 = add nsw i64 %44, %292, !dbg !20
  %294 = icmp ult i64 %293, -4294967295, !dbg !20
  br i1 %294, label %295, label %153, !dbg !21

295:                                              ; preds = %289
  %296 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %296, ptr nonnull @printfFormat_18)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

297:                                              ; preds = %149
  %298 = add i32 %137, %97, !dbg !16
  %299 = sext i32 %97 to i64, !dbg !18
  %300 = add nsw i64 %299, -2147483647, !dbg !19
  %301 = add nsw i64 %300, %142, !dbg !20
  %302 = icmp ult i64 %301, -4294967295, !dbg !20
  br i1 %302, label %303, label %150, !dbg !21

303:                                              ; preds = %297
  %304 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %304, ptr nonnull @printfFormat_17)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

305:                                              ; preds = %120
  %306 = add i32 %122, %121, !dbg !16
  %307 = sext i32 %122 to i64, !dbg !17
  %308 = sext i32 %121 to i64, !dbg !18
  %309 = add nsw i64 %308, -2147483647, !dbg !19
  %310 = add nsw i64 %309, %307, !dbg !20
  %311 = icmp ult i64 %310, -4294967295, !dbg !20
  br i1 %311, label %312, label %123, !dbg !21

312:                                              ; preds = %305
  %313 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %313, ptr nonnull @printfFormat_15)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

314:                                              ; preds = %117
  %315 = add i32 %119, %118, !dbg !16
  %316 = sext i32 %119 to i64, !dbg !17
  %317 = sext i32 %118 to i64, !dbg !18
  %318 = add nsw i64 %317, -2147483647, !dbg !19
  %319 = add nsw i64 %318, %316, !dbg !20
  %320 = icmp ult i64 %319, -4294967295, !dbg !20
  br i1 %320, label %321, label %120, !dbg !21

321:                                              ; preds = %314
  %322 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %322, ptr nonnull @printfFormat_14)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

323:                                              ; preds = %114
  %324 = add i32 %116, %69, !dbg !16
  %325 = sext i32 %116 to i64, !dbg !17
  %326 = sext i32 %69 to i64, !dbg !18
  %327 = add nsw i64 %326, -2147483647, !dbg !19
  %328 = add nsw i64 %327, %325, !dbg !20
  %329 = icmp ult i64 %328, -4294967295, !dbg !20
  br i1 %329, label %330, label %117, !dbg !21

330:                                              ; preds = %323
  %331 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %331, ptr nonnull @printfFormat_13)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

332:                                              ; preds = %111
  %333 = add i32 %113, %112, !dbg !16
  %334 = sext i32 %113 to i64, !dbg !17
  %335 = sext i32 %112 to i64, !dbg !18
  %336 = add nsw i64 %335, -2147483647, !dbg !19
  %337 = add nsw i64 %336, %334, !dbg !20
  %338 = icmp ult i64 %337, -4294967295, !dbg !20
  br i1 %338, label %339, label %114, !dbg !21

339:                                              ; preds = %332
  %340 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %340, ptr nonnull @printfFormat_12)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

341:                                              ; preds = %108
  %342 = add i32 %110, %109, !dbg !16
  %343 = sext i32 %110 to i64, !dbg !17
  %344 = sext i32 %109 to i64, !dbg !18
  %345 = add nsw i64 %344, -2147483647, !dbg !19
  %346 = add nsw i64 %345, %343, !dbg !20
  %347 = icmp ult i64 %346, -4294967295, !dbg !20
  br i1 %347, label %348, label %111, !dbg !21

348:                                              ; preds = %341
  %349 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %349, ptr nonnull @printfFormat_11)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

350:                                              ; preds = %105
  %351 = add i32 %107, %60, !dbg !16
  %352 = sext i32 %107 to i64, !dbg !17
  %353 = sext i32 %60 to i64, !dbg !18
  %354 = add nsw i64 %353, -2147483647, !dbg !19
  %355 = add nsw i64 %354, %352, !dbg !20
  %356 = icmp ult i64 %355, -4294967295, !dbg !20
  br i1 %356, label %357, label %108, !dbg !21

357:                                              ; preds = %350
  %358 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %358, ptr nonnull @printfFormat_10)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

359:                                              ; preds = %102
  %360 = add i32 %104, %103, !dbg !16
  %361 = sext i32 %104 to i64, !dbg !17
  %362 = sext i32 %103 to i64, !dbg !18
  %363 = add nsw i64 %362, -2147483647, !dbg !19
  %364 = add nsw i64 %363, %361, !dbg !20
  %365 = icmp ult i64 %364, -4294967295, !dbg !20
  br i1 %365, label %366, label %105, !dbg !21

366:                                              ; preds = %359
  %367 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %367, ptr nonnull @printfFormat_9)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

368:                                              ; preds = %99
  %369 = add i32 %101, %100, !dbg !16
  %370 = sext i32 %101 to i64, !dbg !17
  %371 = sext i32 %100 to i64, !dbg !18
  %372 = add nsw i64 %371, -2147483647, !dbg !19
  %373 = add nsw i64 %372, %370, !dbg !20
  %374 = icmp ult i64 %373, -4294967295, !dbg !20
  br i1 %374, label %375, label %102, !dbg !21

375:                                              ; preds = %368
  %376 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %376, ptr nonnull @printfFormat_8)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

377:                                              ; preds = %96
  %378 = add i32 %98, %51, !dbg !16
  %379 = sext i32 %98 to i64, !dbg !17
  %380 = sext i32 %51 to i64, !dbg !18
  %381 = add nsw i64 %380, -2147483647, !dbg !19
  %382 = add nsw i64 %381, %379, !dbg !20
  %383 = icmp ult i64 %382, -4294967295, !dbg !20
  br i1 %383, label %384, label %99, !dbg !21

384:                                              ; preds = %377
  %385 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %385, ptr nonnull @printfFormat_7)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

386:                                              ; preds = %89
  %387 = add i32 %94, %90, !dbg !16
  %388 = sext i32 %94 to i64, !dbg !17
  %389 = sext i32 %90 to i64, !dbg !18
  %390 = add nsw i64 %389, -2147483647, !dbg !19
  %391 = add nsw i64 %390, %388, !dbg !20
  %392 = icmp ult i64 %391, -4294967295, !dbg !20
  br i1 %392, label %393, label %96, !dbg !21

393:                                              ; preds = %386
  %394 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %394, ptr nonnull @printfFormat_6)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

395:                                              ; preds = %81
  %396 = add i32 %87, %82, !dbg !16
  %397 = sext i32 %87 to i64, !dbg !17
  %398 = sext i32 %82 to i64, !dbg !18
  %399 = add nsw i64 %398, -2147483647, !dbg !19
  %400 = add nsw i64 %399, %397, !dbg !20
  %401 = icmp ult i64 %400, -4294967295, !dbg !20
  br i1 %401, label %402, label %89, !dbg !21

402:                                              ; preds = %395
  %403 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %403, ptr nonnull @printfFormat_5)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21

404:                                              ; preds = %77
  %405 = add i32 %80, %41, !dbg !16
  %406 = sext i32 %80 to i64, !dbg !17
  %407 = sext i32 %41 to i64, !dbg !18
  %408 = add nsw i64 %407, -2147483647, !dbg !19
  %409 = add nsw i64 %408, %406, !dbg !20
  %410 = icmp ult i64 %409, -4294967295, !dbg !20
  br i1 %410, label %411, label %81, !dbg !21

411:                                              ; preds = %404
  %412 = tail call fastcc i64 @__ockl_fprintf_stderr_begin()
  tail call fastcc void @__ockl_printf_append_string_n(i64 %412, ptr nonnull @printfFormat_4)
  fence syncscope("workgroup") release, !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  tail call void @llvm.trap(), !dbg !21
  unreachable, !dbg !21
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #2

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent norecurse nounwind
define internal fastcc i64 @__ockl_fprintf_stderr_begin() unnamed_addr #5 {
  %1 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %2 = getelementptr inbounds i8, ptr addrspace(4) %1, i64 24
  %3 = load i64, ptr addrspace(4) %2, align 8, !tbaa !26
  %4 = inttoptr i64 %3 to ptr addrspace(1)
  %5 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %6 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %5)
  %7 = tail call i32 asm sideeffect "", "=v,0"(i32 %6) #10, !srcloc !30
  %8 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %7)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %10, label %.loopexit4.i.i

10:                                               ; preds = %0
  %11 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 24
  %12 = load atomic i64, ptr addrspace(1) %11 syncscope("one-as") acquire, align 8
  %13 = getelementptr i8, ptr addrspace(1) %4, i64 40
  %14 = load ptr addrspace(1), ptr addrspace(1) %4, align 8, !tbaa !31
  %15 = load i64, ptr addrspace(1) %13, align 8, !tbaa !35
  %16 = and i64 %15, %12
  %17 = getelementptr inbounds %0, ptr addrspace(1) %14, i64 %16
  %18 = load atomic i64, ptr addrspace(1) %17 syncscope("one-as") monotonic, align 8
  %19 = cmpxchg ptr addrspace(1) %11, i64 %12, i64 %18 syncscope("one-as") acquire monotonic, align 8
  %20 = extractvalue { i64, i1 } %19, 1
  %21 = extractvalue { i64, i1 } %19, 0
  br i1 %20, label %.loopexit4.i.i, label %.preheader3.i.i

.preheader3.i.i:                                  ; preds = %10, %.preheader3.i.i
  %22 = phi i64 [ %30, %.preheader3.i.i ], [ %21, %10 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %23 = load ptr addrspace(1), ptr addrspace(1) %4, align 8, !tbaa !31
  %24 = load i64, ptr addrspace(1) %13, align 8, !tbaa !35
  %25 = and i64 %24, %22
  %26 = getelementptr inbounds %0, ptr addrspace(1) %23, i64 %25
  %27 = load atomic i64, ptr addrspace(1) %26 syncscope("one-as") monotonic, align 8
  %28 = cmpxchg ptr addrspace(1) %11, i64 %22, i64 %27 syncscope("one-as") acquire monotonic, align 8
  %29 = extractvalue { i64, i1 } %28, 1
  %30 = extractvalue { i64, i1 } %28, 0
  br i1 %29, label %.loopexit4.i.i, label %.preheader3.i.i

.loopexit4.i.i:                                   ; preds = %.preheader3.i.i, %10, %0
  %31 = phi i64 [ 0, %0 ], [ %21, %10 ], [ %30, %.preheader3.i.i ]
  %32 = trunc i64 %31 to i32
  %33 = lshr i64 %31, 32
  %34 = trunc nuw i64 %33 to i32
  %35 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %32)
  %36 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %34)
  %37 = zext i32 %36 to i64
  %38 = shl nuw i64 %37, 32
  %39 = zext i32 %35 to i64
  %40 = or disjoint i64 %38, %39
  %41 = load ptr addrspace(1), ptr addrspace(1) %4, align 8, !tbaa !31
  %42 = getelementptr i8, ptr addrspace(1) %4, i64 40
  %43 = load i64, ptr addrspace(1) %42, align 8, !tbaa !35
  %44 = and i64 %40, %43
  %45 = getelementptr inbounds %0, ptr addrspace(1) %41, i64 %44
  %46 = getelementptr i8, ptr addrspace(1) %4, i64 8
  %47 = load ptr addrspace(1), ptr addrspace(1) %46, align 8, !tbaa !36
  %48 = getelementptr inbounds %1, ptr addrspace(1) %47, i64 %44
  %49 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true)
  br i1 %9, label %50, label %53

50:                                               ; preds = %.loopexit4.i.i
  %51 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 8
  %52 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 16
  store i64 %49, ptr addrspace(1) %51, align 8, !tbaa !37
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %52, align 8, !tbaa !40
  br label %53

53:                                               ; preds = %50, %.loopexit4.i.i
  %54 = zext i32 %7 to i64
  %55 = getelementptr inbounds [64 x [8 x i64]], ptr addrspace(1) %48, i64 0, i64 %54
  store i64 33, ptr addrspace(1) %55, align 8, !tbaa !26
  %56 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 8
  store i64 1, ptr addrspace(1) %56, align 8, !tbaa !26
  %57 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 16
  store i64 0, ptr addrspace(1) %57, align 8, !tbaa !26
  %58 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 24
  store i64 0, ptr addrspace(1) %58, align 8, !tbaa !26
  %59 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 32
  store i64 0, ptr addrspace(1) %59, align 8, !tbaa !26
  %60 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 40
  store i64 0, ptr addrspace(1) %60, align 8, !tbaa !26
  %61 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 48
  store i64 0, ptr addrspace(1) %61, align 8, !tbaa !26
  %62 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 56
  store i64 0, ptr addrspace(1) %62, align 8, !tbaa !26
  br i1 %9, label %63, label %__ockl_hsa_signal_add.exit.i.i

63:                                               ; preds = %53
  %64 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 32
  %65 = load atomic i64, ptr addrspace(1) %64 syncscope("one-as") monotonic, align 8
  %66 = load i64, ptr addrspace(1) %42, align 8, !tbaa !35
  %67 = and i64 %66, %40
  %68 = getelementptr inbounds %0, ptr addrspace(1) %41, i64 %67
  store i64 %65, ptr addrspace(1) %68, align 8, !tbaa !41
  %69 = cmpxchg ptr addrspace(1) %64, i64 %65, i64 %40 syncscope("one-as") release monotonic, align 8
  %70 = extractvalue { i64, i1 } %69, 1
  br i1 %70, label %.loopexit2.i.i, label %.preheader1.i.i

.preheader1.i.i:                                  ; preds = %63, %.preheader1.i.i
  %.pn = phi { i64, i1 } [ %72, %.preheader1.i.i ], [ %69, %63 ]
  %71 = extractvalue { i64, i1 } %.pn, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %71, ptr addrspace(1) %68, align 8, !tbaa !41
  %72 = cmpxchg ptr addrspace(1) %64, i64 %71, i64 %40 syncscope("one-as") release monotonic, align 8
  %73 = extractvalue { i64, i1 } %72, 1
  br i1 %73, label %.loopexit2.i.i, label %.preheader1.i.i

.loopexit2.i.i:                                   ; preds = %.preheader1.i.i, %63
  %74 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 16
  %75 = load i64, ptr addrspace(1) %74, align 8
  %76 = inttoptr i64 %75 to ptr addrspace(1)
  %77 = getelementptr inbounds i8, ptr addrspace(1) %76, i64 8
  %78 = atomicrmw add ptr addrspace(1) %77, i64 1 syncscope("one-as") release, align 8
  %79 = getelementptr inbounds i8, ptr addrspace(1) %76, i64 16
  %80 = load i64, ptr addrspace(1) %79, align 16, !tbaa !42
  %81 = icmp eq i64 %80, 0
  br i1 %81, label %__ockl_hsa_signal_add.exit.i.i, label %82

82:                                               ; preds = %.loopexit2.i.i
  %83 = inttoptr i64 %80 to ptr addrspace(1)
  %84 = getelementptr inbounds i8, ptr addrspace(1) %76, i64 24
  %85 = load i32, ptr addrspace(1) %84, align 8, !tbaa !44
  %86 = zext i32 %85 to i64
  store atomic i64 %86, ptr addrspace(1) %83 syncscope("one-as") release, align 8
  %87 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %85)
  %88 = and i32 %87, 255
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %88)
  br label %__ockl_hsa_signal_add.exit.i.i

__ockl_hsa_signal_add.exit.i.i:                   ; preds = %82, %.loopexit2.i.i, %53
  %89 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 20
  br label %90

90:                                               ; preds = %98, %__ockl_hsa_signal_add.exit.i.i
  br i1 %9, label %91, label %94

91:                                               ; preds = %90
  %92 = load atomic i32, ptr addrspace(1) %89 syncscope("one-as") acquire, align 4
  %93 = and i32 %92, 1
  br label %94

94:                                               ; preds = %91, %90
  %95 = phi i32 [ %93, %91 ], [ 1, %90 ]
  %96 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %95)
  %97 = icmp eq i32 %96, 0
  br i1 %97, label %99, label %98

98:                                               ; preds = %94
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %90

99:                                               ; preds = %94
  %100 = load i64, ptr addrspace(1) %55, align 8, !tbaa !26
  br i1 %9, label %101, label %__ockl_hostcall_preview.exit

101:                                              ; preds = %99
  %102 = load i64, ptr addrspace(1) %42, align 8, !tbaa !35
  %103 = add i64 %102, 1
  %104 = add i64 %103, %40
  %105 = icmp eq i64 %104, 0
  %106 = select i1 %105, i64 %103, i64 %104
  %107 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 24
  %108 = load atomic i64, ptr addrspace(1) %107 syncscope("one-as") monotonic, align 8
  %109 = load ptr addrspace(1), ptr addrspace(1) %4, align 8, !tbaa !31
  %110 = and i64 %106, %102
  %111 = getelementptr inbounds %0, ptr addrspace(1) %109, i64 %110
  store i64 %108, ptr addrspace(1) %111, align 8, !tbaa !41
  %112 = cmpxchg ptr addrspace(1) %107, i64 %108, i64 %106 syncscope("one-as") release monotonic, align 8
  %113 = extractvalue { i64, i1 } %112, 1
  br i1 %113, label %__ockl_hostcall_preview.exit, label %.preheader.i.i

.preheader.i.i:                                   ; preds = %101, %.preheader.i.i
  %.pn2 = phi { i64, i1 } [ %115, %.preheader.i.i ], [ %112, %101 ]
  %114 = extractvalue { i64, i1 } %.pn2, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %114, ptr addrspace(1) %111, align 8, !tbaa !41
  %115 = cmpxchg ptr addrspace(1) %107, i64 %114, i64 %106 syncscope("one-as") release monotonic, align 8
  %116 = extractvalue { i64, i1 } %115, 1
  br i1 %116, label %__ockl_hostcall_preview.exit, label %.preheader.i.i

__ockl_hostcall_preview.exit:                     ; preds = %.preheader.i.i, %99, %101
  ret i64 %100
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.sleep(i32 immarg) #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #4

; Function Attrs: nounwind
declare void @llvm.amdgcn.s.sendmsg(i32 immarg, i32) #7

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #8

; Function Attrs: convergent norecurse nounwind
define internal fastcc void @__ockl_printf_append_string_n(i64 noundef %0, ptr noundef readonly %1) unnamed_addr #5 {
  %3 = icmp eq ptr %1, null
  %4 = and i64 %0, -227
  br i1 %3, label %5, label %.loopexit32

5:                                                ; preds = %2
  %6 = or disjoint i64 %4, 34
  %7 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %8 = getelementptr inbounds i8, ptr addrspace(4) %7, i64 24
  %9 = load i64, ptr addrspace(4) %8, align 8, !tbaa !26
  %10 = inttoptr i64 %9 to ptr addrspace(1)
  %11 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %12 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %11)
  %13 = tail call i32 asm sideeffect "", "=v,0"(i32 %12) #10, !srcloc !30
  %14 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %13)
  %15 = icmp eq i32 %13, %14
  br i1 %15, label %16, label %.loopexit4.i.i

16:                                               ; preds = %5
  %17 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 24
  %18 = load atomic i64, ptr addrspace(1) %17 syncscope("one-as") acquire, align 8
  %19 = getelementptr i8, ptr addrspace(1) %10, i64 40
  %20 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !31
  %21 = load i64, ptr addrspace(1) %19, align 8, !tbaa !35
  %22 = and i64 %21, %18
  %23 = getelementptr inbounds %0, ptr addrspace(1) %20, i64 %22
  %24 = load atomic i64, ptr addrspace(1) %23 syncscope("one-as") monotonic, align 8
  %25 = cmpxchg ptr addrspace(1) %17, i64 %18, i64 %24 syncscope("one-as") acquire monotonic, align 8
  %26 = extractvalue { i64, i1 } %25, 1
  %27 = extractvalue { i64, i1 } %25, 0
  br i1 %26, label %.loopexit4.i.i, label %.preheader3.i.i

.preheader3.i.i:                                  ; preds = %16, %.preheader3.i.i
  %28 = phi i64 [ %36, %.preheader3.i.i ], [ %27, %16 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %29 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !31
  %30 = load i64, ptr addrspace(1) %19, align 8, !tbaa !35
  %31 = and i64 %30, %28
  %32 = getelementptr inbounds %0, ptr addrspace(1) %29, i64 %31
  %33 = load atomic i64, ptr addrspace(1) %32 syncscope("one-as") monotonic, align 8
  %34 = cmpxchg ptr addrspace(1) %17, i64 %28, i64 %33 syncscope("one-as") acquire monotonic, align 8
  %35 = extractvalue { i64, i1 } %34, 1
  %36 = extractvalue { i64, i1 } %34, 0
  br i1 %35, label %.loopexit4.i.i, label %.preheader3.i.i

.loopexit4.i.i:                                   ; preds = %.preheader3.i.i, %16, %5
  %37 = phi i64 [ 0, %5 ], [ %27, %16 ], [ %36, %.preheader3.i.i ]
  %38 = trunc i64 %37 to i32
  %39 = lshr i64 %37, 32
  %40 = trunc nuw i64 %39 to i32
  %41 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38)
  %42 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %40)
  %43 = zext i32 %42 to i64
  %44 = shl nuw i64 %43, 32
  %45 = zext i32 %41 to i64
  %46 = or disjoint i64 %44, %45
  %47 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !31
  %48 = getelementptr i8, ptr addrspace(1) %10, i64 40
  %49 = load i64, ptr addrspace(1) %48, align 8, !tbaa !35
  %50 = and i64 %46, %49
  %51 = getelementptr inbounds %0, ptr addrspace(1) %47, i64 %50
  %52 = getelementptr i8, ptr addrspace(1) %10, i64 8
  %53 = load ptr addrspace(1), ptr addrspace(1) %52, align 8, !tbaa !36
  %54 = getelementptr inbounds %1, ptr addrspace(1) %53, i64 %50
  %55 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true)
  br i1 %15, label %56, label %59

56:                                               ; preds = %.loopexit4.i.i
  %57 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 8
  %58 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 16
  store i64 %55, ptr addrspace(1) %57, align 8, !tbaa !37
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %58, align 8, !tbaa !40
  br label %59

59:                                               ; preds = %56, %.loopexit4.i.i
  %60 = zext i32 %13 to i64
  %61 = getelementptr inbounds [64 x [8 x i64]], ptr addrspace(1) %54, i64 0, i64 %60
  store i64 %6, ptr addrspace(1) %61, align 8, !tbaa !26
  %62 = getelementptr inbounds i8, ptr addrspace(1) %61, i64 8
  store i64 0, ptr addrspace(1) %62, align 8, !tbaa !26
  %63 = getelementptr inbounds i8, ptr addrspace(1) %61, i64 16
  store i64 0, ptr addrspace(1) %63, align 8, !tbaa !26
  %64 = getelementptr inbounds i8, ptr addrspace(1) %61, i64 24
  store i64 0, ptr addrspace(1) %64, align 8, !tbaa !26
  %65 = getelementptr inbounds i8, ptr addrspace(1) %61, i64 32
  store i64 0, ptr addrspace(1) %65, align 8, !tbaa !26
  %66 = getelementptr inbounds i8, ptr addrspace(1) %61, i64 40
  store i64 0, ptr addrspace(1) %66, align 8, !tbaa !26
  %67 = getelementptr inbounds i8, ptr addrspace(1) %61, i64 48
  store i64 0, ptr addrspace(1) %67, align 8, !tbaa !26
  %68 = getelementptr inbounds i8, ptr addrspace(1) %61, i64 56
  store i64 0, ptr addrspace(1) %68, align 8, !tbaa !26
  br i1 %15, label %69, label %__ockl_hsa_signal_add.exit.i.i

69:                                               ; preds = %59
  %70 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 32
  %71 = load atomic i64, ptr addrspace(1) %70 syncscope("one-as") monotonic, align 8
  %72 = load i64, ptr addrspace(1) %48, align 8, !tbaa !35
  %73 = and i64 %72, %46
  %74 = getelementptr inbounds %0, ptr addrspace(1) %47, i64 %73
  store i64 %71, ptr addrspace(1) %74, align 8, !tbaa !41
  %75 = cmpxchg ptr addrspace(1) %70, i64 %71, i64 %46 syncscope("one-as") release monotonic, align 8
  %76 = extractvalue { i64, i1 } %75, 1
  br i1 %76, label %.loopexit2.i.i, label %.preheader1.i.i

.preheader1.i.i:                                  ; preds = %69, %.preheader1.i.i
  %.pn12 = phi { i64, i1 } [ %78, %.preheader1.i.i ], [ %75, %69 ]
  %77 = extractvalue { i64, i1 } %.pn12, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %77, ptr addrspace(1) %74, align 8, !tbaa !41
  %78 = cmpxchg ptr addrspace(1) %70, i64 %77, i64 %46 syncscope("one-as") release monotonic, align 8
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %.loopexit2.i.i, label %.preheader1.i.i

.loopexit2.i.i:                                   ; preds = %.preheader1.i.i, %69
  %80 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 16
  %81 = load i64, ptr addrspace(1) %80, align 8
  %82 = inttoptr i64 %81 to ptr addrspace(1)
  %83 = getelementptr inbounds i8, ptr addrspace(1) %82, i64 8
  %84 = atomicrmw add ptr addrspace(1) %83, i64 1 syncscope("one-as") release, align 8
  %85 = getelementptr inbounds i8, ptr addrspace(1) %82, i64 16
  %86 = load i64, ptr addrspace(1) %85, align 16, !tbaa !42
  %87 = icmp eq i64 %86, 0
  br i1 %87, label %__ockl_hsa_signal_add.exit.i.i, label %88

88:                                               ; preds = %.loopexit2.i.i
  %89 = inttoptr i64 %86 to ptr addrspace(1)
  %90 = getelementptr inbounds i8, ptr addrspace(1) %82, i64 24
  %91 = load i32, ptr addrspace(1) %90, align 8, !tbaa !44
  %92 = zext i32 %91 to i64
  store atomic i64 %92, ptr addrspace(1) %89 syncscope("one-as") release, align 8
  %93 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %91)
  %94 = and i32 %93, 255
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %94)
  br label %__ockl_hsa_signal_add.exit.i.i

__ockl_hsa_signal_add.exit.i.i:                   ; preds = %88, %.loopexit2.i.i, %59
  %95 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 20
  br label %96

96:                                               ; preds = %104, %__ockl_hsa_signal_add.exit.i.i
  br i1 %15, label %97, label %100

97:                                               ; preds = %96
  %98 = load atomic i32, ptr addrspace(1) %95 syncscope("one-as") acquire, align 4
  %99 = and i32 %98, 1
  br label %100

100:                                              ; preds = %97, %96
  %101 = phi i32 [ %99, %97 ], [ 1, %96 ]
  %102 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101)
  %103 = icmp eq i32 %102, 0
  br i1 %103, label %105, label %104

104:                                              ; preds = %100
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %96

105:                                              ; preds = %100
  br i1 %15, label %106, label %.loopexit33

106:                                              ; preds = %105
  %107 = load i64, ptr addrspace(1) %48, align 8, !tbaa !35
  %108 = add i64 %107, 1
  %109 = add i64 %108, %46
  %110 = icmp eq i64 %109, 0
  %111 = select i1 %110, i64 %108, i64 %109
  %112 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 24
  %113 = load atomic i64, ptr addrspace(1) %112 syncscope("one-as") monotonic, align 8
  %114 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !31
  %115 = and i64 %111, %107
  %116 = getelementptr inbounds %0, ptr addrspace(1) %114, i64 %115
  store i64 %113, ptr addrspace(1) %116, align 8, !tbaa !41
  %117 = cmpxchg ptr addrspace(1) %112, i64 %113, i64 %111 syncscope("one-as") release monotonic, align 8
  %118 = extractvalue { i64, i1 } %117, 1
  br i1 %118, label %.loopexit33, label %.preheader.i.i

.preheader.i.i:                                   ; preds = %106, %.preheader.i.i
  %.pn14 = phi { i64, i1 } [ %120, %.preheader.i.i ], [ %117, %106 ]
  %119 = extractvalue { i64, i1 } %.pn14, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %119, ptr addrspace(1) %116, align 8, !tbaa !41
  %120 = cmpxchg ptr addrspace(1) %112, i64 %119, i64 %111 syncscope("one-as") release monotonic, align 8
  %121 = extractvalue { i64, i1 } %120, 1
  br i1 %121, label %.loopexit33, label %.preheader.i.i

.loopexit32:                                      ; preds = %2
  %122 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %123 = getelementptr inbounds i8, ptr addrspace(4) %122, i64 24
  %124 = load i64, ptr addrspace(4) %123, align 8, !tbaa !26
  %125 = inttoptr i64 %124 to ptr addrspace(1)
  %126 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %127 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %126)
  %128 = getelementptr inbounds i8, ptr addrspace(1) %125, i64 24
  %129 = getelementptr i8, ptr addrspace(1) %125, i64 40
  %130 = getelementptr i8, ptr addrspace(1) %125, i64 8
  %131 = getelementptr inbounds i8, ptr addrspace(1) %125, i64 32
  %132 = getelementptr inbounds i8, ptr addrspace(1) %125, i64 16
  %133 = load i8, ptr %1, align 1, !tbaa !45
  %134 = zext i8 %133 to i64
  %135 = getelementptr inbounds i8, ptr %1, i64 1
  %136 = load i8, ptr %135, align 1, !tbaa !45
  %137 = zext i8 %136 to i64
  %138 = shl nuw nsw i64 %137, 8
  %139 = or disjoint i64 %138, %134
  %140 = getelementptr inbounds i8, ptr %1, i64 2
  %141 = load i8, ptr %140, align 1, !tbaa !45
  %142 = zext i8 %141 to i64
  %143 = shl nuw nsw i64 %142, 16
  %144 = or disjoint i64 %139, %143
  %145 = getelementptr inbounds i8, ptr %1, i64 3
  %146 = load i8, ptr %145, align 1, !tbaa !45
  %147 = zext i8 %146 to i64
  %148 = shl nuw nsw i64 %147, 24
  %149 = or disjoint i64 %144, %148
  %150 = getelementptr inbounds i8, ptr %1, i64 4
  %151 = load i8, ptr %150, align 1, !tbaa !45
  %152 = zext i8 %151 to i64
  %153 = shl nuw nsw i64 %152, 32
  %154 = or disjoint i64 %149, %153
  %155 = getelementptr inbounds i8, ptr %1, i64 5
  %156 = load i8, ptr %155, align 1, !tbaa !45
  %157 = zext i8 %156 to i64
  %158 = shl nuw nsw i64 %157, 40
  %159 = or i64 %154, %158
  %160 = getelementptr inbounds i8, ptr %1, i64 6
  %161 = load i8, ptr %160, align 1, !tbaa !45
  %162 = zext i8 %161 to i64
  %163 = shl nuw nsw i64 %162, 48
  %164 = or i64 %159, %163
  %165 = getelementptr inbounds i8, ptr %1, i64 7
  %166 = load i8, ptr %165, align 1, !tbaa !45
  %167 = zext i8 %166 to i64
  %168 = shl nuw i64 %167, 56
  %169 = or i64 %164, %168
  %170 = getelementptr inbounds i8, ptr %1, i64 8
  %171 = load i8, ptr %170, align 1, !tbaa !45
  %172 = zext i8 %171 to i64
  %173 = getelementptr inbounds i8, ptr %1, i64 9
  %174 = load i8, ptr %173, align 1, !tbaa !45
  %175 = zext i8 %174 to i64
  %176 = shl nuw nsw i64 %175, 8
  %177 = or disjoint i64 %176, %172
  %178 = getelementptr inbounds i8, ptr %1, i64 10
  %179 = load i8, ptr %178, align 1, !tbaa !45
  %180 = zext i8 %179 to i64
  %181 = shl nuw nsw i64 %180, 16
  %182 = or disjoint i64 %177, %181
  %183 = getelementptr inbounds i8, ptr %1, i64 11
  %184 = load i8, ptr %183, align 1, !tbaa !45
  %185 = zext i8 %184 to i64
  %186 = shl nuw nsw i64 %185, 24
  %187 = or disjoint i64 %182, %186
  %188 = getelementptr inbounds i8, ptr %1, i64 12
  %189 = load i8, ptr %188, align 1, !tbaa !45
  %190 = zext i8 %189 to i64
  %191 = shl nuw nsw i64 %190, 32
  %192 = or disjoint i64 %187, %191
  %193 = getelementptr inbounds i8, ptr %1, i64 13
  %194 = load i8, ptr %193, align 1, !tbaa !45
  %195 = zext i8 %194 to i64
  %196 = shl nuw nsw i64 %195, 40
  %197 = or i64 %192, %196
  %198 = getelementptr inbounds i8, ptr %1, i64 14
  %199 = load i8, ptr %198, align 1, !tbaa !45
  %200 = zext i8 %199 to i64
  %201 = shl nuw nsw i64 %200, 48
  %202 = or i64 %197, %201
  %203 = getelementptr inbounds i8, ptr %1, i64 15
  %204 = load i8, ptr %203, align 1, !tbaa !45
  %205 = zext i8 %204 to i64
  %206 = shl nuw i64 %205, 56
  %207 = or i64 %202, %206
  %208 = getelementptr inbounds i8, ptr %1, i64 16
  %209 = load i8, ptr %208, align 1, !tbaa !45
  %210 = zext i8 %209 to i64
  %211 = getelementptr inbounds i8, ptr %1, i64 17
  %212 = load i8, ptr %211, align 1, !tbaa !45
  %213 = zext i8 %212 to i64
  %214 = shl nuw nsw i64 %213, 8
  %215 = or disjoint i64 %214, %210
  %216 = getelementptr inbounds i8, ptr %1, i64 18
  %217 = load i8, ptr %216, align 1, !tbaa !45
  %218 = zext i8 %217 to i64
  %219 = shl nuw nsw i64 %218, 16
  %220 = or disjoint i64 %215, %219
  %221 = getelementptr inbounds i8, ptr %1, i64 19
  %222 = load i8, ptr %221, align 1, !tbaa !45
  %223 = zext i8 %222 to i64
  %224 = shl nuw nsw i64 %223, 24
  %225 = or disjoint i64 %220, %224
  %226 = getelementptr inbounds i8, ptr %1, i64 20
  %227 = load i8, ptr %226, align 1, !tbaa !45
  %228 = zext i8 %227 to i64
  %229 = shl nuw nsw i64 %228, 32
  %230 = or disjoint i64 %225, %229
  %231 = getelementptr inbounds i8, ptr %1, i64 21
  %232 = load i8, ptr %231, align 1, !tbaa !45
  %233 = zext i8 %232 to i64
  %234 = shl nuw nsw i64 %233, 40
  %235 = or i64 %230, %234
  %236 = getelementptr inbounds i8, ptr %1, i64 22
  %237 = load i8, ptr %236, align 1, !tbaa !45
  %238 = zext i8 %237 to i64
  %239 = shl nuw nsw i64 %238, 48
  %240 = or i64 %235, %239
  %241 = getelementptr inbounds i8, ptr %1, i64 23
  %242 = load i8, ptr %241, align 1, !tbaa !45
  %243 = zext i8 %242 to i64
  %244 = shl nuw i64 %243, 56
  %245 = or i64 %240, %244
  %246 = getelementptr inbounds i8, ptr %1, i64 24
  %247 = load i8, ptr %246, align 1, !tbaa !45
  %248 = zext i8 %247 to i64
  %249 = getelementptr inbounds i8, ptr %1, i64 25
  %250 = load i8, ptr %249, align 1, !tbaa !45
  %251 = zext i8 %250 to i64
  %252 = shl nuw nsw i64 %251, 8
  %253 = or disjoint i64 %252, %248
  %254 = getelementptr inbounds i8, ptr %1, i64 26
  %255 = load i8, ptr %254, align 1, !tbaa !45
  %256 = zext i8 %255 to i64
  %257 = shl nuw nsw i64 %256, 16
  %258 = or disjoint i64 %253, %257
  %259 = getelementptr inbounds i8, ptr %1, i64 27
  %260 = load i8, ptr %259, align 1, !tbaa !45
  %261 = zext i8 %260 to i64
  %262 = shl nuw nsw i64 %261, 24
  %263 = or disjoint i64 %258, %262
  %264 = getelementptr inbounds i8, ptr %1, i64 28
  %265 = load i8, ptr %264, align 1, !tbaa !45
  %266 = zext i8 %265 to i64
  %267 = shl nuw nsw i64 %266, 32
  %268 = or disjoint i64 %263, %267
  %269 = getelementptr inbounds i8, ptr %1, i64 29
  %270 = load i8, ptr %269, align 1, !tbaa !45
  %271 = zext i8 %270 to i64
  %272 = shl nuw nsw i64 %271, 40
  %273 = or i64 %268, %272
  %274 = getelementptr inbounds i8, ptr %1, i64 30
  %275 = load i8, ptr %274, align 1, !tbaa !45
  %276 = zext i8 %275 to i64
  %277 = shl nuw nsw i64 %276, 48
  %278 = or i64 %273, %277
  %279 = getelementptr inbounds i8, ptr %1, i64 31
  %280 = load i8, ptr %279, align 1, !tbaa !45
  %281 = zext i8 %280 to i64
  %282 = shl nuw i64 %281, 56
  %283 = or i64 %278, %282
  %284 = getelementptr inbounds i8, ptr %1, i64 32
  %285 = load i8, ptr %284, align 1, !tbaa !45
  %286 = zext i8 %285 to i64
  %287 = getelementptr inbounds i8, ptr %1, i64 33
  %288 = load i8, ptr %287, align 1, !tbaa !45
  %289 = zext i8 %288 to i64
  %290 = shl nuw nsw i64 %289, 8
  %291 = or disjoint i64 %290, %286
  %292 = getelementptr inbounds i8, ptr %1, i64 34
  %293 = load i8, ptr %292, align 1, !tbaa !45
  %294 = zext i8 %293 to i64
  %295 = shl nuw nsw i64 %294, 16
  %296 = or disjoint i64 %291, %295
  %297 = getelementptr inbounds i8, ptr %1, i64 35
  %298 = load i8, ptr %297, align 1, !tbaa !45
  %299 = zext i8 %298 to i64
  %300 = shl nuw nsw i64 %299, 24
  %301 = or disjoint i64 %296, %300
  %302 = getelementptr inbounds i8, ptr %1, i64 36
  %303 = load i8, ptr %302, align 1, !tbaa !45
  %304 = zext i8 %303 to i64
  %305 = shl nuw nsw i64 %304, 32
  %306 = or disjoint i64 %301, %305
  %307 = getelementptr inbounds i8, ptr %1, i64 37
  %308 = load i8, ptr %307, align 1, !tbaa !45
  %309 = zext i8 %308 to i64
  %310 = shl nuw nsw i64 %309, 40
  %311 = or i64 %306, %310
  %312 = getelementptr inbounds i8, ptr %1, i64 38
  %313 = load i8, ptr %312, align 1, !tbaa !45
  %314 = zext i8 %313 to i64
  %315 = shl nuw nsw i64 %314, 48
  %316 = or i64 %311, %315
  %317 = getelementptr inbounds i8, ptr %1, i64 39
  %318 = load i8, ptr %317, align 1, !tbaa !45
  %319 = zext i8 %318 to i64
  %320 = shl nuw i64 %319, 56
  %321 = or i64 %316, %320
  %322 = getelementptr inbounds i8, ptr %1, i64 40
  %323 = load i8, ptr %322, align 1, !tbaa !45
  %324 = zext i8 %323 to i64
  %325 = getelementptr inbounds i8, ptr %1, i64 41
  %326 = load i8, ptr %325, align 1, !tbaa !45
  %327 = zext i8 %326 to i64
  %328 = shl nuw nsw i64 %327, 8
  %329 = or disjoint i64 %328, %324
  %330 = getelementptr inbounds i8, ptr %1, i64 42
  %331 = load i8, ptr %330, align 1, !tbaa !45
  %332 = zext i8 %331 to i64
  %333 = shl nuw nsw i64 %332, 16
  %334 = or disjoint i64 %329, %333
  %335 = getelementptr inbounds i8, ptr %1, i64 43
  %336 = load i8, ptr %335, align 1, !tbaa !45
  %337 = zext i8 %336 to i64
  %338 = shl nuw nsw i64 %337, 24
  %339 = or disjoint i64 %334, %338
  %340 = getelementptr inbounds i8, ptr %1, i64 44
  %341 = load i8, ptr %340, align 1, !tbaa !45
  %342 = zext i8 %341 to i64
  %343 = shl nuw nsw i64 %342, 32
  %344 = or disjoint i64 %339, %343
  %345 = getelementptr inbounds i8, ptr %1, i64 45
  %346 = load i8, ptr %345, align 1, !tbaa !45
  %347 = zext i8 %346 to i64
  %348 = shl nuw nsw i64 %347, 40
  %349 = or i64 %344, %348
  %350 = getelementptr inbounds i8, ptr %1, i64 46
  %351 = load i8, ptr %350, align 1, !tbaa !45
  %352 = zext i8 %351 to i64
  %353 = shl nuw nsw i64 %352, 48
  %354 = or i64 %349, %353
  %355 = getelementptr inbounds i8, ptr %1, i64 47
  %356 = load i8, ptr %355, align 1, !tbaa !45
  %357 = zext i8 %356 to i64
  %358 = shl nuw i64 %357, 56
  %359 = or i64 %354, %358
  %360 = getelementptr inbounds i8, ptr %1, i64 48
  %361 = load i8, ptr %360, align 1, !tbaa !45
  %362 = zext i8 %361 to i64
  %363 = getelementptr inbounds i8, ptr %1, i64 49
  %364 = load i8, ptr %363, align 1, !tbaa !45
  %365 = zext i8 %364 to i64
  %366 = shl nuw nsw i64 %365, 8
  %367 = or disjoint i64 %366, %362
  %368 = getelementptr inbounds i8, ptr %1, i64 50
  %369 = load i8, ptr %368, align 1, !tbaa !45
  %370 = zext i8 %369 to i64
  %371 = shl nuw nsw i64 %370, 16
  %372 = or disjoint i64 %367, %371
  %373 = getelementptr inbounds i8, ptr %1, i64 51
  %374 = load i8, ptr %373, align 1, !tbaa !45
  %375 = zext i8 %374 to i64
  %376 = shl nuw nsw i64 %375, 24
  %377 = or disjoint i64 %372, %376
  %378 = getelementptr inbounds i8, ptr %1, i64 52
  %379 = load i8, ptr %378, align 1, !tbaa !45
  %380 = zext i8 %379 to i64
  %381 = shl nuw nsw i64 %380, 32
  %382 = or disjoint i64 %377, %381
  %383 = getelementptr inbounds i8, ptr %1, i64 53
  %384 = load i8, ptr %383, align 1, !tbaa !45
  %385 = zext i8 %384 to i64
  %386 = shl nuw nsw i64 %385, 40
  %387 = or i64 %382, %386
  %388 = getelementptr inbounds i8, ptr %1, i64 54
  %389 = load i8, ptr %388, align 1, !tbaa !45
  %390 = zext i8 %389 to i64
  %391 = shl nuw nsw i64 %390, 48
  %392 = or i64 %387, %391
  %393 = getelementptr inbounds i8, ptr %1, i64 55
  %394 = load i8, ptr %393, align 1, !tbaa !45
  %395 = zext i8 %394 to i64
  %396 = shl nuw i64 %395, 56
  %397 = or i64 %392, %396
  %398 = or disjoint i64 %4, 224
  %399 = tail call i32 asm sideeffect "", "=v,0"(i32 %127) #10, !srcloc !30
  %400 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %399)
  %401 = icmp eq i32 %399, %400
  br i1 %401, label %402, label %.loopexit4.i.i14

402:                                              ; preds = %.loopexit32
  %403 = load atomic i64, ptr addrspace(1) %128 syncscope("one-as") acquire, align 8
  %404 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %405 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %406 = and i64 %405, %403
  %407 = getelementptr inbounds %0, ptr addrspace(1) %404, i64 %406
  %408 = load atomic i64, ptr addrspace(1) %407 syncscope("one-as") monotonic, align 8
  %409 = cmpxchg ptr addrspace(1) %128, i64 %403, i64 %408 syncscope("one-as") acquire monotonic, align 8
  %410 = extractvalue { i64, i1 } %409, 1
  %411 = extractvalue { i64, i1 } %409, 0
  br i1 %410, label %.loopexit4.i.i14, label %.preheader3.i.i19

.preheader3.i.i19:                                ; preds = %402, %.preheader3.i.i19
  %412 = phi i64 [ %420, %.preheader3.i.i19 ], [ %411, %402 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %413 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %414 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %415 = and i64 %414, %412
  %416 = getelementptr inbounds %0, ptr addrspace(1) %413, i64 %415
  %417 = load atomic i64, ptr addrspace(1) %416 syncscope("one-as") monotonic, align 8
  %418 = cmpxchg ptr addrspace(1) %128, i64 %412, i64 %417 syncscope("one-as") acquire monotonic, align 8
  %419 = extractvalue { i64, i1 } %418, 1
  %420 = extractvalue { i64, i1 } %418, 0
  br i1 %419, label %.loopexit4.i.i14, label %.preheader3.i.i19

.loopexit4.i.i14:                                 ; preds = %.preheader3.i.i19, %402, %.loopexit32
  %421 = phi i64 [ 0, %.loopexit32 ], [ %411, %402 ], [ %420, %.preheader3.i.i19 ]
  %422 = trunc i64 %421 to i32
  %423 = lshr i64 %421, 32
  %424 = trunc nuw i64 %423 to i32
  %425 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %422)
  %426 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %424)
  %427 = zext i32 %426 to i64
  %428 = shl nuw i64 %427, 32
  %429 = zext i32 %425 to i64
  %430 = or disjoint i64 %428, %429
  %431 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %432 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %433 = and i64 %430, %432
  %434 = getelementptr inbounds %0, ptr addrspace(1) %431, i64 %433
  %435 = load ptr addrspace(1), ptr addrspace(1) %130, align 8, !tbaa !36
  %436 = getelementptr inbounds %1, ptr addrspace(1) %435, i64 %433
  %437 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true)
  br i1 %401, label %438, label %441

438:                                              ; preds = %.loopexit4.i.i14
  %439 = getelementptr inbounds i8, ptr addrspace(1) %434, i64 8
  %440 = getelementptr inbounds i8, ptr addrspace(1) %434, i64 16
  store i64 %437, ptr addrspace(1) %439, align 8, !tbaa !37
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %440, align 8, !tbaa !40
  br label %441

441:                                              ; preds = %438, %.loopexit4.i.i14
  %442 = zext i32 %399 to i64
  %443 = getelementptr inbounds [64 x [8 x i64]], ptr addrspace(1) %436, i64 0, i64 %442
  store i64 %398, ptr addrspace(1) %443, align 8, !tbaa !26
  %444 = getelementptr inbounds i8, ptr addrspace(1) %443, i64 8
  store i64 %169, ptr addrspace(1) %444, align 8, !tbaa !26
  %445 = getelementptr inbounds i8, ptr addrspace(1) %443, i64 16
  store i64 %207, ptr addrspace(1) %445, align 8, !tbaa !26
  %446 = getelementptr inbounds i8, ptr addrspace(1) %443, i64 24
  store i64 %245, ptr addrspace(1) %446, align 8, !tbaa !26
  %447 = getelementptr inbounds i8, ptr addrspace(1) %443, i64 32
  store i64 %283, ptr addrspace(1) %447, align 8, !tbaa !26
  %448 = getelementptr inbounds i8, ptr addrspace(1) %443, i64 40
  store i64 %321, ptr addrspace(1) %448, align 8, !tbaa !26
  %449 = getelementptr inbounds i8, ptr addrspace(1) %443, i64 48
  store i64 %359, ptr addrspace(1) %449, align 8, !tbaa !26
  %450 = getelementptr inbounds i8, ptr addrspace(1) %443, i64 56
  store i64 %397, ptr addrspace(1) %450, align 8, !tbaa !26
  br i1 %401, label %451, label %__ockl_hsa_signal_add.exit.i.i15

451:                                              ; preds = %441
  %452 = load atomic i64, ptr addrspace(1) %131 syncscope("one-as") monotonic, align 8
  %453 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %454 = and i64 %453, %430
  %455 = getelementptr inbounds %0, ptr addrspace(1) %431, i64 %454
  store i64 %452, ptr addrspace(1) %455, align 8, !tbaa !41
  %456 = cmpxchg ptr addrspace(1) %131, i64 %452, i64 %430 syncscope("one-as") release monotonic, align 8
  %457 = extractvalue { i64, i1 } %456, 1
  br i1 %457, label %.loopexit2.i.i18, label %.preheader1.i.i17

.preheader1.i.i17:                                ; preds = %451, %.preheader1.i.i17
  %.pn = phi { i64, i1 } [ %459, %.preheader1.i.i17 ], [ %456, %451 ]
  %458 = extractvalue { i64, i1 } %.pn, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %458, ptr addrspace(1) %455, align 8, !tbaa !41
  %459 = cmpxchg ptr addrspace(1) %131, i64 %458, i64 %430 syncscope("one-as") release monotonic, align 8
  %460 = extractvalue { i64, i1 } %459, 1
  br i1 %460, label %.loopexit2.i.i18, label %.preheader1.i.i17

.loopexit2.i.i18:                                 ; preds = %.preheader1.i.i17, %451
  %461 = load i64, ptr addrspace(1) %132, align 8
  %462 = inttoptr i64 %461 to ptr addrspace(1)
  %463 = getelementptr inbounds i8, ptr addrspace(1) %462, i64 8
  %464 = atomicrmw add ptr addrspace(1) %463, i64 1 syncscope("one-as") release, align 8
  %465 = getelementptr inbounds i8, ptr addrspace(1) %462, i64 16
  %466 = load i64, ptr addrspace(1) %465, align 16, !tbaa !42
  %467 = icmp eq i64 %466, 0
  br i1 %467, label %__ockl_hsa_signal_add.exit.i.i15, label %468

468:                                              ; preds = %.loopexit2.i.i18
  %469 = inttoptr i64 %466 to ptr addrspace(1)
  %470 = getelementptr inbounds i8, ptr addrspace(1) %462, i64 24
  %471 = load i32, ptr addrspace(1) %470, align 8, !tbaa !44
  %472 = zext i32 %471 to i64
  store atomic i64 %472, ptr addrspace(1) %469 syncscope("one-as") release, align 8
  %473 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %471)
  %474 = and i32 %473, 255
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %474)
  br label %__ockl_hsa_signal_add.exit.i.i15

__ockl_hsa_signal_add.exit.i.i15:                 ; preds = %468, %.loopexit2.i.i18, %441
  %475 = getelementptr inbounds i8, ptr addrspace(1) %434, i64 20
  br label %476

476:                                              ; preds = %484, %__ockl_hsa_signal_add.exit.i.i15
  br i1 %401, label %477, label %480

477:                                              ; preds = %476
  %478 = load atomic i32, ptr addrspace(1) %475 syncscope("one-as") acquire, align 4
  %479 = and i32 %478, 1
  br label %480

480:                                              ; preds = %477, %476
  %481 = phi i32 [ %479, %477 ], [ 1, %476 ]
  %482 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %481)
  %483 = icmp eq i32 %482, 0
  br i1 %483, label %485, label %484

484:                                              ; preds = %480
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %476

485:                                              ; preds = %480
  %486 = load i64, ptr addrspace(1) %443, align 8, !tbaa !26
  br i1 %401, label %487, label %__ockl_hostcall_preview.exit20

487:                                              ; preds = %485
  %488 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %489 = add i64 %488, 1
  %490 = add i64 %489, %430
  %491 = icmp eq i64 %490, 0
  %492 = select i1 %491, i64 %489, i64 %490
  %493 = load atomic i64, ptr addrspace(1) %128 syncscope("one-as") monotonic, align 8
  %494 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %495 = and i64 %492, %488
  %496 = getelementptr inbounds %0, ptr addrspace(1) %494, i64 %495
  store i64 %493, ptr addrspace(1) %496, align 8, !tbaa !41
  %497 = cmpxchg ptr addrspace(1) %128, i64 %493, i64 %492 syncscope("one-as") release monotonic, align 8
  %498 = extractvalue { i64, i1 } %497, 1
  br i1 %498, label %__ockl_hostcall_preview.exit20, label %.preheader.i.i16

.preheader.i.i16:                                 ; preds = %487, %.preheader.i.i16
  %.pn2 = phi { i64, i1 } [ %500, %.preheader.i.i16 ], [ %497, %487 ]
  %499 = extractvalue { i64, i1 } %.pn2, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %499, ptr addrspace(1) %496, align 8, !tbaa !41
  %500 = cmpxchg ptr addrspace(1) %128, i64 %499, i64 %492 syncscope("one-as") release monotonic, align 8
  %501 = extractvalue { i64, i1 } %500, 1
  br i1 %501, label %__ockl_hostcall_preview.exit20, label %.preheader.i.i16

__ockl_hostcall_preview.exit20:                   ; preds = %.preheader.i.i16, %487, %485
  %502 = getelementptr inbounds i8, ptr %1, i64 56
  %503 = load i8, ptr %502, align 1, !tbaa !45
  %504 = zext i8 %503 to i64
  %505 = getelementptr inbounds i8, ptr %1, i64 57
  %506 = load i8, ptr %505, align 1, !tbaa !45
  %507 = zext i8 %506 to i64
  %508 = shl nuw nsw i64 %507, 8
  %509 = or disjoint i64 %508, %504
  %510 = getelementptr inbounds i8, ptr %1, i64 58
  %511 = load i8, ptr %510, align 1, !tbaa !45
  %512 = zext i8 %511 to i64
  %513 = shl nuw nsw i64 %512, 16
  %514 = or disjoint i64 %509, %513
  %515 = getelementptr inbounds i8, ptr %1, i64 59
  %516 = load i8, ptr %515, align 1, !tbaa !45
  %517 = zext i8 %516 to i64
  %518 = shl nuw nsw i64 %517, 24
  %519 = or disjoint i64 %514, %518
  %520 = getelementptr inbounds i8, ptr %1, i64 60
  %521 = load i8, ptr %520, align 1, !tbaa !45
  %522 = zext i8 %521 to i64
  %523 = shl nuw nsw i64 %522, 32
  %524 = or disjoint i64 %519, %523
  %525 = getelementptr inbounds i8, ptr %1, i64 61
  %526 = load i8, ptr %525, align 1, !tbaa !45
  %527 = zext i8 %526 to i64
  %528 = shl nuw nsw i64 %527, 40
  %529 = or i64 %524, %528
  %530 = getelementptr inbounds i8, ptr %1, i64 62
  %531 = load i8, ptr %530, align 1, !tbaa !45
  %532 = zext i8 %531 to i64
  %533 = shl nuw nsw i64 %532, 48
  %534 = or i64 %529, %533
  %535 = getelementptr inbounds i8, ptr %1, i64 63
  %536 = load i8, ptr %535, align 1, !tbaa !45
  %537 = zext i8 %536 to i64
  %538 = shl nuw i64 %537, 56
  %539 = or i64 %534, %538
  %540 = getelementptr inbounds i8, ptr %1, i64 64
  %541 = load i8, ptr %540, align 1, !tbaa !45
  %542 = zext i8 %541 to i64
  %543 = getelementptr inbounds i8, ptr %1, i64 65
  %544 = load i8, ptr %543, align 1, !tbaa !45
  %545 = zext i8 %544 to i64
  %546 = shl nuw nsw i64 %545, 8
  %547 = or disjoint i64 %546, %542
  %548 = getelementptr inbounds i8, ptr %1, i64 66
  %549 = load i8, ptr %548, align 1, !tbaa !45
  %550 = zext i8 %549 to i64
  %551 = shl nuw nsw i64 %550, 16
  %552 = or disjoint i64 %547, %551
  %553 = getelementptr inbounds i8, ptr %1, i64 67
  %554 = load i8, ptr %553, align 1, !tbaa !45
  %555 = zext i8 %554 to i64
  %556 = shl nuw nsw i64 %555, 24
  %557 = or disjoint i64 %552, %556
  %558 = getelementptr inbounds i8, ptr %1, i64 68
  %559 = load i8, ptr %558, align 1, !tbaa !45
  %560 = zext i8 %559 to i64
  %561 = shl nuw nsw i64 %560, 32
  %562 = or disjoint i64 %557, %561
  %563 = getelementptr inbounds i8, ptr %1, i64 69
  %564 = load i8, ptr %563, align 1, !tbaa !45
  %565 = zext i8 %564 to i64
  %566 = shl nuw nsw i64 %565, 40
  %567 = or i64 %562, %566
  %568 = getelementptr inbounds i8, ptr %1, i64 70
  %569 = load i8, ptr %568, align 1, !tbaa !45
  %570 = zext i8 %569 to i64
  %571 = shl nuw nsw i64 %570, 48
  %572 = or i64 %567, %571
  %573 = getelementptr inbounds i8, ptr %1, i64 71
  %574 = load i8, ptr %573, align 1, !tbaa !45
  %575 = zext i8 %574 to i64
  %576 = shl nuw i64 %575, 56
  %577 = or i64 %572, %576
  %578 = getelementptr inbounds i8, ptr %1, i64 72
  %579 = load i8, ptr %578, align 1, !tbaa !45
  %580 = zext i8 %579 to i64
  %581 = getelementptr inbounds i8, ptr %1, i64 73
  %582 = load i8, ptr %581, align 1, !tbaa !45
  %583 = zext i8 %582 to i64
  %584 = shl nuw nsw i64 %583, 8
  %585 = or disjoint i64 %584, %580
  %586 = getelementptr inbounds i8, ptr %1, i64 74
  %587 = load i8, ptr %586, align 1, !tbaa !45
  %588 = zext i8 %587 to i64
  %589 = shl nuw nsw i64 %588, 16
  %590 = or disjoint i64 %585, %589
  %591 = getelementptr inbounds i8, ptr %1, i64 75
  %592 = load i8, ptr %591, align 1, !tbaa !45
  %593 = zext i8 %592 to i64
  %594 = shl nuw nsw i64 %593, 24
  %595 = or disjoint i64 %590, %594
  %596 = getelementptr inbounds i8, ptr %1, i64 76
  %597 = load i8, ptr %596, align 1, !tbaa !45
  %598 = zext i8 %597 to i64
  %599 = shl nuw nsw i64 %598, 32
  %600 = or disjoint i64 %595, %599
  %601 = getelementptr inbounds i8, ptr %1, i64 77
  %602 = load i8, ptr %601, align 1, !tbaa !45
  %603 = zext i8 %602 to i64
  %604 = shl nuw nsw i64 %603, 40
  %605 = or i64 %600, %604
  %606 = getelementptr inbounds i8, ptr %1, i64 78
  %607 = load i8, ptr %606, align 1, !tbaa !45
  %608 = zext i8 %607 to i64
  %609 = shl nuw nsw i64 %608, 48
  %610 = or i64 %605, %609
  %611 = getelementptr inbounds i8, ptr %1, i64 79
  %612 = load i8, ptr %611, align 1, !tbaa !45
  %613 = zext i8 %612 to i64
  %614 = shl nuw i64 %613, 56
  %615 = or i64 %610, %614
  %616 = getelementptr inbounds i8, ptr %1, i64 80
  %617 = load i8, ptr %616, align 1, !tbaa !45
  %618 = zext i8 %617 to i64
  %619 = getelementptr inbounds i8, ptr %1, i64 81
  %620 = load i8, ptr %619, align 1, !tbaa !45
  %621 = zext i8 %620 to i64
  %622 = shl nuw nsw i64 %621, 8
  %623 = or disjoint i64 %622, %618
  %624 = getelementptr inbounds i8, ptr %1, i64 82
  %625 = load i8, ptr %624, align 1, !tbaa !45
  %626 = zext i8 %625 to i64
  %627 = shl nuw nsw i64 %626, 16
  %628 = or disjoint i64 %623, %627
  %629 = getelementptr inbounds i8, ptr %1, i64 83
  %630 = load i8, ptr %629, align 1, !tbaa !45
  %631 = zext i8 %630 to i64
  %632 = shl nuw nsw i64 %631, 24
  %633 = or disjoint i64 %628, %632
  %634 = getelementptr inbounds i8, ptr %1, i64 84
  %635 = load i8, ptr %634, align 1, !tbaa !45
  %636 = zext i8 %635 to i64
  %637 = shl nuw nsw i64 %636, 32
  %638 = or disjoint i64 %633, %637
  %639 = getelementptr inbounds i8, ptr %1, i64 85
  %640 = load i8, ptr %639, align 1, !tbaa !45
  %641 = zext i8 %640 to i64
  %642 = shl nuw nsw i64 %641, 40
  %643 = or i64 %638, %642
  %644 = getelementptr inbounds i8, ptr %1, i64 86
  %645 = load i8, ptr %644, align 1, !tbaa !45
  %646 = zext i8 %645 to i64
  %647 = shl nuw nsw i64 %646, 48
  %648 = or i64 %643, %647
  %649 = getelementptr inbounds i8, ptr %1, i64 87
  %650 = load i8, ptr %649, align 1, !tbaa !45
  %651 = zext i8 %650 to i64
  %652 = shl nuw i64 %651, 56
  %653 = or i64 %648, %652
  %654 = getelementptr inbounds i8, ptr %1, i64 88
  %655 = load i8, ptr %654, align 1, !tbaa !45
  %656 = zext i8 %655 to i64
  %657 = getelementptr inbounds i8, ptr %1, i64 89
  %658 = load i8, ptr %657, align 1, !tbaa !45
  %659 = zext i8 %658 to i64
  %660 = shl nuw nsw i64 %659, 8
  %661 = or disjoint i64 %660, %656
  %662 = getelementptr inbounds i8, ptr %1, i64 90
  %663 = load i8, ptr %662, align 1, !tbaa !45
  %664 = zext i8 %663 to i64
  %665 = shl nuw nsw i64 %664, 16
  %666 = or disjoint i64 %661, %665
  %667 = getelementptr inbounds i8, ptr %1, i64 91
  %668 = load i8, ptr %667, align 1, !tbaa !45
  %669 = zext i8 %668 to i64
  %670 = shl nuw nsw i64 %669, 24
  %671 = or disjoint i64 %666, %670
  %672 = getelementptr inbounds i8, ptr %1, i64 92
  %673 = load i8, ptr %672, align 1, !tbaa !45
  %674 = zext i8 %673 to i64
  %675 = shl nuw nsw i64 %674, 32
  %676 = or disjoint i64 %671, %675
  %677 = getelementptr inbounds i8, ptr %1, i64 93
  %678 = load i8, ptr %677, align 1, !tbaa !45
  %679 = zext i8 %678 to i64
  %680 = shl nuw nsw i64 %679, 40
  %681 = or i64 %676, %680
  %682 = getelementptr inbounds i8, ptr %1, i64 94
  %683 = load i8, ptr %682, align 1, !tbaa !45
  %684 = zext i8 %683 to i64
  %685 = shl nuw nsw i64 %684, 48
  %686 = or i64 %681, %685
  %687 = getelementptr inbounds i8, ptr %1, i64 95
  %688 = load i8, ptr %687, align 1, !tbaa !45
  %689 = zext i8 %688 to i64
  %690 = shl nuw i64 %689, 56
  %691 = or i64 %686, %690
  %692 = getelementptr inbounds i8, ptr %1, i64 96
  %693 = load i8, ptr %692, align 1, !tbaa !45
  %694 = zext i8 %693 to i64
  %695 = getelementptr inbounds i8, ptr %1, i64 97
  %696 = load i8, ptr %695, align 1, !tbaa !45
  %697 = zext i8 %696 to i64
  %698 = shl nuw nsw i64 %697, 8
  %699 = or disjoint i64 %698, %694
  %700 = getelementptr inbounds i8, ptr %1, i64 98
  %701 = load i8, ptr %700, align 1, !tbaa !45
  %702 = zext i8 %701 to i64
  %703 = shl nuw nsw i64 %702, 16
  %704 = or disjoint i64 %699, %703
  %705 = getelementptr inbounds i8, ptr %1, i64 99
  %706 = load i8, ptr %705, align 1, !tbaa !45
  %707 = zext i8 %706 to i64
  %708 = shl nuw nsw i64 %707, 24
  %709 = or disjoint i64 %704, %708
  %710 = getelementptr inbounds i8, ptr %1, i64 100
  %711 = load i8, ptr %710, align 1, !tbaa !45
  %712 = zext i8 %711 to i64
  %713 = shl nuw nsw i64 %712, 32
  %714 = or disjoint i64 %709, %713
  %715 = getelementptr inbounds i8, ptr %1, i64 101
  %716 = load i8, ptr %715, align 1, !tbaa !45
  %717 = zext i8 %716 to i64
  %718 = shl nuw nsw i64 %717, 40
  %719 = or i64 %714, %718
  %720 = getelementptr inbounds i8, ptr %1, i64 102
  %721 = load i8, ptr %720, align 1, !tbaa !45
  %722 = zext i8 %721 to i64
  %723 = shl nuw nsw i64 %722, 48
  %724 = or i64 %719, %723
  %725 = getelementptr inbounds i8, ptr %1, i64 103
  %726 = load i8, ptr %725, align 1, !tbaa !45
  %727 = zext i8 %726 to i64
  %728 = shl nuw i64 %727, 56
  %729 = or i64 %724, %728
  %730 = getelementptr inbounds i8, ptr %1, i64 104
  %731 = load i8, ptr %730, align 1, !tbaa !45
  %732 = zext i8 %731 to i64
  %733 = getelementptr inbounds i8, ptr %1, i64 105
  %734 = load i8, ptr %733, align 1, !tbaa !45
  %735 = zext i8 %734 to i64
  %736 = shl nuw nsw i64 %735, 8
  %737 = or disjoint i64 %736, %732
  %738 = getelementptr inbounds i8, ptr %1, i64 106
  %739 = load i8, ptr %738, align 1, !tbaa !45
  %740 = zext i8 %739 to i64
  %741 = shl nuw nsw i64 %740, 16
  %742 = or disjoint i64 %737, %741
  %743 = getelementptr inbounds i8, ptr %1, i64 107
  %744 = load i8, ptr %743, align 1, !tbaa !45
  %745 = zext i8 %744 to i64
  %746 = shl nuw nsw i64 %745, 24
  %747 = or disjoint i64 %742, %746
  %748 = getelementptr inbounds i8, ptr %1, i64 108
  %749 = load i8, ptr %748, align 1, !tbaa !45
  %750 = zext i8 %749 to i64
  %751 = shl nuw nsw i64 %750, 32
  %752 = or disjoint i64 %747, %751
  %753 = getelementptr inbounds i8, ptr %1, i64 109
  %754 = load i8, ptr %753, align 1, !tbaa !45
  %755 = zext i8 %754 to i64
  %756 = shl nuw nsw i64 %755, 40
  %757 = or i64 %752, %756
  %758 = getelementptr inbounds i8, ptr %1, i64 110
  %759 = load i8, ptr %758, align 1, !tbaa !45
  %760 = zext i8 %759 to i64
  %761 = shl nuw nsw i64 %760, 48
  %762 = or i64 %757, %761
  %763 = getelementptr inbounds i8, ptr %1, i64 111
  %764 = load i8, ptr %763, align 1, !tbaa !45
  %765 = zext i8 %764 to i64
  %766 = shl nuw i64 %765, 56
  %767 = or i64 %762, %766
  %768 = or i64 %486, 224
  %769 = tail call i32 asm sideeffect "", "=v,0"(i32 %127) #10, !srcloc !30
  %770 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %769)
  %771 = icmp eq i32 %769, %770
  br i1 %771, label %772, label %.loopexit4.i.i14.1

772:                                              ; preds = %__ockl_hostcall_preview.exit20
  %773 = load atomic i64, ptr addrspace(1) %128 syncscope("one-as") acquire, align 8
  %774 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %775 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %776 = and i64 %775, %773
  %777 = getelementptr inbounds %0, ptr addrspace(1) %774, i64 %776
  %778 = load atomic i64, ptr addrspace(1) %777 syncscope("one-as") monotonic, align 8
  %779 = cmpxchg ptr addrspace(1) %128, i64 %773, i64 %778 syncscope("one-as") acquire monotonic, align 8
  %780 = extractvalue { i64, i1 } %779, 1
  %781 = extractvalue { i64, i1 } %779, 0
  br i1 %780, label %.loopexit4.i.i14.1, label %.preheader3.i.i19.1

.preheader3.i.i19.1:                              ; preds = %772, %.preheader3.i.i19.1
  %782 = phi i64 [ %790, %.preheader3.i.i19.1 ], [ %781, %772 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %783 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %784 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %785 = and i64 %784, %782
  %786 = getelementptr inbounds %0, ptr addrspace(1) %783, i64 %785
  %787 = load atomic i64, ptr addrspace(1) %786 syncscope("one-as") monotonic, align 8
  %788 = cmpxchg ptr addrspace(1) %128, i64 %782, i64 %787 syncscope("one-as") acquire monotonic, align 8
  %789 = extractvalue { i64, i1 } %788, 1
  %790 = extractvalue { i64, i1 } %788, 0
  br i1 %789, label %.loopexit4.i.i14.1, label %.preheader3.i.i19.1

.loopexit4.i.i14.1:                               ; preds = %.preheader3.i.i19.1, %772, %__ockl_hostcall_preview.exit20
  %791 = phi i64 [ 0, %__ockl_hostcall_preview.exit20 ], [ %781, %772 ], [ %790, %.preheader3.i.i19.1 ]
  %792 = trunc i64 %791 to i32
  %793 = lshr i64 %791, 32
  %794 = trunc nuw i64 %793 to i32
  %795 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %792)
  %796 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %794)
  %797 = zext i32 %796 to i64
  %798 = shl nuw i64 %797, 32
  %799 = zext i32 %795 to i64
  %800 = or disjoint i64 %798, %799
  %801 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %802 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %803 = and i64 %800, %802
  %804 = getelementptr inbounds %0, ptr addrspace(1) %801, i64 %803
  %805 = load ptr addrspace(1), ptr addrspace(1) %130, align 8, !tbaa !36
  %806 = getelementptr inbounds %1, ptr addrspace(1) %805, i64 %803
  %807 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true)
  br i1 %771, label %808, label %811

808:                                              ; preds = %.loopexit4.i.i14.1
  %809 = getelementptr inbounds i8, ptr addrspace(1) %804, i64 8
  %810 = getelementptr inbounds i8, ptr addrspace(1) %804, i64 16
  store i64 %807, ptr addrspace(1) %809, align 8, !tbaa !37
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %810, align 8, !tbaa !40
  br label %811

811:                                              ; preds = %808, %.loopexit4.i.i14.1
  %812 = zext i32 %769 to i64
  %813 = getelementptr inbounds [64 x [8 x i64]], ptr addrspace(1) %806, i64 0, i64 %812
  store i64 %768, ptr addrspace(1) %813, align 8, !tbaa !26
  %814 = getelementptr inbounds i8, ptr addrspace(1) %813, i64 8
  store i64 %539, ptr addrspace(1) %814, align 8, !tbaa !26
  %815 = getelementptr inbounds i8, ptr addrspace(1) %813, i64 16
  store i64 %577, ptr addrspace(1) %815, align 8, !tbaa !26
  %816 = getelementptr inbounds i8, ptr addrspace(1) %813, i64 24
  store i64 %615, ptr addrspace(1) %816, align 8, !tbaa !26
  %817 = getelementptr inbounds i8, ptr addrspace(1) %813, i64 32
  store i64 %653, ptr addrspace(1) %817, align 8, !tbaa !26
  %818 = getelementptr inbounds i8, ptr addrspace(1) %813, i64 40
  store i64 %691, ptr addrspace(1) %818, align 8, !tbaa !26
  %819 = getelementptr inbounds i8, ptr addrspace(1) %813, i64 48
  store i64 %729, ptr addrspace(1) %819, align 8, !tbaa !26
  %820 = getelementptr inbounds i8, ptr addrspace(1) %813, i64 56
  store i64 %767, ptr addrspace(1) %820, align 8, !tbaa !26
  br i1 %771, label %821, label %__ockl_hsa_signal_add.exit.i.i15.1

821:                                              ; preds = %811
  %822 = load atomic i64, ptr addrspace(1) %131 syncscope("one-as") monotonic, align 8
  %823 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %824 = and i64 %823, %800
  %825 = getelementptr inbounds %0, ptr addrspace(1) %801, i64 %824
  store i64 %822, ptr addrspace(1) %825, align 8, !tbaa !41
  %826 = cmpxchg ptr addrspace(1) %131, i64 %822, i64 %800 syncscope("one-as") release monotonic, align 8
  %827 = extractvalue { i64, i1 } %826, 1
  br i1 %827, label %.loopexit2.i.i18.1, label %.preheader1.i.i17.1

.preheader1.i.i17.1:                              ; preds = %821, %.preheader1.i.i17.1
  %.pn4 = phi { i64, i1 } [ %829, %.preheader1.i.i17.1 ], [ %826, %821 ]
  %828 = extractvalue { i64, i1 } %.pn4, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %828, ptr addrspace(1) %825, align 8, !tbaa !41
  %829 = cmpxchg ptr addrspace(1) %131, i64 %828, i64 %800 syncscope("one-as") release monotonic, align 8
  %830 = extractvalue { i64, i1 } %829, 1
  br i1 %830, label %.loopexit2.i.i18.1, label %.preheader1.i.i17.1

.loopexit2.i.i18.1:                               ; preds = %.preheader1.i.i17.1, %821
  %831 = load i64, ptr addrspace(1) %132, align 8
  %832 = inttoptr i64 %831 to ptr addrspace(1)
  %833 = getelementptr inbounds i8, ptr addrspace(1) %832, i64 8
  %834 = atomicrmw add ptr addrspace(1) %833, i64 1 syncscope("one-as") release, align 8
  %835 = getelementptr inbounds i8, ptr addrspace(1) %832, i64 16
  %836 = load i64, ptr addrspace(1) %835, align 16, !tbaa !42
  %837 = icmp eq i64 %836, 0
  br i1 %837, label %__ockl_hsa_signal_add.exit.i.i15.1, label %838

838:                                              ; preds = %.loopexit2.i.i18.1
  %839 = inttoptr i64 %836 to ptr addrspace(1)
  %840 = getelementptr inbounds i8, ptr addrspace(1) %832, i64 24
  %841 = load i32, ptr addrspace(1) %840, align 8, !tbaa !44
  %842 = zext i32 %841 to i64
  store atomic i64 %842, ptr addrspace(1) %839 syncscope("one-as") release, align 8
  %843 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %841)
  %844 = and i32 %843, 255
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %844)
  br label %__ockl_hsa_signal_add.exit.i.i15.1

__ockl_hsa_signal_add.exit.i.i15.1:               ; preds = %838, %.loopexit2.i.i18.1, %811
  %845 = getelementptr inbounds i8, ptr addrspace(1) %804, i64 20
  br label %846

846:                                              ; preds = %854, %__ockl_hsa_signal_add.exit.i.i15.1
  br i1 %771, label %847, label %850

847:                                              ; preds = %846
  %848 = load atomic i32, ptr addrspace(1) %845 syncscope("one-as") acquire, align 4
  %849 = and i32 %848, 1
  br label %850

850:                                              ; preds = %847, %846
  %851 = phi i32 [ %849, %847 ], [ 1, %846 ]
  %852 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %851)
  %853 = icmp eq i32 %852, 0
  br i1 %853, label %855, label %854

854:                                              ; preds = %850
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %846

855:                                              ; preds = %850
  %856 = load i64, ptr addrspace(1) %813, align 8, !tbaa !26
  br i1 %771, label %857, label %__ockl_hostcall_preview.exit20.1

857:                                              ; preds = %855
  %858 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %859 = add i64 %858, 1
  %860 = add i64 %859, %800
  %861 = icmp eq i64 %860, 0
  %862 = select i1 %861, i64 %859, i64 %860
  %863 = load atomic i64, ptr addrspace(1) %128 syncscope("one-as") monotonic, align 8
  %864 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %865 = and i64 %862, %858
  %866 = getelementptr inbounds %0, ptr addrspace(1) %864, i64 %865
  store i64 %863, ptr addrspace(1) %866, align 8, !tbaa !41
  %867 = cmpxchg ptr addrspace(1) %128, i64 %863, i64 %862 syncscope("one-as") release monotonic, align 8
  %868 = extractvalue { i64, i1 } %867, 1
  br i1 %868, label %__ockl_hostcall_preview.exit20.1, label %.preheader.i.i16.1

.preheader.i.i16.1:                               ; preds = %857, %.preheader.i.i16.1
  %.pn6 = phi { i64, i1 } [ %870, %.preheader.i.i16.1 ], [ %867, %857 ]
  %869 = extractvalue { i64, i1 } %.pn6, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %869, ptr addrspace(1) %866, align 8, !tbaa !41
  %870 = cmpxchg ptr addrspace(1) %128, i64 %869, i64 %862 syncscope("one-as") release monotonic, align 8
  %871 = extractvalue { i64, i1 } %870, 1
  br i1 %871, label %__ockl_hostcall_preview.exit20.1, label %.preheader.i.i16.1

__ockl_hostcall_preview.exit20.1:                 ; preds = %.preheader.i.i16.1, %855, %857
  %872 = getelementptr inbounds i8, ptr %1, i64 112
  %873 = load i8, ptr %872, align 1, !tbaa !45
  %874 = zext i8 %873 to i64
  %875 = getelementptr inbounds i8, ptr %1, i64 113
  %876 = load i8, ptr %875, align 1, !tbaa !45
  %877 = zext i8 %876 to i64
  %878 = shl nuw nsw i64 %877, 8
  %879 = or disjoint i64 %878, %874
  %880 = getelementptr inbounds i8, ptr %1, i64 114
  %881 = load i8, ptr %880, align 1, !tbaa !45
  %882 = zext i8 %881 to i64
  %883 = shl nuw nsw i64 %882, 16
  %884 = or disjoint i64 %879, %883
  %885 = getelementptr inbounds i8, ptr %1, i64 115
  %886 = load i8, ptr %885, align 1, !tbaa !45
  %887 = zext i8 %886 to i64
  %888 = shl nuw nsw i64 %887, 24
  %889 = or disjoint i64 %884, %888
  %890 = getelementptr inbounds i8, ptr %1, i64 116
  %891 = load i8, ptr %890, align 1, !tbaa !45
  %892 = zext i8 %891 to i64
  %893 = shl nuw nsw i64 %892, 32
  %894 = or disjoint i64 %889, %893
  %895 = getelementptr inbounds i8, ptr %1, i64 117
  %896 = load i8, ptr %895, align 1, !tbaa !45
  %897 = zext i8 %896 to i64
  %898 = shl nuw nsw i64 %897, 40
  %899 = or i64 %894, %898
  %900 = getelementptr inbounds i8, ptr %1, i64 118
  %901 = load i8, ptr %900, align 1, !tbaa !45
  %902 = zext i8 %901 to i64
  %903 = shl nuw nsw i64 %902, 48
  %904 = or i64 %899, %903
  %905 = getelementptr inbounds i8, ptr %1, i64 119
  %906 = load i8, ptr %905, align 1, !tbaa !45
  %907 = zext i8 %906 to i64
  %908 = shl nuw i64 %907, 56
  %909 = or i64 %904, %908
  %910 = getelementptr inbounds i8, ptr %1, i64 120
  %911 = load i8, ptr %910, align 1, !tbaa !45
  %912 = zext i8 %911 to i64
  %913 = getelementptr inbounds i8, ptr %1, i64 121
  %914 = load i8, ptr %913, align 1, !tbaa !45
  %915 = zext i8 %914 to i64
  %916 = shl nuw nsw i64 %915, 8
  %917 = or disjoint i64 %916, %912
  %918 = getelementptr inbounds i8, ptr %1, i64 122
  %919 = load i8, ptr %918, align 1, !tbaa !45
  %920 = zext i8 %919 to i64
  %921 = shl nuw nsw i64 %920, 16
  %922 = or disjoint i64 %917, %921
  %923 = getelementptr inbounds i8, ptr %1, i64 123
  %924 = load i8, ptr %923, align 1, !tbaa !45
  %925 = zext i8 %924 to i64
  %926 = shl nuw nsw i64 %925, 24
  %927 = or disjoint i64 %922, %926
  %928 = getelementptr inbounds i8, ptr %1, i64 124
  %929 = load i8, ptr %928, align 1, !tbaa !45
  %930 = zext i8 %929 to i64
  %931 = shl nuw nsw i64 %930, 32
  %932 = or disjoint i64 %927, %931
  %933 = getelementptr inbounds i8, ptr %1, i64 125
  %934 = load i8, ptr %933, align 1, !tbaa !45
  %935 = zext i8 %934 to i64
  %936 = shl nuw nsw i64 %935, 40
  %937 = or i64 %932, %936
  %938 = getelementptr inbounds i8, ptr %1, i64 126
  %939 = load i8, ptr %938, align 1, !tbaa !45
  %940 = zext i8 %939 to i64
  %941 = shl nuw nsw i64 %940, 48
  %942 = or i64 %937, %941
  %943 = getelementptr inbounds i8, ptr %1, i64 127
  %944 = load i8, ptr %943, align 1, !tbaa !45
  %945 = zext i8 %944 to i64
  %946 = shl nuw i64 %945, 56
  %947 = or i64 %942, %946
  %948 = getelementptr inbounds i8, ptr %1, i64 128
  %949 = load i8, ptr %948, align 1, !tbaa !45
  %950 = zext i8 %949 to i64
  %951 = getelementptr inbounds i8, ptr %1, i64 129
  %952 = load i8, ptr %951, align 1, !tbaa !45
  %953 = zext i8 %952 to i64
  %954 = shl nuw nsw i64 %953, 8
  %955 = or disjoint i64 %954, %950
  %956 = getelementptr inbounds i8, ptr %1, i64 130
  %957 = load i8, ptr %956, align 1, !tbaa !45
  %958 = zext i8 %957 to i64
  %959 = shl nuw nsw i64 %958, 16
  %960 = or disjoint i64 %955, %959
  %961 = getelementptr inbounds i8, ptr %1, i64 131
  %962 = load i8, ptr %961, align 1, !tbaa !45
  %963 = zext i8 %962 to i64
  %964 = shl nuw nsw i64 %963, 24
  %965 = or disjoint i64 %960, %964
  %966 = getelementptr inbounds i8, ptr %1, i64 132
  %967 = load i8, ptr %966, align 1, !tbaa !45
  %968 = zext i8 %967 to i64
  %969 = shl nuw nsw i64 %968, 32
  %970 = or disjoint i64 %965, %969
  %971 = getelementptr inbounds i8, ptr %1, i64 133
  %972 = load i8, ptr %971, align 1, !tbaa !45
  %973 = zext i8 %972 to i64
  %974 = shl nuw nsw i64 %973, 40
  %975 = or i64 %970, %974
  %976 = getelementptr inbounds i8, ptr %1, i64 134
  %977 = load i8, ptr %976, align 1, !tbaa !45
  %978 = zext i8 %977 to i64
  %979 = shl nuw nsw i64 %978, 48
  %980 = or i64 %975, %979
  %981 = getelementptr inbounds i8, ptr %1, i64 135
  %982 = load i8, ptr %981, align 1, !tbaa !45
  %983 = zext i8 %982 to i64
  %984 = shl nuw i64 %983, 56
  %985 = or i64 %980, %984
  %986 = getelementptr inbounds i8, ptr %1, i64 136
  %987 = load i8, ptr %986, align 1, !tbaa !45
  %988 = zext i8 %987 to i64
  %989 = getelementptr inbounds i8, ptr %1, i64 137
  %990 = load i8, ptr %989, align 1, !tbaa !45
  %991 = zext i8 %990 to i64
  %992 = shl nuw nsw i64 %991, 8
  %993 = or disjoint i64 %992, %988
  %994 = getelementptr inbounds i8, ptr %1, i64 138
  %995 = load i8, ptr %994, align 1, !tbaa !45
  %996 = zext i8 %995 to i64
  %997 = shl nuw nsw i64 %996, 16
  %998 = or disjoint i64 %993, %997
  %999 = getelementptr inbounds i8, ptr %1, i64 139
  %1000 = load i8, ptr %999, align 1, !tbaa !45
  %1001 = zext i8 %1000 to i64
  %1002 = shl nuw nsw i64 %1001, 24
  %1003 = or disjoint i64 %998, %1002
  %1004 = getelementptr inbounds i8, ptr %1, i64 140
  %1005 = load i8, ptr %1004, align 1, !tbaa !45
  %1006 = zext i8 %1005 to i64
  %1007 = shl nuw nsw i64 %1006, 32
  %1008 = or disjoint i64 %1003, %1007
  %1009 = getelementptr inbounds i8, ptr %1, i64 141
  %1010 = load i8, ptr %1009, align 1, !tbaa !45
  %1011 = zext i8 %1010 to i64
  %1012 = shl nuw nsw i64 %1011, 40
  %1013 = or i64 %1008, %1012
  %1014 = getelementptr inbounds i8, ptr %1, i64 142
  %1015 = load i8, ptr %1014, align 1, !tbaa !45
  %1016 = zext i8 %1015 to i64
  %1017 = shl nuw nsw i64 %1016, 48
  %1018 = or i64 %1013, %1017
  %1019 = getelementptr inbounds i8, ptr %1, i64 143
  %1020 = load i8, ptr %1019, align 1, !tbaa !45
  %1021 = zext i8 %1020 to i64
  %1022 = shl nuw i64 %1021, 56
  %1023 = or i64 %1018, %1022
  %1024 = getelementptr inbounds i8, ptr %1, i64 144
  %1025 = load i8, ptr %1024, align 1, !tbaa !45
  %1026 = zext i8 %1025 to i64
  %1027 = getelementptr inbounds i8, ptr %1, i64 145
  %1028 = load i8, ptr %1027, align 1, !tbaa !45
  %1029 = zext i8 %1028 to i64
  %1030 = shl nuw nsw i64 %1029, 8
  %1031 = or disjoint i64 %1030, %1026
  %1032 = getelementptr inbounds i8, ptr %1, i64 146
  %1033 = load i8, ptr %1032, align 1, !tbaa !45
  %1034 = zext i8 %1033 to i64
  %1035 = shl nuw nsw i64 %1034, 16
  %1036 = or disjoint i64 %1035, %1031
  %1037 = getelementptr inbounds i8, ptr %1, i64 147
  %1038 = load i8, ptr %1037, align 1, !tbaa !45
  %1039 = zext i8 %1038 to i64
  %1040 = shl nuw nsw i64 %1039, 24
  %1041 = or disjoint i64 %1040, %1036
  %1042 = getelementptr inbounds i8, ptr %1, i64 148
  %1043 = load i8, ptr %1042, align 1, !tbaa !45
  %1044 = zext i8 %1043 to i64
  %1045 = shl nuw nsw i64 %1044, 32
  %1046 = or disjoint i64 %1045, %1041
  %1047 = getelementptr inbounds i8, ptr %1, i64 149
  %1048 = load i8, ptr %1047, align 1, !tbaa !45
  %1049 = zext i8 %1048 to i64
  %1050 = shl nuw nsw i64 %1049, 40
  %1051 = or i64 %1050, %1046
  %1052 = and i64 %856, -227
  %1053 = or disjoint i64 %1052, 162
  %1054 = tail call i32 asm sideeffect "", "=v,0"(i32 %127) #10, !srcloc !30
  %1055 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1054)
  %1056 = icmp eq i32 %1054, %1055
  br i1 %1056, label %1057, label %.loopexit4.i.i14.2

1057:                                             ; preds = %__ockl_hostcall_preview.exit20.1
  %1058 = load atomic i64, ptr addrspace(1) %128 syncscope("one-as") acquire, align 8
  %1059 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %1060 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %1061 = and i64 %1060, %1058
  %1062 = getelementptr inbounds %0, ptr addrspace(1) %1059, i64 %1061
  %1063 = load atomic i64, ptr addrspace(1) %1062 syncscope("one-as") monotonic, align 8
  %1064 = cmpxchg ptr addrspace(1) %128, i64 %1058, i64 %1063 syncscope("one-as") acquire monotonic, align 8
  %1065 = extractvalue { i64, i1 } %1064, 1
  %1066 = extractvalue { i64, i1 } %1064, 0
  br i1 %1065, label %.loopexit4.i.i14.2, label %.preheader3.i.i19.2

.preheader3.i.i19.2:                              ; preds = %1057, %.preheader3.i.i19.2
  %1067 = phi i64 [ %1075, %.preheader3.i.i19.2 ], [ %1066, %1057 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %1068 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %1069 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %1070 = and i64 %1069, %1067
  %1071 = getelementptr inbounds %0, ptr addrspace(1) %1068, i64 %1070
  %1072 = load atomic i64, ptr addrspace(1) %1071 syncscope("one-as") monotonic, align 8
  %1073 = cmpxchg ptr addrspace(1) %128, i64 %1067, i64 %1072 syncscope("one-as") acquire monotonic, align 8
  %1074 = extractvalue { i64, i1 } %1073, 1
  %1075 = extractvalue { i64, i1 } %1073, 0
  br i1 %1074, label %.loopexit4.i.i14.2, label %.preheader3.i.i19.2

.loopexit4.i.i14.2:                               ; preds = %.preheader3.i.i19.2, %1057, %__ockl_hostcall_preview.exit20.1
  %1076 = phi i64 [ 0, %__ockl_hostcall_preview.exit20.1 ], [ %1066, %1057 ], [ %1075, %.preheader3.i.i19.2 ]
  %1077 = trunc i64 %1076 to i32
  %1078 = lshr i64 %1076, 32
  %1079 = trunc nuw i64 %1078 to i32
  %1080 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1077)
  %1081 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1079)
  %1082 = zext i32 %1081 to i64
  %1083 = shl nuw i64 %1082, 32
  %1084 = zext i32 %1080 to i64
  %1085 = or disjoint i64 %1083, %1084
  %1086 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %1087 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %1088 = and i64 %1085, %1087
  %1089 = getelementptr inbounds %0, ptr addrspace(1) %1086, i64 %1088
  %1090 = load ptr addrspace(1), ptr addrspace(1) %130, align 8, !tbaa !36
  %1091 = getelementptr inbounds %1, ptr addrspace(1) %1090, i64 %1088
  %1092 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true)
  br i1 %1056, label %1093, label %1096

1093:                                             ; preds = %.loopexit4.i.i14.2
  %1094 = getelementptr inbounds i8, ptr addrspace(1) %1089, i64 8
  %1095 = getelementptr inbounds i8, ptr addrspace(1) %1089, i64 16
  store i64 %1092, ptr addrspace(1) %1094, align 8, !tbaa !37
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %1095, align 8, !tbaa !40
  br label %1096

1096:                                             ; preds = %1093, %.loopexit4.i.i14.2
  %1097 = zext i32 %1054 to i64
  %1098 = getelementptr inbounds [64 x [8 x i64]], ptr addrspace(1) %1091, i64 0, i64 %1097
  store i64 %1053, ptr addrspace(1) %1098, align 8, !tbaa !26
  %1099 = getelementptr inbounds i8, ptr addrspace(1) %1098, i64 8
  store i64 %909, ptr addrspace(1) %1099, align 8, !tbaa !26
  %1100 = getelementptr inbounds i8, ptr addrspace(1) %1098, i64 16
  store i64 %947, ptr addrspace(1) %1100, align 8, !tbaa !26
  %1101 = getelementptr inbounds i8, ptr addrspace(1) %1098, i64 24
  store i64 %985, ptr addrspace(1) %1101, align 8, !tbaa !26
  %1102 = getelementptr inbounds i8, ptr addrspace(1) %1098, i64 32
  store i64 %1023, ptr addrspace(1) %1102, align 8, !tbaa !26
  %1103 = getelementptr inbounds i8, ptr addrspace(1) %1098, i64 40
  store i64 %1051, ptr addrspace(1) %1103, align 8, !tbaa !26
  %1104 = getelementptr inbounds i8, ptr addrspace(1) %1098, i64 48
  store i64 0, ptr addrspace(1) %1104, align 8, !tbaa !26
  %1105 = getelementptr inbounds i8, ptr addrspace(1) %1098, i64 56
  store i64 0, ptr addrspace(1) %1105, align 8, !tbaa !26
  br i1 %1056, label %1106, label %__ockl_hsa_signal_add.exit.i.i15.2

1106:                                             ; preds = %1096
  %1107 = load atomic i64, ptr addrspace(1) %131 syncscope("one-as") monotonic, align 8
  %1108 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %1109 = and i64 %1108, %1085
  %1110 = getelementptr inbounds %0, ptr addrspace(1) %1086, i64 %1109
  store i64 %1107, ptr addrspace(1) %1110, align 8, !tbaa !41
  %1111 = cmpxchg ptr addrspace(1) %131, i64 %1107, i64 %1085 syncscope("one-as") release monotonic, align 8
  %1112 = extractvalue { i64, i1 } %1111, 1
  br i1 %1112, label %.loopexit2.i.i18.2, label %.preheader1.i.i17.2

.preheader1.i.i17.2:                              ; preds = %1106, %.preheader1.i.i17.2
  %.pn8 = phi { i64, i1 } [ %1114, %.preheader1.i.i17.2 ], [ %1111, %1106 ]
  %1113 = extractvalue { i64, i1 } %.pn8, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %1113, ptr addrspace(1) %1110, align 8, !tbaa !41
  %1114 = cmpxchg ptr addrspace(1) %131, i64 %1113, i64 %1085 syncscope("one-as") release monotonic, align 8
  %1115 = extractvalue { i64, i1 } %1114, 1
  br i1 %1115, label %.loopexit2.i.i18.2, label %.preheader1.i.i17.2

.loopexit2.i.i18.2:                               ; preds = %.preheader1.i.i17.2, %1106
  %1116 = load i64, ptr addrspace(1) %132, align 8
  %1117 = inttoptr i64 %1116 to ptr addrspace(1)
  %1118 = getelementptr inbounds i8, ptr addrspace(1) %1117, i64 8
  %1119 = atomicrmw add ptr addrspace(1) %1118, i64 1 syncscope("one-as") release, align 8
  %1120 = getelementptr inbounds i8, ptr addrspace(1) %1117, i64 16
  %1121 = load i64, ptr addrspace(1) %1120, align 16, !tbaa !42
  %1122 = icmp eq i64 %1121, 0
  br i1 %1122, label %__ockl_hsa_signal_add.exit.i.i15.2, label %1123

1123:                                             ; preds = %.loopexit2.i.i18.2
  %1124 = inttoptr i64 %1121 to ptr addrspace(1)
  %1125 = getelementptr inbounds i8, ptr addrspace(1) %1117, i64 24
  %1126 = load i32, ptr addrspace(1) %1125, align 8, !tbaa !44
  %1127 = zext i32 %1126 to i64
  store atomic i64 %1127, ptr addrspace(1) %1124 syncscope("one-as") release, align 8
  %1128 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1126)
  %1129 = and i32 %1128, 255
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %1129)
  br label %__ockl_hsa_signal_add.exit.i.i15.2

__ockl_hsa_signal_add.exit.i.i15.2:               ; preds = %1123, %.loopexit2.i.i18.2, %1096
  %1130 = getelementptr inbounds i8, ptr addrspace(1) %1089, i64 20
  br label %1131

1131:                                             ; preds = %1139, %__ockl_hsa_signal_add.exit.i.i15.2
  br i1 %1056, label %1132, label %1135

1132:                                             ; preds = %1131
  %1133 = load atomic i32, ptr addrspace(1) %1130 syncscope("one-as") acquire, align 4
  %1134 = and i32 %1133, 1
  br label %1135

1135:                                             ; preds = %1132, %1131
  %1136 = phi i32 [ %1134, %1132 ], [ 1, %1131 ]
  %1137 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1136)
  %1138 = icmp eq i32 %1137, 0
  br i1 %1138, label %1140, label %1139

1139:                                             ; preds = %1135
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %1131

1140:                                             ; preds = %1135
  br i1 %1056, label %1141, label %.loopexit33

1141:                                             ; preds = %1140
  %1142 = load i64, ptr addrspace(1) %129, align 8, !tbaa !35
  %1143 = add i64 %1142, 1
  %1144 = add i64 %1143, %1085
  %1145 = icmp eq i64 %1144, 0
  %1146 = select i1 %1145, i64 %1143, i64 %1144
  %1147 = load atomic i64, ptr addrspace(1) %128 syncscope("one-as") monotonic, align 8
  %1148 = load ptr addrspace(1), ptr addrspace(1) %125, align 8, !tbaa !31
  %1149 = and i64 %1146, %1142
  %1150 = getelementptr inbounds %0, ptr addrspace(1) %1148, i64 %1149
  store i64 %1147, ptr addrspace(1) %1150, align 8, !tbaa !41
  %1151 = cmpxchg ptr addrspace(1) %128, i64 %1147, i64 %1146 syncscope("one-as") release monotonic, align 8
  %1152 = extractvalue { i64, i1 } %1151, 1
  br i1 %1152, label %.loopexit33, label %.preheader.i.i16.2

.preheader.i.i16.2:                               ; preds = %1141, %.preheader.i.i16.2
  %.pn10 = phi { i64, i1 } [ %1154, %.preheader.i.i16.2 ], [ %1151, %1141 ]
  %1153 = extractvalue { i64, i1 } %.pn10, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %1153, ptr addrspace(1) %1150, align 8, !tbaa !41
  %1154 = cmpxchg ptr addrspace(1) %128, i64 %1153, i64 %1146 syncscope("one-as") release monotonic, align 8
  %1155 = extractvalue { i64, i1 } %1154, 1
  br i1 %1155, label %.loopexit33, label %.preheader.i.i16.2

.loopexit33:                                      ; preds = %.preheader.i.i16.2, %.preheader.i.i, %1140, %1141, %106, %105
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.usub.sat.i32(i32, i32) #9

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="1" "denormal-fp-math-f32"="ieee" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #3 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #6 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #7 = { nounwind }
attributes #8 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #9 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.dbg.cu = !{!4}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 400}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 0}
!4 = distinct !DICompileUnit(language: DW_LANG_C, file: !5, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!5 = !DIFile(filename: "test_scan_layouts.ttgir", directory: "/tmp/pytest-of-root/pytest-293/test_scan_layouts_True_1_src_l0")
!6 = distinct !DISubprogram(name: "kernel_0d1d", linkageName: "kernel_0d1d", scope: !5, file: !5, line: 4, type: !7, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4)
!7 = !DISubroutineType(cc: DW_CC_normal, types: !8)
!8 = !{}
!9 = !DILocation(line: 6, column: 12, scope: !6)
!10 = !DILocation(line: 8, column: 12, scope: !6)
!11 = !DILocation(line: 10, column: 12, scope: !6)
!12 = !DILocation(line: 11, column: 12, scope: !6)
!13 = !DILocation(line: 15, column: 12, scope: !6)
!14 = !DILocation(line: 16, column: 13, scope: !6)
!15 = !DILocation(line: 17, column: 13, scope: !6)
!16 = !DILocation(line: 19, column: 15, scope: !6)
!17 = !DILocation(line: 20, column: 15, scope: !6)
!18 = !DILocation(line: 21, column: 15, scope: !6)
!19 = !DILocation(line: 22, column: 15, scope: !6)
!20 = !DILocation(line: 27, column: 15, scope: !6)
!21 = !DILocation(line: 28, column: 9, scope: !6)
!22 = !DILocation(line: 33, column: 13, scope: !6)
!23 = !DILocation(line: 35, column: 13, scope: !6)
!24 = !DILocation(line: 36, column: 7, scope: !6)
!25 = !DILocation(line: 37, column: 7, scope: !6)
!26 = !{!27, !27, i64 0}
!27 = !{!"long", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C/C++ TBAA"}
!30 = !{i64 2662}
!31 = !{!32, !33, i64 0}
!32 = !{!"", !33, i64 0, !33, i64 8, !34, i64 16, !27, i64 24, !27, i64 32, !27, i64 40}
!33 = !{!"any pointer", !28, i64 0}
!34 = !{!"hsa_signal_s", !27, i64 0}
!35 = !{!32, !27, i64 40}
!36 = !{!32, !33, i64 8}
!37 = !{!38, !27, i64 8}
!38 = !{!"", !27, i64 0, !27, i64 8, !39, i64 16, !39, i64 20}
!39 = !{!"int", !28, i64 0}
!40 = !{!39, !39, i64 0}
!41 = !{!38, !27, i64 0}
!42 = !{!43, !27, i64 16}
!43 = !{!"amd_signal_s", !27, i64 0, !28, i64 8, !27, i64 16, !39, i64 24, !39, i64 28, !27, i64 32, !27, i64 40, !28, i64 48, !28, i64 56}
!44 = !{!43, !39, i64 24}
!45 = !{!28, !28, i64 0}
