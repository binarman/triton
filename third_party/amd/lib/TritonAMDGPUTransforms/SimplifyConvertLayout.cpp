#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-simplify-convert-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUSIMPLIFYCONVERTLAYOUT
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

class LoadConvertPattern : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  LoadConvertPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Checking operation " << cvtOp);
    auto srcVal = cvtOp.getSrc();
    auto srcType = srcVal.getType();
    auto dstVal = cvtOp.getResult();
    auto dstType = dstVal.getType();
    auto dstEnc = dstType.getEncoding();

    auto srcDefiningOp = srcVal.getDefiningOp();
    bool matchLoadConvertPattern =
        isa<triton::LoadOp>(srcDefiningOp) && srcVal.getNumUses() == 1;

    triton::LinearLayout srcLL = triton::gpu::toLinearLayout(srcType);
    triton::LinearLayout dstLL = triton::gpu::toLinearLayout(dstType);
    StringAttr kWarp = StringAttr::get(cvtOp.getContext(), "warp");
    auto srcWarps = *srcLL.getBases().find(kWarp);
    auto dstWarps = *dstLL.getBases().find(kWarp);
    bool interWarpConversion = (srcWarps != dstWarps);
    if (matchLoadConvertPattern && interWarpConversion) {
      LDBG("Operation is suitable " << cvtOp);
      auto loadOp = dyn_cast<triton::LoadOp>(srcDefiningOp);
      // TODO make custom layout
      auto newLoadEnc = dstEnc;
      auto newLoadType = srcType.cloneWithEncoding(newLoadEnc);
      loadOp->getResult(0).setType(newLoadType);
      rewriter.setInsertionPoint(loadOp);
      for (int operandIdx = 0; operandIdx < loadOp.getNumOperands();
           operandIdx++) {
        auto operand = loadOp.getOperand(operandIdx);
        RankedTensorType oldOperandType =
            dyn_cast<RankedTensorType>(operand.getType());
        RankedTensorType newOperandType =
            oldOperandType.cloneWithEncoding(newLoadEnc);
        auto newOperand = rewriter.create<triton::gpu::ConvertLayoutOp>(
            operand.getLoc(), newOperandType, operand);
        loadOp.setOperand(operandIdx, newOperand);
      }
    }

    return success();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct TritonAMDGPUSimplifyConvertLayoutPass
    : public impl::TritonAMDGPUSimplifyConvertLayoutBase<
          TritonAMDGPUSimplifyConvertLayoutPass> {
  void runOnOperation() override {
    mlir::triton::FuncOp f = getOperation();
    auto ctx = f.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LoadConvertPattern>(ctx, /*benefit=*/1);
    walkAndApplyPatterns(f, std::move(patterns));
  }
};

} // namespace mlir
