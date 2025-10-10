#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-simplify-convert-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;

using mlir::triton::LinearLayout;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUSIMPLIFYCONVERTLAYOUT
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

class LoadConvertPattern : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  LoadConvertPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  // Create new src layout for convert layout operation, which has same
  // warp/block distribution as convertLL layout Algorithm is designed to
  // maximize vectorization and minimize number of final load instrcutions.
  LinearLayout generateNewLoadLayout(LinearLayout oldLoadLL,
                                     LinearLayout convertLL,
                                     int max_vectorization) const {
    // Algorithm works like this:
    // 1. find bases not covered by warp and block dimensions of convertLL
    // 2. push as much as possible bases from fastest dimension into registers
    // 3. push as much as possible bases into lanes
    // 4. push the rest of bases into registers, they will become repeats.
    // 5. combine computed register+lanes and warps+blocks from convertLL
    //
    // Consider this example:
    // oldLoadLL = #ttg.linear<{
    //     register = [[128, 0]],
    //     lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0]],
    //     warp = [[16, 0], [32, 0], [64, 0]],
    //     block = []
    // }>
    // convertLL = #ttg.linear<{
    //     register = [[0, 1], [0, 2], [64, 0], [128, 0]],
    //     lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]],
    //     warp = [[16, 0], [32, 0], [ 0, 0]],
    //     block = []
    // }>
    //
    // oldLoadLL represents #ttg.blocked<{sizePerThread = [1, 1],
    //                                    threadsPerWarp = [16, 4],
    //                                    warpsPerCTA = [8, 1],
    //                                    order = [1, 0]}>
    //
    // 1. Bases not covered by warps and blocks in convertLL:
    //     [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [64, 0], [128, 0]
    // 2. register = [[0, 1], [0, 2]]
    // 3. lanes = [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0], [128, 0]]
    // 4. there are no more bases left, no repeats.
    // 5. Combined result layout:
    // #ttg.linear<{
    //     register = [[0, 1], [0, 2]],
    //     lane = [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0], [128, 0]],
    //     warp = [[16, 0], [32, 0], [0, 0]],
    //     block = []
    // }>

    // Heuristic considers that oldLoadLL is coalesced and we want to save as
    // much as possible from this original layout, First we try to bread
    // registers into tow parts: repeats and vector part, set of oldLoadLL bases
    // is L(Lr - register part, Ll - lane part, Lw - warp part, Lb - block part)
    // set of convertLL bases is C(same sub-indexing as in L)
    // compute repeats

    // auto rank = oldLoadLL.getNumOutDims();
    // SmallVector<unsigned> order(rank);
    // std::iota(order.rbegin(), order.rend(), 0);

    // return orderPerDim(StringAttr::get(getContext(), "register"), order);
    // auto globalMemOrder = triton::gpu::getOrder(oldLoadLL, shape);
    return convertLL;
  }

  bool isEqualBasis(LinearLayout a, LinearLayout b,
                    const std::string &dim) const {
    StringAttr attr = StringAttr::get(getContext(), dim);
    auto srcBasis = *a.getBases().find(attr);
    auto dstBasis = *b.getBases().find(attr);
    return srcBasis != dstBasis;
  }

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Checking operation " << cvtOp);
    auto srcVal = cvtOp.getSrc();
    auto srcType = srcVal.getType();
    auto dstVal = cvtOp.getResult();
    auto dstType = cast<RankedTensorType>(dstVal.getType());

    auto srcDefiningOp = srcVal.getDefiningOp();
    bool matchLoadConvertPattern =
        isa<triton::LoadOp>(srcDefiningOp) && srcVal.getNumUses() == 1;
    if (!matchLoadConvertPattern) {
      LDBG("Operation does not have load as a direct predecessor");
      return success();
    }

    auto srcLL = ttg::toLinearLayout(srcType);
    auto dstLL = ttg::toLinearLayout(dstType);
    bool interWarpConversion = isEqualBasis(srcLL, dstLL, "warp") &&
                               isEqualBasis(srcLL, dstLL, "block");
    if (interWarpConversion) {
      LDBG("Operation is already intra-warp");
      return success();
    }
    LDBG("Processing suitable operation");
    auto loadOp = dyn_cast<triton::LoadOp>(srcDefiningOp);
    constexpr int maxLoadBitWidth = 128;
    const int max_vectorization =
        maxLoadBitWidth / srcType.getElementTypeBitWidth();
    auto newLoadEnc = ttg::LinearEncodingAttr::get(
        getContext(), generateNewLoadLayout(srcLL, dstLL, max_vectorization));
    auto newLoadType = srcType.cloneWithEncoding(newLoadEnc);

    if (cvtNeedsSharedMemory(newLoadType, dstType)) {
      LDBG("Optimal layout requires LDS transfer, aborting processing");
      return success();
    }

    loadOp->getResult(0).setType(newLoadType);
    rewriter.setInsertionPoint(loadOp);
    for (int operandIdx = 0; operandIdx < loadOp.getNumOperands();
         operandIdx++) {
      auto operand = loadOp.getOperand(operandIdx);
      RankedTensorType oldOperandType =
          dyn_cast<RankedTensorType>(operand.getType());
      RankedTensorType newOperandType =
          oldOperandType.cloneWithEncoding(newLoadEnc);
      auto newOperand = rewriter.create<ttg::ConvertLayoutOp>(
          operand.getLoc(), newOperandType, operand);
      loadOp.setOperand(operandIdx, newOperand);
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
