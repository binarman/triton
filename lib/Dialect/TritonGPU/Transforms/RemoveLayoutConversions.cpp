#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#ifdef USE_ROCM
#include "triton/Analysis/Utility.h"
#endif

#include <memory>

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SliceEncodingAttr;

struct PatternSharedInfo {
  // If a conversion cannot be eliminated with a high-benefit pattern (e.g.,
  // SimplifyConversion, RematerializeBackward), it will be pushed forward in
  // the hope that this will enable the elimination of these conversions later.
  // However, pushing a conversion forward can introduce more conversions
  // (op(cvt(arg_0), arg_1, ..., arg_n) -> cvt(op(arg_0, cvt(arg_1), ...,
  // cvt(arg_n))). This is why the RematerializeForward pattern performs an
  // analysis to determine whether these added conversions can be eliminated
  // later. The RematerializeBackward pattern, applied after pushing this
  // conversion forward, will eliminate these newly added conversions by
  // reversing the process achieved with RematerializeForward. This can create
  // an infinite loop between these two optimizations. To avoid this, we keep
  // track of the conversions that were pushed forward and skip them in the
  // RematerializeBackward pattern. A similar kind of loop can occur with the
  // RematerializeForward and MoveConvertOutOfLoop patterns.
  llvm::DenseMap<Operation *, Operation *> cvtsPushedForwardMap;
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// convert(blocked, dot_operand) ->
// convert(blocked, mma) + convert(mma,  dot_operand)
// if this value is itself the result of a dot operation
// this is a heuristic to accommodate some pattern seen in fused attention
// kernels.
// TODO: replace this by something more generic, i.e. layout-aware CSE
class DecomposeDotOperand : public mlir::RewritePattern {

public:
  explicit DecomposeDotOperand(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  template <typename encTy>
  mlir::LogicalResult processEncoding(encTy encoding,
                                      triton::gpu::ConvertLayoutOp convert,
                                      RankedTensorType &dstType,
                                      mlir::PatternRewriter &rewriter) const {
    SetVector<Operation *> bwdSlices;
    mlir::getBackwardSlice(convert.getResult(), &bwdSlices);
    if (llvm::find_if(bwdSlices, [](Operation *op) {
          return isa<triton::DotOp>(op);
        }) == bwdSlices.end())
      return mlir::failure();

    auto tmpType = RankedTensorType::get(dstType.getShape(),
                                         dstType.getElementType(), encoding);
    auto tmp = rewriter.create<triton::gpu::ConvertLayoutOp>(
        convert.getLoc(), tmpType, convert.getOperand());
    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(convert, dstType,
                                                                tmp);
    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    auto srcType = convert.getOperand().getType().cast<RankedTensorType>();
    auto dstType = convert.getType().cast<RankedTensorType>();
    if (srcType.getEncoding().isa<triton::gpu::BlockedEncodingAttr>() &&
        dstType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>()) {
      auto dstDotOperand =
          dstType.getEncoding().cast<triton::gpu::DotOperandEncodingAttr>();
      auto dstParent = dstDotOperand.getParent();
      if (dstDotOperand.getOpIdx() == 1 ||
          (!dstParent.isa<triton::gpu::MmaEncodingAttr>() &&
           !dstParent.isa<triton::gpu::MfmaEncodingAttr>()))
        return mlir::failure();

      if (dstParent.isa<triton::gpu::MmaEncodingAttr>()) {
        auto dstParentMma = dstParent.cast<triton::gpu::MmaEncodingAttr>();
        if (dstParentMma.isVolta() || dstParentMma.getWarpsPerCTA()[1] > 1)
          return mlir::failure();
        return processEncoding(dstParentMma, convert, dstType, rewriter);
      }

      if (dstParent.isa<triton::gpu::MfmaEncodingAttr>()) {
        auto dstParentMfma = dstParent.cast<triton::gpu::MfmaEncodingAttr>();
        if (dstParentMfma.getWarpsPerCTA()[1] > 1)
          return mlir::failure();
        return processEncoding(dstParentMfma, convert, dstType, rewriter);
      }
    }
    return mlir::failure();
  }
};

// It's beneficial to move the conversion
// to after the reduce if necessary since it will be
// done on a rank-reduced tensor hence cheaper
class SimplifyReduceCvt : public mlir::RewritePattern {
public:
  explicit SimplifyReduceCvt(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    triton::ReduceOp reduce;
    for (auto &use : convert.getResult().getUses()) {
      auto owner = llvm::dyn_cast<triton::ReduceOp>(use.getOwner());
      if (!owner) {
        continue;
      }

      // TODO: This only moves conversions from the first argument which is
      // fine for argmin/argmax but may not be optimal generally
      if (convert.getResult() != owner.getOperands()[0]) {
        continue;
      }
      reduce = owner;
      break;
    }
    if (!reduce)
      return mlir::failure();

    SmallVector<Value> newOperands = reduce.getOperands();

    newOperands[0] = convert.getOperand();
    auto newEncoding =
        newOperands[0].getType().cast<RankedTensorType>().getEncoding();

    // this may generate unsupported conversions in the LLVM codegen
    if (newEncoding.isa<triton::gpu::MmaEncodingAttr>() ||
        newEncoding.isa<triton::gpu::MfmaEncodingAttr>()) {
      return failure();
    }

    for (unsigned i = 1; i < newOperands.size(); ++i) {
      auto oldTy = newOperands[i].getType().cast<RankedTensorType>();
      RankedTensorType newTy =
          RankedTensorType::Builder(oldTy).setEncoding(newEncoding);

      newOperands[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newTy, newOperands[i]);
    }

    rewriter.setInsertionPoint(reduce);
    auto newReduce = rewriter.create<triton::ReduceOp>(
        op->getLoc(), newOperands, reduce.getAxis());
    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(reduce.getCombineOp(), newCombineOp,
                               newCombineOp.end());

    SmallVector<Value> newRet = newReduce.getResult();
    auto oldTypes = reduce.getResult().getType();
    for (unsigned i = 0; i < reduce.getNumOperands(); ++i) {
      // it's still beneficial to move the conversion
      // to after the reduce if necessary since it will be
      // done on a rank-reduced tensor hence cheaper
      if (newRet[i].getType() != oldTypes[i])
        newRet[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), oldTypes[i], newRet[i]);
    }
    rewriter.replaceAllUsesWith(reduce.getResult(), newRet);

    return success();
  }
};

// Layout conversions can't deduce their return type automatically.
// IIUC they are therefore not handled by DRR right now
class SimplifyConversion : public mlir::RewritePattern {
public:
  explicit SimplifyConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             4, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    return ConvertLayoutOp::canonicalize(convert, rewriter);
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// op(cvt(arg_0), arg_1, ..., arg_n)
// -> cvt(op(arg_0, cvt(arg_1), ..., cvt(arg_n)))
void pushConversionForward(triton::gpu::ConvertLayoutOp cvt,
                           SetVector<Operation *> &cvtSlices,
                           PatternSharedInfo &sharedInfo,
                           mlir::PatternRewriter &rewriter) {
  auto srcEncoding =
      cvt.getOperand().getType().cast<RankedTensorType>().getEncoding();
  auto dstEncoding =
      cvt.getResult().getType().cast<RankedTensorType>().getEncoding();
  IRMapping mapping;
  auto op = cvtSlices.front();
  for (Value arg : op->getOperands()) {
    if (arg.getDefiningOp() == cvt)
      mapping.map(arg, cvt.getOperand());
    else {
      auto oldType = arg.getType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(
          oldType.getShape(), oldType.getElementType(), srcEncoding);
      auto cvtI = rewriter.create<triton::gpu::ConvertLayoutOp>(arg.getLoc(),
                                                                newType, arg);
      if (Operation *argOp = arg.getDefiningOp())
        cvtI->moveAfter(argOp);
      mapping.map(arg, cvtI);
    }
  }
  rewriter.setInsertionPoint(op);
  if (op->getNumResults() == 0) {
    Operation *newOp = rewriter.clone(*op, mapping);
    rewriter.eraseOp(op);
    return;
  }
  auto *newOp = cloneWithInferType(rewriter, op, mapping);
  auto newType = newOp->getResult(0).getType().cast<RankedTensorType>();
  auto newCvtType = RankedTensorType::get(
      newType.getShape(), newType.getElementType(), dstEncoding);
  auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
      newOp->getLoc(), newCvtType, newOp->getResult(0));
  sharedInfo.cvtsPushedForwardMap[newCvt] = newCvt->getOperand(0).getDefiningOp();
  rewriter.replaceOp(op, newCvt->getResults());
}

//
class MoveConvertOutOfIf : public mlir::RewritePattern {
public:
  explicit MoveConvertOutOfIf(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::IfOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ifOp = cast<scf::IfOp>(*op);
    // If “scf.if” defines no values, “scf.yield” will be inserted implicitly.
    // However, "scf.else" is not required to be present, so we need to check
    // if it exists.
    auto thenYield = ifOp.thenYield();
    int numOps = thenYield.getNumOperands();
    SmallVector<Value> newThenYieldOps = thenYield.getOperands();
    SetVector<Operation *> thenCvts;
    SmallVector<Type> newRetTypes;

    bool hasElse = !ifOp.getElseRegion().empty();

    scf::YieldOp elseYield;
    SmallVector<Value> newElseYieldOps;
    SetVector<Operation *> elseCvts;
    if (hasElse) {
      elseYield = ifOp.elseYield();
      newElseYieldOps = elseYield.getOperands();
    }

    IRMapping mapping;
    for (size_t i = 0; i < numOps; i++) {
      auto thenCvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
          thenYield.getOperand(i).getDefiningOp());
      if (hasElse) {
        auto elseYield = ifOp.elseYield();
        auto elseCvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
            elseYield.getOperand(i).getDefiningOp());
        if (thenCvt && elseCvt &&
            std::distance(elseCvt->user_begin(), elseCvt->user_end()) == 1 &&
            std::distance(thenCvt->user_begin(), thenCvt->user_end()) == 1 &&
            thenCvt.getOperand().getType() == elseCvt.getOperand().getType()) {
          // If thenCvt and elseCvt's type are the same, it means a single
          // conversion is enough to replace both of them. We can move the
          // conversion out of scf.if and replace both thenCvt and elseCvt with
          // the new conversion.
          mapping.map(thenCvt.getResult(), thenCvt.getOperand());
          thenCvts.insert((Operation *)thenCvt);
          newRetTypes.push_back(thenCvt.getOperand().getType());
          mapping.map(elseCvt.getResult(), elseCvt.getOperand());
          elseCvts.insert((Operation *)elseCvt);
        } else
          // Cannot move out of scf.if because thenCvt != elseCvt
          // Moving it out of scf.if will introduce a new conversion
          newRetTypes.push_back(thenYield.getOperand(i).getType());
      } else {
        if (thenCvt &&
            std::distance(thenCvt->user_begin(), thenCvt->user_end()) == 1) {
          // If there's only a single use of the conversion then we can move it
          mapping.map(thenCvt.getResult(), thenCvt.getOperand());
          thenCvts.insert((Operation *)thenCvt);
          newRetTypes.push_back(thenCvt.getOperand().getType());
        } else
          // Cannot move out of scf.if because either there's another use of
          // the conversion or there's no conversion at all
          newRetTypes.push_back(thenYield.getOperand(i).getType());
      }
    }
    if (mapping.getValueMap().empty())
      return mlir::failure();

    auto newIfOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newRetTypes,
                                              ifOp.getCondition(), hasElse);
    auto rematerialize = [&](Block *block, SetVector<Operation *> &cvts) {
      for (Operation &op : block->getOperations()) {
        if (cvts.contains(&op)) {
          if (mapping.contains(op.getOperand(0)))
            mapping.map(op.getResult(0), mapping.lookup(op.getOperand(0)));
          continue;
        }
        rewriter.clone(op, mapping);
      }
    };
    rewriter.setInsertionPointToEnd(newIfOp.thenBlock());
    rematerialize(ifOp.thenBlock(), thenCvts);
    if (hasElse) {
      rewriter.setInsertionPointToEnd(newIfOp.elseBlock());
      rematerialize(ifOp.elseBlock(), elseCvts);
    }

    rewriter.setInsertionPointAfter(newIfOp);
    SmallVector<Value> newRetValues = newIfOp.getResults();
    for (size_t i = 0; i < numOps; i++) {
      if (newIfOp.getResult(i).getType() != ifOp.getResult(i).getType()) {
        newRetValues[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
            newIfOp.getLoc(), ifOp.getResult(i).getType(),
            newIfOp.getResult(i));
      }
    }

    rewriter.replaceOp(op, newRetValues);
    return mlir::success();
  }
};

//
class RematerializeForward : public mlir::RewritePattern {
  PatternSharedInfo &sharedInfo;

public:
  explicit RematerializeForward(mlir::MLIRContext *context, PatternSharedInfo &sharedInfo)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context), sharedInfo(sharedInfo) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvtOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(*cvtOp);
    auto srcEncoding =
        cvt.getOperand().getType().cast<RankedTensorType>().getEncoding();
    auto dstEncoding =
        cvt.getResult().getType().cast<RankedTensorType>().getEncoding();
    if (srcEncoding.isa<triton::gpu::SharedEncodingAttr>() ||
        dstEncoding.isa<triton::gpu::SharedEncodingAttr>())
      return failure();
    // heuristics for flash attention
    if (srcEncoding.isa<triton::gpu::SliceEncodingAttr>())
      return failure();
    // For cases like:
    // %0 = convert_layout %arg0
    // We should try to move %0 out of scf.for first, if it couldn't be moved
    // out additional conversions will be added to the loop body.
    if (!cvt.getOperand().getDefiningOp() &&
        isa<scf::ForOp>(cvt->getParentOp()))
      return failure();

    SetVector<Operation *> cvtSlices;
    auto filter = [&](Operation *op) {
      return op->getBlock() == cvt->getBlock() &&
             !isa<triton::gpu::ConvertLayoutOp, scf::YieldOp>(op) &&
             !(isa<triton::ReduceOp>(op) &&
               !op->getResult(0).getType().isa<RankedTensorType>());
    };
    mlir::getForwardSlice(cvt.getResult(), &cvtSlices, {filter});
    if (cvtSlices.empty())
      return failure();

    for (Operation *op : cvtSlices) {
      // don't rematerialize anything expensive
      if (isExpensiveToRemat(op, srcEncoding))
        return failure();
      // don't rematerialize non-element-wise
      if (!op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() &&
          !op->hasTrait<mlir::OpTrait::Elementwise>() &&
          !isa<triton::StoreOp, triton::AssertOp, triton::PrintOp,
               triton::ReduceOp>(op))
        return failure();
      // don't rematerialize if it adds an extra conversion that can't
      // be removed
      for (Value arg : op->getOperands()) {
        Operation *argOp = arg.getDefiningOp();
        SetVector<Operation *> processed;
        SetVector<Attribute> layout;
        llvm::MapVector<Value, Attribute> toConvert;
        int numAddedConvs = simulateBackwardRematerialization(
            argOp, processed, layout, toConvert, srcEncoding);
        if (argOp && !isa<triton::gpu::ConvertLayoutOp>(argOp) &&
            cvtSlices.count(argOp) == 0 && numAddedConvs > 0)
          return failure();
      }
    }

    // Call SimplifyReduceCvt instead of the general push conversion forward
    if (isa<triton::ReduceOp>(cvtSlices.front()))
      return failure();

    pushConversionForward(cvt, cvtSlices, sharedInfo, rewriter);
    return success();
  }
};

// Layout conversions are expensive. They require going through
// shared memory, which is orders of magnitude slower than
// other non-i/o operations in the dialect.
// It therefore makes sense to remove them whenever possible,
// even if it means rematerializing all values whose definitions
// are reachable from it without passing through any memory operation.
class RematerializeBackward : public mlir::RewritePattern {
  PatternSharedInfo &sharedInfo;

public:
  explicit RematerializeBackward(mlir::MLIRContext *context, PatternSharedInfo &sharedInfo)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             3, context), sharedInfo(sharedInfo) {}


  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvt,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(cvt))
      return mlir::failure();

    auto it = sharedInfo.cvtsPushedForwardMap.find(cvt);
    if (it != sharedInfo.cvtsPushedForwardMap.end() &&
        it->second == cvt->getOperand(0).getDefiningOp())
      return mlir::failure();

    // we don't touch block arguments
    Operation *op = cvt->getOperand(0).getDefiningOp();
    if (!op)
      return mlir::failure();
    // we don't want to rematerialize any conversion to/from shared
    if (triton::gpu::isSharedEncoding(cvt->getResults()[0]) ||
        triton::gpu::isSharedEncoding(cvt->getOperand(0)))
      return mlir::failure();
    // we don't handle conversions to DotOperandEncodingAttr
    // this is a heuristics to accommodate fused attention
    auto targetType = cvt->getResultTypes()[0].cast<RankedTensorType>();
    if (targetType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>())
      return mlir::failure();
    // DFS
    SetVector<Operation *> processed;
    SetVector<Attribute> layout;
    llvm::MapVector<Value, Attribute> toConvert;
    if (simulateBackwardRematerialization(cvt, processed, layout, toConvert,
                                          targetType.getEncoding()) > 0)
      return mlir::failure();

    IRMapping mapping;
    rematerializeConversionChain(toConvert, rewriter, processed, mapping);
    rewriter.replaceOp(cvt, mapping.lookup(cvt->getOperand(0)));

    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

class MoveConvertOutOfLoop : public mlir::RewritePattern {
  PatternSharedInfo &sharedInfo;

public:
  explicit MoveConvertOutOfLoop(mlir::MLIRContext *context,
                                PatternSharedInfo &sharedInfo)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 1, context),
        sharedInfo(sharedInfo) {}

  SmallVector<Value, 4>
  rematerializeForLoop(mlir::PatternRewriter &rewriter, scf::ForOp &forOp,
                       size_t i, RankedTensorType newType,
                       triton::gpu::ConvertLayoutOp origConversion) const {
    // Rewrite init argument
    Type origType = forOp.getInitArgs()[i].getType();
    SmallVector<Value, 4> newInitArgs = forOp.getInitArgs();
    newInitArgs[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newInitArgs[i].getLoc(), newType, newInitArgs[i]);
    // Clone for loop
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->moveBefore(forOp);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    IRMapping mapping;
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    mapping.map(origConversion.getResult(), newForOp.getRegionIterArgs()[i]);

    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (&op == (Operation *)(&origConversion))
        continue;
      Operation *newOp = rewriter.clone(op, mapping);
    }
    // create yield, inserting conversions if necessary
    auto yieldOp = forOp.getBody()->getTerminator();
    SmallVector<Value, 4> newYieldArgs;
    for (Value arg : yieldOp->getOperands())
      newYieldArgs.push_back(mapping.lookup(arg));
    if (newYieldArgs[i].getType() != newType)
      newYieldArgs[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
          yieldOp->getLoc(), newType, newYieldArgs[i]);
    rewriter.create<scf::YieldOp>(forOp.getLoc(), newYieldArgs);

    // replace
    SmallVector<Value, 4> newResults = newForOp->getResults();
    newResults[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newForOp.getLoc(), origType, newForOp->getResult(i));
    newResults[i].getDefiningOp()->moveAfter(newForOp);
    return newResults;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);
    auto iterArgs = forOp.getRegionIterArgs();
    for (const auto &iterArg : llvm::enumerate(iterArgs)) {
      // skip non-tensor types
      if (!iterArg.value().getType().isa<RankedTensorType>())
        continue;
      SmallVector<Operation *> cvts;
      if (canMoveOutOfLoop(iterArg.value(), cvts).failed())
        continue;
      // check
      for (auto *op : cvts) {
        auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
        auto it = sharedInfo.cvtsPushedForwardMap.find(cvt);
        if (it != sharedInfo.cvtsPushedForwardMap.end())
          return mlir::failure();
        auto targetType = op->getResultTypes()[0].cast<RankedTensorType>();
        auto newFor = rematerializeForLoop(rewriter, forOp, iterArg.index(),
                                           targetType, cvt);
        rewriter.replaceOp(forOp, newFor);
        return success();
      }
    }
    return failure();
  }
};

//
class ConvertDotConvert : public mlir::RewritePattern {
public:
  ConvertDotConvert(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstOp = cast<triton::gpu::ConvertLayoutOp>(op);
    auto dotOp =
        dyn_cast_or_null<triton::DotOp>(dstOp.getSrc().getDefiningOp());
    if (!dotOp)
      return mlir::failure();
    if (std::distance(dstOp->user_begin(), dstOp->user_end()) != 1 ||
        std::distance(dotOp->user_begin(), dotOp->user_end()) != 1)
      return mlir::failure();
    auto cvtOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        dotOp.getOperand(2).getDefiningOp());
    if (!cvtOp)
      return mlir::failure();
    auto loadOp =
        dyn_cast_or_null<triton::LoadOp>(cvtOp.getSrc().getDefiningOp());
    if (!loadOp)
      return mlir::failure();
    auto dstTy = dstOp.getResult().getType().cast<RankedTensorType>();
    auto srcTy = cvtOp.getOperand().getType().cast<RankedTensorType>();
    if (dstTy != srcTy)
      return mlir::failure();

    // TODO: int tensor cores
    auto out_dtype = dstTy.getElementType().cast<FloatType>();
    APFloat value(0.0f);
    if (out_dtype.isBF16())
      value = APFloat(APFloat::IEEEhalf(), APInt(16, 0));
    else if (out_dtype.isF16())
      value = APFloat(APFloat::IEEEhalf(), APInt(16, 0));
    else if (out_dtype.isF32())
      value = APFloat(0.0f);
    else
      llvm_unreachable("unsupported data type");

    auto _0f =
        rewriter.create<arith::ConstantFloatOp>(op->getLoc(), value, out_dtype);
    auto _0 = rewriter.create<triton::SplatOp>(
        op->getLoc(), dotOp.getResult().getType(), _0f);
    auto newDot = rewriter.create<triton::DotOp>(
        op->getLoc(), dotOp.getResult().getType(), dotOp.getOperand(0),
        dotOp.getOperand(1), _0, dotOp.getAllowTF32());
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op->getLoc(), dstTy, newDot.getResult());
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, newCvt, cvtOp.getOperand());
    return mlir::success();
  }
};

#ifdef USE_ROCM

// Following pattern looking for TransOp between ConvertOps
// If these ConvertOps process tensors with compatible mfma encodings,
// replace TransOp with simple convert layout operation
//
// This pattern works as an optional preparation step for
// FuseConversionWithMFMADot pattern
//
// looking for following pattern:
//   entryConvert = ConvertLayoutOp(x) MFMA -> SharedLayout1
//   trans = transOp(c1) SharedLayout1 -> SharedLayout2
//   outputConvert = ConvertLayout(t) SharedLayout2 -> dotOp(MFMA)
//
// transforms to
//   transMFMA = ConvertLayoutOp(x) MFMA -> transposed MFMA
//   transDotMFMA = ConvertLayout(c1) transposed MFMA -> dotOp(transposed MFMA)
//   dotMFMA = ConvertLayout(c1) dotOp(transposed MFMA) -> dotOp(MFMA)
class EliminateMFMATrans : public mlir::RewritePattern {
public:
  EliminateMFMATrans(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto outputConvert = cast<triton::gpu::ConvertLayoutOp>(op);
    auto outputType = cast<RankedTensorType>(outputConvert.getResult().getType());
    auto outputLayout = dyn_cast_or_null<DotOperandEncodingAttr>(outputType.getEncoding());
    if (!outputLayout)
      return mlir::failure();
    auto outputParentLayout = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(outputLayout.getParent());
    if (!outputParentLayout)
      return mlir::failure();

    auto trans = dyn_cast_or_null<triton::TransOp>(outputConvert.getSrc().getDefiningOp());
    if (!trans)
      return mlir::failure();
    assert(llvm::isa<triton::gpu::SharedEncodingAttr>(cast<RankedTensorType>(trans.getResult().getType()).getEncoding()));
    assert(llvm::isa<triton::gpu::SharedEncodingAttr>(cast<RankedTensorType>(trans.getSrc().getType()).getEncoding()));

    auto entryConvert = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(trans.getSrc().getDefiningOp());
    if (!entryConvert)
      return mlir::failure();
    auto inputType = cast<RankedTensorType>(entryConvert.getSrc().getType());
    auto inputLayout = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(inputType.getEncoding());
    if (!inputLayout)
      return mlir::failure();
    
    assert(inputType.getElementType() == outputType.getElementType());
    assert(inputType.getShape() == outputType.getShape());

    // In following cases we can not eliminate layout conversions
    if (outputParentLayout.getWarpsPerCTA() != inputLayout.getWarpsPerCTA())
      return mlir::failure();
    if (outputParentLayout.getNonKDim() != inputLayout.getNonKDim())
      return mlir::failure();

    auto loc = trans.getLoc();

    auto dtype = inputType.getElementType();
    auto tranposedShape = outputType.getShape();
    auto transposedMfma = triton::gpu::MfmaEncodingAttr::get(getContext(), inputLayout.getNonKDim(), inputLayout.getWarpsPerCTA(), !inputLayout.getIsTransposed());
    auto transposedMFMAType = RankedTensorType::get(tranposedShape, dtype, transposedMfma);

    auto transMFMA = rewriter.create<ConvertLayoutOp>(loc, transposedMFMAType, entryConvert.getSrc());
    auto transDotMFMA = rewriter.create<ConvertLayoutOp>(loc, outputConvert.getResult().getType(), transMFMA.getResult());

    rewriter.replaceOp(op, transDotMFMA.getResult());

    return mlir::success();
  }
};

// Following pattern searches for mfma DotOp with ConvertOps as arguments
// chains of convert layouts with operands which 
class FuseConversionWithMFMADot : public mlir::RewritePattern {
public:
  FuseConversionWithMFMADot(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::DotOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dot = cast<triton::DotOp>(op);
    auto dotLoc = dot.getLoc();
    auto dotType = cast<RankedTensorType>(dot.getD().getType());
    int dotFlavor = getMFMAFlavor(dotType);
    if (dotFlavor == 0)
      return mlir::failure();

    auto [aEarlySrc, aDist] = earliestMFMACompatibleValue(dot.getA());
    auto [bEarlySrc, bDist] = earliestMFMACompatibleValue(dot.getB());
    llvm::outs() << "Analyzing: " << dot << "\n";
    llvm::outs() << "  srcA: " << aEarlySrc << "\n";
    llvm::outs() << "  srcB: " << bEarlySrc << "\n";

    if (aEarlySrc == dot.getA() && bEarlySrc == dot.getB())
      return mlir::failure();

    auto aEarlyType = cast<RankedTensorType>(aEarlySrc.getType());
    auto bEarlyType = cast<RankedTensorType>(bEarlySrc.getType());
    int aFlavor = getMFMAFlavor(aEarlyType);
    int bFlavor = getMFMAFlavor(bEarlyType);

    llvm::outs() << "  flavors: " << aFlavor << " " << bFlavor << " " << dotFlavor << "\n";

    // case 1, no need to tranpose anything, just reuse earlier found values
    if (aFlavor != bFlavor && bFlavor == dotFlavor) {
      // if distance between curren dot operand and proposed operand is 1
      // we will create layout conversion operation with adjustLayout,
      // so data flow chain will not get shorter.
      if (aDist <= 1 && bDist <= 1)
        return mlir::failure();
      rewriter.setInsertionPoint(dot);

      auto newA = adjustLayout(rewriter, dotLoc, aEarlySrc, dot.getA().getType());

      auto newB = adjustLayout(rewriter, dotLoc, bEarlySrc, dot.getB().getType());

      rewriter.replaceOpWithNewOp<triton::DotOp>(dot, newA, newB, dot.getC(), dot.getAllowTF32());
      return mlir::success();
    }

    // case 2, can transpose dot and use operand layouts as-is
    if (aFlavor != bFlavor && aFlavor == dotFlavor) {
      rewriter.setInsertionPoint(dot);

      auto newAType = transposeLayout(dot.getA().getType());
      Value newA = adjustLayout(rewriter, dotLoc, aEarlySrc, newAType);

      auto newBType = transposeLayout(dot.getB().getType());
      Value newB = adjustLayout(rewriter, dotLoc, bEarlySrc, newAType);

      auto newCType = transposeLayout(dot.getC().getType());
      Value newC = adjustLayout(rewriter, dotLoc, dot.getC(), newCType);
      
      auto newDot = rewriter.create<triton::DotOp>(dotLoc, newA, newB, newC, dot.getAllowTF32());

      rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(dot, newCType, newDot.getD());
      return mlir::success();
    }
    // case 3
    // aFlavor == bFlavor
    // 
    // try to transpose a or b if succeed, try to apply case 1 and 2

    return mlir::failure();
  }

private:

  static Type transposeLayout(Type srcType) {
    auto tensorType = cast<RankedTensorType>(srcType);
    auto enc = tensorType.getEncoding();
    auto shape = tensorType.getShape();
    auto dtype = tensorType.getElementType();
    mlir::Attribute newEnc;
    triton::gpu::MfmaEncodingAttr srcMfmaEnc;
    auto dotOpEnc = dyn_cast_or_null<triton::gpu::DotOperandEncodingAttr>(enc);
    if (dotOpEnc)
      srcMfmaEnc = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(dotOpEnc.getParent());
    else
      srcMfmaEnc = cast<triton::gpu::MfmaEncodingAttr>(enc);
    
    auto ctx = srcMfmaEnc.getContext();
    auto nonKDim = srcMfmaEnc.getNonKDim();
    auto warpsPerCTA = srcMfmaEnc.getWarpsPerCTA();
    auto isTransposed = !srcMfmaEnc.getIsTransposed();
    auto newMfmaEnc = triton::gpu::MfmaEncodingAttr::get(ctx, nonKDim, warpsPerCTA, isTransposed);
    if (dotOpEnc)
      newEnc = triton::gpu::DotOperandEncodingAttr::get(ctx, dotOpEnc.getOpIdx(), newMfmaEnc);
    else
      newEnc = newMfmaEnc;
    auto newType = RankedTensorType::get(shape, dtype, newEnc);
    return newType;
  }

  static Value adjustLayout(mlir::PatternRewriter &rewriter, mlir::Location loc, Value src, Type targetType){
    assert(isa<RankedTensorType>(targetType));
    auto srcType = cast<RankedTensorType>(src.getType());
    if (srcType == targetType)
      return src;
    auto convert = rewriter.create<triton::gpu::ConvertLayoutOp>(loc, targetType, src);
    return convert.getResult();
  }

  // mfma layout can be one of two flavors:
  // flavor 0 - encoding is not mfma related
  // flavor 1 includes: input A, transposed dot output, transposed input B
  // flavor 2 includes: input B, dot output, transposed input A
  static int getMFMAFlavor(RankedTensorType type) {
    auto encoding = type.getEncoding();
    auto dotOpEncoding = dyn_cast_or_null<triton::gpu::DotOperandEncodingAttr>(encoding);
    if (dotOpEncoding) {
      auto mfmaEncoding = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(dotOpEncoding.getParent());
      if (mfmaEncoding) {
        bool isInputA = dotOpEncoding.getOpIdx() == 0;
        bool isTransposed = mfmaEncoding.getIsTransposed();
        // input A && not transposed  =>  flavor 1
        // input A && transposed  =>  flavor 2
        // input B && not transposed  =>  flavor 1
        // input B && transposed  =>  flavor 2
        int flavor = (isInputA != isTransposed ? 1 : 2);
        return flavor;
      }
    }
    auto mfmaEncoding = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(encoding);
    if (mfmaEncoding) {
      return mfmaEncoding.getIsTransposed() ? 1 : 2;
    }
    return 0;
  }

  static bool areEncodingsCompatible(mlir::Attribute query, triton::gpu::DotOperandEncodingAttr targetDotOperand) {
    auto targetMfma = cast<triton::gpu::MfmaEncodingAttr>(targetDotOperand.getParent());

    triton::gpu::MfmaEncodingAttr queryMfma;
    bool kDimCompatible = false;

    auto queryDotOperand = dyn_cast_or_null<triton::gpu::DotOperandEncodingAttr>(query);
    if (queryDotOperand) {
      queryMfma = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(queryDotOperand.getParent());
      kDimCompatible = queryDotOperand.getKWidth() == targetDotOperand.getKWidth();
    } else {
      queryMfma = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(query);
      if (targetMfma.getNonKDim() == 32) {
        kDimCompatible = targetDotOperand.getKWidth() == 8;
      } else {
        assert(targetMfma.getNonKDim() == 16);
        // TODO implement for mfma16
        kDimCompatible = false;
      }
    }

    if (!queryMfma)
      return false;

    bool nonKDimCompatible = queryMfma.getNonKDim() == targetMfma.getNonKDim();
    bool warpsPerCTACompatible = queryMfma.getWarpsPerCTA() == targetMfma.getWarpsPerCTA();

    return nonKDimCompatible && warpsPerCTACompatible && kDimCompatible;
  }

  // trace chain of layout conversions and find earliest compatible value,
  // which can be used as a dot operand
  // returns pair: found value and distance to this value (0 if returned value equal to input val).
  static std::tuple<mlir::Value, int> earliestMFMACompatibleValue(mlir::Value val) {
    auto tensorType = dyn_cast_or_null<RankedTensorType>(val.getType());
    auto dotEnc = cast<triton::gpu::DotOperandEncodingAttr>(tensorType.getEncoding());
    auto mfmaEnc = cast<triton::gpu::MfmaEncodingAttr>(dotEnc.getParent());
    mlir::Operation *op;
    triton::gpu::ConvertLayoutOp convertOp;
    Value earliestDef;
    int earliestDist;
    for (int dist = 0; ; ++dist) {
      auto valType = cast<RankedTensorType>(val.getType());
      if (areEncodingsCompatible(valType.getEncoding(), dotEnc)) {
        earliestDef = val;
        earliestDist = dist;
      }
      op = val.getDefiningOp();
      convertOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(op);
      if (!convertOp)
        return {earliestDef, earliestDist};
      val = convertOp.getSrc();
    }
  }

};

// Following pattern searches for mfma DotOp with ConvertOps as arguments
// chains of convert layouts with operands which
class SimplifyMFMADotOpConversions : public mlir::RewritePattern {
public:
  SimplifyMFMADotOpConversions(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto candidate = cast<triton::gpu::ConvertLayoutOp>(op);
    auto dstType = cast<RankedTensorType>(candidate.getResult().getType());
    auto dotOpEnc = dyn_cast_or_null<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
    if (!dotOpEnc)
      return mlir::failure();
    auto mfmaEnc = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(dotOpEnc.getParent());
    if (!mfmaEnc)
      return mlir::failure();
    auto loc = candidate.getLoc();

    auto [earlySrc, dist] = earliestMFMACompatibleValue(candidate.getResult());

    if (dist > 1) {
      rewriter.setInsertionPoint(candidate);
      auto newCvt = adjustLayout(rewriter, loc, earlySrc, dstType);
      rewriter.replaceOp(candidate, newCvt);
      return mlir::success();
    }

    return mlir::failure();
  }

private:

  static Value adjustLayout(mlir::PatternRewriter &rewriter, mlir::Location loc, Value src, Type targetType){
    assert(isa<RankedTensorType>(targetType));
    auto srcType = cast<RankedTensorType>(src.getType());
    if (srcType == targetType)
      return src;
    auto convert = rewriter.create<triton::gpu::ConvertLayoutOp>(loc, targetType, src);
    return convert.getResult();
  }

  // trace chain of layout conversions and find earliest compatible value,
  // which can be used as a dot operand
  // returns pair: found value and distance to this value (0 if returned value equal to input val).
  static std::tuple<mlir::Value, int> earliestMFMACompatibleValue(mlir::Value val) {
    auto dstType = cast<RankedTensorType>(val.getType());
    mlir::Operation *op;
    triton::gpu::ConvertLayoutOp convertOp;
    Value earliestDef = val;
    int earliestDist = 0;
    for (int dist = 0; ; ++dist) {
      auto valType = cast<RankedTensorType>(val.getType());
      auto mfmaLayout = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(valType.getEncoding());
      if (mfmaLayout && isMfmaToDotShortcut(valType, dstType)) {
        earliestDef = val;
        earliestDist = dist;
      }
      op = val.getDefiningOp();
      convertOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(op);
      if (!convertOp)
        return {earliestDef, earliestDist};
      val = convertOp.getSrc();
    }
  }
};

#endif // USE_ROCM

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPURemoveLayoutConversionsPass
    : public TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
  TritonGPURemoveLayoutConversionsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    PatternSharedInfo sharedInfo;

    patterns.add<SimplifyConversion>(context);
    patterns.add<SimplifyReduceCvt>(context);
    patterns.add<RematerializeBackward>(context, sharedInfo);
    patterns.add<RematerializeForward>(context, sharedInfo);
    patterns.add<MoveConvertOutOfLoop>(context, sharedInfo);
    patterns.add<MoveConvertOutOfIf>(context);
    patterns.add<DecomposeDotOperand>(context);
    patterns.add<ConvertDotConvert>(context);
#ifdef USE_ROCM
    patterns.add<EliminateMFMATrans>(context);
    // patterns.add<FuseConversionWithMFMADot>(context);
    patterns.add<SimplifyMFMADotOpConversions>(context);
#endif

    if (mlir::applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    if (fixupLoops(m).failed()) {
      signalPassFailure();
    }
    llvm::outs() << "after remove layoutConversions:\n" << m << "\n";
  }
};

std::unique_ptr<Pass> mlir::createTritonGPURemoveLayoutConversionsPass() {
  return std::make_unique<TritonGPURemoveLayoutConversionsPass>();
}
