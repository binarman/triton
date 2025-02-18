#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-in-thread-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

static Type getNewType(Type type, Attribute encoding) {
  RankedTensorType tensorType = dyn_cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

void convertLayout(Attribute encoding, Operation *op) {
  OpBuilder builder(op);
  // Convert operands
  // For load/store with tensor pointers, we don't have to change the
  // operands' type, we do this by changing the outputs' type of
  // `make_tensor_ptr`
  SmallVector<Value, 4> newArgs;
  for (auto operand : op->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType && !isa<triton::gpu::SwizzledSharedEncodingAttr>(
                          tensorType.getEncoding())) {
      Type newType = getNewType(tensorType, encoding);
      newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newType, operand));
    } else {
      newArgs.push_back(operand);
    }
  }

  // Convert output types
  SmallVector<Type, 4> newTypes;
  for (auto t : op->getResultTypes()) {
    bool isAsync = isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
    newTypes.push_back(isAsync ? t : getNewType(t, encoding));
  }

  // Construct new op with the new encoding
  Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(),
                                    newArgs, newTypes, op->getAttrs());

  // Cast the results back to the original layout
  for (size_t i = 0; i < op->getNumResults(); i++) {
    Value newResult = newOp->getResult(i);
    if (newTypes[i] != op->getResultTypes()[i]) {
      newResult = builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), op->getResult(i).getType(), newResult);
    }
    op->getResult(i).replaceAllUsesWith(newResult);
  }
  op->erase();
}

ttg::LinearEncodingAttr
createInThreadTransposedEncoding(ArrayRef<int64_t> shape,
                                 ttg::BlockedEncodingAttr srcEncoding) {
  auto srcLL = srcEncoding.toLinearLayout(shape);
  SmallVector<unsigned> newInRegOrder(srcEncoding.getOrder());
  int rank = shape.size();
  std::swap(newInRegOrder[rank - 2], newInRegOrder[rank - 1]);

  // Make in-register transposed tile
  auto ctx = srcEncoding.getContext();
  auto regDimName = StringAttr::get(ctx, "register");
  auto inRegTransposeTile = tt::identityStandardND(
      regDimName, srcEncoding.getSizePerThread(), newInRegOrder);
  // make sure basis in same order as in srcLayout
  SmallVector<StringAttr> outDimNames(srcLL.getOutDimNames());
  inRegTransposeTile = inRegTransposeTile.transposeOuts(outDimNames);

  // Copy original bases, and replace register tile with transposed one
  tt::LinearLayout::BasesT bases = srcLL.getBases();
  auto &regBase = *bases.find(regDimName);
  int regsTransposed = inRegTransposeTile.getInDimSizeLog2(regDimName);
  for (int i = 0; i < regsTransposed; ++i)
    regBase.second[i] = inRegTransposeTile.getBasis(regDimName, i);

  tt::LinearLayout transposedLL(bases, SmallVector<StringAttr>(outDimNames));
  return ttg::LinearEncodingAttr::get(ctx, transposedLL);
}

template <typename LocalMemOp>
void transposeInRegsitersBeforeStoreInLocalMemory(LocalMemOp alloc) {
  auto operand = alloc.getSrc();
  OpBuilder builder(alloc);

  auto data = alloc.getSrc();
  // local alloc has optional src
  // if it is not provided, nothing to do
  if (!data)
    return;
  auto operandType = data.getType();
  auto operandEncoding =
      cast<ttg::BlockedEncodingAttr>(operandType.getEncoding());
  auto transposedEncoding =
      createInThreadTransposedEncoding(operandType.getShape(), operandEncoding);
  auto newType = getNewType(operand.getType(), transposedEncoding);
  auto inThreadTransposed =
      builder.create<ttg::ConvertLayoutOp>(alloc->getLoc(), newType, operand);
  alloc.setOperand(0, inThreadTransposed);
}

template <typename LocalMemOp> void changeSharedEncoding(LocalMemOp alloc) {
  auto originalType = cast<ttg::MemDescType>(alloc.getResult().getType());
  auto sharedEnc =
      cast<ttg::SwizzledSharedEncodingAttr>(originalType.getEncoding());
  auto ctx = sharedEnc.getContext();
  auto sharedVec = sharedEnc.getVec();
  auto perPhase = sharedEnc.getPerPhase();
  auto maxPhase = sharedEnc.getMaxPhase();
  auto order = sharedEnc.getOrder();
  auto ctaLayout = sharedEnc.getCTALayout();

  // TODO replace SwizzledSharedEncodingAttr with special swizzling pattern
  auto newSharedEnc = ttg::SwizzledSharedEncodingAttr::get(
      ctx, sharedVec, perPhase, maxPhase, order, ctaLayout);
  auto newType = ttg::MemDescType::get(
      originalType.getShape(), originalType.getElementType(), newSharedEnc,
      originalType.getMemorySpace(), originalType.getMutableMemory());

  alloc.getResult().setType(newType);
}

/// For a given value return all operations that define it.
///
/// If val is a result of operation, return definingOp.
/// If val is a result of some control flow operation or block argument,
/// traverse control flow instructions.
FailureOr<SmallVector<mlir::Operation *>> getNonCFDefOps(Value val) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    Block *block = blockArg.getOwner();

    // Get parent operation (e.g., scf.for, scf.if, scf.while)
    Operation *parentOp = block->getParentOp();
    if (!parentOp) {
      LDBG("block without parent op, can not analyze further");
      return failure();
    }

    // If block belongs to a function, stop tracking (function arguments)
    if (isa<triton::FuncOp>(parentOp)) {
      LDBG("can not traverse def-use chains, found function argument");
      return failure();
    }

    int argIdx = blockArg.getArgNumber();

    // Handle `scf.for`
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // Continue traversing this op, even if it is visited
      // It could be visited with different arg idx
      int iterArgIdx = argIdx - 1; // Skip induction variable
      if (iterArgIdx >= 0) {
        Value yieldVal =
            forOp.getBody()->getTerminator()->getOperand(iterArgIdx);
        // look inside loop and outside of a loop
        auto inLoop = getNonCFDefOps(yieldVal);
        auto outLoop = getNonCFDefOps(forOp.getOperand(iterArgIdx));
        if (failed(inLoop) || failed(outLoop))
          return failure();

        SmallVector<mlir::Operation *> totalOps(std::move(inLoop.value()));
        totalOps.append(outLoop.value());
        return totalOps;
      } else {
        // Induction variable
        return getNonCFDefOps(forOp.getOperand(0));
      }
    }

    // Handle `scf.if`
    if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
      auto thenYield = ifOp.thenYield();
      auto elseYield = ifOp.elseYield();

      // Track all possible yielded values from then/else blocks
      SmallVector<mlir::Operation *> totalOps;
      if (thenYield) {
        auto ops = getNonCFDefOps(thenYield->getOperand(argIdx));
        if (failed(ops))
          return failure();
        totalOps.append(ops.value());
      }
      if (elseYield) {
        auto ops = getNonCFDefOps(elseYield->getOperand(argIdx));
        if (failed(ops))
          return failure();
        totalOps.append(ops.value());
      }
      return totalOps;
    }

    // Handle `scf.while`
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      auto terminator = whileOp.getBefore().front().getTerminator();
      return getNonCFDefOps(terminator->getOperand(argIdx));
    }

    if (isa<RegionBranchOpInterface>(parentOp)) {
      // Deal with the case that convert_layout intakes from scf.if, etc.
      llvm::SmallVector<scf::YieldOp> yieldOps;
      parentOp->walk([&](Operation *op) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
          yieldOps.push_back(yieldOp);
        }
      });

      SmallVector<mlir::Operation *> totalOps;
      for (auto yieldOp : yieldOps) {
        auto ops = getNonCFDefOps(yieldOp->getOperand(argIdx));
        if (failed(ops))
          return failure();
        totalOps.append(ops.value());
      }
      return totalOps;
    }
    assert(false && "unexpected control flow operation");
  } else {
    return SmallVector<Operation *>{val.getDefiningOp()};
  }
}

/// For a given value return all operations that uses it.
///
/// Traverses control flow instructions forward.
FailureOr<SmallVector<mlir::Operation *>> getNonCFUserOps(Value val) {
  SmallVector<mlir::Operation *> users;
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    if (isa<triton::ReturnOp>(user)) {
      LDBG("Reached return from function");
      return failure();
    }
    if (isa<scf::YieldOp>(user)) {
      auto opIdx = use.getOperandNumber();
      auto cfSearch = getNonCFUserOps(user->getParentOp()->getResult(opIdx));
      if (failed(cfSearch)) {
        LDBG("Failed nested forward analysis");
        return failure();
      }
      users.append(cfSearch.value());
    } else {
      users.push_back(user);
    }
  }
  return users;
}

/// Look for defining operation, hopping over control flow.
///
/// Gather all operations of type T within one def-use hop from val,
/// control flow constructions are not considered as an operations.
/// \returns true on success, false if analysis failed
template <typename Op>
FailureOr<SmallVector<Op>> findAllDefiningOps(Value val) {
  auto candidates = getNonCFDefOps(val);
  if (failed(candidates))
    return failure();
  SmallVector<Op> result;
  for (auto candidate : candidates.value()) {
    if (auto typedOp = dyn_cast<Op>(candidate))
      result.push_back(typedOp);
  }
  return result;
}

/// Look for all operations with one of OpTy types in def-use chains in both
/// forward and backward directions.
///
/// Traversal goes through control flow operations and and stops at non OpTy
/// operation. For example: findAllDefUseOps<local_load, mem_subview,
/// local_store>(dot_operand)
///
///                                                    ----------------->
///                                                    local_store | traversed
/// global_load -> local_store -> mem_subview -> local_load -> dot
///                 traversed      traversed      traversed
///
/// \returns true on success, false if analysis failed
template <typename... OpTy>
FailureOr<SmallVector<Operation *>>
findAllDefUseOps(Operation *op, SetVector<mlir::Operation *> &visited) {
  // breadth-first search for reachable opeations of given types
  SmallVector<Operation *> foundNetwork;
  SmallVector<Operation *> traversalStep{op};
  while (!traversalStep.empty()) {
    SmallVector<Operation *> nextTraversalStep;
    for (auto candidate : traversalStep) {
      if (visited.contains(candidate) || !(isa<OpTy>(candidate) || ...))
        continue;
      visited.insert(candidate);
      foundNetwork.push_back(candidate);

      // Look backward
      if (candidate->getNumOperands() > 0) {
        auto backwardSearch = getNonCFDefOps(candidate->getOperand(0));
        if (failed(backwardSearch))
          return failure();
        nextTraversalStep.append(backwardSearch.value());
      }

      // Look forward
      if (candidate->getNumResults() > 0) {
        auto forwardSearch = getNonCFUserOps(candidate->getResult(0));
        if (failed(forwardSearch))
          return failure();
        nextTraversalStep.append(forwardSearch.value());
      }
    }
    traversalStep = std::move(nextTraversalStep);
  }
  return foundNetwork;
}

/// Structure describes operations involved in local_alloc->local_load pattern
struct loadStoreLoadPatternComponents {
  SmallVector<tt::LoadOp> globalLoads;
  SmallVector<ttg::LocalAllocOp> localAllocs;
  SmallVector<ttg::LocalStoreOp> localStores;
  SmallVector<ttg::MemDescSubviewOp> subviews;
  SmallVector<ttg::LocalLoadOp> localLoads;
};

llvm::FailureOr<loadStoreLoadPatternComponents>
matchThreadRakePattern(Value operand) {
  // TODO implement general heuristic,
  // analyzing local load/store vectorization and estimating bank conflicts
  auto opTensorTy = cast<RankedTensorType>(operand.getType());
  auto opEnc = opTensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  if (!opDotOpEnc)
    return failure();

  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  // TODO: support wmma
  if (!isa<ttg::AMDMfmaEncodingAttr, ttg::AMDWmmaEncodingAttr>(
          opDotOpEnc.getParent())) {
    LDBG("Operand's parent encoding is not MFMA");
    return failure();
  }

  // Find nearest local_load
  loadStoreLoadPatternComponents pattern;

  auto localLoadSearch = findAllDefiningOps<ttg::LocalLoadOp>(operand);
  if (failed(localLoadSearch)) {
    LDBG("Failed to traverse local loads");
    return failure();
  }

  if (localLoadSearch.value().size() == 0) {
    LDBG("Did not find local load operation");
    return failure();
  }

  SetVector<Operation *> visited;
  for (auto lLoad : localLoadSearch.value()) {
    // find local_alloc, local_store, local_load and ttg.memdesc_subview
    // operations
    auto sharedMemSearch =
        findAllDefUseOps<ttg::LocalAllocOp, ttg::LocalStoreOp, ttg::LocalLoadOp,
                         ttg::MemDescSubviewOp>(lLoad, visited);
    if (failed(sharedMemSearch)) {
      LDBG("Failed to traverse shared memmory operation network");
      return failure();
    }
    // Fill pattern description
    for (Operation *op : sharedMemSearch.value()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
        pattern.localAllocs.push_back(alloc);
      if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(op))
        pattern.localLoads.push_back(localLoad);
      if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
        pattern.localStores.push_back(store);
      if (auto view = dyn_cast<ttg::MemDescSubviewOp>(op))
        pattern.subviews.push_back(view);
    }
  }

  if (pattern.localAllocs.empty()) {
    LDBG("Did not find local alloc operations");
    return failure();
  }

  SmallVector<Value> loadCandidates;
  for (auto lAlloc : pattern.localAllocs) {
    if (lAlloc.getSrc())
      loadCandidates.push_back(lAlloc.getSrc());
  }
  for (auto lStore : pattern.localStores)
    loadCandidates.push_back(lStore.getSrc());

  for (auto loadCandidate : loadCandidates) {
    auto loadedEnc =
        cast<RankedTensorType>(loadCandidate.getType()).getEncoding();
    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(loadedEnc);
    if (!blockedEnc)
      return failure();
    auto order = blockedEnc.getOrder();
    if (order[0] != kDimNum) {
      return failure();
    }
    auto globalLoadSearch = findAllDefiningOps<triton::LoadOp>(loadCandidate);
    if (failed(globalLoadSearch)) {
      LDBG("Failed to traverse path to global loads");
      return failure();
    }
    pattern.globalLoads = std::move(globalLoadSearch.value());
  }
  if (pattern.globalLoads.empty()) {
    LDBG("Did not find global load operation");
    return failure();
  }

  return pattern;
}

ttg::BlockedEncodingAttr
getThreadRakedBlockedEnc(Value dotOperand, tt::LoadOp load, ModuleOp &mod) {
  // get the K dim according to dotOp operand's index
  auto tensorTy = cast<RankedTensorType>(dotOperand.getType());
  auto shape = tensorTy.getShape();
  auto opEnc = tensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  // get the current blocked encoding
  auto loadResult = load.getResult();
  auto loadEnc = cast<RankedTensorType>(loadResult.getType()).getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(loadEnc);
  // compute the sizePerThread for the new encoding
  auto sizePerThread = blockedEnc.getSizePerThread();
  auto elemsPerIter = product(sizePerThread);
  auto elemsTotal = blockedEnc.getTotalElemsPerThread(shape, tensorTy);
  // we need to know how many iteration each thread will load
  LDBG("elemsPerIter = " << elemsPerIter << "; elemsTotal = " << elemsTotal);
  auto numMaxIters = elemsTotal / elemsPerIter;
  auto bitwidth = tensorTy.getElementType().getIntOrFloatBitWidth();
  // LDBG("bitwidth = " << bitwidth);
  // Current the widest is set to ds_write_b64
  auto newKOuterDim = std::min(numMaxIters, 64 / bitwidth);
  LDBG("Choose the minimum of numIters: " << numMaxIters << " and numDtype: "
                                          << 64 / bitwidth);
  SmallVector<unsigned> newSizePerThread(sizePerThread);
  newSizePerThread[kDimNum] = newKOuterDim;

  // return the new blocked encoding
  auto order = blockedEnc.getOrder();
  int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  return ttg::BlockedEncodingAttr::get(mod.getContext(), shape,
                                       newSizePerThread, order, numWarps,
                                       threadsPerWarp, numCTAs);
}

} // namespace

class TritonAMDGPUInThreadTransposePass
    : public TritonAMDGPUInThreadTransposeBase<
          TritonAMDGPUInThreadTransposePass> {

public:
  TritonAMDGPUInThreadTransposePass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](tt::DotOp dotOp) {
      LDBG("DotOp under inspection: " << dotOp);
      auto mod = dotOp->getParentOfType<ModuleOp>();

      auto tryToConvertToThreadRaked = [&](Value operand) {
        LDBG("Consider " << operand);
        // Dot operand
        auto matchResult = matchThreadRakePattern(operand);
        if (!llvm::succeeded(matchResult)) {
          LDBG("operand is K-inner and nothing to be done");
          return;
        }
        auto pattern = matchResult.value();
        LDBG("operand is K-outer");
        for (auto gLoad : pattern.globalLoads) {
          auto newBlockedEnc = getThreadRakedBlockedEnc(operand, gLoad, mod);
          LDBG("operand newBlockedEnc = " << newBlockedEnc);
          convertLayout(newBlockedEnc, (Operation *)gLoad);
        }

        for (auto lAlloc : pattern.localAllocs) {
          transposeInRegsitersBeforeStoreInLocalMemory(lAlloc);
          changeSharedEncoding(lAlloc);
        }
        for (auto lStore : pattern.localStores) {
          transposeInRegsitersBeforeStoreInLocalMemory(lStore);
        }
        for (auto view : pattern.subviews) {
          changeSharedEncoding(view);
        }
      };
      // Check opA
      tryToConvertToThreadRaked(dotOp.getA());

      // Check opB
      tryToConvertToThreadRaked(dotOp.getB());
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUInThreadTransposePass() {
  return std::make_unique<TritonAMDGPUInThreadTransposePass>();
}
