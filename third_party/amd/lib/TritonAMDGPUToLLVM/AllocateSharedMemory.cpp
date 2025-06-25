#include "TritonAMDGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATEAMDGPUSHAREDMEMORY
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

constexpr int globalPtrBitWidth = 64;
constexpr int bufferPtrBitWidth = 32;

static unsigned getBitwidth(RankedTensorType ty) {
  auto ptrTy = dyn_cast<PointerType>(ty.getElementType());
  if (ptrTy) {
    // TODO check if buffer load ptr is 32 bits
    return ptrTy.getAddressSpace() == 1 ? globalPtrBitWidth : bufferPtrBitWidth;
  }
  return std::max(ty.getElementTypeBitWidth(), 8u);
}

unsigned allocationAnalysisScratchSizeFn(Operation *op) {
  if (auto cvtLayout = dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    if (!cvtNeedsSharedMemory(srcTy, dstTy))
      return 0;
    auto scratchConfig = getScratchConfigForCvt(srcTy, dstTy);
    auto elems = getNumScratchElements(scratchConfig.paddedRepShape);
    return elems * getBitwidth(srcTy) / 8;
  }
  return defaultAllocationAnalysisScratchSizeFn(op);
}

struct AllocateAMDGPUSharedMemory
    : public mlir::triton::impl::AllocateAMDGPUSharedMemoryBase<
          AllocateAMDGPUSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    ModuleAllocation allocation(mod, allocationAnalysisScratchSizeFn);

    mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
      auto *funcAllocation = allocation.getFuncData(funcOp);
      funcOp.walk([&](Operation *op) {
        auto oBufferId = funcAllocation->getBufferId(op);
        int offset = -1;
        if (oBufferId != Allocation::InvalidBufferId)
          offset = funcAllocation->getOffset(oBufferId);
        else if (op->getNumResults() == 1) {
          Value value = op->getResult(0);
          auto vBufferId = funcAllocation->getBufferId(value);
          if (vBufferId != Allocation::InvalidBufferId)
            offset = funcAllocation->getOffset(vBufferId);
        }
        if (offset == -1)
          return;
        op->setAttr("allocation.offset",
                    IntegerAttr::get(IntegerType::get(ctx, 32), offset));
      });
      return WalkResult::skip();
    });
    mod->setAttr("ttg.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        allocation.getSharedMemorySize()));
  }
};
} // namespace
