#include "Analysis/AMDGPUAllocation.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {

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

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy) {
  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return 0;
  auto scratchConfig = getScratchConfigForCvt(srcTy, dstTy);
  auto elems = getNumScratchElements(scratchConfig.paddedRepShape);
  return elems * getBitwidth(srcTy) / 8;
}

unsigned allocationAnalysisScratchSizeFn(Operation *op) {
  if (auto cvtLayout = dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    return getConvertLayoutScratchInBytes(srcTy, dstTy);
  }
  return defaultAllocationAnalysisScratchSizeFn(op);
}
} // namespace mlir::triton::AMD
