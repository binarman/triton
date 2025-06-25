#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::AMD {

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy);

unsigned allocationAnalysisScratchSizeFn(Operation *op);

} // namespace mlir::triton::AMD
