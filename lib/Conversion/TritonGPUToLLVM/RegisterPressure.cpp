#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/Liveness.h"

#include <numeric>

#include "mlir/Pass/Pass.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {

namespace triton {

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

// filter out type info with 
// %s/reg_cost = \(\d*\) : i32}.*$/register_cost = \1/

class TritonGPURegisterPressurePass
    : public TritonGPURegisterPressureBase<
          TritonGPURegisterPressurePass> {
public:
  TritonGPURegisterPressurePass() = default;

  int estimateTypeByteSize(mlir::Type type) {
    int numElems = 0;
    if (auto vectorTy = type.dyn_cast<VectorType>())
      return vectorTy.getNumElements() * estimateTypeByteSize(vectorTy.getElementType());
    if (auto structTy = type.dyn_cast<LLVM::LLVMStructType>()){
      int totalSize = 0;
      for (auto subTy: structTy.getBody())
        totalSize += estimateTypeByteSize(subTy);
      return totalSize;
    }
    if (auto ptrTy = type.dyn_cast<mlir::triton::PointerType>())
      return ptrTy.getAddressSpace() == 3 ? 4 : 8;
    else
      return type.getIntOrFloatBitWidth();
  }

  int estimateTensorRegisterUsage(mlir::RankedTensorType type) {
    auto encoding = type.getEncoding();
    if (encoding.isa<mlir::triton::gpu::SharedEncodingAttr>())
      return 0;
    int elemsPerThread = 0;
    if (auto dotOpLayout = encoding.dyn_cast<mlir::triton::gpu::DotOperandEncodingAttr>()) {
      if (auto mfmaParent = dotOpLayout.getParent().dyn_cast<mlir::triton::gpu::MfmaEncodingAttr>()) {
        auto rep = dotOpLayout.getMFMARep(type.getShape());
        elemsPerThread = rep[0] * rep[1] * dotOpLayout.getKWidth();
      } else {
        assert(false);
      }
    } else {
      auto elemsShape = mlir::triton::gpu::getElemsPerThread(type);
      elemsPerThread = std::reduce(elemsShape.begin(), elemsShape.end(), 1, std::multiplies<unsigned>());
    }
    auto elemTypeSize = estimateTypeByteSize(type.getElementType());
    return std::max(1, elemsPerThread * elemTypeSize / 4);
  }

  int estimateScalarRegisterUsage(mlir::Type type) {
    return estimateTypeByteSize(type) / 4;
  }

  bool isTTGIR(ModuleOp m) {
    auto &ops = m.getBody()->getOperations();
    auto pred = [](const mlir::Operation &op){return isa<const mlir::triton::FuncOp>(op);};
    return std::any_of(ops.begin(), ops.end(), pred);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    bool isTTG = isTTGIR(m);

    Liveness liveness(m);

    std::map<mlir::Operation *, int> registerUsage;

    m.walk([&](mlir::Operation *op) {
      for (auto res: op->getResults()) {
        if (auto t = res.getType().dyn_cast<RankedTensorType>()) {
          auto liveOperations = liveness.resolveLiveness(res);
          int registers = isTTG ? estimateTensorRegisterUsage(t) : estimateScalarRegisterUsage(t);
          for (auto liveOps: liveOperations)
            registerUsage[liveOps] += registers;
        }
      }
      // gather register usage
    }
    );
    for (auto item: registerUsage)
      item.first->setDiscardableAttr("ttg.reg_cost", IntegerAttr::get(mlir::IntegerType::get(context, 32), item.second));
  }
};

std::unique_ptr<Pass> createTritonGPURegisterPressurePass() {
  return std::make_unique<TritonGPURegisterPressurePass>();
}

} // namespace triton

} // namespace mlir
