#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/Liveness.h"

#include <numeric>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"


namespace mlir {

class TritonGPURegisterPressurePass
    : public TritonGPURegisterPressureBase<
          TritonGPURegisterPressurePass> {
public:
  TritonGPURegisterPressurePass() = default;

  int estimateRegisterUsage(mlir::RankedTensorType type) {
    auto shape = type.getShape();
    auto enc = type.getEncoding();
    int redundancy = 1;
    if (auto mfmaLayout = enc.dyn_cast<triton::gpu::MfmaEncodingAttr>()){
      if (mfmaLayout.getMDim() == 4 && mfmaLayout.getNDim() == 4)
        redundancy = 16;
    }
    if (auto dotLayout = enc.dyn_cast<triton::gpu::DotOperandEncodingAttr>()){
      if (auto mfmaLayout = dotLayout.getParent().dyn_cast<triton::gpu::MfmaEncodingAttr>()) {
        if (mfmaLayout.getMDim() == 4 && dotLayout.getOpIdx() == 0)
          redundancy = 16;
        if (mfmaLayout.getNDim() == 4 && dotLayout.getOpIdx() == 1)
          redundancy = 16;
      }
    }
    int tSize = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int>() );
    int waveSize = 64;
    return tSize * redundancy / waveSize;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    Liveness liveness(m);

    std::map<mlir::Operation *, int> registerUsage;

    m.walk([&](mlir::Operation *op) {
      for (auto res: op->getResults()) {
        if (auto t = res.getType().dyn_cast<RankedTensorType>()) {
          auto liveOperations = liveness.resolveLiveness(res);
          int registers = estimateRegisterUsage(t);
          for (auto liveOps: liveOperations)
            registerUsage[liveOps] += registers;
        }
      }
      // gather register usage
    }
    );
    for (auto item: registerUsage) {
      llvm::errs() << *item.first << " cost: " << item.second << "\n";
    }
  }
};

std::unique_ptr<Pass> createTritonGPURegisterPressurePass() {
  return std::make_unique<TritonGPURegisterPressurePass>();
}

} // namespace mlir
