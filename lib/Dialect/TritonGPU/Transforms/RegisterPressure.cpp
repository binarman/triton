#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/Liveness.h"

#include <numeric>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// filter out type info with 
// %s/register_cost = \(\d*\) : i32}.*$/register_cost = \1/

namespace mlir {

class TritonGPURegisterPressurePass
    : public TritonGPURegisterPressureBase<
          TritonGPURegisterPressurePass> {
public:
  TritonGPURegisterPressurePass() = default;

  int estimateTensorRegisterUsage(mlir::RankedTensorType type) {
    auto encoding = type.getEncoding();
    if (encoding.isa<triton::gpu::SharedEncodingAttr>())
      return 0;
    int elemsPerThread = 0;
    if (auto dotOpLayout = encoding.dyn_cast<triton::gpu::DotOperandEncodingAttr>()) {
      if (auto mfmaParent = dotOpLayout.getParent().dyn_cast<triton::gpu::MfmaEncodingAttr>()) {
        auto rep = dotOpLayout.getMFMARep(type.getShape());
        elemsPerThread = rep[0] * rep[1] * dotOpLayout.getKWidth();
      } else {
        assert(false);
      }
    } else {
      auto elemsShape = triton::gpu::getElemsPerThread(type);
      elemsPerThread = std::reduce(elemsShape.begin(), elemsShape.end(), 1, std::multiplies<unsigned>());
    }
    auto elemType = type.getElementType();
    auto scalarBitWidth = 0;
    if (auto ptrTy = elemType.dyn_cast<mlir::triton::PointerType>())
      scalarBitWidth = ptrTy.getAddressSpace() == 3 ? 32 : 64;
    else
      scalarBitWidth = elemType.getIntOrFloatBitWidth();
    return std::max(1, elemsPerThread * scalarBitWidth / 32);
  }

  bool isTTGIR(ModuleOp m) {
    auto &ops = m.getBody()->getOperations();
    return std::any_of(ops.begin(), ops.end(), [](auto op){return op.isa<triton::FuncOp>()});
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
          int registers = isTTG ? estimateTensorRegisterUsage(t);
          for (auto liveOps: liveOperations)
            registerUsage[liveOps] += registers;
        }
      }
      // gather register usage
    }
    );
    for (auto item: registerUsage)
      item.first->setDiscardableAttr("ttg.register_cost", IntegerAttr::get(mlir::IntegerType::get(context, 32), item.second));
  }
};

std::unique_ptr<Pass> createTritonGPURegisterPressurePass() {
  return std::make_unique<TritonGPURegisterPressurePass>();
}

} // namespace mlir
