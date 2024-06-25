#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <numeric>

using namespace mlir;
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_INSERTAMDSCHEDULERBARRIERS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct InsertAMDSchedulerBarriers
    : public mlir::triton::impl::InsertAMDSchedulerBarriersBase<
          InsertAMDSchedulerBarriers> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod.walk([&](scf::ForOp forOp) -> void {
      auto ctx = forOp.getContext();
      auto loc = forOp.getLoc();
      mlir::OpBuilder opbuilder(ctx);

      // From https://llvm.org/docs/AMDGPUUsage.html
      // llvm.amdgcn.sched_barrier
      // 0x0000: No instructions may be scheduled across sched_barrier
      uint32_t permittingMask = 0x0;

      opbuilder.setInsertionPoint(forOp);
      opbuilder.create<mlir::ROCDL::SchedBarrier>(loc, permittingMask);

      opbuilder.setInsertionPointAfter(forOp);
      opbuilder.create<mlir::ROCDL::SchedBarrier>(loc, permittingMask);
    });
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createInsertAMDSchedulerBarriersPass() {
  return std::make_unique<InsertAMDSchedulerBarriers>();
}

} // namespace mlir::triton
