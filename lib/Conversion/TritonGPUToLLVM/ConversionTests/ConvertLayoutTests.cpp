#include "ConvertLayoutTestsLib.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include <gtest/gtest.h>

#include "../ConvertLayoutOpToLLVM.h"

constexpr int threadIdPos = 0;
constexpr int waveSize = 64;
constexpr int ldsSize = 65536;
constexpr int tensorSize = 1024;

mlir::ModuleOp createTestModule(mlir::MLIRContext *ctx, const char *functionName) {
  ctx->getOrLoadDialect<mlir::triton::TritonDialect>();
  ctx->getOrLoadDialect<mlir::triton::gpu::TritonGPUDialect>();
  ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
  ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlir::OpBuilder builder(ctx);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<int64_t> tensorShape{16, 16};
  auto elementType = builder.getF32Type();
  unsigned versionMajor = 2;
  unsigned versionMinor = 0;
  llvm::SmallVector<unsigned> warpsPerCTA{1, 1};
  unsigned mDim = 4;
  unsigned nDim = 4;
  bool isTransposed = false;
  auto srcLayout = mlir::triton::gpu::MfmaEncodingAttr::get(ctx, versionMajor, versionMinor, warpsPerCTA, mDim, nDim, isTransposed);
  auto srcType = mlir::RankedTensorType::get(tensorShape, elementType, srcLayout);

  auto mlirModule = mlir::ModuleOp::create(loc);
  mlirModule->setAttr("triton_gpu.num-warps", builder.getI32IntegerAttr(1));
  mlirModule->setAttr("triton_gpu.threads-per-warp", builder.getI32IntegerAttr(64));
  mlirModule->setAttr("triton_gpu.num-ctas", builder.getI32IntegerAttr(1));

  auto threadIdTy = builder.getI32Type();
  auto offsetType = builder.getI32Type();
  auto offsetPtrTy = mlir::LLVM::LLVMPointerType::get(ctx, offsetType, 3);
  std::vector<mlir::Type> inTypes{threadIdTy, srcType};
  auto mlirFuncTy = builder.getFunctionType(inTypes, {});

  auto mlirFunc = builder.create<mlir::triton::FuncOp>(loc, functionName, mlirFuncTy);
  mlirModule.push_back(mlirFunc);
  llvm::errs() << "module created:\n" << mlirModule << "\n";

  auto block = mlirFunc.addEntryBlock();

  // call function for index compute and generate llvmIR
  unsigned vecSize = 1;
  unsigned perPhase = 1;
  unsigned maxPhase = 1;
  llvm::SmallVector<unsigned> order{1, 0};
  llvm::SmallVector<unsigned> CTAsPerCGA{1, 1};
  llvm::SmallVector<unsigned> CTASplitNum{1, 1};
  llvm::SmallVector<unsigned> CTAOrder{1, 0};
  auto CTALayout = mlir::triton::gpu::CTALayoutAttr::get(ctx, CTAsPerCGA, CTASplitNum, CTAOrder);
  bool hasLeadingOffset = false;
  auto dstLayout = mlir::triton::gpu::SharedEncodingAttr::get(ctx, vecSize, perPhase, maxPhase, order, CTALayout, hasLeadingOffset);
  auto dstType = mlir::RankedTensorType::get(tensorShape, elementType, dstLayout);

  auto elementAttr = builder.getFloatAttr(elementType, 0);
  auto denseAttr = mlir::DenseElementsAttr::get(srcType, elementAttr);
  auto inputTensor = block->getArgument(1);

  auto convertLayout = builder.create<mlir::triton::gpu::ConvertLayoutOp>(loc, dstType, inputTensor);
  block->push_back(convertLayout);

  auto returnOp = builder.create<mlir::triton::ReturnOp>(loc);
  block->push_back(returnOp);

  llvm::errs() << "module filled:\n" << mlirModule << "\n";

  return mlirModule;
}

void convertTTGToLLVM(mlir::ModuleOp &mod) {
  auto ctx = mod.getContext();
  mlir::PassManager pm(ctx);
  pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(90, mlir::triton::Target::ROCDL, nullptr));
  pm.addPass(mlir::createCanonicalizerPass());

  if (failed(pm.run(mod))) {
    llvm::errs() << "Pass execution failed\n";
    return;
  }
}

void replaceGPUSpecificEntities(mlir::ModuleOp &mod) {
  class EliminateThreadId : public mlir::RewritePattern {

  public:
    EliminateThreadId(mlir::MLIRContext *context)
        : mlir::RewritePattern(
              mlir::ROCDL::ThreadIdXOp::getOperationName(), 1,
              context) {}

    LogicalResult
    matchAndRewrite(mlir::Operation *op,
                    mlir::PatternRewriter &rewriter) const override {
      auto threadIdx = cast<mlir::ROCDL::ThreadIdXOp>(op);
      auto func = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
      auto threadIdArgument = func.getArgument(threadIdPos);
      rewriter.replaceOp(op, {threadIdArgument});
      return mlir::success();
    }
  };

  auto ctx = mod.getContext();
  mlir::OpBuilder builder(ctx);
  mlir::Location loc = mod.getLoc();

  RewritePatternSet patterns(ctx);
  patterns.add<EliminateThreadId>(ctx);

  SmallVector<Operation *> candidates;
  mod.walk([&candidates](mlir::ROCDL::ThreadIdXOp op) {
    candidates.push_back(op);
  });
  if (mlir::applyOpPatternsAndFold(candidates, std::move(patterns)).failed()) {
    llvm::errs() << "failed to eliminate threadId operations\n";
    return;
  }

  mod.walk([&builder, &loc](mlir::LLVM::GlobalOp op) {
    if (op.getNameAttr().str() == "global_smem") {
      auto ldsType = builder.getType<mlir::LLVM::LLVMArrayType>(builder.getIntegerType(8), ldsSize);
      op.setGlobalType(ldsType);
      op.setLinkage(mlir::LLVM::linkage::Linkage::Private);
    }
  });
}

std::unique_ptr<llvm::Module> convertMLIRToLLVMIR(mlir::ModuleOp &mod, llvm::LLVMContext &llvmCtx) {
  auto mlirCtx = mod.getContext();

  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  mlirCtx->appendDialectRegistry(registry);

  auto llvmModule = mlir::translateModuleToLLVMIR(mod, llvmCtx);
  assert(llvmModule.get() != nullptr);
  return std::move(llvmModule);
}

void RunInterpreter(std::unique_ptr<llvm::Module> mod, const char *functionName) {
  // interpret llvm ir
  auto llvmModulePtr = mod.get();
  std::string errorMsg;
  llvm::EngineBuilder interpreterBuilder(std::move(mod));
  interpreterBuilder.setEngineKind(llvm::EngineKind::Kind::Interpreter);
  interpreterBuilder.setErrorStr(&errorMsg);

  std::unique_ptr<llvm::ExecutionEngine> EE(interpreterBuilder.create());
  if (!EE) {
    if (!errorMsg.empty())
      llvm::errs() << "error creating EE: " << errorMsg << "\n";
    else
      llvm::errs() << "unknown error creating EE!\n";
    exit(1);
  }

  llvm::Function *llvmTestFunction = llvmModulePtr->getFunction(functionName);

  std::vector<llvm::GenericValue> inputs(2);
  // todo adjust to llvm func signature
  int elemsPerThread = 64;
  inputs[1].AggregateVal.resize(elemsPerThread);

  // Initizlize LDS
  auto *ldsMem = reinterpret_cast<char *>(EE->getPointerToGlobalIfAvailable("global_smem"));
  std::fill(ldsMem, ldsMem + ldsSize, 0xcc);

  for (int threadId = 0; threadId < waveSize; ++threadId) {
    inputs[0].IntVal = llvm::APInt(32, threadId, true);

    for (int i = 0; i < elemsPerThread; ++i)
      inputs[1].AggregateVal[i].FloatVal = threadId * elemsPerThread + i;
    auto result = EE->runFunction(llvmTestFunction, inputs);
    llvm::outs() << "mem contents:\n";
    for (int i = 0; i < tensorSize; ++i)
      if (ldsMem[i] == 0xcc)
        llvm::outs() << "_";
      else
        llvm::outs() << "*";
    llvm::outs() << "\n";
  }
}

TEST(LayoutConversions, MFMAtoShared) {
  const std::string functionName = "test_func";
  mlir::MLIRContext mlirCtx;
  auto mlirModule = createTestModule(&mlirCtx, functionName.c_str());

  convertTTGToLLVM(mlirModule);

  replaceGPUSpecificEntities(mlirModule);

  llvm::LLVMContext llvmCtx;

  std::unique_ptr<llvm::Module> llvmModule = convertMLIRToLLVMIR(mlirModule, llvmCtx);

  RunInterpreter(std::move(llvmModule), functionName.c_str());
  SUCCEED();
}
