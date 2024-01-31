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
constexpr int ldsSize = 65536;

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

TEST(LayoutConversions, MFMAtoShared) {
  // create mlir module and function
  auto ctx = std::make_unique<mlir::MLIRContext>();
  ctx->getOrLoadDialect<mlir::triton::TritonDialect>();
  ctx->getOrLoadDialect<mlir::triton::gpu::TritonGPUDialect>();
  ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
  ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
  // ctx->getOrLoadDialect<mlir::index::IndexDialect>();
  ctx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlir::OpBuilder builder(ctx.get());
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<int64_t> tensorShape{16, 16};
  auto elementType = builder.getF32Type();
  unsigned versionMajor = 2;
  unsigned versionMinor = 0;
  llvm::SmallVector<unsigned> warpsPerCTA{1, 1};
  unsigned mDim = 4;
  unsigned nDim = 4;
  bool isTransposed = false;
  auto srcLayout = mlir::triton::gpu::MfmaEncodingAttr::get(ctx.get(), versionMajor, versionMinor, warpsPerCTA, mDim, nDim, isTransposed);
  auto srcType = mlir::RankedTensorType::get(tensorShape, elementType, srcLayout);

  auto mlirModule = mlir::ModuleOp::create(loc);
  mlirModule->setAttr("triton_gpu.num-warps", builder.getI32IntegerAttr(1));
  mlirModule->setAttr("triton_gpu.threads-per-warp", builder.getI32IntegerAttr(64));
  mlirModule->setAttr("triton_gpu.num-ctas", builder.getI32IntegerAttr(1));

  auto threadIdTy = builder.getI32Type();
  auto offsetType = builder.getI32Type();
  auto offsetPtrTy = mlir::LLVM::LLVMPointerType::get(ctx.get(), offsetType, 3);
  std::vector<mlir::Type> inTypes{threadIdTy, srcType};
  // auto mlirFuncTy = mlir::LLVM::LLVMFunctionType::get(offsetType, inTypes);
  auto mlirFuncTy = builder.getFunctionType(inTypes, {});

  const std::string functionName = "test_func";
  // auto mlirFunc = builder.create<mlir::LLVM::LLVMFuncOp>(loc, functionName, mlirFuncTy);
  auto mlirFunc = builder.create<mlir::triton::FuncOp>(loc, functionName, mlirFuncTy);
  mlirModule.push_back(mlirFunc);

  auto block = mlirFunc.addEntryBlock();

  // call function for index compute and generate llvmIR

  // ::mlir::MLIRContext *context, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CTALayoutAttr CTALayout, bool hasLeadingOffset);
  unsigned vecSize = 1;
  unsigned perPhase = 1;
  unsigned maxPhase = 1;
  llvm::SmallVector<unsigned> order{1, 0};
  llvm::SmallVector<unsigned> CTAsPerCGA{1, 1};
  llvm::SmallVector<unsigned> CTASplitNum{1, 1};
  llvm::SmallVector<unsigned> CTAOrder{1, 0};
  auto CTALayout = mlir::triton::gpu::CTALayoutAttr::get(ctx.get(), CTAsPerCGA, CTASplitNum, CTAOrder);
  bool hasLeadingOffset = false;
  auto dstLayout = mlir::triton::gpu::SharedEncodingAttr::get(ctx.get(), vecSize, perPhase, maxPhase, order, CTALayout, hasLeadingOffset);
  auto dstType = mlir::RankedTensorType::get(tensorShape, elementType, dstLayout);

  auto elementAttr = builder.getFloatAttr(elementType, 0);
  auto denseAttr = mlir::DenseElementsAttr::get(srcType, elementAttr);
  // auto dummyTensorInput = builder.create<mlir::arith::ConstantOp>(loc, denseAttr);
  auto inputTensor = block->getArgument(1);
  // block->push_back(dummyTensorInput);

  auto convertLayout = builder.create<mlir::triton::gpu::ConvertLayoutOp>(loc, dstType, inputTensor);
  block->push_back(convertLayout);

  auto returnOp = builder.create<mlir::triton::ReturnOp>(loc);
  block->push_back(returnOp);

  // void populateConvertLayoutOpToLLVMPatterns(
  //   TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
  //   int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
  //   ModuleAllocation &allocation,
  //   ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
  //   PatternBenefit benefit);
  // mlir::LowerToLLVMOptions option(ctx.get());
  // TritonGPUToLLVMTypeConverter typeConverter(ctx.get(), option);
  // RewritePatternSet patterns(ctx.get());
  // int numWarps = 1;
  // mlir::ModuleAxisInfoAnalysis axisInfoAnalysis(mlirModule);
  // mlir::ModuleAllocation allocation(mlirModule);
  // ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo indexCacheInfo;
  // int benefit = 1;
  // populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, numWarps, axisInfoAnalysis, allocation, indexCacheInfo, benefit);

  // if (failed(applyPartialConversion(mlirModule, convTarget, std::move(patterns))))
  //   FAIL();

  llvm::outs() << "before conversion:\n" << mlirModule << "\n";

  mlir::PassManager pm(ctx.get());
  pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(90, mlir::triton::Target::ROCDL, nullptr));
  pm.addPass(mlir::createCanonicalizerPass());

  if (failed(pm.run(mlirModule))) {
    llvm::errs() << "Pass execution failed";
    return;
  }

  // create test example for now
  // auto returnValue = builder.create<mlir::LLVM::ConstantOp>(loc, offsetType, mlir::IntegerAttr::get(offsetType, 42));
  // block->push_back(returnValue);

  // auto addr = builder.create<mlir::LLVM::ZeroOp>(loc, offsetPtrTy);
  // block->push_back(addr);
  // auto store = builder.create<mlir::LLVM::StoreOp>(loc, returnValue, mlirFunc.getArgument(1));
  // block->push_back(store);

  // auto returnOp = builder.create<mlir::LLVM::ReturnOp>(loc, returnValue);
  // block->push_back(returnOp);

  llvm::outs() << "after conversion:\n" << mlirModule << "\n";

  RewritePatternSet patterns(ctx.get());
  patterns.add<EliminateThreadId>(ctx.get());

  // SmallVector<Operation *> threadIds;
  // mlirModule.walk([&threadIds, &builder](mlir::ROCDL::ThreadIdXOp op) {
  //   threadIds.push_back(op);
  //   auto func = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  //   auto threadIdArgument = func.getArgument(threadIdPos);

  //   llvm::errs() << "Op has "
  //                  << std::distance(op.getRes().getUses().begin(),
  //                                   op.getRes().getUses().end())
  //                  << " uses:\n";
  //   for (Operation *userOp : op->getUsers())
  //     llvm::errs() << "    - " << userOp->getName() << "\n";

  //   for (OpOperand &use: op->getUses()) {
  //     auto owner = use.getOwner();
  //     auto opId = use.getOperandNumber();
  //     owner->setOperand(opId, threadIdArgument);
  //   }
  // });


  // for (auto op: threadIds)
  //   op->erase();

  // postprocess generated code, check that all operations are supported, replace arch specific values with function arguments
  SmallVector<Operation *> candidates;
  mlirModule.walk([&candidates](mlir::ROCDL::ThreadIdXOp op) {
    candidates.push_back(op);
  });
  if (mlir::applyOpPatternsAndFold(candidates, std::move(patterns)).failed())
    llvm::errs() << "failed to eliminate threadId operations\n";

  // llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  mlirModule.walk([&builder, &loc](mlir::LLVM::GlobalOp op) {
    llvm::errs() << "found symbol" << op.getNameAttr() << "\n";
    if (op.getNameAttr().str() == "global_smem") {
      auto ldsType = builder.getType<mlir::LLVM::LLVMArrayType>(builder.getIntegerType(8), ldsSize);
      op.setGlobalType(ldsType);
      op.setLinkage(mlir::LLVM::linkage::Linkage::Private);
      // auto zeroOp = builder.create<mlir::LLVM::ZeroOp>(loc, ldsType);
      // mlir::Block zeroBlock;
      // FlatSymbolRefAttr zeroAttr;
      // op.setInitializer(zeroBlock);
    }
    llvm::errs() << "transformed to: " << op << "\n";
  });

  llvm::outs() << "after llvm preparation:\n" << *mlirModule << "\n";

  // Convert llvm mlir to llvm ir
  llvm::LLVMContext llvmContext;

  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  ctx->appendDialectRegistry(registry);

  auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmContext);
  ASSERT_NE(llvmModule, nullptr);
  llvm::outs() << "after llvm translation:\n" << *llvmModule << "\n";
  // interpret llvm ir
  // TBD
  auto llvmModulePtr = llvmModule.get();
  std::string errorMsg;
  llvm::EngineBuilder interpreterBuilder(std::move(llvmModule));
  interpreterBuilder.setEngineKind(llvm::EngineKind::Kind::Interpreter);
  interpreterBuilder.setErrorStr(&errorMsg);
  // auto ldsObject = interpreterBuilder.FindGlobalVariableNamed("global_smem");
  // auto ldsMemory = interpreterBuilder.getMemoryForGV(ldsObject);

  std::unique_ptr<llvm::ExecutionEngine> EE(interpreterBuilder.create());
  if (!EE) {
    if (!errorMsg.empty())
      llvm::errs() << "error creating EE: " << errorMsg << "\n";
    else
      llvm::errs() << "unknown error creating EE!\n";
    exit(1);
  }

  llvm::Function *llvmTestFunction = llvmModulePtr->getFunction(functionName);

  // std::vector<int> mem(16, 0xDEADBEEF);
  std::vector<llvm::GenericValue> inputs(2);
  inputs[0].IntVal = llvm::APInt(32, 123, true);
  // inputs[1].PointerVal = mem.data();
  int elemsPerThread = 64;
  inputs[1].AggregateVal.resize(elemsPerThread);

  auto *ldsMem = reinterpret_cast<int *>(EE->getPointerToGlobalIfAvailable("global_smem"));
  for (int i = 0; i < ldsSize/sizeof(int); ++i)
    ldsMem[i] = 0;

  for (int i = 0; i < elemsPerThread; ++i)
    inputs[1].AggregateVal[i].FloatVal = 1.0f;
  auto result = EE->runFunction(llvmTestFunction, inputs);
  // llvm::outs() << "returned value: " << result.IntVal.getSExtValue() << "\n";
  llvm::outs() << "mem contents:\n";
  // auto *mem = reinterpret_cast<int *>(EE->getGlobalValueAddress("global_smem"));
  for (int i = 0; i < ldsSize/sizeof(int); ++i)
    llvm::outs() << ldsMem[i] << "\n";
  SUCCEED();
}
