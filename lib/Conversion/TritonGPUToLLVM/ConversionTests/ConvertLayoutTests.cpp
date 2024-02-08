#include "ConvertLayoutTestsLib.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"

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

#include <iomanip>

constexpr int threadIdPos = 0;
constexpr int waveSize = 64;
constexpr int ldsSize = 65536;
constexpr int tensorSize = 256;

// mlir::ModuleOp createTestModuleFromScratch(mlir::MLIRContext *ctx, const char *functionName) {
//   ctx->getOrLoadDialect<mlir::triton::TritonDialect>();
//   ctx->getOrLoadDialect<mlir::triton::gpu::TritonGPUDialect>();
//   ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
//   ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
//   ctx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
//   mlir::OpBuilder builder(ctx);
//   mlir::Location loc = builder.getUnknownLoc();

//   llvm::SmallVector<int64_t> tensorShape{16, 16};
//   auto elementType = builder.getF32Type();
//   unsigned versionMajor = 2;
//   unsigned versionMinor = 0;
//   llvm::SmallVector<unsigned> warpsPerCTA{1, 1};
//   unsigned mDim = 4;
//   unsigned nDim = 4;
//   bool isTransposed = false;
//   auto srcLayout = mlir::triton::gpu::MfmaEncodingAttr::get(ctx, versionMajor, versionMinor, warpsPerCTA, mDim, nDim, isTransposed);
//   auto srcType = mlir::RankedTensorType::get(tensorShape, elementType, srcLayout);

//   auto mlirModule = mlir::ModuleOp::create(loc);
//   mlirModule->setAttr("triton_gpu.num-warps", builder.getI32IntegerAttr(1));
//   mlirModule->setAttr("triton_gpu.threads-per-warp", builder.getI32IntegerAttr(64));
//   mlirModule->setAttr("triton_gpu.num-ctas", builder.getI32IntegerAttr(1));

//   auto threadIdTy = builder.getI32Type();
//   auto offsetType = builder.getI32Type();
//   auto offsetPtrTy = mlir::LLVM::LLVMPointerType::get(ctx, offsetType, 3);
//   std::vector<mlir::Type> inTypes{threadIdTy, srcType};
//   auto mlirFuncTy = builder.getFunctionType(inTypes, {});

//   auto mlirFunc = builder.create<mlir::triton::FuncOp>(loc, functionName, mlirFuncTy);
//   mlirModule.push_back(mlirFunc);
//   llvm::errs() << "module created:\n" << mlirModule << "\n";

//   auto block = mlirFunc.addEntryBlock();

//   // call function for index compute and generate llvmIR
//   unsigned vecSize = 1;
//   unsigned perPhase = 1;
//   unsigned maxPhase = 1;
//   llvm::SmallVector<unsigned> order{1, 0};
//   llvm::SmallVector<unsigned> CTAsPerCGA{1, 1};
//   llvm::SmallVector<unsigned> CTASplitNum{1, 1};
//   llvm::SmallVector<unsigned> CTAOrder{1, 0};
//   auto CTALayout = mlir::triton::gpu::CTALayoutAttr::get(ctx, CTAsPerCGA, CTASplitNum, CTAOrder);
//   bool hasLeadingOffset = false;
//   auto dstLayout = mlir::triton::gpu::SharedEncodingAttr::get(ctx, vecSize, perPhase, maxPhase, order, CTALayout, hasLeadingOffset);
//   auto dstType = mlir::RankedTensorType::get(tensorShape, elementType, dstLayout);

//   auto elementAttr = builder.getFloatAttr(elementType, 0);
//   auto denseAttr = mlir::DenseElementsAttr::get(srcType, elementAttr);
//   auto inputTensor = block->getArgument(1);

//   auto convertLayout = builder.create<mlir::triton::gpu::ConvertLayoutOp>(loc, dstType, inputTensor);
//   block->push_back(convertLayout);

//   auto returnOp = builder.create<mlir::triton::ReturnOp>(loc);
//   block->push_back(returnOp);

//   llvm::errs() << "module filled:\n" << mlirModule << "\n";

//   return mlirModule;
// }

mlir::ModuleOp createTestModule(mlir::MLIRContext *ctx, const char *src) {
  ctx->getOrLoadDialect<mlir::triton::TritonDialect>();
  ctx->getOrLoadDialect<mlir::triton::gpu::TritonGPUDialect>();
  ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
  ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  ParserConfig cfg(ctx);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(src, cfg);
  if (!module) {
    llvm::errs() << "failed to parse module\n";
    exit(1);
  }
  return module.release();
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
  llvm::errs() << *mod << "\n";
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
    llvm::outs() << "memory (thread: " << threadId << "):\n";
    auto floatMem = reinterpret_cast<float *>(ldsMem);
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        std::cout << std::setw(10) << floatMem[row*16 + col] << " ";
      }
      std::cout << "\n";
    }
    // for (int i = 0; i < tensorSize; ++i)
      
    //   if (ldsMem[i] == 0xcc)
    //     llvm::outs() << "_";
    //   else
    //     llvm::outs() << "*";
    // llvm::outs() << "\n";
  }
}

std::vector<llvm::GenericValue> RunInterpreterWithArgs(std::unique_ptr<llvm::Module> mod, const char *functionName, std::vector<std::vector<llvm::GenericValue>> &inputs) {
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

  // Initialize LDS
  auto *ldsMem = reinterpret_cast<char *>(EE->getPointerToGlobalIfAvailable("global_smem"));
  std::fill(ldsMem, ldsMem + ldsSize, 0xcc);

  const int numThreads = inputs.size();
  std::vector<std::vector<llvm::GenericValue>> augmentedInputs(numThreads);
  for (int threadId = 0; threadId < numThreads; ++threadId) {
    augmentedInputs[threadId].resize(1);
    augmentedInputs[threadId][0].IntVal = llvm::APInt(32, threadId, true);
    augmentedInputs[threadId].insert(augmentedInputs[threadId].end(), inputs[threadId].begin(), inputs[threadId].end());
  }
  std::vector<llvm::GenericValue> outputs(numThreads);

  for (int threadId = 0; threadId < numThreads; ++threadId)
    outputs[threadId] = EE->runFunction(llvmTestFunction, augmentedInputs[threadId]);
  
  return outputs;
}

TEST(LayoutConversions, DISABLED_MFMA44toShared) {
  const std::string functionName = "test_func";
  mlir::MLIRContext mlirCtx;
  const char *src =
"#mfma = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [4, 4], isTransposed = false}>"
"#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset = false}>"
""
"module attributes {\"triton_gpu.num-ctas\" = 1 : i32, \"triton_gpu.num-warps\" = 1 : i32, \"triton_gpu.threads-per-warp\" = 64 : i32} {\n"
"  tt.func @test_func(%arg0: i32, %arg1: tensor<16x16xf32, #mfma>) -> tensor<16x16xf32, #shared> {\n"
"    %0 = triton_gpu.convert_layout %arg1 : (tensor<16x16xf32, #mfma>) -> tensor<16x16xf32, #shared>\n"
"    tt.return %0: tensor<16x16xf32, #shared>\n"
"  }\n"
"}\n";

  auto mlirModule = createTestModule(&mlirCtx, src);

  convertTTGToLLVM(mlirModule);

  replaceGPUSpecificEntities(mlirModule);

  llvm::LLVMContext llvmCtx;

  std::unique_ptr<llvm::Module> llvmModule = convertMLIRToLLVMIR(mlirModule, llvmCtx);

  RunInterpreter(std::move(llvmModule), functionName.c_str());
  SUCCEED();
}

TEST(LayoutConversions, DISABLED_SharedToMFMA44) {
  const char *functionName = "test_func";
  mlir::MLIRContext mlirCtx;
  const char *src =
"#mfma = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [4, 4], isTransposed = false}>"
"#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset = false}>"
""
"module attributes {\"triton_gpu.num-ctas\" = 1 : i32, \"triton_gpu.num-warps\" = 1 : i32, \"triton_gpu.threads-per-warp\" = 64 : i32} {\n"
"  tt.func @test_func(%arg0: i32, %arg1: tensor<16x16xf32, #shared>) -> tensor<16x16xf32, #mfma> {\n"
"    %0 = triton_gpu.convert_layout %arg1 : (tensor<16x16xf32, #shared>) -> tensor<16x16xf32, #mfma>\n"
"    tt.return %0: tensor<16x16xf32, #mfma>\n"
"  }\n"
"}\n";

  auto mlirModule = createTestModule(&mlirCtx, src);

  convertTTGToLLVM(mlirModule);

  replaceGPUSpecificEntities(mlirModule);

  llvm::LLVMContext llvmCtx;

  std::unique_ptr<llvm::Module> llvmModule = convertMLIRToLLVMIR(mlirModule, llvmCtx);

  float sharedObject[256] = {};
  std::iota(sharedObject, sharedObject+256, 0.0f);

  std::vector<std::vector<llvm::GenericValue>> inputs(waveSize);
  for (int i = 0; i < waveSize; ++i) {
    inputs[i].resize(1);
    inputs[i][0].AggregateVal.resize(5);
    inputs[i][0].AggregateVal[0].PointerVal = sharedObject;
    inputs[i][0].AggregateVal[1].IntVal = llvm::APInt(32, 16, true);
    inputs[i][0].AggregateVal[2].IntVal = llvm::APInt(32, 1, true);
    inputs[i][0].AggregateVal[3].IntVal = llvm::APInt(32, 0, true);
    inputs[i][0].AggregateVal[4].IntVal = llvm::APInt(32, 0, true);
  }

  auto outputs = RunInterpreterWithArgs(std::move(llvmModule), functionName, inputs);
  for (int i = 0; i < waveSize; ++i) {
    std::cout << "thread " << i << ": ";
    for (int j = 0; j < outputs[i].AggregateVal.size(); ++j) {
      std::cout << std::setw(4) << outputs[i].AggregateVal[j].FloatVal;
    }
    std::cout << "\n";
  }
  SUCCEED();
}

TEST(LayoutConversions, SharedToTransposedMFMA464) {
  const char *functionName = "test_func";
  mlir::MLIRContext mlirCtx;
  const char *src =
"#mfma = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [4, 64], isTransposed = true}>"
"#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset = false}>"
""
"module attributes {\"triton_gpu.num-ctas\" = 1 : i32, \"triton_gpu.num-warps\" = 4 : i32, \"triton_gpu.threads-per-warp\" = 64 : i32} {\n"
"  tt.func @test_func(%arg0: i32, %arg1: tensor<16x64xf32, #shared>) -> tensor<16x64xf32, #mfma> {\n"
"    %0 = triton_gpu.convert_layout %arg1 : (tensor<16x64xf32, #shared>) -> tensor<16x64xf32, #mfma>\n"
"    tt.return %0: tensor<16x64xf32, #mfma>\n"
"  }\n"
"}\n";

  auto mlirModule = createTestModule(&mlirCtx, src);

  convertTTGToLLVM(mlirModule);

  replaceGPUSpecificEntities(mlirModule);

  llvm::LLVMContext llvmCtx;

  std::unique_ptr<llvm::Module> llvmModule = convertMLIRToLLVMIR(mlirModule, llvmCtx);

  float sharedObject[1024*4] = {};
  std::iota(sharedObject, sharedObject+1024*4, 0.0f);

  constexpr int waves = 4;
  constexpr int numThreads = waves * waveSize;

  std::vector<std::vector<llvm::GenericValue>> inputs(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    inputs[i].resize(1);
    inputs[i][0].AggregateVal.resize(5);
    inputs[i][0].AggregateVal[0].PointerVal = sharedObject;
    inputs[i][0].AggregateVal[1].IntVal = llvm::APInt(32, 64, true);
    inputs[i][0].AggregateVal[2].IntVal = llvm::APInt(32, 1, true);
    inputs[i][0].AggregateVal[3].IntVal = llvm::APInt(32, 0, true);
    inputs[i][0].AggregateVal[4].IntVal = llvm::APInt(32, 0, true);
  }

  auto outputs = RunInterpreterWithArgs(std::move(llvmModule), functionName, inputs);
  for (int i = 0; i < numThreads; ++i) {
    std::cout << "thread " << i << ": ";
    for (int j = 0; j < outputs[i].AggregateVal.size(); ++j) {
      std::cout << std::setw(4) << outputs[i].AggregateVal[j].FloatVal;
    }
    std::cout << "\n";
  }
  SUCCEED();
}

TEST(LayoutConversions, DISABLED_SharedToTransposedMFMA44OpA) {
  const char *functionName = "test_func";
  mlir::MLIRContext mlirCtx;
  const char *src =
"#mfma = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [4, 4], isTransposed = false}>"
"#dotop = #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>"
"#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset = false}>"
""
"module attributes {\"triton_gpu.num-ctas\" = 1 : i32, \"triton_gpu.num-warps\" = 4 : i32, \"triton_gpu.threads-per-warp\" = 64 : i32} {\n"
"  tt.func @test_func(%arg0: i32, %arg1: tensor<16x64xf32, #shared>) -> tensor<16x64xf32, #dotop> {\n"
"    %0 = triton_gpu.convert_layout %arg1 : (tensor<16x64xf32, #shared>) -> tensor<16x64xf32, #dotop>\n"
"    tt.return %0: tensor<16x64xf32, #dotop>\n"
"  }\n"
"}\n";

  auto mlirModule = createTestModule(&mlirCtx, src);

  convertTTGToLLVM(mlirModule);

  replaceGPUSpecificEntities(mlirModule);

  llvm::LLVMContext llvmCtx;

  std::unique_ptr<llvm::Module> llvmModule = convertMLIRToLLVMIR(mlirModule, llvmCtx);

  float sharedObject[1024] = {};
  std::iota(sharedObject, sharedObject+1024, 0.0f);

  constexpr int waves = 4;
  constexpr int numThreads = waves * waveSize;

  std::vector<std::vector<llvm::GenericValue>> inputs(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    inputs[i].resize(1);
    inputs[i][0].AggregateVal.resize(5);
    inputs[i][0].AggregateVal[0].PointerVal = sharedObject;
    inputs[i][0].AggregateVal[1].IntVal = llvm::APInt(32, 64, true);
    inputs[i][0].AggregateVal[2].IntVal = llvm::APInt(32, 1, true);
    inputs[i][0].AggregateVal[3].IntVal = llvm::APInt(32, 0, true);
    inputs[i][0].AggregateVal[4].IntVal = llvm::APInt(32, 0, true);
  }

  auto outputs = RunInterpreterWithArgs(std::move(llvmModule), functionName, inputs);
  for (int i = 0; i < numThreads; ++i) {
    std::cout << "thread " << i << ": ";
    for (int j = 0; j < outputs[i].AggregateVal[0].AggregateVal.size(); ++j) {
      std::cout << std::setw(4) << outputs[i].AggregateVal[0].AggregateVal[j].FloatVal;
    }
    std::cout << "\n";
  }
  SUCCEED();
}
