#include "ConvertLayoutTestsLib.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include <gtest/gtest.h>

TEST(Conversions, MFMAtoMFMAOpA) {
  // create module and function
  auto ctx = std::make_unique<mlir::MLIRContext>();
  ctx->getOrLoadDialect<mlir::triton::TritonDialect>();
  ctx->getOrLoadDialect<mlir::index::IndexDialect>();
  ctx->getOrLoadDialect<mlir::triton::TritonDialect>();
  ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
  mlir::OpBuilder builder(ctx.get());
  mlir::Location loc = builder.getUnknownLoc();

  auto mlirModule = mlir::ModuleOp::create(loc);

  auto threadIdTy = builder.getI32Type();
  auto offsetType = builder.getI32Type();
  auto offsetPtrTy = mlir::LLVM::LLVMPointerType::get(ctx.get(), offsetType, 3);
  std::vector<mlir::Type> inTypes{threadIdTy, offsetPtrTy};
  auto mlirFuncTy = mlir::LLVM::LLVMFunctionType::get(offsetType, inTypes);
  // auto funcTy = builder.getFunctionType(inTypes, outTypes);

  const std::string functionName = "test_func";
  auto mlirFunc = builder.create<mlir::LLVM::LLVMFuncOp>(loc, functionName, mlirFuncTy);
  mlirModule.push_back(mlirFunc);

  auto block = mlirFunc.addEntryBlock();

  // call function for index compute and generate llvmIR
  // TBD

  // create test example for now
  auto returnValue = builder.create<mlir::LLVM::ConstantOp>(loc, offsetType, mlir::IntegerAttr::get(offsetType, 42));
  block->push_back(returnValue);

  // auto addr = builder.create<mlir::LLVM::ZeroOp>(loc, offsetPtrTy);
  // block->push_back(addr);
  auto store = builder.create<mlir::LLVM::StoreOp>(loc, returnValue, mlirFunc.getArgument(1));
  block->push_back(store);

  auto returnOp = builder.create<mlir::LLVM::ReturnOp>(loc, returnValue);
  block->push_back(returnOp);

  llvm::outs() << mlirModule << "\n";


  // postprocess generated code, check that all operations are supported, replace arch specific values with function arguments
  // TBD
  // Convert llvm mlir to llvm ir
  llvm::LLVMContext llvmContext;

  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  ctx->appendDialectRegistry(registry);

  auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmContext);
  ASSERT_NE(llvmModule, nullptr);
  llvm::outs() << *llvmModule << "\n";
  // interpret llvm ir
  // TBD
  auto llvmModulePtr = llvmModule.get();
  std::string errorMsg;
  llvm::EngineBuilder interpreterBuilder(std::move(llvmModule));
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

  std::vector<int> mem(16, 0xDEADBEEF);
  std::vector<llvm::GenericValue> inputs(2);
  inputs[0].IntVal = llvm::APInt(32, 123, true);
  inputs[1].PointerVal = mem.data();
  auto result = EE->runFunction(llvmTestFunction, inputs);
  llvm::outs() << "returned value: " << result.IntVal.getSExtValue() << "\n";
  llvm::outs() << "mem contents:\n";
  for (int i = 0; i < mem.size(); ++i)
    llvm::outs() << mem[i] << "\n";
  SUCCEED();
}
