#include "ConvertLayoutTestsLib.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

  auto mlirModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto threadIdTy = builder.getI32Type();
  auto offsetType = builder.getI32Type();
  std::vector<mlir::Type> inTypes{threadIdTy};
  std::vector<mlir::Type> outTypes{offsetType};
  auto funcTy = builder.getFunctionType(inTypes, outTypes);

  auto func = builder.create<mlir::triton::FuncOp>(loc, "test_func", funcTy);
  mlirModule.push_back(func);

  auto block = func.addEntryBlock();
  auto returnValue = builder.create<mlir::LLVM::ConstantOp>(loc, offsetType, mlir::IntegerAttr::get(offsetType, 42));
  auto returnOp = builder.create<mlir::LLVM::ReturnOp>(loc, returnValue);
  block->push_back(returnOp);

  llvm::outs() << mlirModule << "\n";
  // call function for index compute and generate llvmIR
  // TBD
  // postprocess generated code, check that all operations are supported, replace arch specific values with function arguments
  // TBD
  // Convert llvm mlir to llvm ir
  // TBD
  // interpret llvm ir
  // TBD
  SUCCEED();
}

TEST(Conversions, MFMAtoMFMAOpB) {
  FAIL();
}
