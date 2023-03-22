#ifndef TRITONGPU_DIALECT
#define TRITONGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPU_Dialect : Dialect {
  let name = "triton_gpu";

  let cppNamespace = "::mlir::triton::gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "triton_gpu.num-warps"; }
    static int getNumWarps(ModuleOp mod) { 
      if(!mod->hasAttr("triton_gpu.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-warps attribute");
      return mod->getAttr("triton_gpu.num-warps").cast<IntegerAttr>().getInt();
    }

    static std::string getTripleAttrName() { return "triton_gpu.triple"; }
    static std::string getTriple(ModuleOp mod) { 
      if(!mod->hasAttr(getTripleAttrName()))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.triple attribute");
      return mod->getAttrOfType<StringAttr>(getTripleAttrName()).getValue().str();
    }

    static std::string getComputeCapabilityAttrName() { return "triton_gpu.compute_capability"; }
    static int getComputeCapability(ModuleOp mod) {
      if (!mod->hasAttr(getComputeCapabilityAttrName()))
        llvm::report_fatal_error(
            "TritonGPU module does not contain triton_gpu.compute_capability attribute");
      return mod->getAttrOfType<IntegerAttr>(getComputeCapabilityAttrName()).getInt();
    }

    static std::string getGFXArchAttrName() { return "triton_gpu.gfx_arch"; }
    static int getGFXArch(ModuleOp mod) {
      if (!mod->hasAttr(getGFXArchAttrName()))
        llvm::report_fatal_error(
            "TritonGPU module does not contain triton_gpu.gfx_arch attribute");
      return mod->getAttrOfType<IntegerAttr>(getGFXArchAttrName()).getInt();
    }

    static std::string getFeaturesAttrName() { return "triton_gpu.features"; }
    static int getFeatures(ModuleOp mod) {
      if (!mod->hasAttr(getFeaturesAttrName()))
        llvm::report_fatal_error(
            "TritonGPU module does not contain triton_gpu.features attribute");
      return mod->getAttrOfType<IntegerAttr>(getFeaturesAttrName()).getInt();
    }

  }];
  
  let useDefaultAttributePrinterParser = 1;
}

#endif
