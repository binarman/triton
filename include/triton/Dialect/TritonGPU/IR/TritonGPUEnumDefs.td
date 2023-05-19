include "mlir/IR/EnumAttr.td"

class TritonGPU_EnumAttrCase<string cppSym, int val, string irSym>
  : I64EnumAttrCase<cppSym, val, irSym>;

class TritonGPU_EnumAttr<string name, string description,
    list<TritonGPU_EnumAttrCase> cases>
  : I64EnumAttr<name, description, cases> {

  let cppNamespace = "::mlir::triton::gpu";
}

def NVMMA : TritonGPU_EnumAttrCase<"nvmma", 0, "nvmma">;
def AMDMMA  : TritonGPU_EnumAttrCase<"amdmma", 1, "amdmma">;

def MMAFamily : TritonGPU_EnumAttr<"MMAFamily", "Type of matrix multiply layout", [NVMMA, AMDMMA]>;
