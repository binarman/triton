#ifndef TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

constexpr static char AttrNumWarpsName[] = "triton_gpu.num-warps";

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUPass();

class CompilationTargetBase {
public:
  enum class Vendor {
    INVALID,
    AMD,
    NVIDIA
  };

  CompilationTargetBase() : triple(""), vendor(Vendor::INVALID) {}

  CompilationTargetBase(const std::string &triple) : triple(triple), vendor(Vendor::INVALID) {}

  std::string getTriple() const { return triple; }

  Vendor getVendor() const { return vendor; }

  virtual ~CompilationTargetBase() = default;

private:
  std::string triple;

protected:
  Vendor vendor;
};

class CompilationTargetAMD: public CompilationTargetBase {
public:
  CompilationTargetAMD(const std::string &triple,
                       const std::string &arch,
                       const std::string &features)
     : CompilationTargetBase(triple), arch(arch), features(features) {
        this->vendor = CompilationTargetBase::Vendor::AMD;
     }

  std::string getArch() const { return arch; }

  std::string getFeatures() const { return features; }

private:
  std::string arch;
  std::string features;
};

class CompilationTargetNvidia: public CompilationTargetBase {
public:
  CompilationTargetNvidia(const std::string &triple,
                          int computeCapability)
     : CompilationTargetBase(triple), computeCapability(computeCapability) {
        this->vendor = CompilationTargetBase::Vendor::NVIDIA;
     }
  
  int getComputeCapability() const { return computeCapability; }

private:
  int computeCapability;
};

// Create the pass with parameters set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(int numWarps, const CompilationTargetAMD &target);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(int numWarps, const CompilationTargetNvidia &target);


} // namespace triton
} // namespace mlir

#endif
