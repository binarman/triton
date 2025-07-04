#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/amd/include/Utils/Utility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

// In distributed layouts, tensors are divided into CTA tiles.
// A CTA tile represents the smallest contiguous portion of a tensor that is
// distributed across all threads and warps within a workgroup. The ExtractSlice
// operation extracts a portion of the tensor that is a multiple of CTA tiles.

namespace {

struct ExtractSliceOpConversion
    : public ConvertOpToLLVMPattern<amdgpu::ExtractSliceOp> {
  explicit ExtractSliceOpConversion(LLVMTypeConverter &typeConverter,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<amdgpu::ExtractSliceOp>(typeConverter, benefit) {
  }

  LogicalResult processLayout(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSource().getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcShape = srcTy.getShape();
    auto dstShape = dstTy.getShape();

    auto vals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(srcTy);
    auto srcCTAShape = LLVM::AMD::multiDimElementwise<int64_t, unsigned>(
        srcShape, shapePerCTATile, std::divides<unsigned>());
    auto dstCTAShape = LLVM::AMD::multiDimElementwise<int64_t, unsigned>(
        dstShape, shapePerCTATile, std::divides<unsigned>());

    auto numCTATiles = std::accumulate(dstCTAShape.begin(), dstCTAShape.end(),
                                       1, std::multiplies<>());
    auto offsets = op.getStaticOffsets();
    auto firstTileCoordinate =
        LLVM::AMD::multiDimElementwise<int64_t, unsigned>(
            offsets, shapePerCTATile, std::divides<unsigned>());

    Attribute srcEncoding = srcTy.getEncoding();
    Attribute dstEncoding = dstTy.getEncoding();
    auto linearLayoutSrc = triton::gpu::toLinearLayout(srcShape, srcEncoding);
    auto linearLayoutDst = triton::gpu::toLinearLayout(dstShape, dstEncoding);

    auto srcCTAOrder =
        LLVM::AMD::getCTATileOrder(srcTy.getContext(), linearLayoutSrc);
    auto dstCTAOrder =
        LLVM::AMD::getCTATileOrder(srcTy.getContext(), linearLayoutDst);

    unsigned elemsPerThreadPerCTA =
        triton::gpu::getTotalElemsPerThread(srcTy) /
        std::accumulate(srcCTAShape.begin(), srcCTAShape.end(), 1,
                        std::multiplies<>());

    // get number of output registers
    // for every input register:
    //   get element coords
    //   map element coords -> src register no
    // for every output register
    //   get element coords
    //   copy from corresponding src register
    auto ctx = rewriter.getContext();
    int rank = srcTy.getRank();
    StringAttr kReg = StringAttr::get(ctx, "register");
    auto srcRegBases = linearLayoutSrc.getBases().lookup(kReg);
    auto dstRegBases = linearLayoutDst.getBases().lookup(kReg);

    // Mapping from tensors element location to src register id
    using ElemLocationKey = decltype(linearLayoutSrc.apply({}));
    llvm::MapVector<ElemLocationKey, unsigned> srcElemToReg;
    int srcRegNum = 1 << srcRegBases.size();
    for (int regId = 0; regId < srcRegNum; ++regId) {
      SmallVector<std::pair<StringAttr, int32_t>> hardwareLocation;
      for (auto dimName : linearLayoutSrc.getInDimNames()) {
        if (dimName == kReg)
          hardwareLocation.push_back({dimName, regId});
        else
          hardwareLocation.push_back({dimName, 0});
      }
      auto elemCoords = linearLayoutSrc.apply(hardwareLocation);
      srcElemToReg[elemCoords] = regId;
    }
    // for every output register get element coords, copy corresponding src
    // register
    int dstRegNum = 1 << dstRegBases.size();
    SmallVector<Value> resultVals;
    for (int regId = 0; regId < dstRegNum; ++regId) {
      SmallVector<std::pair<StringAttr, int32_t>> hardwareLocation;
      for (auto dimName : linearLayoutSrc.getInDimNames()) {
        if (dimName == kReg)
          hardwareLocation.push_back({dimName, regId});
        else
          hardwareLocation.push_back({dimName, 0});
      }
      auto elemCoords = linearLayoutDst.apply(hardwareLocation);
      for (int i = 0; i < rank; ++i)
        elemCoords[i].second += offsets[i];
      assert(srcElemToReg.contains(elemCoords));
      auto srcRegId = srcElemToReg.lookup(elemCoords);
      resultVals.push_back(vals[srcRegId]);
    }

    Value ret = packLLElements(loc, this->getTypeConverter(), resultVals,
                               rewriter, dstTy);

    rewriter.replaceOp(op, ret);
    return success();
  }

  LogicalResult
  matchAndRewrite(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSource().getType();
    return processLayout(op, adaptor, rewriter);
  }
};
} // namespace

namespace mlir::triton::AMD {

void populateExtractSliceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit) {
  patterns.add<ExtractSliceOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
