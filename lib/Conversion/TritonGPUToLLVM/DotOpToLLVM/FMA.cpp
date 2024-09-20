#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;

using ValueTableFMA = std::map<std::tuple<int, int, int>, Value>;

static ValueTableFMA
getValueTableFromStructFMA(Value val, ArrayRef<unsigned> perTileShape,
                           unsigned kDim, unsigned nonKDim,
                           ConversionPatternRewriter &rewriter, Location loc,
                           ArrayRef<unsigned> order) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  assert(perTileShape.size() == 3);
  assert(elems.size() == product(perTileShape));
  assert(kDim == 1 || kDim == 2);
  assert(nonKDim == 1 || nonKDim == 2);
  const unsigned bDim = 0;

  for (unsigned idx = 0; idx < elems.size(); ++idx) {
    unsigned spatialIdx[3];
    unsigned curIdx = idx;
    for (auto dim : order) {
      spatialIdx[dim] = curIdx % perTileShape[dim];
      curIdx /= perTileShape[dim];
    }
    assert(curIdx == 0);
    res[{spatialIdx[bDim], spatialIdx[nonKDim], spatialIdx[kDim]}] = elems[idx];
  }
  return res;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());
  auto dElemTy = dTensorTy.getElementType();

  SmallVector<int64_t> aShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(aTensorTy)));
  auto dShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(dTensorTy)));

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = expandMatrixOrderWithBatch(dLayout.getOrder());
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread =
      expandMatrixShapeWithBatch(ArrayRef(getSizePerThread(dLayout)));
  auto shapePerCTATile =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTATile(dLayout)));

  unsigned K = aShapePerCTA[2];

  unsigned retSize[3];
  for (int i = 0; i < 3; ++i) {
    unsigned numRep = dShapePerCTA[i] / shapePerCTATile[i];
    numRep = std::max(static_cast<unsigned>(1), numRep);
    retSize[i] = numRep * sizePerThread[i];
  }

  auto has = getValueTableFromStructFMA(llA, {retSize[0], retSize[1], K}, 2, 1,
                                        rewriter, loc, order);
  auto hbs = getValueTableFromStructFMA(llB, {retSize[0], K, retSize[2]}, 1, 2,
                                        rewriter, loc, order);

  SmallVector<Value> ret = cc;

  for (unsigned b = 0; b < retSize[0]; ++b)
    for (unsigned m = 0; m < retSize[1]; ++m)
      for (unsigned n = 0; n < retSize[2]; ++n) {
        unsigned idx[] = {b, m, n};
        unsigned linearIdx = 0;
        for (auto dim : llvm::reverse(order)) {
          linearIdx = linearIdx * retSize[dim] + idx[dim];
        }
        for (unsigned k = 0; k < K; ++k) {
          ret[linearIdx] = rewriter.create<LLVM::FMulAddOp>(
              loc, has[{b, m, k}], hbs[{b, n, k}], ret[linearIdx]);
        }
      }

  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
