#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;

namespace {

Type getShemPtrTy(Type operandTy) {
  auto tensorTy = operandTy.cast<RankedTensorType>();
  auto elemTy = tensorTy.getElementType();
  if (elemTy.isBF16()) {
    auto ctx = elemTy.getContext();
    return ptr_ty(type::i16Ty(ctx), 3);
  }
  return ptr_ty(elemTy, 3);
}

SmallVector<int64_t> getMFMAInstrShape(Type operandTy) {
  auto tensorTy = operandTy.cast<RankedTensorType>();
  auto elemTy = tensorTy.getElementType();
  if (elemTy.isF16())
    return {32l, 32l, 8l}; // FP32_FP16_FP16_FP32
  if (elemTy.isF32())
    return {32l, 32l, 2l}; // FP32_FP32_FP32_FP32;
  if (elemTy.isBF16())
    return {32l, 32l, 4l}; // FP32_BF16_BF16_FP32;
  if (elemTy.isInteger(8))
    return {32l, 32l, 8l}; // INT32_INT8_INT8_INT32;
  if (elemTy.isF64())
    return {16l, 16l, 4l}; // FP64_FP64_FP64_FP64;
  assert(false && "unsupported operand data type");
  return {};
}

static int getNumOfElems(Type operand) {
  auto mfmainstrShape = getMFMAInstrShape(operand);
  int instrM = mfmainstrShape[0];
  int instrK = mfmainstrShape[2];
  return std::max<int>(instrM * instrK / 64, 1);
}

// Get a waveId for M axis.
Value getWaveM(ConversionPatternRewriter &rewriter, Location loc, Value wave, const ArrayRef<unsigned int> &wpt, int elemPerInstr, int M) {
  return urem(urem(wave, i32_val(wpt[0])), i32_val(M / elemPerInstr));
}
// Get a waveId for N axis.
Value getWaveN(ConversionPatternRewriter &rewriter, Location loc, Value wave, const ArrayRef<unsigned int> &wpt, int elemPerInstr, int N) {
  Value waveMN = udiv(wave, i32_val(wpt[0]));
  return urem(urem(waveMN, i32_val(wpt[1])), i32_val(N / elemPerInstr));
}

} // namespace

namespace SharedToDotOperandMMAv3 {

llvm::SmallVector<Value> computeOffsetsA(ConversionPatternRewriter &rewriter,
                                         Location loc,
                                         const ArrayRef<int64_t> &mfmaShape,
                                         Value waveM, Value laneId, int wptA,
                                         int numOfElems, ArrayRef<int64_t> reps,
                                         Value cSwizzleOffset) {
  auto numM = reps[0];
  auto numK = reps[1];
  SmallVector<Value> offsets(numM * numK * numOfElems);
  int lineSize = mfmaShape[2] * numK;
  int blockSize = mfmaShape[0] * numM * lineSize;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveHalf = udiv(laneId, _32);

  // Value waveOffset = mul(waveM, i32_val(blockSize));
  Value waveOffset = wptA > 1 ? mul(waveM, i32_val(mfmaShape[0] * lineSize))
                              : mul(waveM, i32_val(blockSize));
  Value colOffset = select(icmp_uge(laneId, _32), i32_val(numOfElems), _0);

  for (int block = 0; block < numM; ++block) {
    // Value blockOffset = i32_val(block * mfmaShape[0] * lineSize);
    Value blockOffset = wptA > 1 ? i32_val(block * blockSize)
                                 : i32_val(block * mfmaShape[0] * lineSize);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * mfmaShape[2]);
      for (int elem = 0; elem < numOfElems; ++elem) {
        Value rowOffset =
            add(mul(urem(laneId, _32), i32_val(lineSize)), i32_val(elem));
        Value elemOffset = add(rowOffset, colOffset);
        Value offset =
            add(add(add(waveOffset, blockOffset), tileOffset), elemOffset);
        offsets[numK * numOfElems * block + numOfElems * tile + elem] = offset;
      }
    }
  }
  return offsets;
}

llvm::SmallVector<Value> computeOffsetsB(ConversionPatternRewriter &rewriter,
                                         Location loc,
                                         const ArrayRef<int64_t> &mfmaShape,
                                         int warpsPerGroupN,
                                         Value waveN, Value laneId, int wptB,
                                         int numOfElems, ArrayRef<int64_t> reps,
                                         Value cSwizzleOffset) {
  auto numK = reps[0];
  auto numN = reps[1];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  int lineSize = warpsPerGroupN * mfmaShape[1] * numN;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveOffset = wptB > 1 ? mul(waveN, i32_val(mfmaShape[1]))
                              : mul(waveN, i32_val(mfmaShape[1] * numN));
  // Value waveOffset = mul(waveN, i32_val(mfmaShape[1] * numN));
  Value colOffset = urem(laneId, _32);

  for (int block = 0; block < numN; ++block) {
    // Value blockOffset = i32_val(block * mfmaShape[1]);
    Value blockOffset = wptB > 1 ? i32_val(block * mfmaShape[1] * numN)
                                 : i32_val(block * mfmaShape[1]);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * mfmaShape[2] * lineSize);
      for (int elem = 0; elem < numOfElems; ++elem) {
        Value halfOffset =
            select(icmp_uge(laneId, _32), i32_val(numOfElems * lineSize), _0);
        Value rowOffset = add(i32_val(elem * lineSize), halfOffset);
        Value elemOffset = add(rowOffset, colOffset);
        Value offset =
            add(add(add(waveOffset, blockOffset), tileOffset), elemOffset);
        offsets[numK * numOfElems * block + numOfElems * tile + elem] = offset;
      }
    }
  }
  return offsets;
}

Value loadA(ConversionPatternRewriter &rewriter,
    Location loc, Value thread, DotOperandEncodingAttr encoding,
    TritonGPUToLLVMTypeConverter *typeConverter,
    Value tensor, const SharedMemoryObject &smemObj) {
  auto mmaLayout = encoding.getParent().cast<MmaEncodingAttr>();
  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  SmallVector<Value> ha;
  auto mfmaInstrShape = getMFMAInstrShape(aTensorTy);
  auto mfmaInstrM = mfmaInstrShape[0];

  auto numReps = encoding.getMMAv3Rep(shape, mfmaInstrShape);
  auto numRepM = numReps[0];
  auto numRepK = numReps[1];

  Value waveSize = i32_val(64);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveM = getWaveM(rewriter, loc, wave, warpsPerCTA, mfmaInstrM, shape[0]);
  int numOfElems = getNumOfElems(aTensorTy);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
  // TODO make macro tile size granilarity configurable
  int macroTileM =
      std::max<int>(shape[0] / (mmaLayout.getWarpsPerCTA()[0] * 32), 1);
  int wptM = std::min<int>(mmaLayout.getWarpsPerCTA()[0], macroTileM);
  int macroTileN =
      std::max<int>(shape[1] / (mmaLayout.getWarpsPerCTA()[1] * 32), 1);
  int wptN = std::min<int>(mmaLayout.getWarpsPerCTA()[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);
  auto mfmaShape = getMFMAInstrShape(aTensorTy);
  auto offsets = computeOffsetsA(rewriter, loc, mfmaShape, waveM, lane, wpt, numOfElems, numReps,
                                 cSwizzleOffset);

  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  Type smemPtrTy = getShemPtrTy(aTensorTy);

  Type elemTy = aTensorTy.getElementType();
  for (int m = 0; m < numRepM; ++m) {
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(elemTy, numOfElems);
      Value valVec = undef(vecTy);
      for (unsigned elem = 0; elem < numOfElems; ++elem) {
        Value elemOffset =
            offsets[m * numOfElems * numRepK + k * numOfElems + elem];
        Value elemValue = load(gep(smemPtrTy, smemBase, elemOffset));
        if (numOfElems > 1)
          valVec = insert_element(vecTy, valVec, elemValue, i32_val(elem));
        else
          valVec = elemValue;
      }
      if (elemTy == i8_ty)
        valVec = bitcast(valVec, i32_ty);
      ha.push_back(valVec);
    }
  }

  elemTy = ha[0].getType();
  MLIRContext *ctx = mmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(ha.size(), elemTy));
  auto result = typeConverter->packLLElements(loc, ha, rewriter, structTy);
  return result;
}

Value loadB(ConversionPatternRewriter &rewriter,
    Location loc, Value thread, DotOperandEncodingAttr encoding,
    TritonGPUToLLVMTypeConverter *typeConverter,
    Value tensor, const SharedMemoryObject &smemObj) {
  auto mmaLayout = encoding.getParent().cast<MmaEncodingAttr>();
  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();

  auto bTensorTy = tensor.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = bTensorTy.getShape();
  auto sharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  SmallVector<Value> hb;
  auto mfmaInstrShape = getMFMAInstrShape(bTensorTy);
  auto mfmaInstrN = mfmaInstrShape[1];

  auto numReps = encoding.getMMAv3Rep(shape, mfmaInstrShape);
  auto numRepK = numReps[0];
  auto numRepN = numReps[1];

  Value waveSize = i32_val(64);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveN = getWaveN(rewriter, loc, wave, mmaLayout.getWarpsPerCTA(), mfmaInstrN, shape[1]);
  int numOfElems = getNumOfElems(bTensorTy);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);

  int macroTileM =
      std::max<int>(shape[0] / (mmaLayout.getWarpsPerCTA()[0] * 32), 1);
  int wptM = std::min<int>(mmaLayout.getWarpsPerCTA()[0], macroTileM);
  int macroTileN =
      std::max<int>(shape[1] / (mmaLayout.getWarpsPerCTA()[1] * 32), 1);
  int wptN = std::min<int>(mmaLayout.getWarpsPerCTA()[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);

  auto mfmaShape = getMFMAInstrShape(bTensorTy);
  int warpsPerGroupN = mmaLayout.getWarpsPerCTA()[1];
  auto offsets = computeOffsetsB(rewriter, loc, mfmaShape, warpsPerGroupN, waveN, lane, wpt, numOfElems, numReps,
                                 cSwizzleOffset);

  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  Type smemPtrTy = getShemPtrTy(bTensorTy);

  Type elemTy = bTensorTy.getElementType();
  for (int n = 0; n < numRepN; ++n) {
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(bTensorTy.getElementType(), numOfElems);
      Value valVec = undef(vecTy);
      for (unsigned elem = 0; elem < numOfElems; ++elem) {
        Value elemOffset =
            offsets[n * numOfElems * numRepK + k * numOfElems + elem];
        Value elemValue = load(gep(smemPtrTy, smemBase, elemOffset));
        if (numOfElems > 1)
          valVec = insert_element(vecTy, valVec, elemValue, i32_val(elem));
        else
          valVec = elemValue;
      }
      if (elemTy == i8_ty)
        valVec = bitcast(valVec, i32_ty);
      hb.push_back(valVec);
    }
  }

  elemTy = hb[0].getType();
  MLIRContext *ctx = mmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(hb.size(), elemTy));
  auto result = typeConverter->packLLElements(loc, hb, rewriter, structTy);
  return result;
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  switch (opIdx) {
  case 0:
    // operand $a
    return loadA(rewriter, loc, thread, encoding, typeConverter, tensor, smemObj);
  case 1:
    // operand $b
    return loadB(rewriter, loc, thread, encoding, typeConverter, tensor, smemObj);
  default:
    assert(false && "unexpected operand idx");
    return Value();
  }
}

} // namespace SharedToDotOperandMMAv3
