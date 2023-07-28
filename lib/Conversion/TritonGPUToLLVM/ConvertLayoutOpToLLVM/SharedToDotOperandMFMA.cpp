#ifdef USE_ROCM

#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

Type getShemPtrTy(Type elemTy) {
  if (elemTy.isBF16()) {
    auto ctx = elemTy.getContext();
    return ptr_ty(type::i16Ty(ctx), 3);
  }
  return ptr_ty(elemTy, 3);
}

// Get a waveId for M axis.
Value getWaveM(ConversionPatternRewriter &rewriter, Location loc, Value wave,
               const ArrayRef<unsigned int> &wpt, int elemPerInstr, int M) {
  return urem(urem(wave, i32_val(wpt[0])), i32_val(M / elemPerInstr));
}
// Get a waveId for N axis.
Value getWaveN(ConversionPatternRewriter &rewriter, Location loc, Value wave,
               const ArrayRef<unsigned int> &wpt, int elemPerInstr, int N) {
  Value waveMN = udiv(wave, i32_val(wpt[0]));
  return urem(urem(waveMN, i32_val(wpt[1])), i32_val(N / elemPerInstr));
}

} // namespace

namespace SharedToDotOperandMFMA {

/**
 * @brief swizzling tensor element indexes according pattern encoded in
 * SharedEncodingAttr
 *
 * @param rewriter
 * @param loc
 * @param row row of target tensor element related to the start of smemObj
 * @param col col of target tensor element related to the start of smemObj
 * @param smemObj shared memory object, contains info about tensor in LDS
 * @param attr layout attribute, contains swizzling info
 * @return swizzled row, col indexes in tensor notation
 */
std::pair<mlir::Value, mlir::Value>
swizzleIndexes(ConversionPatternRewriter &rewriter, Location loc, Value row,
               Value col, SharedMemoryObject smemObj, SharedEncodingAttr attr) {
  (void)smemObj; // unused in current pattern
  bool transposed = (attr.getOrder()[0] != 1);
  if (transposed) {
    // tensor is column-wise, so swapping col and row in computations
    std::swap(row, col);
  }
  auto vec = i32_val(attr.getVec());
  auto perPhase = i32_val(attr.getPerPhase());
  auto maxPhase = i32_val(attr.getMaxPhase());

  // Original algorithm taken from getSwizzledSharedPtrs function
  // (TritonGPUToLLVMBase.h): Basic algorithm for row-major tensor is following:
  //
  // phase = (row // perPhase) % maxPhase
  // colOffSwizzled = ((col // vec) ^ phase) * vec
  // colOffOrdered = col % vec
  // colOff = colOffSwizzled + colOffOrdered
  auto phase = urem(udiv(row, perPhase), maxPhase);
  auto colOffSwizzled = mul(xor_(udiv(col, vec), phase), vec);
  auto colOffOrdered = urem(col, vec);
  auto colOff = add(colOffSwizzled, colOffOrdered);

  if (transposed)
    return {colOff, row};
  else
    return {row, colOff};
}

// Position of element in dot operand
// nonKDim is M for A op and N for B op
struct ElementOffsetSpatial {
  Value nonKDim;
  Value kDim;
};

using ElementOffsetLinear = Value;

// Helper structure to store offsets for loads
// map load operation to elements of matrix or memory offsets that it should process
template <class Offset>
class LoadOffsetMap {
  std::vector<Offset> offsets;
  int vectorSize; // number of elements which one load should process
  int numBlocks;
  int numKTiles;
  int loadsPerThread;

public:
  LoadOffsetMap(int numBlocks, int numKTiles, int numElemsPerThread, int vectorSize):
      numBlocks(numBlocks),
      numKTiles(numKTiles),
      loadsPerThread(numElemsPerThread/vectorSize),
      vectorSize(vectorSize),
      offsets(numBlocks * numKTiles * numElemsPerThread / vectorSize) {
    assert(numElemsPerThread % vectorSize == 0);
  }

  int getNumBlocks() {
    return numBlocks;
  }

  int getNumKTiles() {
    return numKTiles;
  }

  int getNumLoadsPerThread() {
    return loadsPerThread;
  }

  int getNumElemPerLoad() {
    return vectorSize;
  }

  Offset &offset(int block, int kTile, int loadId) {
    assert(block < numBlocks);
    assert(kTile < numKTiles);
    assert(loadId < loadsPerThread);
    return offsets[numKTiles * loadsPerThread * block + loadsPerThread * kTile + loadId];
  }
};

using LoadOffsetMapSpatial = LoadOffsetMap<ElementOffsetSpatial>;
using LoadOffsetMapLinear = LoadOffsetMap<ElementOffsetLinear>;

/**
 * @brief This function maps particular load of mfma dot operand to element
 * indexes(row, col)
 *
 * Whole tensor is broken into "blocks" of waves along "non-K" axis.
 * One block could be processed by multiple waves.
 * One wave works on a piece of tensor size elemsPerInstr[0] x K.
 * Each of these pieces is broken into "tiles" of size elemsPerInstr[0] x
 * elemsPerInstr[1].
 *
 * Total offset of element is a sum of following values:
 * 1. Offset of wave block in tensor
 * 2. Offset of wave inside one wave block
 * 3. Offset of tile in one wave
 * 4. Offset of one lane data in a tile
 * 5. Offset of particular element of tensor processed by one lane
 *
 * This function computes these offsets for axies independently
 *
 * @param rewriter
 * @param loc
 * @param elemsPerInstr operand tile shape consumed by one MFMA instruction
 * @param waveId id component of 2d wave grid along nono-K axis
 * @param laneId lane id in warp [0..63]
 * @param warpsPerGroup number of warps in one block
 * @param numOfElems number of elements accessed by thread per repetition
 * @param reps number of instructions repretition to fully cover dot operand (nonK, K dim repetitions)
 * @param smemStrides strides in LDS tensor
 * @return mapping from loads to row and col of operand element it processes
 */
LoadOffsetMapSpatial
computeTensorElemMapping(ConversionPatternRewriter &rewriter, Location loc,
                         const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                         Value laneId, int warpsPerGroup, int numOfElems,
                         ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
                         int loadVecSize) {
  auto numBlocks = reps[0];
  auto kTiles = reps[1];
  LoadOffsetMapSpatial mapping(numBlocks, kTiles, numOfElems, loadVecSize);

  Value _0 = i32_val(0);
  Value _32 = i32_val(32);

  for (int block = 0; block < numBlocks; ++block) {
    Value blockVOffset = i32_val(block * elemsPerInstr[0] * warpsPerGroup);
    Value blockHOffset = _0;
    Value waveVOffset = mul(waveId, i32_val(elemsPerInstr[0]));
    Value waveHOffset = _0;
    for (int tile = 0; tile < kTiles; ++tile) {
      Value tileVOffset = _0;
      Value tileHOffset = i32_val(tile * elemsPerInstr[1]);

      Value laneVOffset = urem(laneId, _32);
      Value laneHOffset = mul(udiv(laneId, _32), i32_val(numOfElems));
      for (int elem = 0; elem < numOfElems/loadVecSize; ++elem) {
        Value elemVOffset = _0;
        Value elemHOffset = i32_val(elem*loadVecSize);

        Value sliceVOffset = add(
            add(add(add(blockVOffset, waveVOffset), tileVOffset), laneVOffset),
            elemVOffset);
        Value sliceHOffset = add(
            add(add(add(blockHOffset, waveHOffset), tileHOffset), laneHOffset),
            elemHOffset);

        Value row = add(sliceVOffset, smemOffsets[0]);
        Value col = add(sliceHOffset, smemOffsets[1]);

        mapping.offset(block, tile, elem) = {row, col};
      }
    }
  }
  return mapping;
}

bool isSwizzled(SharedEncodingAttr layout) {
  return layout.getMaxPhase() != 1;
}

Value computeOffset(ConversionPatternRewriter &rewriter, Location loc,
                    Value row, Value col, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  auto [swizzledRow, swizzledCol] =
      swizzleIndexes(rewriter, loc, row, col, smemObj, srcLayout);
  auto &strides = smemObj.strides;
  Value rowOffset = mul(swizzledRow, strides[0]);
  Value colOffset = mul(swizzledCol, strides[1]);
  return add(rowOffset, colOffset);
}

LoadOffsetMapLinear
computeOffsetsAType(ConversionPatternRewriter &rewriter, Location loc,
                    const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                    Value laneId, int warpsPerGroup, int numOfElems,
                    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  SmallVector<Value> strides{smemObj.strides[0], smemObj.strides[1]};
  SmallVector<Value> offsets{smemObj.offsets[0], smemObj.offsets[1]};

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == 1) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping =
      computeTensorElemMapping(rewriter, loc, elemsPerInstr, waveId, laneId,
                               warpsPerGroup, numOfElems, reps, offsets, vectorSize);
  LoadOffsetMapLinear aOffsets(reps[0], reps[1], numOfElems, vectorSize);
  for (int block = 0; block < reps[0]; ++block)
    for (int tile = 0; tile < reps[1]; ++tile)
      for (int loadId = 0; loadId < numOfElems/vectorSize; ++loadId){
        auto spatialOffset = mapping.offset(block, tile, loadId);
        Value row = spatialOffset.nonKDim;
        Value col = spatialOffset.kDim;
        aOffsets.offset(block, tile, loadId) = computeOffset(rewriter, loc, row, col, smemObj, srcLayout);
      }
  return aOffsets;
}

LoadOffsetMapLinear
computeOffsetsBType(ConversionPatternRewriter &rewriter, Location loc,
                    const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                    Value laneId, int warpsPerGroup, int numOfElems,
                    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  // transpose reps and offsets, because operand B has layout equal to
  // transposed operand A layout
  SmallVector<int64_t> tElemsPerInstr{elemsPerInstr[1], elemsPerInstr[0]};
  SmallVector<int64_t> tReps{reps[1], reps[0]};
  SmallVector<Value> toffsets{smemObj.offsets[1], smemObj.offsets[0]};

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == 0) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping =
      computeTensorElemMapping(rewriter, loc, tElemsPerInstr, waveId, laneId,
                               warpsPerGroup, numOfElems, tReps, toffsets, vectorSize);
  LoadOffsetMapLinear bOffsets(reps[1], reps[0], numOfElems, vectorSize);
  for (int block = 0; block < reps[1]; ++block)
    for (int tile = 0; tile < reps[0]; ++tile)
      for (int loadId = 0; loadId < numOfElems/vectorSize; ++loadId){
        auto spatialOffset = mapping.offset(block, tile, loadId);
        Value row = spatialOffset.kDim;
        Value col = spatialOffset.nonKDim;
        bOffsets.offset(block, tile, loadId) = computeOffset(rewriter, loc, row, col, smemObj, srcLayout);
      }
  return bOffsets;
}

Value computeBasePtr(ConversionPatternRewriter &rewriter, Location loc,
                     const SharedMemoryObject &smemObj) {
  Value base = smemObj.base;
  Type type = base.getType();
  for (int i = 0; i < smemObj.strides.size(); ++i) {
    Value offset = sub(i32_val(0), mul(smemObj.offsets[i], smemObj.strides[i]));
    base = gep(type, base, offset);
  }
  return base;
}

/**
 * @brief try find if value is an integer constant
 * 
 * Trace def-use chain and return integer in case we can proof it is constant.
 * Current implementation can trace chains of insertValue->extractValue operations.
 * 
 * @param val Value for that we want to get constant
 * @return std::optional on found integer value or empty std::optional
*/
std::optional<int> findConstValue(Value val) {
  while (val && !val.getDefiningOp<LLVM::ConstantOp>()) {
    LLVM::ExtractValueOp extractValOp = val.getDefiningOp<LLVM::ExtractValueOp>();
    if (!extractValOp)
      return std::optional<int>();
    auto extractPosArr = extractValOp.getPosition();
    if (extractPosArr.size() > 1)
      return std::optional<int>();
    int extractPos = extractPosArr[0];

    int insertPos = -1;
    LLVM::InsertValueOp insertValOp;
    Value container = extractValOp.getOperand();
    do {
      insertValOp = container.getDefiningOp<LLVM::InsertValueOp>();
      if (!insertValOp)
        return std::optional<int>();
      auto insertPosArr = insertValOp.getPosition();
      if (insertPosArr.size() > 1)
        return std::optional<int>();
      insertPos = insertPosArr[0];
      container = insertValOp.getContainer();
    } while(insertPos != extractPos);
    val = insertValOp.getValue();
  }
  if (!val)
    return std::optional<int>();
  auto cOp = val.getDefiningOp<LLVM::ConstantOp>();
  assert(cOp);
  auto valAttr = cOp.getValueAttr();
  auto intAttr = dyn_cast<mlir::IntegerAttr>(valAttr);
  assert(intAttr);
  return intAttr.getInt();
}

bool fastPathAvailable(const SharedMemoryObject &smemObj, const SharedEncodingAttr &srcEncoding, const MfmaEncodingAttr &dstEncoding) {
  if (dstEncoding.getNonKDim() != 32)
    return false;
  if (srcEncoding.getMaxPhase() > 1)
    return false;
  auto stride0 = findConstValue(smemObj.strides[0]);
  auto stride1 = findConstValue(smemObj.strides[1]);
  auto offset0 = findConstValue(smemObj.offsets[0]);
  auto offset1 = findConstValue(smemObj.offsets[1]);
  bool allValuesDefined =
      stride0.has_value() &&
      stride1.has_value() &&
      offset0.has_value() &&
      offset1.has_value();
  if (!allValuesDefined)
    return false;
  if (offset0.value() != 0 || offset1.value() != 0)
    return false;
  return true;
}

// Computes offsets for operand A or transposed operand B
// @param rewriter
// @param loc
// @param elemsPerInstr operand tile shape consumed by one MFMA instruction
// @param waveM wave id for the "non K" axis
// @param laneId lane id in warp [0..63]
// @param warpsPerGroup number of warps in one block
// @param numOfElems number of elements accessed by thread per repetition
// @param reps number of instructions repretition to fully cover dot operand
// @param cSwizzleOffset
llvm::SmallVector<Value>
fastPathComputeOffsetsTy1(ConversionPatternRewriter &rewriter, Location loc,
                  const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                  Value laneId, int warpsPerGroup, int numOfElems,
                  ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  auto numM = reps[0];
  auto numK = reps[1];
  SmallVector<Value> offsets(numM * numK * numOfElems);
  int lineSize = elemsPerInstr[1] * numK;
  int blockSize = elemsPerInstr[0] * warpsPerGroup * lineSize;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveHalf = udiv(laneId, _32);

  Value waveOffset = mul(waveId, i32_val(elemsPerInstr[0] * lineSize));
  Value colOffset = select(icmp_uge(laneId, _32), i32_val(numOfElems), _0);

  for (int block = 0; block < numM; ++block) {
    Value blockOffset = i32_val(block * blockSize);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * elemsPerInstr[1]);
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

// Computes offsets for operand B or transposed operand A
// @param rewriter
// @param loc
// @param elemsPerInstr operand tile shape consumed by one MFMA instruction
// @param waveId wave id for the "non K" axis
// @param laneId lane id in warp [0..63]
// @param warpsPerGroup number of warps per horizontal axis
// @param numOfElems number of elements accessed by threads per repetition
// @param reps number of instructions repretition to fully cover dot operand
// @param cSwizzleOffset
llvm::SmallVector<Value>
fastPathComputeOffsetsTy2(ConversionPatternRewriter &rewriter, Location loc,
                  const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                  Value laneId, int warpsPerGroup, int numOfElems,
                  ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  auto numK = reps[0];
  auto numN = reps[1];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  int lineSize = warpsPerGroup * elemsPerInstr[1] * numN;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveOffset = mul(waveId, i32_val(elemsPerInstr[1]));
  Value colOffset = urem(laneId, _32);

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = i32_val(block * elemsPerInstr[1] * warpsPerGroup);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * elemsPerInstr[0] * lineSize);
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

bool isTransposed(::llvm::ArrayRef<unsigned> order) {
  assert(order.size() == 2 && (order[0] & ~1ul) == 0 &&
         order[0] + order[1] == 1);
  return order[0] == 0;
}

Value loadA(ConversionPatternRewriter &rewriter, Location loc, Value thread,
            DotOperandEncodingAttr encoding,
            TritonGPUToLLVMTypeConverter *typeConverter, Value tensor,
            const SharedMemoryObject &smemObj) {
  auto mfmaLayout = encoding.getParent().cast<MfmaEncodingAttr>();
  assert(mfmaLayout.getNonKDim() == 32);
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto aElemTy = aTensorTy.getElementType();
  auto aElemsPerInstr = encoding.getMFMAElemsPerThread(aElemTy);
  auto mfmaInstrM = aElemsPerInstr[0];
  auto mfmaInstrK = aElemsPerInstr[1];

  auto numReps = encoding.getMFMARep(shape, aElemTy);
  auto numRepM = numReps[0];
  auto numRepK = numReps[1];

  unsigned iWaveSize = triton::gpu::getWarpSize(mfmaLayout);
  Value waveSize = i32_val(iWaveSize);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveM =
      getWaveM(rewriter, loc, wave, warpsPerCTA, mfmaInstrM, shape[0]);
  int numOfElems =
      std::max<int>(mfmaInstrM * mfmaInstrK / iWaveSize /*wave size*/, 1);
  unsigned int maxNumWarps = shape[0] / mfmaInstrM;
  int warpsPerGroupM = std::min(warpsPerCTA[0], maxNumWarps);

  SmallVector<Value> ha;

  if (fastPathAvailable(smemObj, sharedLayout, mfmaLayout)) {
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    SmallVector<Value> offsets;
    if (isTransposed(order)) {
      SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrM};
      SmallVector<int64_t> reps{numReps[1], numReps[0]};
      offsets =
          fastPathComputeOffsetsTy2(rewriter, loc, elemsPerInstr, waveM, lane,
                            warpsPerGroupM, numOfElems, reps, cSwizzleOffset);
    } else {
      offsets =
          fastPathComputeOffsetsTy1(rewriter, loc, aElemsPerInstr, waveM, lane,
                            warpsPerGroupM, numOfElems, numReps, cSwizzleOffset);
    }
    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);

    Type smemPtrTy = getShemPtrTy(aElemTy);

    for (int m = 0; m < numRepM; ++m) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(aElemTy, numOfElems);
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
        if (aElemTy == i8_ty)
          valVec = bitcast(valVec, i32_ty);
        ha.push_back(valVec);
      }
    }
  } else { // normal path
    Value smemBase = computeBasePtr(rewriter, loc, smemObj);

    Type smemPtrTy = getShemPtrTy(aElemTy);

    LoadOffsetMapLinear offsets = computeOffsetsAType(
      rewriter, loc, aElemsPerInstr, waveM, lane, warpsPerGroupM, numOfElems,
      numReps, smemObj, sharedLayout);

    int loadsPerThread = offsets.getNumLoadsPerThread();
    int elemsPerLoad = offsets.getNumElemPerLoad();
    for (int m = 0; m < numRepM; ++m) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(aElemTy, numOfElems);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(aElemTy, elemsPerLoad);
          Value loadOffset = offsets.offset(m, k, loadId);
          Value loadAddress = bitcast(gep(smemPtrTy, smemBase, loadOffset), getShemPtrTy(loadVecTy));
          // llvm::errs() <<  loadAddress << "\n";
          Value vectorValue = load(loadAddress);
          // llvm::errs() << vectorValue << "\n" << loadVecTy << "\n";
          if (numOfElems > 1) {
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal = extract_element(aElemTy, vectorValue, i32_val(elemId));
              // llvm::errs() << elemVal << "\n";
              valVec = insert_element(vecTy, valVec, elemVal, i32_val(loadId * elemsPerLoad + elemId));
              // llvm::errs() << valVec << "\n";
            }
          } else {
            valVec = extract_element(aElemTy, vectorValue, i32_val(0));
          }
        }
        if (aElemTy == i8_ty)
          valVec = bitcast(valVec, i32_ty);
        ha.push_back(valVec);
      }
    }
  }

  MLIRContext *ctx = mfmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(ha.size(), ha[0].getType()));
  auto result = typeConverter->packLLElements(loc, ha, rewriter, structTy);
  return result;
}

Value loadB(ConversionPatternRewriter &rewriter, Location loc, Value thread,
            DotOperandEncodingAttr encoding,
            TritonGPUToLLVMTypeConverter *typeConverter, Value tensor,
            const SharedMemoryObject &smemObj) {
  auto mfmaLayout = encoding.getParent().cast<MfmaEncodingAttr>();
  assert(mfmaLayout.getNonKDim() == 32);
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto bTensorTy = tensor.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = bTensorTy.getShape();
  auto sharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto bElemTy = bTensorTy.getElementType();
  auto bElemsPerInstr = encoding.getMFMAElemsPerThread(bElemTy);
  auto mfmaInstrK = bElemsPerInstr[0];
  auto mfmaInstrN = bElemsPerInstr[1];

  auto numReps = encoding.getMFMARep(shape, bElemTy);
  auto numRepK = numReps[0];
  auto numRepN = numReps[1];

  unsigned iWaveSize = triton::gpu::getWarpSize(mfmaLayout);
  Value waveSize = i32_val(iWaveSize);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveN =
      getWaveN(rewriter, loc, wave, warpsPerCTA, mfmaInstrN, shape[1]);
  int numOfElems =
      std::max<int>(mfmaInstrK * mfmaInstrN / iWaveSize /*wave size*/, 1);

  int macroTileM = std::max<int>(shape[0] / (warpsPerCTA[0] * 32), 1);
  int wptM = std::min<int>(warpsPerCTA[0], macroTileM);
  int macroTileN = std::max<int>(shape[1] / (warpsPerCTA[1] * 32), 1);
  int wptN = std::min<int>(warpsPerCTA[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);

  unsigned int maxNumWarps = shape[1] / mfmaInstrN;
  int warpsPerGroupN = std::min(warpsPerCTA[1], maxNumWarps);

  SmallVector<Value> hb;

  if (fastPathAvailable(smemObj, sharedLayout, mfmaLayout)) {
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);

    llvm::SmallVector<Value> offsets;
    unsigned int maxNumWarps = shape[1] / mfmaInstrN;
    int warpsPerGroupN = std::min(warpsPerCTA[1], maxNumWarps);
    if (isTransposed(order)) {
      SmallVector<int64_t> elemsPerInstr{mfmaInstrN, mfmaInstrK};
      SmallVector<int64_t> reps{numReps[1], numReps[0]};
      offsets =
          fastPathComputeOffsetsTy1(rewriter, loc, elemsPerInstr, waveN, lane,
                            warpsPerGroupN, numOfElems, reps, cSwizzleOffset);
    } else {
      offsets =
          fastPathComputeOffsetsTy2(rewriter, loc, bElemsPerInstr, waveN, lane,
                            warpsPerGroupN, numOfElems, numReps, cSwizzleOffset);
    }

    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);

    Type smemPtrTy = getShemPtrTy(bElemTy);

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
        if (bElemTy == i8_ty)
          valVec = bitcast(valVec, i32_ty);
        hb.push_back(valVec);
      }
    }
  } else { // normal path
    LoadOffsetMapLinear offsets = computeOffsetsBType(
      rewriter, loc, bElemsPerInstr, waveN, lane, warpsPerGroupN, numOfElems,
      numReps, smemObj, sharedLayout);

    Value smemBase = computeBasePtr(rewriter, loc, smemObj);

    Type smemPtrTy = getShemPtrTy(bElemTy);

    int loadsPerThread = offsets.getNumLoadsPerThread();
    int elemsPerLoad = offsets.getNumElemPerLoad();
    for (int n = 0; n < numRepN; ++n) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(bElemTy, numOfElems);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(bElemTy, elemsPerLoad);
          Value loadOffset = offsets.offset(n, k, loadId);
          Value loadAddress = bitcast(gep(smemPtrTy, smemBase, loadOffset), getShemPtrTy(loadVecTy));
          Value vectorValue = load(loadAddress);
          if (numOfElems > 1) {
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal = extract_element(bElemTy, vectorValue, i32_val(elemId));
              valVec = insert_element(vecTy, valVec, elemVal, i32_val(loadId * elemsPerLoad + elemId));
            }
          } else {
            valVec = extract_element(bElemTy, vectorValue, i32_val(0));
          }
        }
        if (bElemTy == i8_ty)
          valVec = bitcast(valVec, i32_ty);
        hb.push_back(valVec);
      }
    }
  }

  MLIRContext *ctx = mfmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(hb.size(), hb[0].getType()));
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
    return loadA(rewriter, loc, thread, encoding, typeConverter, tensor,
                 smemObj);
  case 1:
    // operand $b
    return loadB(rewriter, loc, thread, encoding, typeConverter, tensor,
                 smemObj);
  default:
    assert(false && "unexpected operand idx");
    return Value();
  }
}

} // namespace SharedToDotOperandMFMA

#endif // ifdef USE_ROCM
