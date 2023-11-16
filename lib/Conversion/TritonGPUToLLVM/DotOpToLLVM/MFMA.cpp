/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifdef USE_ROCM

#include "../DotOpToLLVM.h"
#include "../Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MfmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

enum class MatrixCoreType : uint8_t {
  // D = AB + C
  FP32_FP8_FP8_FP32,
  FP32_FP8_BF8_FP32,
  FP32_BF8_FP8_FP32,
  FP32_BF8_BF8_FP32,
  FP32_FP16_FP16_FP32,
  FP32_BF16_BF16_FP32,
  FP32_BF16_BF16_FP32_1K,
  FP32_FP32_FP32_FP32,
  FP64_FP64_FP64_FP64,
  INT32_INT8_INT8_INT32,
  INT32_INT8_INT8_INT32_CDNA3,  
  NOT_APPLICABLE,
};

struct MFMAInstrDescr {
  MatrixCoreType coreType;
  unsigned size;
};

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

struct DotOpMFMAConversionHelper {
  MfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(
      MfmaEncodingAttr mfmaLayout, ConversionPatternRewriter &rewriter,
      TritonGPUToLLVMTypeConverter *typeConverter, Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  Value generateMFMA32Op(MatrixCoreType coreType, Value valA, Value valB, Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    switch (coreType) {
    case MatrixCoreType::FP32_FP8_FP8_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x16_fp8_fp8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP8_BF8_FP32:
        return rewriter.create<ROCDL::mfma_f32_32x32x16_fp8_bf8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF8_FP8_FP32:
        return rewriter.create<ROCDL::mfma_f32_32x32x16_bf8_fp8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF8_BF8_FP32:
        return rewriter.create<ROCDL::mfma_f32_32x32x16_bf8_bf8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP16_FP16_FP32:
        return rewriter.create<ROCDL::mfma_f32_32x32x8f16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32:
        return rewriter.create<ROCDL::mfma_f32_32x32x4bf16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32_1K:
        return rewriter.create<ROCDL::mfma_f32_32x32x8bf16_1k>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP32_FP32_FP32:
        return rewriter.create<ROCDL::mfma_f32_32x32x2f32>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::INT32_INT8_INT8_INT32:
        return rewriter.create<ROCDL::mfma_i32_32x32x8i8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::INT32_INT8_INT8_INT32_CDNA3:
        return rewriter.create<ROCDL::mfma_i32_32x32x16_i8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP64_FP64_FP64_FP64:
      return rewriter.create<ROCDL::mfma_f64_16x16x4f64>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    default:
      llvm::report_fatal_error("MFMA 32x32 data type not supported");
    }
  }

  Value generateMFMA16Op(MatrixCoreType coreType, Value valA, Value valB, Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    switch (coreType) {
    case MatrixCoreType::FP32_FP16_FP16_FP32:
        return rewriter.create<ROCDL::mfma_f32_16x16x16f16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32:
        return rewriter.create<ROCDL::mfma_f32_16x16x8bf16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32_1K:
        return rewriter.create<ROCDL::mfma_f32_16x16x16bf16_1k>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP32_FP32_FP32:
        return rewriter.create<ROCDL::mfma_f32_16x16x4f32>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::INT32_INT8_INT8_INT32:
        return rewriter.create<ROCDL::mfma_i32_16x16x16i8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP64_FP64_FP64_FP64:
      return rewriter.create<ROCDL::mfma_f64_16x16x4f64>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    default:
      llvm::report_fatal_error("MFMA data type not supported");
    }
  }

  Value generateMFMA4Op(MatrixCoreType coreType, Value valA, Value valB, Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    switch (coreType) {
    case MatrixCoreType::FP32_FP16_FP16_FP32:
        return rewriter.create<ROCDL::mfma_f32_4x4x4f16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32:
        return rewriter.create<ROCDL::mfma_f32_4x4x2bf16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32_1K:
        return rewriter.create<ROCDL::mfma_f32_4x4x4bf16_1k>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP32_FP32_FP32:
        return rewriter.create<ROCDL::mfma_f32_4x4x1f32>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::INT32_INT8_INT8_INT32:
        return rewriter.create<ROCDL::mfma_i32_4x4x4i8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    default:
      llvm::report_fatal_error("MFMA4 data type not supported");
    }

  }

  Value generateMFMAOp(MFMAInstrDescr mfmaDescr, Value valA, Value valB,
                       Value valC) const {
    switch (mfmaDescr.size) {
      case 32:
      return generateMFMA32Op(mfmaDescr.coreType, valA, valB, valC);
      break;
      case 16:
      return generateMFMA16Op(mfmaDescr.coreType, valA, valB, valC);
      break;
      case 4:
      return generateMFMA4Op(mfmaDescr.coreType, valA, valB, valC);
      default:
      llvm::report_fatal_error("MFMA nonkDim size is not supported");
    }
    return Value();
  }

  int getNumSubmatrices(Type elementType, int nonKDim) const {
    switch (nonKDim) {
      case 32:
      case 16:
        return 1;
        break;
      case 4:
        assert(elementType.getIntOrFloatBitWidth() <= 32 && "fp64 is not supported yet");
        assert(elementType.getIntOrFloatBitWidth() != 8 || elementType.isInteger(8) && "fp8 is not supported yet");
        return 16;
        break;
      default:
        llvm::report_fatal_error("unsupported nonKDim in MFMA dot");
    }
    return -1;
  }

  // TODO unify this function with Utility.cpp:supportMFMATypes
  static MatrixCoreType getMatrixCoreTypeFromDot(DotOp op) {
    auto aOperandTy = op.getA().getType();
    auto aTensorTy = aOperandTy.cast<RankedTensorType>();
    auto aElemTy = aTensorTy.getElementType();
    auto bOperandTy = op.getB().getType();
    auto bTensorTy = bOperandTy.cast<RankedTensorType>();
    auto bElemTy = bTensorTy.getElementType();

    auto dotOpEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto mfmaEncoding = dotOpEncoding.getParent().cast<MfmaEncodingAttr>();
    if (aElemTy.isFloat8E4M3FNUZ() && bElemTy.isFloat8E4M3FNUZ())
      return MatrixCoreType::FP32_FP8_FP8_FP32;
    if (aElemTy.isFloat8E4M3FNUZ() && bElemTy.isFloat8E5M2FNUZ())
      return MatrixCoreType::FP32_FP8_BF8_FP32;
    if (aElemTy.isFloat8E5M2FNUZ() && bElemTy.isFloat8E4M3FNUZ())
      return MatrixCoreType::FP32_BF8_FP8_FP32;
    if (aElemTy.isFloat8E5M2FNUZ() && bElemTy.isFloat8E5M2FNUZ())
      return MatrixCoreType::FP32_BF8_BF8_FP32;
    if (aElemTy.isF16())
      return MatrixCoreType::FP32_FP16_FP16_FP32;
    if (aElemTy.isF32())
      return MatrixCoreType::FP32_FP32_FP32_FP32;
    if (aElemTy.isBF16()) {
      auto nonKDim = mfmaEncoding.getNonKDim();
      auto kWidth = dotOpEncoding.getKWidth();
      if ((nonKDim == 32 ||nonKDim == 16 || nonKDim == 4) && kWidth == 4) {
        return MatrixCoreType::FP32_BF16_BF16_FP32_1K;
      } else {
        assert((nonKDim == 32 && kWidth == 2) ||
               (nonKDim == 16 && kWidth == 2) ||
               (nonKDim == 4 && kWidth == 2));
        return MatrixCoreType::FP32_BF16_BF16_FP32;
      }
    }
    if (aElemTy.isInteger(8)) {
      auto nonKDim = mfmaEncoding.getNonKDim();
      auto kWidth = dotOpEncoding.getKWidth();
      if ((nonKDim == 32 ||nonKDim == 16 || nonKDim == 4) && kWidth == 8) {
        return MatrixCoreType::INT32_INT8_INT8_INT32_CDNA3;
      }
      else {
        assert((nonKDim == 32 ||nonKDim == 16 || nonKDim == 4) && kWidth == 4);
        return MatrixCoreType::INT32_INT8_INT8_INT32;
      }
    }
    if (aElemTy.isF64())
      return MatrixCoreType::FP64_FP64_FP64_FP64;
    return MatrixCoreType::NOT_APPLICABLE;
  }

  static MFMAInstrDescr getMatrixInstrDescr(DotOp op) {
    MFMAInstrDescr descr;
    auto tensorTy = op.getD().getType().cast<RankedTensorType>();
    auto encoding = tensorTy.getEncoding().cast<MfmaEncodingAttr>();
    descr.coreType = getMatrixCoreTypeFromDot(op);
    descr.size = encoding.getNonKDim();
    return descr;
  }

  Value bflSwizzle(Value val, int stride, Value laneId) const {
    GCNBuilder builder;
    if (stride > 0) {
      laneId = xor_(laneId, i32_val(stride));
      auto shfl = builder.create("ds_permute_b32");
      auto dOpr = builder.newOperand("=v");
      // Multiple lineId by 4. (More on permute instruction semantics:
      // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=180
      Value byteOffset = i32_val(2);
      Value permuteAddr = shl(laneId, byteOffset);
      auto addrOpr = builder.newOperand(permuteAddr, "v");
      auto aOpr = builder.newOperand(val, "v");
      (*shfl)(dOpr, addrOpr, aOpr);
    } else {
      // This map facilates the butterfly shuffle pattern for a stride less
      // than 16. The pattern stride is the key of the map.
      DenseMap<short, unsigned int> masks{
          {16, 0x401F}, {8, 0x201F}, {4, 0x101F}, {2, 0x081F}, {1, 0x041F}};
      auto shfl = builder.create("ds_swizzle_b32");
      auto dOpr = builder.newOperand("=v");
      auto aOpr = builder.newOperand(val, "v");
      auto maskOpr =
          builder.newConstantOperand("offset:" + std::to_string(masks[stride]));
      (*shfl)(dOpr, aOpr, maskOpr);
    }
    auto swait = builder.create("s_waitcnt lgkmcnt(0)");
    (*swait)();
    return builder.launch(rewriter, loc, val.getType(), true);
  }

  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 && "numSubBlocks in not pow 2!");
    constexpr int waveSize = 64;
    int subBlockSize = waveSize / numSubBlocks;
    Value laneId = getThreadId();
    laneId = and_(laneId, i32_val(waveSize - 1));
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = extract_element(elemType, acc, i32_val(i));

    while (subBlockSize < waveSize) {
      for (int i = 0; i < numScalars; ++i) {
        Value other_acc = bflSwizzle(accScalar[i], subBlockSize, laneId);
        if (elemType.isInteger(32))
          accScalar[i] = add(accScalar[i], other_acc);
        else
          accScalar[i] = fadd(accScalar[i], other_acc);
      }
      subBlockSize *= 2;
    }
    Value reducedAcc = undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc = insert_element(vecTy, reducedAcc, accScalar[i], i32_val(i));
    return reducedAcc;
  }

  void printValue(std::string prefix, Value v) const {
    auto ctx = v.getContext();
    std::vector<Value> values;
    auto vTy = v.getType();
    if (auto vecTy = dyn_cast<VectorType>(vTy)) {
      auto elemTy = vecTy.getElementType();
      for (int i = 0; i < vecTy.getNumElements(); ++i) {
        values.push_back(extract_element(elemTy, v, i32_val(i)));
      }
    } else {
      values.push_back(v);
    }
    auto prefixAttr = mlir::StringAttr::get(ctx, prefix);
    rewriter.create<triton::PrintOp>(loc, prefixAttr, values);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto nonKDim = mfmaLayout.getNonKDim();
    assert(nonKDim == 32 || nonKDim == 16 || nonKDim == 4);
    auto mfmaInstrDescr = getMatrixInstrDescr(op);

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto bTensorTy = b.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();
    auto elemTy = aTensorTy.getElementType();

    auto aEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto bEncoding = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();

    auto repA = aEncoding.getMFMARep(aTensorTy.getShape(), elemTy);
    auto repB = bEncoding.getMFMARep(bTensorTy.getShape(), elemTy);

    assert(repA[1] == repB[0]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[0];
    auto numRepN = repB[1];
    auto numRepK = repA[1];

    ValueTable ha = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepM, numRepK, aTensorTy.getElementType());
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepN, numRepK, aTensorTy.getElementType());
    auto dstElemTy = dTensorTy.getElementType();
    auto fc =
        typeConverter->unpackLLElements(loc, loadedC, rewriter, dstElemTy);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks = getNumSubmatrices(aTensorTy.getElementType(), nonKDim);
    auto elemsPerVec = nonKDim * nonKDim * subBlocks / warpSize;

    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        Value acc = undef(vecTy);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          acc = insert_element(
              vecTy, acc, fc[m * numRepN * elemsPerVec + n * elemsPerVec + v],
              i32_val(v));
        }

        for (size_t k = 0; k < numRepK; k++) {
          printValue("Arg A (k: " + std::to_string(k) + ", m: " + std::to_string(m) + ") ", ha[{m, k}]);
          printValue("Arg B (k: " + std::to_string(k) + ", n: " + std::to_string(n) + ") ", hb[{n, k}]);
          acc =
              mfmaLayout.getIsTransposed()
                  ? generateMFMAOp(mfmaInstrDescr, hb[{n, k}], ha[{m, k}], acc)
                  : generateMFMAOp(mfmaInstrDescr, ha[{m, k}], hb[{n, k}], acc);
        }
        printValue("Acc no reduce (m: " + std::to_string(m) + ", n: " + std::to_string(n) + ") ", acc);
        acc = reduceSubBlocks(subBlocks, acc);
        printValue("Acc reduced (m: " + std::to_string(m) + ", n: " + std::to_string(n) + ") ", acc);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          fc[m * numRepN * elemsPerVec + n * elemsPerVec + v] =
              extract_element(dstElemTy, acc, i32_val(v));
        }
      }
    }

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = typeConverter->packLLElements(loc, fc, rewriter, structTy);
    rewriter.replaceOp(op, res);

    return success();
  }

  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int n0, int n1,
                                                 Type type) const {
    auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);
    ValueTable vals;
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        vals[{i, j}] = elems[n1 * i + j];
      }
    }
    return vals;
  }
};

} // namespace

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return tensor.getType().cast<RankedTensorType>();
  };

  assert(rankedTType(op.getA()).getEncoding().isa<DotOperandEncodingAttr>() &&
         rankedTType(op.getB()).getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(cTensorTy.getEncoding().isa<MfmaEncodingAttr>() &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = op.getResult()
                        .getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<MfmaEncodingAttr>();

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}

#endif // ifdef USE_ROCM
