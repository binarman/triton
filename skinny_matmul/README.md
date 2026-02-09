## Potential use cases

### llama.cpp

TBD how to run some test/benchmarks

Kernel implemneting https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu#L143

GEMM shapes MxNxK: 2880x1x2880, 4096x1x14336

Targeted data types:
C/D - fp32
A - custom fixed point 4-bit data type
B - custom fixed point 8-bit data type

TBD: add memory operand mapping.

Used dot variant: int8xint8 -> int32

### hipBLASLt

TBD how to run some test/benchmarks

https://github.com/ROCm/hipBLASLt/pull/1258

TBD shapes

Targeted data types: fp16

TBD: memory

### Triton tutorial


### MxFP8 kernel?
