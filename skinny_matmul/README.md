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


## Experiemnts

### Tutorial based kernel

Utilizing 3d fma dot and reduction across K dim.
Test shape `4096 x 1 x 14336`, dtype fp16xfp16 -> fp16

rocBLAS reports ~**1.1 TFLOPS**

MMA based kernel resports **1.3 TFLOPS**

#### v_dot based approach

Benefits of fma dot and 3d dot multiplication:
- maximizing parallelism using only M dimension.
- no need to use LDS
- no need to use atomics

Theoretical performance on mi308 N=1 is **6 TFLOPS**.
Assuming most of work is required to load matrix A,
i.e. `PERF = 2*M*N*K/T`; `T = 2(bytes in fp16)*M*K/6e12(HBM bandwidth in bytes/sec)`
`PERF = 2*M*N*K/(2*M*K/6e12) = N * 6e12`

Naive implementation: **1.76 TFLOPS**
- No pipelining
- One row in A per warp

Naive implementation, 8 rows per warp: **2.1 TFLOPS**
- Better utilization of memory
- Store process multiple elements
