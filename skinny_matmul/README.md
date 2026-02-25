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

### MxFP8 kernel

TBD

## Experiments

### 4096 x 1 x 16384 fp16 x fp16 -> fp16/fp32

Kernel | torch | dot2d_mma | dot2d_fma | dot3d | gluon_dot3d | gluon_dot3d | gluon_dot3d_local_b | gluon_dot3d_flex_m
Performance(TFLOPS) | 1.152286 | 1.370338 | 0.606801 | 2.120142 | 2.3048 | 1.538471 | 2.024003
Mem bandwidth(TBytes/s) |
LDS | N/A | 24576 | 16640 | 1024 | 0 | 32768 | 512
VGPRs | N/A | 102 | 156 | 60 | 65 | 92 | 18
SGPRS | N/A | 23 | 30 | 30 | 23 | 23 | 23

### 4096 x 1 x 16384 fp8 x fp8 -> fp16

Kernel | torch | dot2d_mma | dot2d_fma | dot3d | gluon_dot3d | gluon_dot3d | gluon_dot3d_local_b | gluon_dot3d_flex_m
Performance(TFLOPS) |
Mem bandwidth(TBytes/s) |
LDS | N/A |
VGPRs | N/A |
SGPRS | N/A |

#### torch kernel

Simple `torch.matmul` invocation.

Benchmark in `kernels/v0_torch.py`

#### dot2d_mma

Kernel from triton tutorial.
Added few additional configs in autotune.

Benchmark in `kernels/v1_dot2d_mma.py`

#### dot2d_fma

Kernel from triton tutorial, but using fma dot instead of mma.
More additional configs added in autotune.

Benchmark in `kernels/v2_dot2d_fma.py`

#### dot3d

Triton kernel, which uses 3d dot to distribute k dim between threads and reduce in epilog.

Benchmark in `kernels/v3_dot3d.py`

#### gluon_dot3d

Gluon kernel, which uses 3d dot to distribute k dim between threads and reduce in epilog.
No LDS used, each workgroup contains only one warp.

Benchmark in `kernels/v4_gluon_dot3d.py`

#### gluon_dot3d_local_b

Gluon kernel, similar to gluon_dot3d, but loads whole b tensor in LDS in prolog.
Each workgroup contains 4 warps, so they share same LDS buffer and there are enough LDS for all workgroups to run at the same time.

Benchmark in `kernels/v5_gluon_dot3d_local_b.py`

#### gluon_dot3d_flex_m

Gluon kernel, similar to gluon_dot3d, but do not use tt.dot [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N], instead use explicit for loop over A rows.
Benefits are: can choose any number of rows per workgroup, not limited to power of 2.

Benchmark in `kernels/v6_gluon_dot3d_flex_m.py`

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
