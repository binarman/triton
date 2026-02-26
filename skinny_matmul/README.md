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

Performance is computed with following formula: `2 * M * N * K * 1e-12 / (ms * 1e-3)`

Bandwidth is computed with following formula: `sizeof(dtype) * (M * K + N * K) * 1e-12 / (ms * 1e-3)`

run `./skinny_matmul/run_all.py` to get following tables.
run `python3 ./skinny_matmul/kernels/v1_dot2d_mma.py` or any other kernel to run one particular kernel.

### 4096 x 1 x 16384 fp16 x fp16 -> fp16

| Kernel              | v0_torch | v1_dot2d_mma | v2_dot2d_fma | v3_dot3d | v4_gluon_dot3d | v5_gluon_dot3d_local_b | v6_gluon_dot3d_flex_m |
|---------------------|----------|--------------|--------------|----------|----------------|------------------------|-----------------------|
| bandwidth(TBytes/s) | 1.1531   | 1.3673       | 0.6071       | 2.1261   | 2.2966         | 1.5308                 | 2.0147                |
| lds                 | N/A      | 24576        | 16640        | 1024     | 0              | 32768                  | 512                   |
| performance(TFLOPS) | 1.1528   | 1.3670       | 0.6069       | 2.1255   | 2.2961         | 1.5304                 | 2.0143                |
| sgpr_count          | N/A      | 23           | 30           | 30       | 23             | 23                     | 23                    |
| vgpr_count          | N/A      | 102          | 156          | 60       | 37             | 92                     | 18                    |

If use constexpr K:

| Metric              | v0_torch | v1_dot2d_mma | v2_dot2d_fma | v3_dot3d | v4_gluon_dot3d | v5_gluon_dot3d_local_b | v6_gluon_dot3d_flex_m |
|---------------------|----------|--------------|--------------|----------|----------------|------------------------|-----------------------|
| bandwidth(TBytes/s) | 1.1509   | 1.3964       | 0.6215       | 2.1396   | 2.3449         | 1.8749                 | 2.0734                |
| vgpr_count          | N/A      | 101          | 374          | 56       | 58             | 92                     | 27                    |

### 4096 x 1 x 16384 fp8 x fp8 -> fp16

| Kernel              | v1_dot2d_mma | v2_dot2d_fma | v3_dot3d | v4_gluon_dot3d | v5_gluon_dot3d_local_b | v6_gluon_dot3d_flex_m |
|---------------------|--------------|--------------|----------|----------------|------------------------|-----------------------|
| bandwidth(TBytes/s) | 0.6694       | 0.1156       | 1.2132   | 1.5870         | 1.3080                 | 1.5070                |
| lds                 | 20480        | 65536        | 0        | 0              | 16384                  | 512                   |
| performance(TFLOPS) | 1.3385       | 0.2312       | 2.4259   | 3.1731         | 2.6154                 | 3.0132                |
| sgpr_count          | 23           | 30           | 23       | 23             | 23                     | 23                    |
| vgpr_count          | 112          | 363          | 25       | 100            | 63                     | 47                    |


If use constexpr K:

| Metric              | v1_dot2d_mma | v2_dot2d_fma | v3_dot3d | v4_gluon_dot3d | v5_gluon_dot3d_local_b | v6_gluon_dot3d_flex_m |
|---------------------|--------------|--------------|----------|----------------|------------------------|-----------------------|
| bandwidth(TBytes/s) | 0.6773       | 0.1231       | 1.5002   | 1.7386         | 1.4225                 | 1.5677                |
| vgpr_count          | 116          | 252          | 22       | 59             | 89                     | 43                    |


#### v0_torch kernel

Simple `torch.matmul` invocation.

Benchmark in `kernels/v0_torch.py`

#### v1_dot2d_mma

Kernel from triton tutorial.
Added few additional configs in autotune.

Benchmark in `kernels/v1_dot2d_mma.py`

#### v2_dot2d_fma

Kernel from triton tutorial, but using fma dot instead of mma.
More additional configs added in autotune.

Benchmark in `kernels/v2_dot2d_fma.py`

#### v3_dot3d

Triton kernel, which uses 3d dot to distribute k dim between threads and reduce in epilog.

Benchmark in `kernels/v3_dot3d.py`

#### v4_gluon_dot3d

Gluon kernel, which uses 3d dot to distribute k dim between threads and reduce in epilog.
No LDS used, each workgroup contains only one warp.

Benchmark in `kernels/v4_gluon_dot3d.py`

#### v5_gluon_dot3d_local_b

Gluon kernel, similar to gluon_dot3d, but loads whole b tensor in LDS in prolog.
Each workgroup contains 4 warps, so they share same LDS buffer and there are enough LDS for all workgroups to run at the same time.

Benchmark in `kernels/v5_gluon_dot3d_local_b.py`

#### v6_gluon_dot3d_flex_m

Gluon kernel, similar to gluon_dot3d, but do not use tt.dot [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N], instead use explicit for loop over A rows.
Benefits are: can choose any number of rows per workgroup, not limited to power of 2.

Benchmark in `kernels/v6_gluon_dot3d_flex_m.py`
