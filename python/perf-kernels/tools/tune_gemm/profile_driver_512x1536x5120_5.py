import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_gemm import gen_rotating_tensors

@triton.jit
def matmul_kernel_BM256_BN256_BK16_GM2_SK1_nW2_nS2_EU0_kP2_mfma16_schedNONE(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, stride_bias, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr,
                  EVEN_K: tl.constexpr, GRID_MN: tl.constexpr, NUM_XCDS: tl.constexpr):


    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        pid = xcd * pids_per_xcd + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    if SPLIT_K == 1:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
    else:
        offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    if BIAS:
        bias_ptrs = bias_ptr + offs_am * stride_bias
        bias = tl.load(bias_ptrs, mask=offs_am < M, other=0.0)
    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    c = accumulator.to(c_ptr.type.element_ty)
    if BIAS:
        c += bias[:, None]
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)



def matmul_BM256_BN256_BK16_GM2_SK1_nW2_nS2_EU0_kP2_mfma16_schedNONE(a, b, c, bias, M, N, K, am, ak, bk, bn, cm, cn, biasn):
    grid = triton.cdiv(M, 256) * triton.cdiv(N, 256), 1
    matmul_kernel_BM256_BN256_BK16_GM2_SK1_nW2_nS2_EU0_kP2_mfma16_schedNONE[grid](
        a, b, c, bias,
        M, N, K,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 2,
        SPLIT_K = 1,
        num_warps = 2,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True,
        GRID_MN = grid[0],
        NUM_XCDS = 8,
        schedule_hint = "none",
    )
    return c

def test_gemm(M, N, K, rotating_buffer_size, bias_size):
    tensors = gen_rotating_tensors(M, N, K, 'fp16', False, 'fp16', False, 'fp16',
                                   1, 'randn', rotating_buffer_size, bias_size, device='cuda')

    a = tensors['input_a'][0]
    b = tensors['input_b'][0]
    c = tensors['output_c'][0]
    assert bias_size == M or bias_size == 0

    stride_bias = tensors['bias'][0].stride(0) if bias_size > 0 else 0

    try:
        with open("/rocm-triton/python/perf-kernels/tools/tune_gemm/output/compile_driver.py.failed_configs", "r") as f:
            failed_configs = [cfg.strip() for cfg in f.readlines()]
    except Exception:
        failed_configs = []

    if  'BM256_BN256_BK16_GM2_SK1_nW2_nS2_EU0_kP2_mfma16_schedNONE' not in failed_configs:
        print('BM256_BN256_BK16_GM2_SK1_nW2_nS2_EU0_kP2_mfma16_schedNONE')
        rotating_num = tensors['rotating_num']
        print("rotating_num", rotating_num)
        for i in range(2):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            print("a tensor", a.data_ptr(), a.shape, a.dtype)
            b = tensors['input_b'][0]
            print("b tensor", b.data_ptr())
            c = tensors['output_c'][0]
            print("c tensor", c.data_ptr())
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            d = matmul_BM256_BN256_BK16_GM2_SK1_nW2_nS2_EU0_kP2_mfma16_schedNONE(a, b, c, bias, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    return d

def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=1, help='number of threads')
    parser.add_argument("-rotating_tensor", type=int, default=512, help='size of rotating buffer (MB), default: 512')
    args = parser.parse_args()
    numThreads = args.n
    rotating_buffer_size = args.rotating_tensor
    test_gemm(512, 1536, 5120, rotating_buffer_size, 0)

if __name__ == '__main__':
   sys.exit(main())
