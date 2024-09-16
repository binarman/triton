#!/opt/conda/envs/py_3.9/bin/python3
import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from myKernels import *

def matmul_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    print(f"BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16")
    matmul_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
    return True

def compile_kernels(M, N, K, num_sms, rotating_buffer_size, bias_size, num_threads):
    thread_pool = multiprocessing.Pool(processes=num_threads)

    assert bias_size == M or bias_size == 0

    stride_bias = 1 if bias_size > 0 else 0
    stride_am, stride_ak = M, 1
    stride_bk, stride_bn = 1, N
    stride_cm, stride_cn = N, 1
    task_args = (M, N, K, num_sms,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 stride_cm, stride_cn, stride_bias)

    results = []
    config_names = []

    try_compile_config_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(*task_args)

def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=32, help='number of threads')
    parser.add_argument("-rotating_tensor", type=int, default=0, help='size of rotating buffer (MB), default: 0')
    args = parser.parse_args()
    numThreads = args.n
    rotating_buffer_size = args.rotating_tensor
    compile_kernels(4864, 8192, 4160, 304, rotating_buffer_size, 0, numThreads)

if __name__ == '__main__':
   sys.exit(main())
