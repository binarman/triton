# This file is a copy of tutorial with some parts cut off for convenience
import torch

from triton.backends.compiler import GPUTarget
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

from triton._filecheck import filecheck_test, run_parser


def make_args(*args, **kwargs):
    return args, kwargs


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_autotune_config():
    sizes = [
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1},
    ]
    return [triton.Config(s | {'matrix_instr_nonkdim': 0}, num_warps=1, num_stages=2) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 0}, num_warps=2, num_stages=2) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 0}, num_warps=4, num_stages=2) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 0}, num_warps=1, num_stages=1) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 0}, num_warps=2, num_stages=1) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 0}, num_warps=4, num_stages=1) for s in sizes]


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K, stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ACTIVATION: tl.constexpr  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0, cache_modifier=".cg")
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


if __name__ == "__main__":
    from common_wrappers import triton_benchmark
    triton_benchmark.run_isolated_triton_bench(matmul_kernel)
