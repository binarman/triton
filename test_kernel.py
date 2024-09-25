#!/usr/bin/env python3
import triton
import triton.language as tl
import torch


# reference kernel
# should be optimized by compiler
@triton.jit
def matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_bn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_cn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)


# load as reference, do all transformations after load
@triton.jit
def matmul_kernel_explicit_dot3d(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        SPLIT_K: tl.constexpr = 64):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_bn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((SPLIT_K, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a = tl.reshape(a, (BLOCK_SIZE_M, SPLIT_K, BLOCK_SIZE_K // SPLIT_K))
        a = tl.permute(a, (1, 0, 2))
        b = tl.reshape(b, (SPLIT_K, BLOCK_SIZE_K // SPLIT_K, BLOCK_SIZE_N))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = tl.sum(acc, axis=0)
    offs_cm = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_cn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)


# load a as in reference, transform later, load b in 3d
@triton.jit
def matmul_kernel_explicit_dot3d_2(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        SPLIT_K: tl.constexpr = 64):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_bn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // SPLIT_K)
    offs_split_k = tl.arange(0, SPLIT_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_split_k[:, None, None] * BLOCK_SIZE_K // SPLIT_K * stride_bk +
                      offs_bk[None, :, None] * stride_bk + offs_bn[None, None, :] * stride_bn)
    acc = tl.zeros((SPLIT_K, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a = tl.reshape(a, (BLOCK_SIZE_M, SPLIT_K, BLOCK_SIZE_K // SPLIT_K))
        a = tl.permute(a, (1, 0, 2))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = tl.sum(acc, axis=0)
    offs_cm = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_cn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)


# load everything in 3d, transform a after load
@triton.jit
def matmul_kernel_explicit_dot3d_3(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        SPLIT_K: tl.constexpr = 64):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_bn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    offs_ak = tl.arange(0, BLOCK_SIZE_K // SPLIT_K)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // SPLIT_K)
    offs_split_k = tl.arange(0, SPLIT_K)
    a_ptrs = a_ptr + (offs_am[:, None, None] * stride_am + offs_split_k[None, :, None] * BLOCK_SIZE_K // SPLIT_K *
                      stride_ak + offs_ak[None, None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_split_k[:, None, None] * BLOCK_SIZE_K // SPLIT_K * stride_bk +
                      offs_bk[None, :, None] * stride_bk + offs_bn[None, None, :] * stride_bn)
    acc = tl.zeros((SPLIT_K, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a = tl.permute(a, (1, 0, 2))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = tl.sum(acc, axis=0)
    offs_cm = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_cn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)


# load in 3d in required layout
@triton.jit
def matmul_kernel_explicit_dot3d_4(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        SPLIT_K: tl.constexpr = 64):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_bn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    offs_ak = tl.arange(0, BLOCK_SIZE_K // SPLIT_K)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // SPLIT_K)
    offs_split_k = tl.arange(0, SPLIT_K)
    a_ptrs = a_ptr + (offs_am[None, :, None] * stride_am + offs_split_k[:, None, None] * BLOCK_SIZE_K // SPLIT_K *
                      stride_ak + offs_ak[None, None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_split_k[:, None, None] * BLOCK_SIZE_K // SPLIT_K * stride_bk +
                      offs_bk[None, :, None] * stride_bk + offs_bn[None, None, :] * stride_bn)
    acc = tl.zeros((SPLIT_K, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = tl.sum(acc, axis=0)
    offs_cm = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_cn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)


if __name__ == "__main__":
    BLOCK_M = 1
    BLOCK_N = 32
    BLOCK_K = 512
    M = 2
    N = 128
    K = 4096
    a = torch.randn((M, K), dtype=torch.float16, device="cuda") * 0.1
    b = torch.randn((K, N), dtype=torch.float16, device="cuda") * 0.1
    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    c_ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    pgm = matmul_kernel[M // BLOCK_M, N // BLOCK_N](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                                    c.stride(0), c.stride(1), K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4)
    print("dot2d automatic:\n", pgm.asm["ttgir"])
    print("triton:", c, "\nreference:", c_ref)
    assert torch.all(torch.abs(c - c_ref) < 0.01)

    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    pgm = matmul_kernel_explicit_dot3d[M // BLOCK_M, N // BLOCK_N](a, b, c, a.stride(0), a.stride(1), b.stride(0),
                                                                   b.stride(1), c.stride(0), c.stride(1), K, BLOCK_M,
                                                                   BLOCK_N, BLOCK_K, num_warps=4)
    print("explicit dot3d version1:\n", pgm.asm["ttgir"])
    assert torch.all(torch.abs(c - c_ref) < 0.01)

    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    pgm = matmul_kernel_explicit_dot3d_2[M // BLOCK_M, N // BLOCK_N](a, b, c, a.stride(0), a.stride(1), b.stride(0),
                                                                     b.stride(1), c.stride(0), c.stride(1), K, BLOCK_M,
                                                                     BLOCK_N, BLOCK_K, num_warps=4)
    print("explicit dot3d version2:\n", pgm.asm["ttgir"])
    assert torch.all(torch.abs(c - c_ref) < 0.01)

    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    pgm = matmul_kernel_explicit_dot3d_3[M // BLOCK_M, N // BLOCK_N](a, b, c, a.stride(0), a.stride(1), b.stride(0),
                                                                     b.stride(1), c.stride(0), c.stride(1), K, BLOCK_M,
                                                                     BLOCK_N, BLOCK_K, num_warps=4)
    print("explicit dot3d version3:\n", pgm.asm["ttgir"])
    assert torch.all(torch.abs(c - c_ref) < 0.01)

    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    pgm = matmul_kernel_explicit_dot3d_4[M // BLOCK_M, N // BLOCK_N](a, b, c, a.stride(0), a.stride(1), b.stride(0),
                                                                     b.stride(1), c.stride(0), c.stride(1), K, BLOCK_M,
                                                                     BLOCK_N, BLOCK_K, num_warps=4)
    print("explicit dot3d version4:\n", pgm.asm["ttgir"])
    assert torch.all(torch.abs(c - c_ref) < 0.01)
