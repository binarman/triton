#!/usr/bin/env python3
import triton
import triton.language as tl
import torch


@triton.jit
def matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accum = tl.dot(a, b, acc=accum)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accum)


@triton.jit
def matmul_kernel_explicit_dot3d(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        SPLIT_K: tl.constexpr = 64):
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
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
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = tl.sum(acc, axis=0)
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)


@triton.jit
def matmul_kernel_explicit_dot3d_2(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        SPLIT_K: tl.constexpr = 64):
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
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
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = tl.sum(acc, axis=0)
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)


if __name__ == "__main__":
    BLOCK_M = 1
    BLOCK_N = 32
    BLOCK_K = 512
    K = 512
    a = torch.ones((BLOCK_M, K), dtype=torch.float16, device="cuda")
    b = torch.zeros((K, BLOCK_N), dtype=torch.float16, device="cuda")
    for k in range(K):
        for n in range(BLOCK_N):
            b[k, n] = n if k == 0 else 0.0
    c = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")
    c_ref = torch.matmul(a, b)
    pgm = matmul_kernel[(1, )](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), K,
                               BLOCK_M, BLOCK_N, BLOCK_K, num_warps=1)
    print("dot2d automatic:\n", pgm.asm["ttgir"])
    print("triton:", c, "\nreference:", c_ref)
    assert torch.all(torch.abs(c - c_ref) < 0.01)

    pgm = matmul_kernel_explicit_dot3d[(1, )](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                                              c.stride(1), K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4)
    print("explicit dot3d version1:\n", pgm.asm["ttgir"])
    assert torch.all(torch.abs(c - c_ref) < 0.01)

    pgm = matmul_kernel_explicit_dot3d_2[(1, )](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                                c.stride(0), c.stride(1), K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4)
    print("explicit dot3d version2:\n", pgm.asm["ttgir"])
    assert torch.all(torch.abs(c - c_ref) < 0.01)
