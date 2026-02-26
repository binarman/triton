# This file is a copy of tutorial with some parts cut off for convenience
import torch

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

THREADS_PER_WARP = triton.language.constexpr(triton.runtime.driver.active.get_current_target().warp_size)


def get_autotune_config():
    elems_in_load = 16 // 2  # 16 bytes per load, 2 bytes in one element
    sizes = [
        {
            'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1,
            'ALIGNED_BLOCK_SIZE_M': 1
        },
        {
            'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 1
        },
        {
            'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 1
        },
        {
            'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1,
            'ALIGNED_BLOCK_SIZE_M': 2
        },
        {
            'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 2
        },
        {
            'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 2
        },
        {
            'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1,
            'ALIGNED_BLOCK_SIZE_M': 4
        },
        {
            'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 4
        },
        {
            'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 4
        },
        {
            'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1,
            'ALIGNED_BLOCK_SIZE_M': 8
        },
        {
            'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 8
        },
        {
            'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 8
        },
        {
            'BLOCK_SIZE_M': 12, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1,
            'ALIGNED_BLOCK_SIZE_M': 16
        },
        {
            'BLOCK_SIZE_M': 12, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 16
        },
        {
            'BLOCK_SIZE_M': 12, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 16
        },
        {
            'BLOCK_SIZE_M': 13, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1,
            'ALIGNED_BLOCK_SIZE_M': 16
        },
        {
            'BLOCK_SIZE_M': 13, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 16
        },
        {
            'BLOCK_SIZE_M': 13, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M':
            1, 'ALIGNED_BLOCK_SIZE_M': 16
        },
    ]
    return [triton.Config(s, num_warps=nw) for s in sizes for nw in [1, 2, 4]]


@gluon.jit
def gluon_leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@gluon.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: ttgl.constexpr, BLOCK_SIZE_N: ttgl.constexpr, BLOCK_SIZE_K: ttgl.constexpr,  #
                  GROUP_SIZE_M: ttgl.constexpr,  #
                  ACTIVATION: ttgl.constexpr,  #
                  ALIGNED_BLOCK_SIZE_M: ttgl.constexpr  #
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

    A_LOAD_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1,
                                                                        8], threads_per_warp=[1, THREADS_PER_WARP, 1],
                                                       warps_per_cta=[ttgl.num_warps(), 1, 1], order=[2, 1, 0])
    # num warps goes in N dim, becase we expect it to be == 1, so data duplicated in warps.
    # fast hack to avoid using explicit linear layout
    B_LOAD_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 8,
                                                                        1], threads_per_warp=[THREADS_PER_WARP, 1, 1],
                                                       warps_per_cta=[1, 1, ttgl.num_warps()], order=[1, 0, 2])
    ACC_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1,
                                                                     1], threads_per_warp=[THREADS_PER_WARP, 1, 1],
                                                    warps_per_cta=[1, ttgl.num_warps(), 1], order=[2, 1, 0])
    offs_bn = (pid_n * BLOCK_SIZE_N +
               ttgl.arange(0, BLOCK_SIZE_N, ttgl.SliceLayout(0, ttgl.SliceLayout(1, B_LOAD_LAYOUT)))) % N
    NUM_SUB_BLOCK_K: ttgl.constexpr = THREADS_PER_WARP
    SUB_BLOCK_SIZE_K: ttgl.constexpr = BLOCK_SIZE_K // NUM_SUB_BLOCK_K
    offs_ak = ttgl.arange(0, SUB_BLOCK_SIZE_K, ttgl.SliceLayout(0, ttgl.SliceLayout(1, A_LOAD_LAYOUT)))
    offs_bk = ttgl.arange(0, SUB_BLOCK_SIZE_K, ttgl.SliceLayout(0, ttgl.SliceLayout(2, B_LOAD_LAYOUT)))

    offs_ak_sub_block = ttgl.arange(0, NUM_SUB_BLOCK_K, ttgl.SliceLayout(0, ttgl.SliceLayout(2, A_LOAD_LAYOUT)))
    offs_bk_sub_block = ttgl.arange(0, NUM_SUB_BLOCK_K, ttgl.SliceLayout(1, ttgl.SliceLayout(2, B_LOAD_LAYOUT)))
    stirde_sub_block_ak = stride_ak * SUB_BLOCK_SIZE_K
    stirde_sub_block_bk = stride_bk * SUB_BLOCK_SIZE_K

    a_ptrs_no_m = a_ptr + (offs_ak_sub_block[None, :, None] * stirde_sub_block_ak + offs_ak[None, None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk_sub_block[:, None, None] * stirde_sub_block_bk + offs_bk[None, :, None] * stride_bk +
                      offs_bn[None, None, :] * stride_bn)

    SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0, 1, 2])
    # ALIGNED_BLOCK_M_SIZE: ttgl.constexpr = ttgl.int32(ttgl.exp2(ttgl.ceil(ttgl.log2(BLOCK_SIZE_M / 1))))
    accumulator_smem = ttgl.allocate_shared_memory(ttgl.float32,
                                                   (ALIGNED_BLOCK_SIZE_M, NUM_SUB_BLOCK_K, 1, BLOCK_SIZE_N),
                                                   SHARED_LAYOUT)
    for m_row in ttgl.static_range(BLOCK_SIZE_M):
        zero_acc = ttgl.zeros((NUM_SUB_BLOCK_K, 1, BLOCK_SIZE_N), dtype=tl.float32, layout=ACC_LAYOUT)
        accumulator_smem.index(m_row).store(zero_acc)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b = ttgl.load(
            b_ptrs, mask=offs_bk_sub_block[:, None, None] * SUB_BLOCK_SIZE_K + offs_bk[None, :, None]
            < K - k * BLOCK_SIZE_K, other=0.0)
        LHS_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=ACC_LAYOUT, operand_index=0, k_width=0)
        RHS_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=ACC_LAYOUT, operand_index=1, k_width=0)
        b = ttgl.convert_layout(b, RHS_LAYOUT).to(tl.float16)

        for m_row in ttgl.static_range(BLOCK_SIZE_M):
            accumulator = accumulator_smem.index(m_row).load(ACC_LAYOUT)
            offs_am = (pid_m * BLOCK_SIZE_M +
                       ttgl.arange(0, 1, ttgl.SliceLayout(1, ttgl.SliceLayout(2, A_LOAD_LAYOUT))) + m_row) % M
            a = ttgl.load(
                a_ptrs_no_m + offs_am[:, None, None] * stride_am,
                mask=offs_ak_sub_block[None, :, None] * SUB_BLOCK_SIZE_K + offs_ak[None, None, :]
                < K - k * BLOCK_SIZE_K, other=0.0)
            a = a.permute(1, 0, 2)
            a = ttgl.convert_layout(a, LHS_LAYOUT).to(tl.float16)
            # We accumulate along the K dimension.
            accumulator = ttgl.dot_fma(a, b, accumulator)
            accumulator_smem.index(m_row).store(accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs_no_m += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    for m_row in ttgl.static_range(BLOCK_SIZE_M):
        accumulator = accumulator_smem.index(m_row).load(ACC_LAYOUT)
        accumulator = ttgl.sum(accumulator, 0)
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        if ACTIVATION == "leaky_relu":
            accumulator = gluon_leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + ttgl.arange(0, 1, ttgl.SliceLayout(1, ttgl.SliceLayout(0, ACC_LAYOUT))) + m_row
        offs_cn = pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N,
                                                     ttgl.SliceLayout(0, ttgl.SliceLayout(0, ACC_LAYOUT)))
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")
