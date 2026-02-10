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

# ****************** MMA kernel ******************


def get_mma_autotune_config():
    sizes = [
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1},
    ]
    return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=1, num_stages=2) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=1, num_stages=3) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=2, num_stages=2) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=2) for s in sizes] + \
           [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=3) for s in sizes]


@triton.autotune(
    configs=get_mma_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
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
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


@triton.autotune(
    configs=get_mma_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def skinny_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    NUM_SUB_BLOCK_K: tl.constexpr = 64
    SUB_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // NUM_SUB_BLOCK_K
    offs_k = tl.arange(0, SUB_BLOCK_SIZE_K)
    offs_k_sub_block = tl.arange(0, NUM_SUB_BLOCK_K)
    stire_sub_block_ak = stride_ak * SUB_BLOCK_SIZE_K
    stire_sub_block_bk = stride_bk * SUB_BLOCK_SIZE_K

    a_ptrs = a_ptr + (offs_am[:, None, None] * stride_am + offs_k_sub_block[None, :, None] * stire_sub_block_ak +
                      offs_k[None, None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_sub_block[:, None, None] * stire_sub_block_bk + offs_k[None, :, None] * stride_bk +
                      offs_bn[None, None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((NUM_SUB_BLOCK_K, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            a_ptrs, mask=offs_k_sub_block[None, :, None] * SUB_BLOCK_SIZE_K + offs_k[None, None, :]
            < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(
            b_ptrs, mask=offs_k_sub_block[:, None, None] * SUB_BLOCK_SIZE_K + offs_k[None, :, None]
            < K - k * BLOCK_SIZE_K, other=0.0)
        a = tl.permute(a, (1, 0, 2))
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    accumulator = tl.sum(accumulator, 0)
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# ****************** gluon kernel naive ******************


def get_fma_warp_autotune_config():
    sizes = [
        {'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 1},
    ]
    return [triton.Config(s, num_warps=1) for s in sizes]


THREADS_PER_WARP: ttgl.constexpr = triton.runtime.driver.active.get_current_target().warp_size


@gluon.jit
def gluon_leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.autotune(
    configs=get_fma_warp_autotune_config(),
    key=['M', 'N', 'K'],
)
@gluon.jit
def gluon_skinny_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn, THREADS_PER_WARP: ttgl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: ttgl.constexpr, BLOCK_SIZE_N: ttgl.constexpr, BLOCK_SIZE_K: ttgl.constexpr,  #
        GROUP_SIZE_M: ttgl.constexpr,  #
        ACTIVATION: ttgl.constexpr  #
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
                                                       warps_per_cta=[1, 1, 1], order=[2, 1, 0])
    B_LOAD_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 8,
                                                                        1], threads_per_warp=[THREADS_PER_WARP, 1, 1],
                                                       warps_per_cta=[1, 1, 1], order=[1, 0, 2])
    ACC_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1,
                                                                     1], threads_per_warp=[THREADS_PER_WARP, 1, 1],
                                                    warps_per_cta=[1, 1, 1], order=[2, 1, 0])

    offs_am = (pid_m * BLOCK_SIZE_M +
               ttgl.arange(0, BLOCK_SIZE_M, ttgl.SliceLayout(1, ttgl.SliceLayout(2, A_LOAD_LAYOUT)))) % M
    offs_bn = (pid_n * BLOCK_SIZE_N +
               ttgl.arange(0, BLOCK_SIZE_N, ttgl.SliceLayout(0, ttgl.SliceLayout(1, B_LOAD_LAYOUT)))) % N
    NUM_SUB_BLOCK_K: ttgl.constexpr = THREADS_PER_WARP
    SUB_BLOCK_SIZE_K: ttgl.constexpr = BLOCK_SIZE_K // NUM_SUB_BLOCK_K
    offs_ak = ttgl.arange(0, SUB_BLOCK_SIZE_K, ttgl.SliceLayout(0, ttgl.SliceLayout(1, A_LOAD_LAYOUT)))
    offs_bk = ttgl.arange(0, SUB_BLOCK_SIZE_K, ttgl.SliceLayout(0, ttgl.SliceLayout(2, B_LOAD_LAYOUT)))

    offs_ak_sub_block = ttgl.arange(0, NUM_SUB_BLOCK_K, ttgl.SliceLayout(0, ttgl.SliceLayout(2, A_LOAD_LAYOUT)))
    offs_bk_sub_block = ttgl.arange(0, NUM_SUB_BLOCK_K, ttgl.SliceLayout(1, ttgl.SliceLayout(2, B_LOAD_LAYOUT)))
    stire_sub_block_ak = stride_ak * SUB_BLOCK_SIZE_K
    stire_sub_block_bk = stride_bk * SUB_BLOCK_SIZE_K

    a_ptrs = a_ptr + (offs_am[:, None, None] * stride_am + offs_ak_sub_block[None, :, None] * stire_sub_block_ak +
                      offs_ak[None, None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk_sub_block[:, None, None] * stire_sub_block_bk + offs_bk[None, :, None] * stride_bk +
                      offs_bn[None, None, :] * stride_bn)

    accumulator = ttgl.zeros((NUM_SUB_BLOCK_K, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32, layout=ACC_LAYOUT)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = ttgl.load(
            a_ptrs, mask=offs_ak_sub_block[None, :, None] * SUB_BLOCK_SIZE_K + offs_ak[None, None, :]
            < K - k * BLOCK_SIZE_K, other=0.0)
        b = ttgl.load(
            b_ptrs, mask=offs_bk_sub_block[:, None, None] * SUB_BLOCK_SIZE_K + offs_bk[None, :, None]
            < K - k * BLOCK_SIZE_K, other=0.0)
        a = a.permute(1, 0, 2)

        LHS_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=ACC_LAYOUT, operand_index=0, k_width=0)
        RHS_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=ACC_LAYOUT, operand_index=1, k_width=0)
        a = ttgl.convert_layout(a, LHS_LAYOUT)
        b = ttgl.convert_layout(b, RHS_LAYOUT)
        # We accumulate along the K dimension.
        accumulator = ttgl.dot_fma(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    accumulator = ttgl.sum(accumulator, 0)
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = gluon_leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, ttgl.SliceLayout(1, ttgl.SliceLayout(0, ACC_LAYOUT)))
    offs_cn = pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, ttgl.SliceLayout(0, ttgl.SliceLayout(0, ACC_LAYOUT)))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


# ****************** gluon kernel with pipelining ******************


def get_fma_warp_autotune_config():
    sizes = [
        {'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 1},
    ]
    return [triton.Config(s, num_warps=1) for s in sizes]


THREADS_PER_WARP: ttgl.constexpr = triton.runtime.driver.active.get_current_target().warp_size


@gluon.jit
def gluon_leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.autotune(
    configs=get_fma_warp_autotune_config(),
    key=['M', 'N', 'K'],
)
@gluon.jit
def gluon_skinny_matmul_pipelined_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn, THREADS_PER_WARP: ttgl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: ttgl.constexpr, BLOCK_SIZE_N: ttgl.constexpr, BLOCK_SIZE_K: ttgl.constexpr,  #
        GROUP_SIZE_M: ttgl.constexpr,  #
        ACTIVATION: ttgl.constexpr  #
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
                                                       warps_per_cta=[1, 1, 1], order=[2, 1, 0])
    B_LOAD_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 8,
                                                                        1], threads_per_warp=[THREADS_PER_WARP, 1, 1],
                                                       warps_per_cta=[1, 1, 1], order=[1, 0, 2])
    ACC_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1,
                                                                     1], threads_per_warp=[THREADS_PER_WARP, 1, 1],
                                                    warps_per_cta=[1, 1, 1], order=[2, 1, 0])

    offs_am = (pid_m * BLOCK_SIZE_M +
               ttgl.arange(0, BLOCK_SIZE_M, ttgl.SliceLayout(1, ttgl.SliceLayout(2, A_LOAD_LAYOUT)))) % M
    offs_bn = (pid_n * BLOCK_SIZE_N +
               ttgl.arange(0, BLOCK_SIZE_N, ttgl.SliceLayout(0, ttgl.SliceLayout(1, B_LOAD_LAYOUT)))) % N
    NUM_SUB_BLOCK_K: ttgl.constexpr = THREADS_PER_WARP
    SUB_BLOCK_SIZE_K: ttgl.constexpr = BLOCK_SIZE_K // NUM_SUB_BLOCK_K
    offs_ak = ttgl.arange(0, SUB_BLOCK_SIZE_K, ttgl.SliceLayout(0, ttgl.SliceLayout(1, A_LOAD_LAYOUT)))
    offs_bk = ttgl.arange(0, SUB_BLOCK_SIZE_K, ttgl.SliceLayout(0, ttgl.SliceLayout(2, B_LOAD_LAYOUT)))

    offs_ak_sub_block = ttgl.arange(0, NUM_SUB_BLOCK_K, ttgl.SliceLayout(0, ttgl.SliceLayout(2, A_LOAD_LAYOUT)))
    offs_bk_sub_block = ttgl.arange(0, NUM_SUB_BLOCK_K, ttgl.SliceLayout(1, ttgl.SliceLayout(2, B_LOAD_LAYOUT)))
    stire_sub_block_ak = stride_ak * SUB_BLOCK_SIZE_K
    stire_sub_block_bk = stride_bk * SUB_BLOCK_SIZE_K

    a_ptrs = a_ptr + (offs_am[:, None, None] * stride_am + offs_ak_sub_block[None, :, None] * stire_sub_block_ak +
                      offs_ak[None, None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk_sub_block[:, None, None] * stire_sub_block_bk + offs_bk[None, :, None] * stride_bk +
                      offs_bn[None, None, :] * stride_bn)

    a_prefetch1 = ttgl.load(
        a_ptrs, mask=offs_ak_sub_block[None, :, None] * SUB_BLOCK_SIZE_K + offs_ak[None, None, :]
        < K - 0 * BLOCK_SIZE_K, other=0.0)
    b_prefetch1 = ttgl.load(
        b_ptrs, mask=offs_bk_sub_block[:, None, None] * SUB_BLOCK_SIZE_K + offs_bk[None, :, None]
        < K - 0 * BLOCK_SIZE_K, other=0.0)
    a_prefetch1 = a_prefetch1.permute(1, 0, 2)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

    # a_prefetch2 = ttgl.load(
    #         a_ptrs, mask=offs_ak_sub_block[None, :, None] * SUB_BLOCK_SIZE_K + offs_ak[None, None, :]
    #         < K - 1 * BLOCK_SIZE_K, other=0.0)
    # b_prefetch2 = ttgl.load(
    #         b_ptrs, mask=offs_bk_sub_block[:, None, None] * SUB_BLOCK_SIZE_K + offs_bk[None, :, None]
    #         < K - 1 * BLOCK_SIZE_K, other=0.0)
    # a_prefetch2 = a_prefetch2.permute(1, 0, 2)
    # a_ptrs += BLOCK_SIZE_K * stride_ak
    # b_ptrs += BLOCK_SIZE_K * stride_bk
    prefetch_depth = 1

    accumulator = ttgl.zeros((NUM_SUB_BLOCK_K, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32, layout=ACC_LAYOUT)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        with ttgl.amd.warp_pipeline_stage("stage0"):
            a = a_prefetch1
            b = b_prefetch1
            # a_prefetch1 = a_prefetch2
            # b_prefetch1 = b_prefetch2
            a_prefetch1 = ttgl.load(
                a_ptrs, mask=offs_ak_sub_block[None, :, None] * SUB_BLOCK_SIZE_K + offs_ak[None, None, :]
                < K - (k + prefetch_depth) * BLOCK_SIZE_K, other=0.0)
            b_prefetch1 = ttgl.load(
                b_ptrs, mask=offs_bk_sub_block[:, None, None] * SUB_BLOCK_SIZE_K + offs_bk[None, :, None]
                < K - (k + prefetch_depth) * BLOCK_SIZE_K, other=0.0)
            a_prefetch1 = a_prefetch1.permute(1, 0, 2)

            LHS_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=ACC_LAYOUT, operand_index=0, k_width=0)
            RHS_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(parent=ACC_LAYOUT, operand_index=1, k_width=0)
            a = ttgl.convert_layout(a, LHS_LAYOUT)
            b = ttgl.convert_layout(b, RHS_LAYOUT)
        with ttgl.amd.warp_pipeline_stage("stage1"):
            # We accumulate along the K dimension.
            accumulator = ttgl.dot_fma(a, b, accumulator)
            accumulator = accumulator.to(ttgl.float16)
            accumulator = accumulator.to(ttgl.float32)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    accumulator = ttgl.sum(accumulator, 0)
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = gluon_leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, ttgl.SliceLayout(1, ttgl.SliceLayout(0, ACC_LAYOUT)))
    offs_cn = pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, ttgl.SliceLayout(0, ttgl.SliceLayout(0, ACC_LAYOUT)))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


# **************** Benchmark wrappers ****************


def matmul_mma(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        # matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation)
    return c


reported_configs = {}


def matmul_fma(a, b, BLOCK_SIZE_M, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    pgm = gluon_skinny_matmul_kernel[grid](
        # matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #
        THREADS_PER_WARP=THREADS_PER_WARP)
    config_id = 1000 + BLOCK_SIZE_M
    global reported_configs
    if config_id not in reported_configs:
        print("FMA with BLOCK_SIZE_M: ", BLOCK_SIZE_M)
        reported_configs[config_id] = True
        for line in pgm.asm["amdgcn"].split("\n"):
            if "gpr_count" in line or "spill_count" in line:
                print(line)
    return c


def matmul_pipelined_fma(a, b, BLOCK_SIZE_M, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # target = GPUTarget("hip", "gfx942", 64)
    # module = run_parser(gluon_skinny_matmul_pipelined_kernel, *make_args(num_warps=1), target=target)
    # ir_str = anonymize_ir(module.str_nodebug())

    pgm = gluon_skinny_matmul_pipelined_kernel[grid](
        # matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #
        THREADS_PER_WARP=THREADS_PER_WARP)
    config_id = 2000 + BLOCK_SIZE_M
    global reported_configs
    if config_id not in reported_configs:
        # print(ir_str)
        print("PIPELINED FMA with BLOCK_SIZE_M: ", BLOCK_SIZE_M)
        reported_configs[config_id] = True
        for line in pgm.asm["amdgcn"].split("\n"):
            if "gpr_count" in line or "spill_count" in line:
                print(line)
    return c


ref_lib = 'rocBLAS'

configs = []
for fp8_inputs in [False]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[(4096, 1, 14336)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=[
                ref_lib.lower(), "triton_mma", "gluon_fma1", "gluon_fma4", "gluon_fma8", "gluon_fma16",
                "gluon_pipelined"
            ],  # Label name for the lines
            line_names=[
                ref_lib, "Triton mma", "Gluon, w1", "Gluon, 4 rows", "Gluon, 8 rows", "Gluon, 16 rows",
                "Gluon pipelined"
            ],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("green", "-"), ("blue", "-"), ("green", "-"), ("blue", "-"),
                    ("green", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    # if provider == 'triton_mma':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_mma(a, b), quantiles=quantiles)
    # if provider == 'gluon_fma1':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_fma(a, b, 1), quantiles=quantiles)
    # if provider == 'gluon_fma4':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_fma(a, b, 4), quantiles=quantiles)
    # if provider == 'gluon_fma8':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_fma(a, b, 8), quantiles=quantiles)
    # if provider == 'gluon_fma16':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_fma(a, b, 16), quantiles=quantiles)
    ms = 1
    max_ms = 1
    min_ms = 1
    if provider == 'gluon_pipelined':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_pipelined_fma(a, b, 1), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# **************** Unit Test ****************
import os
if "ROCPROF_ATT_LIBRARY_PATH" not in os.environ:
    torch.manual_seed(0)
    a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    triton_output = matmul_pipelined_fma(a, b, 8)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

benchmark.run(show_plots=True, print_data=True)
