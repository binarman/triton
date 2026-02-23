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

THREADS_PER_WARP = triton.language.constexpr(triton.runtime.driver.active.get_current_target().warp_size)


def get_autotune_config():
    elems_in_load = 16 // 2  # 16 bytes per load, 2 bytes in one element
    sizes = [
        {'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 2, 'GROUP_SIZE_M': 1},
        {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_K': THREADS_PER_WARP * elems_in_load * 4, 'GROUP_SIZE_M': 1},
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
                                                       warps_per_cta=[ttgl.num_warps(), 1, 1], order=[2, 1, 0])
    # num warps goes in N dim, becase we expect it to be == 1, so data duplicated in warps.
    # fast hack to avoid using explicit linear layout
    B_LOAD_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 8,
                                                                        1], threads_per_warp=[THREADS_PER_WARP, 1, 1],
                                                       warps_per_cta=[1, 1, ttgl.num_warps()], order=[1, 0, 2])
    ACC_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1,
                                                                     1], threads_per_warp=[THREADS_PER_WARP, 1, 1],
                                                    warps_per_cta=[1, ttgl.num_warps(), 1], order=[2, 1, 0])

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
        a = ttgl.convert_layout(a, LHS_LAYOUT).to(tl.float16)
        b = ttgl.convert_layout(b, RHS_LAYOUT).to(tl.float16)
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


def matmul(a, b, activation=""):
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
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


configs = {}
for metric in ["performance", "bandwidth"]:
    configs[metric] = []
    for fp8_inputs in [True, False]:
        configs[metric].append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],
                x_vals=[(4096, 1, 16384)],
                line_arg="provider",
                line_vals=["Triton"],
                line_names=["Triton"],
                ylabel="TFLOPS" if metric == "performance" else "TBYTES/S",
                plot_name=f"matmul-{metric}-" + ("fp16" if not fp8_inputs else "fp8"),
                args={"fp8_inputs": fp8_inputs},
            ))


@triton.testing.perf_report(configs["performance"])
def benchmark_perf(M, N, K, provider, fp8_inputs):
    input_dtype = torch.float8_e5m2 if fp8_inputs else torch.float16
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms)


@triton.testing.perf_report(configs["bandwidth"])
def benchmark_bw(M, N, K, provider, fp8_inputs):
    input_dtype = torch.float8_e5m2 if fp8_inputs else torch.float16
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    bandwidth = lambda ms: input_dtype.itemsize * (M * K + N * K) * 1e-12 / (ms * 1e-3)
    return bandwidth(ms)


benchmark_perf.run(show_plots=False, print_data=True)
benchmark_bw.run(show_plots=False, print_data=True)

# **************** Unit Test ****************
import os
if "ROCPROF_ATT_LIBRARY_PATH" not in os.environ:
    for fp8_inputs in [False, True]:
        torch.manual_seed(0)
        a = ((torch.rand(
            (4096, 16384), device=DEVICE, dtype=torch.float32) - 0.5) * 16).to(torch.int8).to(torch.float16)
        b = ((torch.rand((16384, 1), device=DEVICE, dtype=torch.float32) - 0.5) * 16).to(torch.int8).to(torch.float16)
        torch_output = torch.matmul(a, b)
        if fp8_inputs:
            a = a.to(torch.float8_e5m2)
            b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b)

        dtype_prefix = "fp8" if fp8_inputs else "fp16"
        print(f"{dtype_prefix} inputs: ", end="")
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
