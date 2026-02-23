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
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
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
for metric in ["perf", "bw"]:
    configs[metric] = []
    for fp8_inputs in [True, False]:
        configs[metric].append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],
                x_vals=[(4096, 1, 16384)],
                line_arg="provider",
                line_vals=["Triton"],
                line_names=["Triton"],
                ylabel="TFLOPS" if metric == "perf" else "TBYTES/S",
                plot_name="matmul-performance-" + ("fp16" if not fp8_inputs else "fp8"),
                args={"fp8_inputs": fp8_inputs},
            ))


@triton.testing.perf_report(configs["perf"])
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


@triton.testing.perf_report(configs["bw"])
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
