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

configs = {}
for metric in ["perf", "bw"]:
    configs[metric] = []
    for fp8_inputs in [False]:
        configs[metric].append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],
                x_vals=[(4096, 1, 16384)],
                line_arg="provider",
                line_vals=["rocBLAS"],
                line_names=["rocBLAS"],
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
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
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
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    bandwidth = lambda ms: input_dtype.itemsize * (M * K + N * K) * 1e-12 / (ms * 1e-3)
    return bandwidth(ms)


benchmark_perf.run(show_plots=False, print_data=True)
benchmark_bw.run(show_plots=False, print_data=True)
