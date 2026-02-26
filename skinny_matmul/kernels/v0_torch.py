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


def benchmark_torch():
    M, N, K = (4096, 1, 16384)
    input_dtype = torch.float16
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    bandwidth = lambda ms: input_dtype.itemsize * (M * K + N * K) * 1e-12 / (ms * 1e-3)
    return [{
        "dtype": torch.float16, "name": "v0_torch", "performance(TFLOPS)": perf(ms), "bandwidth(TBytes/s)":
        bandwidth(ms)
    }]
