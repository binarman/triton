# This file is a copy of tutorial with some parts cut off for convenience
import torch

import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def run_all(matmul_kernel):

    first_run = set()

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

        pgm = matmul_kernel[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            ACTIVATION=activation)
        run_key = (a.dtype, b.dtype)
        if run_key not in first_run:
            first_run.add(run_key)
            print("LDS:", pgm.metadata.shared)
            for line in pgm.asm["amdgcn"].split("\n"):
                if ".sgpr" in line or ".vgpr" in line:
                    print(line)

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
            b = ((torch.rand(
                (16384, 1), device=DEVICE, dtype=torch.float32) - 0.5) * 16).to(torch.int8).to(torch.float16)
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
