# This file is a copy of tutorial with some parts cut off for convenience
import torch

import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def run_triton_bench(name, matmul_kernel):

    print(f"Running benchmarking for {name}")
    kernel_stats = {}

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
        kernel_key = (a.dtype, )
        if kernel_key not in kernel_stats:
            lds = pgm.metadata.shared
            v_dot_count = 0
            v_mfma_count = 0
            v_fma_count = 0
            for line in pgm.asm["amdgcn"].split("\n"):
                if ".sgpr_spill_count" in line:
                    sgpr_spills = int(line.split(":")[1].strip())
                elif ".sgpr_count" in line:
                    sgpr_count = int(line.split(":")[1].strip())
                if ".vgpr_spill_count" in line:
                    vgpr_spills = int(line.split(":")[1].strip())
                elif ".vgpr_count" in line:
                    vgpr_count = int(line.split(":")[1].strip())
                if "v_dot" in line:
                    v_dot_count += 1
                if "v_mfma" in line:
                    v_mfma_count += 1
                if "v_fmac" in line:
                    v_fma_count += 1
            kernel_stats[kernel_key] = {
                "lds": lds, "v_dot_count": v_dot_count, "v_mfma_count": v_mfma_count, "v_fmac_count": v_fma_count,
                "sgpr_count": sgpr_count, "vgpr_count": vgpr_count, "sgpr_spills": sgpr_spills, "vgpr_spills":
                vgpr_spills
            }
        return c

    M, N, K = (4096, 1, 16384)
    for input_dtype in [torch.float8_e5m2, torch.float16]:
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        if input_dtype != torch.float16:
            a = a.to(input_dtype)
            b = b.to(input_dtype)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
        kernel_stats[(a.dtype, )]["perf"] = 2 * M * N * K * 1e-12 / (ms * 1e-3)
        kernel_stats[(a.dtype, )]["bw"] = input_dtype.itemsize * (M * K + N * K) * 1e-12 / (ms * 1e-3)

    results = []
    for (dtype, ) in kernel_stats:
        results += [{"name": name, "dtype": dtype, **kernel_stats[(dtype, )]}]
    return results


def run_triton_test(name, matmul_kernel):
    print(f"Running tests for {name}")
    import os

    def matmul(a, b, activation=""):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

        matmul_kernel[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            ACTIVATION=activation)
        return c

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
