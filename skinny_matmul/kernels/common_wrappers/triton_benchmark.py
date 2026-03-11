# This file is a copy of tutorial with some parts cut off for convenience
import torch
import triton
import kernels.common_wrappers.power as power

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def run_isolated_triton_bench(matmul_kernel):
    M, N, K = (4096, 1, 16384)
    input_dtype = torch.float16

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
        return c

    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if input_dtype != torch.float16:
        a = a.to(input_dtype)
        b = b.to(input_dtype)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)


kernel_stats = {}


def run_generic_bench(name, matmul_func, dtypes):
    sizes = [(4096, 1, 16384), (4096, 2, 16384), (4096, 4, 16384), (4096, 8, 16384), (4096, 16, 16384)]

    print(f"Running benchmarks for {name}")
    global kernel_stats
    kernel_stats = {}

    for (M, N, K) in sizes:
        for input_dtype in dtypes:
            variant_key = (input_dtype, M, N, K)
            print("    benchmarking variant", variant_key)
            a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
            b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
            if input_dtype != torch.float16:
                a = a.to(input_dtype)
                b = b.to(input_dtype)
            quantiles = [0.5, 0.2, 0.8]
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_func(a, b), quantiles=quantiles)

            used_energy = 0.0
            num_energy_measure_repeats = 100
            import time
            for i in range(num_energy_measure_repeats):
                start_energy = power.GetEnergy(DEVICE.index)
                c = matmul_func(a, b)
                used_energy += power.GetEnergy(DEVICE.index) - start_energy

            idle_consumed = 0.0
            for i in range(num_energy_measure_repeats):
                start_energy = power.GetEnergy(DEVICE.index)
                # here goes matmul(a, b)
                idle_consumed += power.GetEnergy(DEVICE.index) - start_energy
            used_energy -= idle_consumed
            total_energy_per_run = used_energy / 1e6 / num_energy_measure_repeats

            if variant_key not in kernel_stats:
                kernel_stats[variant_key] = {}
            kernel_stats[variant_key]["performance(TFLOPS)"] = 2 * M * N * K * 1e-12 / (ms * 1e-3)
            kernel_stats[variant_key]["bandwidth(TBytes/s)"] = input_dtype.itemsize * (M * K + N * K) * 1e-12 / (ms *
                                                                                                                 1e-3)
            kernel_stats[variant_key]["energy(Joules)"] = total_energy_per_run
            kernel_stats[variant_key]["average time(ms)"] = ms

    results = []
    for variant_key in kernel_stats:
        results += [{"name": name, "variant": variant_key, **kernel_stats[variant_key]}]
    return results


def run_torch_bench(name):
    return run_generic_bench(name, torch.matmul, [torch.float16])


def run_triton_bench(name, matmul_kernel):

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
        kernel_key = (a.dtype, M, N, K)
        global kernel_stats
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

    return run_generic_bench(name, matmul, [torch.float8_e5m2, torch.float16])


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
