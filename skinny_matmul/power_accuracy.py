# This file is a copy of tutorial with some parts cut off for convenience
import statistics
import torch
import triton
import kernels.common_wrappers.power as power
import kernels.v4_gluon_dot3d as v4_gluon_dot3d

DEVICE = triton.runtime.driver.active.get_active_torch_device()

matmul_kernel = v4_gluon_dot3d.matmul_kernel


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


def run_bench_multiple_experiments(input_dtype):
    M, N, K = (4096, 1, 16384)

    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if input_dtype != torch.float16:
        a = a.to(input_dtype)
        b = b.to(input_dtype)

    num_energy_measure_repeats = 1000
    num_experiments = 1000

    # warmup matmul, so we do not spend time on compilation
    c = matmul(a, b)

    measurements = []
    for i in range(num_experiments):
        used_energy = 0.0
        for i in range(num_energy_measure_repeats):
            start_energy = power.GetEnergy(DEVICE.index)
            c = matmul(a, b)
            used_energy += power.GetEnergy(DEVICE.index) - start_energy

        idle_consumed = 0.0
        for i in range(num_energy_measure_repeats):
            start_energy = power.GetEnergy(DEVICE.index)
            # here goes matmul(a, b)
            idle_consumed += power.GetEnergy(DEVICE.index) - start_energy
        total_energy_per_run = (used_energy - idle_consumed) / 1e6 / num_energy_measure_repeats
        measurements += [total_energy_per_run]

    # Compute variance of measurements
    variance = statistics.variance(measurements)
    mean = statistics.mean(measurements)
    print(measurements)
    measurements.sort()
    print("Batched measurements")
    print(f"Median energy per run: {measurements[len(measurements) // 2]} J")
    print(f"Mean energy per run: {mean:.6f} J")
    print(f"Variance: {variance:.6f} J^2")
    print(f"Standard deviation: {statistics.stdev(measurements):.6f} J")


def run_bench_single_experiment(input_dtype):
    M, N, K = (4096, 1, 16384)

    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if input_dtype != torch.float16:
        a = a.to(input_dtype)
        b = b.to(input_dtype)

    num_repeats = 100

    # warmup matmul, so we do not spend time on compilation
    c = matmul(a, b)

    measurements = []
    for i in range(num_repeats):
        start_energy = power.GetEnergy(DEVICE.index)
        c = matmul(a, b)
        end_energy = power.GetEnergy(DEVICE.index)
        measurements += [(end_energy - start_energy) / 1e6]

    idle_measurements = []
    for i in range(num_repeats):
        start_energy = power.GetEnergy(DEVICE.index)
        # here goes matmul(a, b)
        end_energy = power.GetEnergy(DEVICE.index)
        idle_measurements += [(end_energy - start_energy) / 1e6]

    # Compute variance of measurements
    variance = statistics.variance(measurements)
    mean = statistics.mean(measurements)
    print(measurements)
    print(idle_measurements)
    measurements.sort()
    idle_measurements.sort()
    print("Single measurement")
    print(f"Median total energy per run: {measurements[num_repeats // 2]} J")
    print(f"Median idle energy per run: {idle_measurements[num_repeats // 2]} J")
    print(f"Median peeled energy per run: {measurements[num_repeats // 2] - idle_measurements[num_repeats // 2]} J")
    # print(f"Median idle energy per run")
    # print(f"Mean energy per run: {mean:.6f} J")
    # print(f"Variance: {variance:.6f} J^2")
    # print(f"Standard deviation: {statistics.stdev(measurements):.6f} J")


if __name__ == "__main__":
    # run_bench_multiple_experiments(torch.float16)
    run_bench_single_experiment(torch.float16)
