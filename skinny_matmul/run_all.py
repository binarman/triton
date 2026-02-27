#!/usr/bin/env python3

import torch
from kernels.common_wrappers import triton_benchmark
from kernels import v0_torch
from kernels import v1_dot2d_mma
from kernels import v2_dot2d_fma
from kernels import v3_dot3d
from kernels import v4_gluon_dot3d
from kernels import v5_gluon_dot3d_local_b
from kernels import v6_gluon_dot3d_flex_m


def print_results_table(results, dtype):
    """Print a formatted table for results of a specific dtype.

    Columns are kernel names, rows are metrics.
    """
    # Filter results by dtype
    filtered = [r for r in results if r["dtype"] == dtype]
    if not filtered:
        print(f"\nNo results for {dtype}")
        return

    # Get all kernel names (columns)
    names = [r["name"] for r in filtered]

    # Get all metric keys (rows) - exclude "name" and "dtype"
    all_keys = set()
    for r in filtered:
        all_keys.update(r.keys())
    all_keys.discard("name")
    all_keys.discard("dtype")
    metric_keys = sorted(all_keys)

    # Build a lookup: name -> metrics dict
    name_to_metrics = {r["name"]: r for r in filtered}

    # Calculate column widths
    col_widths = [max(len("Metric"), max(len(k) for k in metric_keys))]
    for name in names:
        max_width = len(name)
        for key in metric_keys:
            val = name_to_metrics[name].get(key, "N/A")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            max_width = max(max_width, len(val_str))
        col_widths.append(max_width)

    # Print header
    print(f"\n{'=' * 60}")
    print(f"Results for {dtype}")
    print(f"{'=' * 60}")

    header = "| " + "Metric".ljust(col_widths[0]) + " |"
    for i, name in enumerate(names):
        header += " " + name.ljust(col_widths[i + 1]) + " |"
    print(header)

    # Print separator
    sep = "|" + "-" * (col_widths[0] + 2) + "|"
    for i in range(len(names)):
        sep += "-" * (col_widths[i + 1] + 2) + "|"
    print(sep)

    # Print rows
    for key in metric_keys:
        row = "| " + key.ljust(col_widths[0]) + " |"
        for i, name in enumerate(names):
            val = name_to_metrics[name].get(key, "N/A")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            row += " " + val_str.ljust(col_widths[i + 1]) + " |"
        print(row)


# list of benchmark results
# each list element is a map with following fields: "dtype", "name", "perf", "bandwidth", "lds", "vgprs", "sgprs", "v_dot_count", "v_fma_count", "v_mfma_count"
results = []
results += v0_torch.benchmark_torch()
results += triton_benchmark.run_triton_bench("v1_dot2d_mma", v1_dot2d_mma.matmul_kernel)
results += triton_benchmark.run_triton_bench("v2_dot2d_fma", v2_dot2d_fma.matmul_kernel)
results += triton_benchmark.run_triton_bench("v3_dot3d", v3_dot3d.matmul_kernel)
results += triton_benchmark.run_triton_bench("v4_gluon_dot3d", v4_gluon_dot3d.matmul_kernel)
results += triton_benchmark.run_triton_bench("v5_gluon_dot3d_local_b", v5_gluon_dot3d_local_b.matmul_kernel)
results += triton_benchmark.run_triton_bench("v6_gluon_dot3d_flex_m", v6_gluon_dot3d_flex_m.matmul_kernel)

triton_benchmark.run_triton_test("v1_dot2d_mma", v1_dot2d_mma.matmul_kernel)
triton_benchmark.run_triton_test("v2_dot2d_fma", v2_dot2d_fma.matmul_kernel)
triton_benchmark.run_triton_test("v3_dot3d", v3_dot3d.matmul_kernel)
triton_benchmark.run_triton_test("v4_gluon_dot3d", v4_gluon_dot3d.matmul_kernel)
triton_benchmark.run_triton_test("v5_gluon_dot3d_local_b", v5_gluon_dot3d_local_b.matmul_kernel)
triton_benchmark.run_triton_test("v6_gluon_dot3d_flex_m", v6_gluon_dot3d_flex_m.matmul_kernel)

# Print formatted tables
print_results_table(results, torch.float16)
print_results_table(results, torch.float8_e5m2)
