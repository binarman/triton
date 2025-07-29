#!/usr/bin/env python3
import torch
import triton
import tempfile
import numpy as np
from numpy.random import RandomState
import pathlib
import concurrent.futures

device = "cuda"


def gen_ll(shape, order, multiplier):
    tileshape = [1, 1]
    num_elems = shape[0] * shape[1]
    bases = []
    while num_elems > 1:
        for dim in order:
            if tileshape[dim] < shape[dim]:
                vector = [0, 0]
                vector[dim] = tileshape[dim] * multiplier[dim]
                tileshape[dim] *= 2
                bases += [vector]
                break
        num_elems //= 2
    return bases


def compile_and_report(file_name, kernel_id):
    kernel = triton.compile(file_name)
    print("{} ".format(kernel_id), end="", flush=True)
    return kernel


def itt_padding():
    num_warps = 1
    # size of one element in bytes
    elem_width = 2
    bank_width = 4
    #num_banks = 32
    kWidth = 4
    # tensor sizes
    sizes = [(32, 128), (32, 256), (32, 512), (64, 64), (64, 128), (64, 256), (128, 32), (128, 64), (128, 128),
             (256, 32), (256, 64)]
    # global load shape per thread
    input_shape_per_thread = [[2, 8], [4, 8], [8, 8]]
    # paddings, try to pad after every tensor row/bank row, then try to add additional paddings after first rank of paddings exhaust exhaust row width.
    # output layouts:
    output_layouts = [
        "#ttg.amd_mfma<{version = 3, warpsPerCTA = [1, " + str(num_warps) +
        "], instrShape = [32, 32], isTransposed = true}>", "#ttg.amd_mfma<{version = 3, warpsPerCTA = [1, " +
        str(num_warps) + "], instrShape = [16, 16], isTransposed = true}>"
    ]

    dtype = "f16" if elem_width == 2 else "i8"

    configs = []
    config_id = 0
    for s in sizes:
        for spt in input_shape_per_thread:
            for output_layout in output_layouts:
                num_banks_per_write = spt[0] * elem_width // bank_width
                num_banks_per_read = kWidth * elem_width // bank_width
                max_banks_per_access = max(num_banks_per_write, num_banks_per_read)
                paddings = [pad for pad in [2, 4, 8, 16] if pad >= (max_banks_per_access * bank_width // elem_width)]

                input_lanes = [0, s[1] // spt[1]]
                input_lanes[0] = 64 // input_lanes[1]

                # generate global load layout
                gl_layout = "#ttg.blocked<{sizePerThread = " + str(spt) + ", threadsPerWarp = " + str(
                    input_lanes) + ", warpsPerCTA = [" + str(num_warps) + ", 1], order = [1, 0]}>"

                # generate shared store layout
                registers = gen_ll(spt, [0, 1], [1, 1])
                lanes = gen_ll(input_lanes, [1, 0], spt)
                warps = gen_ll([num_warps, 1], [1, 0], [spt[0] * lanes[0], spt[1] * lanes[1]])
                ls_layout = "#ttg.linear<{register = " + str(registers) + ", lane = " + str(lanes) + ", warp = " + str(
                    warps) + ", block = []}>"

                # elems_in_bank_row = bank_width // elem_width * num_banks
                # case 1: pad every row of a matrix
                # case 2: pad every time we exhaust line of banks width
                # case 3: pad between every adjacent lanes in different rows
                #intervals1 = list(set([s[0], elems_in_bank_row, spt[1] * s[0], s[0] * 2, s[0] * 4]))
                intervals1 = [32, 64, 128, 256, 512]
                for interval1 in intervals1:
                    for pad1 in paddings:
                        shared_layout = "#ttg.padded_shared<[" + str(interval1) + ":+" + str(
                            pad1) + "] {order = [0, 1]}>"
                        configs += [(config_id, s, gl_layout, ls_layout, output_layout, shared_layout)]
                        # try to add padding between groups of shifts
                        # intervals2 = elems_in_bank_row // pad1 * interval1
                        intervals2 = [i for i in [256, 512, 1024, 2048, 4096, 8192] if i > interval1]
                        for interval2 in intervals2:
                            for pad2 in paddings:
                                shared_layout = "#ttg.padded_shared<[" + str(interval1) + ":+" + str(pad1) + ", " + str(
                                    interval2) + ":+" + str(pad2) + "] {order = [0, 1]}>"
                                configs += [(config_id, s, gl_layout, ls_layout, output_layout, shared_layout)]
                mfma = "32" if "32" in output_layout else "16"
                print(",".join([str(config_id), str(s[0]), str(s[1]), str(spt[0]), str(spt[1]), mfma]))
                config_id += 1

    print("compiling total", len(configs), "configs")

    kernel_futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        for idx, config in enumerate(configs):
            config_id = config[0]
            w = config[1][0]
            h = config[1][1]

            global_load_layout = config[2]
            transposed = config[3]
            mma = config[4]
            shared = config[5]

            shared_id = shared[shared.find('[') + 1:shared.find(']')].replace(':+', '_').replace(',',
                                                                                                 '_').replace(' ', '')
            kernel_name = "kernel_config_" + str(config_id) + "__" + shared_id

            ir = f"""
            #smem = #ttg.shared_memory
            #mma = {mma}
            #linear = {transposed}
            #shared = {shared}
            #blocked = {global_load_layout}
            #dotop = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {kWidth}}}>
            #slice0_load = #ttg.slice<{{dim = 0, parent = #blocked}}>
            #slice1_load = #ttg.slice<{{dim = 1, parent = #blocked}}>
            #slice0_store = #ttg.slice<{{dim = 0, parent = #dotop}}>
            #slice1_store = #ttg.slice<{{dim = 1, parent = #dotop}}>
            module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32}} {{
                tt.func public @{kernel_name}(%x: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}, %y: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}) {{
                    %c{h}_i32 = arith.constant {h} : i32
                    %23 = tt.make_range {{end = {h} : i32, start = 0 : i32}} : tensor<{h}xi32, #slice0_load>
                    %46 = tt.make_range {{end = {w} : i32, start = 0 : i32}} : tensor<{w}xi32, #slice1_load>
                    %47 = tt.expand_dims %46 {{axis = 1 : i32}} : tensor<{w}xi32, #slice1_load> -> tensor<{w}x1xi32, #blocked>
                    %48 = tt.splat %c{h}_i32 : i32 -> tensor<{w}x1xi32, #blocked>
                    %49 = arith.muli %47, %48 : tensor<{w}x1xi32, #blocked>
                    %50 = tt.broadcast %49 : tensor<{w}x1xi32, #blocked> -> tensor<{w}x{h}xi32, #blocked>
                    %51 = tt.expand_dims %23 {{axis = 0 : i32}} : tensor<{h}xi32, #slice0_load> -> tensor<1x{h}xi32, #blocked>
                    %52 = tt.broadcast %51 : tensor<1x{h}xi32, #blocked> -> tensor<{w}x{h}xi32, #blocked>
                    %53 = arith.addi %52, %50 : tensor<{w}x{h}xi32, #blocked>

                    %111 = amdgpu.buffer_load %x[%53]: tensor<{w}x{h}x{dtype}, #blocked>
                    %116 = amdgpu.in_thread_transpose %111 : tensor<{w}x{h}x{dtype}, #blocked> -> tensor<{w}x{h}x{dtype}, #linear>
                    %117 = ttg.local_alloc %116 : (tensor<{w}x{h}x{dtype}, #linear>) -> !ttg.memdesc<{w}x{h}x{dtype}, #shared, #smem>
                    %118 = ttg.local_load %117 : !ttg.memdesc<{w}x{h}x{dtype}, #shared, #smem> -> tensor<{w}x{h}x{dtype}, #dotop>

                    %s_c{h}_i32 = arith.constant {h} : i32
                    %s_23 = tt.make_range {{end = {h} : i32, start = 0 : i32}} : tensor<{h}xi32, #slice0_store>
                    %s_46 = tt.make_range {{end = {w} : i32, start = 0 : i32}} : tensor<{w}xi32, #slice1_store>
                    %s_47 = tt.expand_dims %s_46 {{axis = 1 : i32}} : tensor<{w}xi32, #slice1_store> -> tensor<{w}x1xi32, #dotop>
                    %s_48 = tt.splat %s_c{h}_i32 : i32 -> tensor<{w}x1xi32, #dotop>
                    %s_49 = arith.muli %s_47, %s_48 : tensor<{w}x1xi32, #dotop>
                    %s_50 = tt.broadcast %s_49 : tensor<{w}x1xi32, #dotop> -> tensor<{w}x{h}xi32, #dotop>
                    %s_51 = tt.expand_dims %s_23 {{axis = 0 : i32}} : tensor<{h}xi32, #slice0_store> -> tensor<1x{h}xi32, #dotop>
                    %s_52 = tt.broadcast %s_51 : tensor<1x{h}xi32, #dotop> -> tensor<{w}x{h}xi32, #dotop>
                    %s_53 = arith.addi %s_52, %s_50 : tensor<{w}x{h}xi32, #dotop>

                    amdgpu.buffer_store %118, %y[%s_53]: tensor<{w}x{h}x{dtype}, #dotop>
                    tt.return
                }}
            }}
            """
            tmp_file = "tmp/kernel_" + str(idx) + ".ttgir"
            with open(tmp_file, "w") as f:
                f.write(ir)
            kernel_futures += [executor.submit(compile_and_report, tmp_file, config_id)]

    print("\ncompilation is done, benchmarking")

    for idx, config in enumerate(configs):
        config_id = config[0]
        w = config[1][0]
        h = config[1][1]

        global_load_layout = config[2]
        transposed = config[3]
        mma = config[4]
        shared = config[5]

        shared_id = shared[shared.find('[') + 1:shared.find(']')].replace(':+', '_').replace(',', '_').replace(' ', '')
        kernel_name = "kernel_config_" + str(config_id) + "__" + shared_id

        torch_dtype = torch.float16 if elem_width == 2 else torch.int8
        x = (torch.randn((w, h), dtype=torch.float32, device=device) * 10).to(torch_dtype)
        y = torch.zeros((w, h), dtype=torch_dtype, device=device)
        print("running {}".format(kernel_name), end="")
        pgm = kernel_futures[config_id].result()[(1, 1, 1)](x, y)
        np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())
        print("successfully")


if __name__ == "__main__":
    itt_padding()
