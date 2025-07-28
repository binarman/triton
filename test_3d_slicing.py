#!/usr/bin/env python3
import torch
import triton
import tempfile
import numpy as np
from numpy.random import RandomState
import pathlib

device = "cuda"


def test():
    b = 4
    w = 32
    h = 128

    ir = f"""
    #smem = #ttg.shared_memory
    #shared = #ttg.padded_shared<[32:+4, 512:+4] {{order = [0, 1]}}>
    #blocked = #ttg.blocked<{{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}}>
    #blocked1 = #ttg.blocked<{{sizePerThread = [1,1,8], threadsPerWarp = [1,4,16], warpsPerCTA = [1,1,1], order = [2,1,0]}}>
    #blocked2 = #ttg.blocked<{{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}}>

    module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32}} {{
        tt.func public @kernel(%x: !tt.ptr<i16> {{tt.divisibility = 16 : i32}}) {{
            %c{h}_i32 = arith.constant {h} : i32
            %buf = ttg.local_alloc : () -> !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            %c2_i32 = arith.constant 2 : i32
            %c3_i32 = arith.constant 3 : i32
            %buf_0 = ttg.memdesc_index %buf, %c0_i32 : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            %buf_1 = ttg.memdesc_index %buf, %c1_i32 : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            %buf_2 = ttg.memdesc_index %buf, %c2_i32 : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            %buf_3 = ttg.memdesc_index %buf, %c3_i32 : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>

            %cm1_i16 = arith.constant -1 : i16
            %c0_i16 = arith.constant 0 : i16
            %c1_i16 = arith.constant 1 : i16
            %c2_i16 = arith.constant 2 : i16
            %c3_i16 = arith.constant 3 : i16

            %cm1_splat = tt.splat %cm1_i16 : i16 -> tensor<{b}x{w}x{h}xi16, #blocked1>
            %c0_splat = tt.splat %c0_i16 : i16 -> tensor<{w}x{h}xi16, #blocked>
            %c1_splat = tt.splat %c1_i16 : i16 -> tensor<{w}x{h}xi16, #blocked>
            %c2_splat = tt.splat %c2_i16 : i16 -> tensor<{w}x{h}xi16, #blocked>
            %c3_splat = tt.splat %c3_i16 : i16 -> tensor<{w}x{h}xi16, #blocked>

            ttg.local_store %cm1_splat, %buf : tensor<{b}x{w}x{h}xi16, #blocked1> -> !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable>

            // ttg.local_store %c0_splat, %buf_0 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            ttg.local_store %c1_splat, %buf_1 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            // ttg.local_store %c2_splat, %buf_2 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            // ttg.local_store %c3_splat, %buf_3 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>

            // %buf1_data = ttg.local_load %buf_1 : !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable> -> tensor<{w}x{h}xi16, #blocked>

            %buf_data = ttg.local_load %buf : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> tensor<{b}x{w}x{h}xi16, #blocked1>
            %buf_lin_data = tt.reshape %buf_data : tensor<{b}x{w}x{h}xi16, #blocked1> -> tensor<{b*w*h}xi16, #blocked2>

            %offset = tt.make_range {{end = {b*w*h} : i32, start = 0 : i32}} : tensor<{b*w*h}xi32, #blocked2>
            amdgpu.buffer_store %buf_lin_data, %x[%offset]: tensor<{b*w*h}xi16, #blocked2>
            tt.return
        }}
    }}
    """
    tmp_file = "tmp.ttgir"
    with open(tmp_file, "w") as f:
        f.write(ir)
    kernel = triton.compile(tmp_file)

    x = torch.zeros((w, h), dtype=torch.int16, device=device)
    pgm = kernel[(1, 1, 1)](x)
    np.testing.assert_allclose(x.cpu().numpy(), 1)


if __name__ == "__main__":
    test()
