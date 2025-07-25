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
    #slice0 = #ttg.slice<{{dim = 0, parent = #blocked}}>
    #slice1 = #ttg.slice<{{dim = 1, parent = #blocked}}>
    module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32}} {{
        tt.func public @kernel(%x: !tt.ptr<i16> {{tt.divisibility = 16 : i32}}) {{
            %c{h}_i32 = arith.constant {h} : i32
            %0 = ttg.local_alloc : () -> !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            %c2_i32 = arith.constant 1 : i32
            %c3_i32 = arith.constant 1 : i32
            %buf_0 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            %buf_1 = ttg.memdesc_subview %0[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            %buf_2 = ttg.memdesc_subview %0[%c2_i32, %c0_i32, %c0_i32] : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            %buf_3 = ttg.memdesc_subview %0[%c3_i32, %c0_i32, %c0_i32] : !ttg.memdesc<{b}x{w}x{h}xi16, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>

            %c0_i16 = arith.constant 0 : i16
            %c1_i16 = arith.constant 1 : i16
            %c2_i16 = arith.constant 1 : i16
            %c3_i16 = arith.constant 1 : i16

            %c0_splat = tt.splat %c0_i16 : i16 -> tensor<{w}x{h}xi16, #blocked>
            %c1_splat = tt.splat %c1_i16 : i16 -> tensor<{w}x{h}xi16, #blocked>
            %c2_splat = tt.splat %c2_i16 : i16 -> tensor<{w}x{h}xi16, #blocked>
            %c3_splat = tt.splat %c3_i16 : i16 -> tensor<{w}x{h}xi16, #blocked>

            //ttg.local_store %c0_splat, %buf_0 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable, 4x32x128>
            ttg.local_store %c0_splat, %buf_0 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            ttg.local_store %c1_splat, %buf_1 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            ttg.local_store %c2_splat, %buf_2 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>
            ttg.local_store %c3_splat, %buf_3 : tensor<{w}x{h}xi16, #blocked> -> !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable>

            %buf1_data = ttg.local_load %buf_1 : !ttg.memdesc<{w}x{h}xi16, #shared, #smem, mutable> -> tensor<{w}x{h}xi16, #blocked>

            %23 = tt.make_range {{end = {h} : i32, start = 0 : i32}} : tensor<{h}xi32, #slice0>
            %46 = tt.make_range {{end = {w} : i32, start = 0 : i32}} : tensor<{w}xi32, #slice1>
            %47 = tt.expand_dims %46 {{axis = 1 : i32}} : tensor<{w}xi32, #slice1> -> tensor<{w}x1xi32, #blocked>
            %48 = tt.splat %c{h}_i32 : i32 -> tensor<{w}x1xi32, #blocked>
            %49 = arith.muli %47, %48 : tensor<{w}x1xi32, #blocked>
            %50 = tt.broadcast %49 : tensor<{w}x1xi32, #blocked> -> tensor<{w}x{h}xi32, #blocked>
            %51 = tt.expand_dims %23 {{axis = 0 : i32}} : tensor<{h}xi32, #slice0> -> tensor<1x{h}xi32, #blocked>
            %52 = tt.broadcast %51 : tensor<1x{h}xi32, #blocked> -> tensor<{w}x{h}xi32, #blocked>
            %53 = arith.addi %52, %50 : tensor<{w}x{h}xi32, #blocked>
            amdgpu.buffer_store %buf1_data, %x[%53]: tensor<{w}x{h}xi16, #blocked>
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
