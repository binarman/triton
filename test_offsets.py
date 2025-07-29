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
    w = 1
    h = 64

    ir = f"""
    #smem = #ttg.shared_memory
    #shared = #ttg.padded_shared<[64:+2] {{order = [1, 0]}}>
    #blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}}>
    #blocked1 = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}}>

    module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32}} {{
        tt.func public @kernel(%x: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
            %c{h}_i32 = arith.constant {h} : i32
            %buf = ttg.local_alloc : () -> !ttg.memdesc<{b}x{w}x{h}xi32, #shared, #smem, mutable>
            %c1_i32 = arith.constant 1 : i32
            %buf_1 = ttg.memdesc_index %buf, %c1_i32 : !ttg.memdesc<{b}x{w}x{h}xi32, #shared, #smem, mutable> -> !ttg.memdesc<{w}x{h}xi32, #shared, #smem, mutable>

            %buf_1_data = ttg.local_load %buf_1 : !ttg.memdesc<{w}x{h}xi32, #shared, #smem, mutable> -> tensor<{w}x{h}xi32, #blocked>

            %buf_lin_data = tt.reshape %buf_1_data : tensor<{w}x{h}xi32, #blocked> -> tensor<{w*h}xi32, #blocked1>
            %offset = tt.make_range {{end = {w*h} : i32, start = 0 : i32}} : tensor<{w*h}xi32, #blocked1>
            amdgpu.buffer_store %buf_lin_data, %x[%offset]: tensor<{w*h}xi32, #blocked1>
            tt.return
        }}
    }}
    """
    tmp_file = "tmp.ttgir"
    with open(tmp_file, "w") as f:
        f.write(ir)
    kernel = triton.compile(tmp_file)

    x = torch.zeros((w, h), dtype=torch.int32, device=device)
    pgm = kernel[(1, 1, 1)](x)
    np.set_printoptions(threshold=100500)
    print(x.cpu().numpy())


if __name__ == "__main__":
    test()
