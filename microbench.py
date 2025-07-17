import torch
import triton
import tempfile
from numpy.random import RandomState
import pathlib
import numpy as np

device = "cuda"


def test_itt_padding():

    shape = (64, 128)
    #        K   N
    # global order = [1, 0]
    # local order = [0, 1]

    ir = f"""
    #smem = #ttg.shared_memory
    #mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}}>
    #linear = #ttg.linear<{{register = [[1, 0], [0, 1], [0, 2], [0, 4]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = []}}>
    #shared = #ttg.padded_shared<[64:+4] {{order = [0, 1]}}>
    #blocked = #ttg.blocked<{{sizePerThread = [2, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}}>
    #dotop = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = 4}}>
    #slice0_load = #ttg.slice<{{dim = 0, parent = #blocked}}>
    #slice1_load = #ttg.slice<{{dim = 1, parent = #blocked}}>
    #slice0_store = #ttg.slice<{{dim = 0, parent = #dotop}}>
    #slice1_store = #ttg.slice<{{dim = 1, parent = #dotop}}>
    module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32}} {{
        tt.func public @kernel(%x: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %y: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
            %c128_i32 = arith.constant 128 : i32
            %23 = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #slice0_load>
            %46 = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #slice1_load>
            %47 = tt.expand_dims %46 {{axis = 1 : i32}} : tensor<64xi32, #slice1_load> -> tensor<64x1xi32, #blocked>
            %48 = tt.splat %c128_i32 : i32 -> tensor<64x1xi32, #blocked>
            %49 = arith.muli %47, %48 : tensor<64x1xi32, #blocked>
            %50 = tt.broadcast %49 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
            %51 = tt.expand_dims %23 {{axis = 0 : i32}} : tensor<128xi32, #slice0_load> -> tensor<1x128xi32, #blocked>
            %52 = tt.broadcast %51 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
            %53 = arith.addi %52, %50 : tensor<64x128xi32, #blocked>

            %111 = amdgpu.buffer_load %x[%53]: tensor<64x128xf16, #blocked>
            %116 = amdgpu.in_thread_transpose %111 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #linear>
            %117 = ttg.local_alloc %116 : (tensor<64x128xf16, #linear>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
            %118 = ttg.local_load %117 : !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf16, #dotop>

            %s_c128_i32 = arith.constant 128 : i32
            %s_23 = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #slice0_store>
            %s_46 = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #slice1_store>
            %s_47 = tt.expand_dims %s_46 {{axis = 1 : i32}} : tensor<64xi32, #slice1_store> -> tensor<64x1xi32, #dotop>
            %s_48 = tt.splat %s_c128_i32 : i32 -> tensor<64x1xi32, #dotop>
            %s_49 = arith.muli %s_47, %s_48 : tensor<64x1xi32, #dotop>
            %s_50 = tt.broadcast %s_49 : tensor<64x1xi32, #dotop> -> tensor<64x128xi32, #dotop>
            %s_51 = tt.expand_dims %s_23 {{axis = 0 : i32}} : tensor<128xi32, #slice0_store> -> tensor<1x128xi32, #dotop>
            %s_52 = tt.broadcast %s_51 : tensor<1x128xi32, #dotop> -> tensor<64x128xi32, #dotop>
            %s_53 = arith.addi %s_52, %s_50 : tensor<64x128xi32, #dotop>

            amdgpu.buffer_store %118, %y[%s_53]: tensor<64x128xf16, #dotop>
            tt.return
        }}
    }}
    """
    tmp_file = "tmp.ttgir"
    with open(tmp_file, "w") as f:
        f.write(ir)
    kernel = triton.compile(tmp_file)

    x = torch.randn(shape, dtype=torch.float16, device=device)
    y = torch.zeros(shape, dtype=torch.float16, device=device)
    pgm = kernel[(1, 1, 1)](x, y)
    np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())
