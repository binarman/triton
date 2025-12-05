#blocked = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4], [0, 8]], lane = [[0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[0, 64], [0, 128]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(%K_Buffer: !tt.ptr<f8E4M3FNUZ> {tt.divisibility = 16 : i32}, %stride_buf_kbs: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %kv_loc_53 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_k_c = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_buf_kv = tt.expand_dims %kv_loc_53 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %offs_buf_kv_57 = tt.splat %stride_buf_kbs : i32 -> tensor<64x1xi32, #blocked>
    %offs_buf_kv_58 = arith.muli %offs_buf_kv, %offs_buf_kv_57 : tensor<64x1xi32, #blocked>
    %offs_buf_kv_59 = tt.expand_dims %offs_k_c {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %offs_buf_kv_60 = tt.broadcast %offs_buf_kv_58 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %offs_buf_kv_61 = tt.broadcast %offs_buf_kv_59 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %offs_buf_kv_62 = arith.addi %offs_buf_kv_60, %offs_buf_kv_61 : tensor<64x256xi32, #blocked>

    %kv1 = amdgpu.buffer_load %K_Buffer[%offs_buf_kv_62] : tensor<64x256xf8E4M3FNUZ, #blocked> loc(#loc1)
    %smem_kv1 = ttg.local_alloc : () -> !ttg.memdesc<64x256xf8E4M3FNUZ, #shared, #smem, mutable> loc(#loc2)
    %0 = ttg.convert_layout %kv1 : tensor<64x256xf8E4M3FNUZ, #blocked> -> tensor<64x256xf8E4M3FNUZ, #linear> loc(#loc3)
    ttg.local_store %0, %smem_kv1 : tensor<64x256xf8E4M3FNUZ, #linear> -> !ttg.memdesc<64x256xf8E4M3FNUZ, #shared, #smem, mutable> loc(#loc4)
    tt.return
  }
}
#loc1 = loc("buffer_load":1:1)
#loc2 = loc("local_alloc":1:1)
#loc3 = loc("trans":1:1)
#loc4 = loc("local_store":1:1)
