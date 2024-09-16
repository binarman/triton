import triton
import triton.language as tl


@triton.jit()
def streamk_gemm_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(
    A,
    B,
    C,
    bias_ptr,
    P,
    locks,
    M,
    N,
    K,
    num_sms,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(K, 1, 1):
        rk = BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    rm = tl.arange(0, BLOCK_SIZE_M)
    P_ = P + rm[:, None] + rk[None, :]
    k_mask = rk < K
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    tl.store(P_, acc, mask=k_mask[None, :], cache_modifier=".wt")
