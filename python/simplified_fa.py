#!/opt/conda/envs/py_3.8/bin/python

import pytest
import torch

import triton
import triton.language as tl

@triton.jit
def kernel(Q, K, V, Out,
              stride_qm, stride_qk,
              stride_kn, stride_kk,
              stride_vk, stride_vn,
              stride_om, stride_on,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr,
              ):
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(BLOCK_M, BLOCK_K),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(BLOCK_N, BLOCK_K),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_K, BLOCK_N),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(BLOCK_M, BLOCK_K),
        strides=(stride_om, stride_on),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    q = tl.load(Q_block_ptr)

    # -- compute qk ----
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.dot(q, k, matrix_instr_nonkdim=[4, 64])
    # add some reduction here
    p = qk
    # -- update output accumulator --
    acc = tl.dot(p.to(v.dtype), v, matrix_instr_nonkdim=[4, 4])

    # epilogue
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def forward(q, k, v):
    # shape constraints
    o = torch.empty_like(q)
    if torch.version.hip is None:
        BLOCK_M = 16
        BLOCK_N = 64
    grid = lambda META: (
        1,
        1,
        1
    )
    kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=q.shape[0],
        BLOCK_N=k.shape[0],
        BLOCK_K=q.shape[1],
    )

    return o

# dot1: (MxK) x (KxN) -> (MxN)
# dot2: (MxN) x (NxK) -> (MxK)
@pytest.mark.parametrize('M, N, K',
                         [(4, 64, 128),
                          ])
def test_op_fwd_mfma4(M, N, K):
    torch.manual_seed(20)
    dtype = torch.float16
    q = torch.empty((M, K), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((N, K), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((N, K), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    dout = torch.randn_like(q, dtype=torch.float16)
    # reference implementation
    p = torch.matmul(q.half(), k.transpose(0, 1).half())
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = forward(q, k, v)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)

