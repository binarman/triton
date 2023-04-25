# flake8: noqa: F821,F841
import itertools
import os
import re
from typing import Optional, Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.ops
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']


def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Hack. Never return zero so tests of division don't error out.
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32')
                & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def to_triton(x: np.ndarray, device='cuda', dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


def check_type_supported(dtype):
    '''
    skip test if dtype is not supported on the current device
    '''
    cc = torch.cuda.get_device_capability()
    if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
        pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")


@pytest.mark.parametrize("dtype_x", list(dtypes) + ["bfloat16"])
def test_empty_kernel(dtype_x, device='cuda'):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass
    check_type_supported(dtype_x)
    x = to_triton(numpy_random(SIZE, dtype_str=dtype_x), device=device, dst_type=dtype_x)
    kernel[(1, )](x, SIZE=SIZE, num_warps=4)

# MFMA Test Dot tests
@pytest.mark.parametrize("M, N, K, num_warps, col_a, col_b, epilogue, allow_tf32, dtype",
                         [(*shape, 2, False, False, epilogue, allow_tf32, dtype)
                          for shape in [(64, 64, 64), (32, 32, 32)]
                          for epilogue in ['none', 'trans', 'add-matrix']
                          for allow_tf32 in [False]
                          for dtype in ['float32']
                          if not (allow_tf32 and (dtype in ['float16']))] +

                         [(*shape_nw, col_a, col_b, 'none', allow_tf32, dtype)
                          for shape_nw in [[128, 128, 32, 2]]
                          for allow_tf32 in [False, True]
                          for col_a in [False]
                          for col_b in [False]
                          for dtype in ['float32']])
def test_dot(M, N, K, num_warps, col_a, col_b, epilogue, allow_tf32, dtype, device='cuda'):
    capability = torch.cuda.get_device_capability()
    if torch.version.hip is not None:
        if (M, N, K) == (64, 128, 128):
            pytest.skip("Not supported: memory out of resource.")

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk,
               Y, stride_yk, stride_yn,
               W, stride_wn, stride_wl,
               Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
               ALLOW_TF32: tl.constexpr,
               DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
               COL_A: tl.constexpr, COL_B: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, allow_tf32=ALLOW_TF32)
        if ADD_MATRIX:
            z += tl.load(Zs)
        if ADD_ROWS:
            ZRs = Z + off_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if ADD_COLS:
            ZCs = Z + off_n * stride_zn
            z += tl.load(ZCs)[None, :]
        if DO_SOFTMAX:
            max = tl.max(z, 1)
            z = z - max[:, None]
            num = tl.exp(z)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if CHAIN_DOT:
            w = tl.load(Ws)
            z = tl.dot(z.to(w.dtype), w)
        tl.store(Zs, z)
    # input
    rs = RandomState(17)
    if col_a:
        x = numpy_random((K, M), dtype_str=dtype, rs=rs).T
    else:
        x = numpy_random((M, K), dtype_str=dtype, rs=rs)
    if col_b:
        y = numpy_random((N, K), dtype_str=dtype, rs=rs).T
    else:
        y = numpy_random((K, N), dtype_str=dtype, rs=rs)
    w = numpy_random((N, N), dtype_str=dtype, rs=rs)
    if 'int' not in dtype:
        x *= .1
        y *= .1
    if dtype == 'float32' and allow_tf32:
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    w_tri = to_triton(w, device=device)
    # triton result
    if dtype == 'int8':
        z = 1 + numpy_random((M, N), dtype_str='int32', rs=rs)
    else:
        z = 1 + numpy_random((M, N), dtype_str=dtype, rs=rs) * .1

    z_tri = to_triton(z, device=device)
    if epilogue == 'trans':
        z_tri = torch.as_strided(z_tri, (M, N), z_tri.stride()[::-1])
    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         y_tri, y_tri.stride(0), y_tri.stride(1),
                         w_tri, w_tri.stride(0), w_tri.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         COL_A=col_a, COL_B=col_b,
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX=epilogue == 'add-matrix',
                         ADD_ROWS=epilogue == 'add-rows',
                         ADD_COLS=epilogue == 'add-cols',
                         DO_SOFTMAX=epilogue == 'softmax',
                         CHAIN_DOT=epilogue == 'chain-dot',
                         ALLOW_TF32=allow_tf32,
                         num_warps=num_warps)
    # torch result
    if dtype == 'int8':
        z_ref = np.matmul(x.astype(np.float32),
                          y.astype(np.float32())).astype(np.int32)
    else:
        z_ref = np.matmul(x, y)

    if epilogue == 'add-matrix':
        z_ref += z
    if epilogue == 'add-rows':
        z_ref += z[:, 0][:, None]
    if epilogue == 'add-cols':
        z_ref += z[0, :][None, :]
    if epilogue == 'softmax':
        num = np.exp(z_ref - np.max(z_ref, axis=-1, keepdims=True))
        denom = np.sum(num, axis=-1, keepdims=True)
        z_ref = num / denom
    if epilogue == 'chain-dot':
        z_ref = np.matmul(z_ref, w)
    # compare
    # print(z_ref[:,0], z_tri[:,0])
    if dtype == 'float32':
        # XXX: Somehow there's a larger difference when we use float32
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
    if torch.version.hip is None:
        # make sure ld/st are vectorized
        ptx = pgm.asm['ptx']
        if K > 16 or N > 16 or M > 16:
            # XXX: skip small sizes because they are not vectorized
            assert 'ld.global.v4' in ptx
            assert 'st.global.v4' in ptx
        if dtype == 'float32' and allow_tf32:
            assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
        elif dtype == 'float32' and allow_tf32:
            assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' not in ptx
        elif dtype == 'int8':
            assert 'mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32' in ptx

# TODO: uncomment once DotOperandEncoding::getElemsPerThread is implemented
# @pytest.mark.parametrize("dtype_str", ['float32', 'float16'])
# def test_dot_without_load(dtype_str):
#     @triton.jit
#     def _kernel(out):
#         a = GENERATE_TEST_HERE
#         b = GENERATE_TEST_HERE
#         c = tl.dot(a, b)
#         out_ptr = out + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
#         tl.store(out_ptr, c)

#     kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.full((32, 32), 1.0, tl.{dtype_str})"})
#     a = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device="cuda")
#     b = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device="cuda")
#     out_ref = torch.matmul(a, b)
#     out = torch.zeros((32, 32), dtype=getattr(torch, dtype_str), device="cuda")
#     kernel[(1,)](out)
#     assert torch.all(out == out_ref)

# ---------------
# test gemm
# ---------------

# Simplified matrix multiplication kernel
# Checks Matrix multiplication consists of several blocks is optimized correctly

def get_variant_golden(a, b):
    SIZE_M = a.shape[0]
    SIZE_K = a.shape[1]
    SIZE_N = b.shape[1]
    assert a.shape[1] == b.shape[0]
    zero_M_K = torch.zeros((SIZE_M, SIZE_K)).cuda()
    zero_3M_K = torch.zeros((3 * SIZE_M, SIZE_K)).cuda()
    zero_K_N = torch.zeros((SIZE_K, SIZE_N)).cuda()
    zero_3K_N = torch.zeros((3 * SIZE_K, SIZE_N)).cuda()
    a_padded = torch.cat((a, zero_M_K, zero_M_K), 0)
    a_padded = torch.cat((a_padded, zero_3M_K, zero_3M_K), 1)
    b_padded = torch.cat((b, zero_K_N, zero_K_N), 0)
    b_padded = torch.cat((b_padded, zero_3K_N, zero_3K_N), 1)
    c_padded = torch.matmul(a_padded, b_padded)
    return c_padded[:SIZE_M, :SIZE_N]

@pytest.mark.parametrize('SIZE_M,SIZE_N,SIZE_K,NUM_WARPS,BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,dtype',
    [
        (*shape_mnk, dtype)
        for shape_mnk in [
            [64, 32, 128, 1, 64, 64, 64], # smoke test
            [1136, 1, 4480, 1, 64, 64, 64],
            [1136, 2, 4480, 1, 64, 64, 64],
            [1136, 4, 4480, 1, 64, 64, 64],
            [1136, 8, 4480, 1, 64, 64, 64],
            [1136, 16, 4480, 1, 64, 64, 64],
            [1136, 1303, 4480, 1, 64, 64, 64],

            [16, 1, 4480, 1, 64, 64, 64],
            [16, 2, 4480, 1, 64, 64, 64],
            [16, 4, 4480, 1, 64, 64, 64],
            [16, 8, 4480, 1, 64, 64, 64],
            [16, 1303, 4480, 1, 64, 64, 64],
            [1136, 1303, 4480, 1, 64, 64, 64]]
        for dtype in ['float32', 'float16', 'int8']
    ]
)
def test_gemm(SIZE_M, SIZE_N, SIZE_K, NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype):
    if dtype == 'float32':
        operandTy = torch.float32
        outTy = torch.float32
    elif dtype == 'float16':
        operandTy = torch.float16
        outTy = torch.float32
    elif dtype == 'int8':
        operandTy = torch.int8
        outTy = torch.int32
    a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=operandTy)
    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=operandTy)
    grid = lambda META: (1, )

    # matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
    #                     stride_am=a.stride(0), stride_ak=a.stride(1),
    #                     stride_bk=b.stride(0), stride_bn=b.stride(1),
    #                     stride_cm=c.stride(0), stride_cn=c.stride(1),
    #                     M=a.shape[0], N=b.shape[1], K=a.shape[1],
    #                     BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    #                     num_warps=NUM_WARPS)
    c = triton.ops.matmul(a, b, outTy, NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    golden = torch.matmul(a, b)

    # It's not easy to get a proper error threshold in different size
    # Here the gemm calculation is padded to a different size in order to get
    # a variant version of the golden result. And the error between golden and
    # golden_variant provide reference on selecting the proper rtol / atol.
    golden_variant = get_variant_golden(a, b)
    golden_diff = golden - golden_variant
    golden_abs_err = torch.max(torch.abs(golden_diff)).item()
    golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()

    triton.testing.assert_close(c, golden.to(torch.float32), rtol=max(1e-4, 1.5 * golden_rel_err), atol=max(1e-4, 1.5 * golden_abs_err))
