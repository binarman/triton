#!/usr/bin/python3
import triton
import triton.language as tl
import numpy as np
from numpy.random import RandomState
from typing import Optional, Union
import torch
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret

def numpy_random(shape, dtype_str):
    if isinstance(shape, int):
        shape = (shape, )
    rs = RandomState(seed=17)
    return rs.normal(0, 1, shape).astype(dtype_str)

def to_triton(x: np.ndarray):
    return torch.tensor(x, device='cuda')

def test_exp():
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = tl.exp(x)
        tl.store(Z + off, z)

    # inputs
    x = numpy_random(SIZE, dtype_str="float32")
    # reference result
    z_ref = np.exp(x)
    # triton result
    x_tri = to_triton(x)
    z_tri = to_triton(np.empty_like(z_ref))
    kernel[(1, )](z_tri, x_tri, SIZE=SIZE, num_warps=4)
    # compare
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)


test_exp()