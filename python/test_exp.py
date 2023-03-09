#!/usr/bin/python3
import triton
import triton.language as tl
import numpy as np
from numpy.random import RandomState
from typing import Optional, Union
import torch
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret
import sys

def numpy_random(shape, dtype_str):
    if isinstance(shape, int):
        shape = (shape, )
    rs = RandomState(seed=17)
    return rs.normal(0, 1, shape).astype(dtype_str)

def to_triton(x: np.ndarray):
    return torch.tensor(x, device='cuda')

def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        raise "Not a triton-compatible tensor: {x}"

def test_exp():
    BLOCK_SIZE = 32
    SIZE = 1024*BLOCK_SIZE
    # define the kernel / launch-grid

    @triton.jit
    def kernel_exp(Z, X, SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + off)
        z = tl.exp(x)
        tl.store(Z + off, z)

    # inputs
    x = numpy_random(SIZE, dtype_str="float32")
    # reference result
    z_ref = np.exp(x)
    # triton result
    x_tri = to_triton(x)
    z_tri = to_triton(np.empty_like(x))

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']),)

    ms, min_ms, max_ms = triton.testing.do_bench(lambda: kernel_exp[grid](z_tri, x_tri, SIZE=SIZE, BLOCK_SIZE=BLOCK_SIZE, num_warps=1))
    print("ms:", ms, "min_ms:", min_ms, "max_ms:", max_ms)

    # compare
    np.testing.assert_allclose(z_ref.reshape((-1,)), to_numpy(z_tri).reshape((-1,)), rtol=0.01)
    abs_diff = abs(to_numpy(z_tri) - z_ref)
    relative_diff = abs_diff / (abs(z_ref)+1e-9)
    print("maximum relative diff:", max(relative_diff.reshape((-1, ))))

def test_api():
    BLOCK_SIZE = 64
    SIZE = 2*BLOCK_SIZE
    # define the kernel / launch-grid

    @triton.jit
    def kernel_program(PID, PROGRAMS, SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        programs = tl.num_programs(axis = 0)
        off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        #tl.store(PROGRAMS + off, programs)
        tl.store(PID + off, pid)

    # inputs
    x = numpy_random(SIZE, dtype_str="int32")
    # triton result
    PROG_tri = to_triton(x)
    PID_tri = to_triton(x)

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']),)

    kernel_program[grid](PID_tri, PROG_tri, SIZE=SIZE, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
    np.set_printoptions(threshold=sys.maxsize)
    print("PIDS:", to_numpy(PID_tri))
    print("PROGRAMS:", to_numpy(PROG_tri))


test_exp()
#test_api()
