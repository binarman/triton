#!/usr/bin/python3
import triton
import triton.language as tl
import numpy as np
from numpy.random import RandomState
from typing import Optional, Union
import torch
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret
import sys
import struct
import math

def to_triton(x: np.ndarray):
    return torch.tensor(x, device='cuda')

def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        raise "Not a triton-compatible tensor: {x}"

def test_exp_precision():
    BLOCK_SIZE = 1024
    SAMPLE_SIZE = 2**26
    START_RANGE = 0
    END_RANGE = 2**32

    @triton.jit
    def kernel_exp(Z, X, SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + off)
        z = tl.exp(x)
        tl.store(Z + off, z)

    for sample_start in range(START_RANGE, END_RANGE, SAMPLE_SIZE):
        int_x = np.arange(sample_start, sample_start + SAMPLE_SIZE, dtype=np.uint32)
        float_x = struct.unpack("=" + str(SAMPLE_SIZE) + "f", struct.pack("=" + str(SAMPLE_SIZE) + "I", *int_x))
        ref = np.exp(float_x, dtype="float32")

        x_tri = to_triton(float_x)
        z_tri = to_triton(np.empty_like(float_x))
        grid = lambda meta: (triton.cdiv(SAMPLE_SIZE, meta['BLOCK_SIZE']),)
        kernel_exp[grid](z_tri, x_tri, SIZE=SAMPLE_SIZE, BLOCK_SIZE=BLOCK_SIZE)
        res = to_numpy(z_tri)
        max_diff = 0.0
        abs_diff = abs(res - ref)
        ref_res_nan_mismatch = False
        ref_res_zero_mismatch = False
        ref_res_inf_mismatch = False
        max_diff_id = -1
        for i in range(SAMPLE_SIZE):
            if math.isnan(ref[i]) != math.isnan(res[i]):
                ref_res_nan_mismatch = True
                continue
            if (ref[i] == 0.0) != (res[i] == 0.0):
                ref_res_zero_mismatch = True
                continue
            if math.isinf(ref[i]) != math.isinf(res[i]):
                ref_res_inf_mismatch = True
                continue
            if math.isinf(ref[i]) and math.isinf(res[i]):
                continue
            relative_diff = abs_diff[i] / (abs(ref[i]))
            if relative_diff > max_diff:
                max_diff_id = i + sample_start
            max_diff = max(max_diff, relative_diff)
        print("sample", sample_start)
        print("  maximum relative diff:", max_diff)
        print("  maximum relative diff id", max_diff_id)
        print("  ref_res_nan_mismatch", ref_res_nan_mismatch)
        print("  ref_res_zero_mismatch", ref_res_zero_mismatch)
        print("  ref_res_inf_mismatch", ref_res_inf_mismatch)

test_exp_precision()