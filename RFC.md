# Small dot optimization

Good day!

This is an RFC for small `tl.dot` optimizations.
First proposed optimization is based on splitting K dim idea.
Second proposed optimization is an upgrade of split-k transformation based on bypassing LDS.

## Background

Consider matrix multiplication, with relatively small M/N dimensions and large K dimension.
For example `M=1`, `K=4096`, `N=1024`, i.e. `(1x4096) x (4096x1024) -> (1x1024)`

Let's take following code as an `reference` GEMM implementation:

``` python
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offs_bn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offs_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)
```

This implementation has two issues:
- Matrix `A` is so small, that Triton can not use matrix instructions directly and requires padded operands, wasting part of computation
- Level of available parallelism is limited: only N dimension can be distributed across GPU cores and `BLOCK_N` can not be smaller than certain limit.

If padding is not an option or amount of wasted computations is large, Triton can use FMA based `tl.dot`, which solves first problem.
**Further in text,** `tl.dot` **is FMA based**.

To fully utilize parallelism of thread block, we need to distrubute `<warp_size> * <num_warps>` threads across dot dimensions: `BLOCK_M` and `BLOCK_N`.
For example if `warp_size=64` and `num_warps=4`, Triton needs to distribute `256` threads across `BLOCK_M * BLOCK_N` elements of dot output. Since `M=1`, we have no choice, except set `BLOCK_M=1`, this forces minimal size of `BLOCK_M` to 256.

FMA dot solves wasted computation issue, but does not solve limited parallelism issue. Considering `BLOCK_M` for example, we have only `M/BLOCK_M = 1024/256 = 4` thread blocks we can run.

## Intra-warp split-k optimization

Additional parallelism could be squeezed out of splitting K dimension.
This is typically done with atomic RMW operations, but this approach adds overhead.

Using FMA dot opens possibility to avoid this overhead almost entirely using intra-warp reductions across threads of one warp, instead of inter-warp reductions across warps throgh global memory.

Idea behind this approach is to split BLOCK_K up to `warp_size`, distribute K dimension across threads of one warp and then reduce them to one.
This gives opportunitty to lower minimal `BLOCK_M` size down to `num_warps`.

Idea is to split K dimension into blocks, use this "block" dimension as a batch in dot3dm then reduce over this blocks.
In triton language proposed solution could look like this:

``` python
def matmul_kernel_explicit_dot3d(  #
        a_ptr, b_ptr, c_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        SPLIT_K: tl.constexpr = 64):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_bn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // SPLIT_K)
    offs_split_k = tl.arange(0, SPLIT_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_split_k[:, None, None] * BLOCK_SIZE_K // SPLIT_K * stride_bk +
                      offs_bk[None, :, None] * stride_bk + offs_bn[None, None, :] * stride_bn)
    acc = tl.zeros((SPLIT_K, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a = tl.reshape(a, (BLOCK_SIZE_M, SPLIT_K, BLOCK_SIZE_K // SPLIT_K))
        a = tl.permute(a, (1, 0, 2))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = tl.sum(acc, axis=0)                                   # Reduction
    offs_cm = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offs_cn = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)
```

Using mentioned split k technique `BLOCK_M` could be reduced down to number of warps, theoretical parallelism in example above grows up to `N/BLOCK_M = 1024/num_warps = 256` versus original `4` thread blocks.

### Problems

Currently triton generates dot layouts with multiple warps in batch dimension, which reduces efficiency of reduction.

## Shared memory elimination optimization

Previous optimization solves problems of low parallelism, but in many cases program could be optimized even further by bypassing shared memory.

In example above, by carefully choosing layout of operand B triton can completely eliminate shared mememory conversions for it.
This is extremely beneficial, because this saves both memory accesses and address computations.

Consider following layout:

<Load operand>

## Possible implementations

There are two ways to achieve mentioned transofmrations:
- automatic compiler optimization of `dot2d`
- user defined kernel with `dot3d` + `reduction`

### Compiler dot2d->dot3d transformation

Pros are:
- Simpler kernel code, old kernels could be optimized without changes

Cons are:
- Split-k parameter cound not be tuned, unless it is added as a `tl.dot` parameter.

## User driven transformation

Pros are:
- Split k could be tuned out of the box.

Cons are:
- Kernel developer have to implement non-obvious code, especially hard to justify, if kernel developer does not know Triton implementation details.
