Sequence of transofmrations:

1. Unroll loop in original IR:

  ./build/bin/triton-opt --test-loop-unrolling=unroll-factor=4 orig.ttgir > unrolled.ttgir

  This will generate two loops: one unrolled, one for the tail of iterations

2. manually aggregate loads in unroll.mlir:
  - add aggregated load in unrolled loop body
  - add appropriate layout for aggregated load
  - add extract_slice and replace scale values uses
  - remove old LDS based loads for scales in first loop (?)
  - remove first loop arguments related to scales (?)
  - adjust address computation, move old address computation before second loop

  first loop aggregated scales are not pipeliner, because this will further increase register pressure.
  second loop is a normal one, because it is easier to leave it like this, plus it seems pipelining of scales do not affect performance that much.

3. cleanup IR and update wait coutners:
  ./build/bin/triton-opt --tritonamdgpu-update-async-wait-count=arch-generation-name=gfx950 --canonicalize unrolled.ttgir > final.ttgir
