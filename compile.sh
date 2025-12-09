#!/usr/bin/bash

./build/cmake.linux-x86_64-cpython-*/bin/triton-opt --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm stage1.mlir &> stage2.mlir
~/.triton/llvm/llvm-ubuntu-x64/bin/mlir-translate --mlir-to-llvmir stage2.mlir | sed "s/ptx_kernel/amdgpu_kernel/" > stage3.ll
#~/.triton/llvm/llvm-ubuntu-x64/bin/llc -march=amdgcn -mcpu=gfx942 -O2 -mattr=-sdwa stage3.ll -o stage4.s
~/.triton/llvm/llvm-ubuntu-x64/bin/llc -mtriple=amdgcn-amd-amdhsa -march=amdgcn -mcpu=gfx942 -O3 stage3.ll -o stage4.s
