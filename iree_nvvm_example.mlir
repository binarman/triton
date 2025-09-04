// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-nvvm))))" --iree-gpu-test-target=sm_60 --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @shared_memory_dealloc_elision {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export public @shared_memory_dealloc_elision layout(#pipeline_layout)
    builtin.module {
      func.func @shared_memory_dealloc_elision() {
        %f0 = arith.constant 0.0 : f32
        %f1 = arith.constant 1.0 : f32
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        %0 = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
        %1 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %0) -> (memref<1xf32, #gpu.address_space<workgroup>>) {
          memref.store %f0, %arg2[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
          memref.dealloc %arg2 : memref<1xf32, #gpu.address_space<workgroup>>

          %tmp = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
          memref.store %f1, %tmp[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
          memref.dealloc %tmp : memref<1xf32, #gpu.address_space<workgroup>>

          %2 = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
          scf.yield %2 : memref<1xf32, #gpu.address_space<workgroup>>
        }
        memref.dealloc %1 : memref<1xf32, #gpu.address_space<workgroup>>
        return
      }
    }
  }
}
