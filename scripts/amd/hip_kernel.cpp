#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

template <typename T> __global__ void div_kernel(T *out) {
  out[threadIdx.x] = 2.0/2.0;
}

int main() {
  float *p;
  hipMalloc(&p, 4096);
  hipMemset(p, 0, 4096);
  hipDeviceSynchronize();
  div_kernel<float><<<1, 256>>>(p);
  hipDeviceSynchronize();
  bool pass = true;
  for (int i = 0; i < 10; i++) {
    if (p[i] != 1.0)
      pass = false;
    printf("%f | %f\n", p[i], 1.0);
  }
  printf("Test %s\n", pass ? "PASS" : "FAIL");
}