#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>


template <typename T, typename F>
__device__ T GpuAtomicCasHelper(T* ptr, F accumulate) {
  T old = *ptr;
  T assumed;
  do {
    assumed = old;
    old = atomicCAS(ptr, assumed, accumulate(assumed));
  } while (assumed != old);
  return old;
}

template <typename F>
__device__ __half GpuAtomicCasHelper(__half* ptr, F accumulate) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Not little endian");
#endif
  intptr_t intptr = reinterpret_cast<intptr_t>(ptr);
  assert(!(intptr & 0x1));  // should be 2-aligned.
  if (intptr & 0x2) {
    // The half is in the second part of the uint32_t (upper 16 bits).
    uint32_t* address = reinterpret_cast<uint32_t*>(intptr - 2);
    uint32_t result = GpuAtomicCasHelper(address, [accumulate](uint32_t arg) {
      unsigned short high = static_cast<unsigned short>(arg >> 16);
      __half acc = accumulate(__ushort_as_half(high));
      return (static_cast<uint32_t>(__half_as_ushort(acc)) << 16) | (arg & 0xffff);
    });
    return __ushort_as_half(static_cast<uint16_t>(result >> 16));
  } else {
    // The half is in the first part of the uint32_t (lower 16 bits).
    uint32_t* address = reinterpret_cast<uint32_t*>(intptr);
    uint32_t result = GpuAtomicCasHelper(address, [accumulate](uint32_t arg) {
      unsigned short low = static_cast<unsigned short>(arg & 0xffff);
      __half acc = accumulate(__ushort_as_half(low));
      return (arg & 0xffff0000) | static_cast<uint32_t>(__half_as_ushort(acc));
    });
    return __ushort_as_half(static_cast<uint16_t>(result & 0xffff));
  }
}

__device__ inline __half GpuAtomicAdd(__half* ptr, __half value) {
  return GpuAtomicCasHelper(ptr, [value](__half a) { return a + value; });
}

template <typename T, typename AccT>
__global__ void test_kernel(
    int32_t nthreads, T* out, int32_t bias_size) {
  __shared__ char s_buf[1024];
  AccT* s_data = reinterpret_cast<AccT*>(s_buf);
  if(threadIdx.x < nthreads)
    s_data[threadIdx.x] = 1.0f;
  __syncthreads();

  for (int32_t index = threadIdx.x; index < bias_size; index += blockDim.x)
    GpuAtomicAdd(out + index, T(s_data[index]));
}

int main()
{
  __half* p;
  hipMalloc(&p, 4096);
  hipMemset(p, 0, 4096);
  hipDeviceSynchronize();
  test_kernel<__half, float> <<<1,256>>>(48, p, 12);
  hipDeviceSynchronize();
  bool pass = true;
  for(int i=0; i<10; i++) {
    if(*(uint16_t*)&p[i] != 0x3c00)
      pass = false;
    printf("%04x | %f\n", *(uint16_t*)&p[i], __half2float(p[i]));
  }
  printf("Test %s\n", pass ? "PASS" : "FAIL");
}