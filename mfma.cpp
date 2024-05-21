#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using float16_t = _Float16;

#define HIP_CHECK(command)                                                     \
  {                                                                            \
    hipError_t stat = (command);                                               \
    if (stat != hipSuccess) {                                                  \
      std::cerr << "HIP error: " << hipGetErrorString(stat) << " in file "     \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(-1);                                                                \
    }                                                                          \
  }

std::ostream &operator<<(std::ostream &os, const float16_t &val) {
  os << static_cast<float>(val);
  return os;
}

constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 8;

constexpr int A_size = M * K;
constexpr int B_size = K * N;
constexpr int D_size = M * N;

__global__ void sgemm_32x32x8(const float16_t *A, const float16_t *B,
                              float *D) {
  unsigned lane32Id = threadIdx.x % 32;
  unsigned laneGroupId = threadIdx.x / 32;
  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  floatx16 d = {0};
  float16x4 a;
  float16x4 b;
  for (int elemId = 0; elemId < 4; ++elemId) {
    const int a_idx = lane32Id * K + elemId + laneGroupId * 4;
    a[elemId] = A[a_idx];

    const int b_idx = lane32Id + elemId * N + laneGroupId * 4 * N;
    b[elemId] = B[b_idx];
  }

  d = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, d, 0, 0, 0);

  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      const int d_idx = lane32Id + i * N + laneGroupId * 4 * N + j * 2 * 4 * N;
      D[d_idx] = d[i + 4 * j];
    }
  }
}

int main() {
  std::vector<float16_t> A_h(A_size);
  for (int i = 0; i < A_h.size(); ++i)
    A_h[i] = static_cast<float16_t>(0.0f);
  uint16_t bitRepMinSubNormal = 0x1;
  uint16_t bitRep1 = 0x3c00;
  A_h[0] = reinterpret_cast<float16_t &>(bitRepMinSubNormal);

  std::vector<float16_t> B_h(B_size);
  for (int i = 0; i < B_h.size(); ++i)
    B_h[i] = static_cast<float16_t>(0.0f);
  B_h[0] = reinterpret_cast<float16_t &>(bitRep1);

  // Make and populate device buffers
  float16_t *A_d, *B_d;
  float *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float16_t),
                      hipMemcpyHostToDevice));

  sgemm_32x32x8<<<1, 64>>>(A_d, B_d, D_d);
  HIP_CHECK(hipGetLastError());

  std::vector<float> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << D_h[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
