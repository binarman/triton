// original kernel:
// https://raw.githubusercontent.com/ROCm/rocm-examples/develop/HIP-Basic/matrix_multiplication/main.hip

#include <hip/hip_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>
#include <cstddef>

constexpr int error_exit_code = 1;

#define HIP_CHECK(condition)                                                   \
  {                                                                            \
    const hipError_t error = condition;                                        \
    if (error != hipSuccess) {                                                 \
      std::cerr << "error " << hipGetErrorString(error) << std::endl;          \
      exit(error_exit_code);                                                   \
    }                                                                          \
  }

/// \brief Multiplies matrices \p A and \p B and stores the result to \p C.
/// - The number of rows of the result matrix is equal to the number of rows of
/// matrix A
///   which is \p blockDim.y*gridDim.y.
/// - The number of columns of the result matrix is equal to the number of
/// columns of matrix B
///   which is \p blockDim.x*gridDim.x.
/// - The number of columns of matrix \p A is passed as argument.
/// - The matrix elements are stored in a row-major order.
///
/// - Each thread in the grid is responsible for one element of the result
/// matrix.
/// - Each element is calculated cooperatively in a tiled manner. In each step,
/// a BlockSize*BlockSize
///   tile is loaded to the shared memory so individual threads can address this
///   shared cache instead of loading the same values from the global device
///   memory individually. The end result is accumulated through each step on a
///   per-thread basis.
/// - The matrix dimensions are assumed to be multiples of the block size for
/// simplicity.
template <unsigned int BlockSize>
__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C,
                                             const unsigned int a_cols) {
  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;
  const unsigned int spec = threadIdx.z;
  const unsigned int bx = blockIdx.x;
  const unsigned int by = blockIdx.y;

  // b_cols must match the number of output matrix columns.
  const unsigned int b_cols = blockDim.x * gridDim.x;

  // The number of tiles is determined by A's columns (which is equal to B's
  // rows).
  const unsigned int steps = a_cols / BlockSize;

  // thread_result is the accumulation variable.
  float thread_result = 0.0F;
  for (unsigned int step = 0; step < steps; step++) {
    // Shared memory is used to cache the tile from both input matrices.
    // The tile is a square of BlockSize*BlockSize.
    __shared__ float a_values[BlockSize][BlockSize];
    __shared__ float b_values[BlockSize][BlockSize];

    // Load each element in the tile to shared memory.
    if (spec == 1) {
      // Index of the top-left element of the tile in A.
      // "BlockSize * a_cols * by" is the number of elements to move "down".
      // "BlockSize * step" is the number of elements to move "right".
      const unsigned int a_idx = BlockSize * (a_cols * by + step);

      // Index of the top-left element of the tile in B.
      // "BlockSize * b_cols * step" is the number of elements to move "down".
      // "BlockSize * bx" is the number of elements to move "right".
      const unsigned int b_idx = BlockSize * (b_cols * step + bx);
      a_values[ty][tx] = A[a_idx + a_cols * ty + tx];
      b_values[ty][tx] = B[b_idx + b_cols * ty + tx];
    }

    // Synchronization is needed to make sure that all elements are loaded
    // before starting the calculation.
    __syncthreads();

    // Each thread calculates the scalar product of the tile and increments the
    // thread-individual thread_result.
    if (spec == 0) {
      for (unsigned int i = 0; i < BlockSize; i++)
        thread_result += a_values[ty][i] * b_values[i][tx];
    }

    // Synchronize to ensure that the calculation is finished before the next
    // tile's elements start to load.
    __syncthreads();
  }

  if (spec == 0) {
    // Calculate the index of the top-left element of the output block.
    const unsigned block_offset = b_cols * BlockSize * by + BlockSize * bx;

    // Every thread stores the final result to global memory.
    C[block_offset + b_cols * ty + tx] = thread_result;
  }
}

int main(int argc, const char *argv[]) {
  constexpr unsigned int block_size = 16;

  // Get matrix dimensions from the command line, if provided.
  const unsigned int a_rows = 2048;
  const unsigned int a_cols = 2048;
  const unsigned int b_cols = 2048;

  if ((a_rows % block_size != 0) || (a_cols % block_size != 0) ||
      (b_cols % block_size != 0)) {
    std::cout
        << "Matrix dimensions must be positive multiples of block_size (" +
               std::to_string(block_size) + ")"
        << std::endl;
    exit(error_exit_code);
  }

  // Outer matrix dimensions must match.
  const unsigned int b_rows = a_cols;
  const unsigned int c_cols = b_cols;
  const unsigned int c_rows = a_rows;

  std::vector<float> A(a_cols * a_rows);
  std::vector<float> B(b_cols * b_rows);
  std::vector<float> C(c_cols * c_rows);

  // Set matrix elements to a constant on the host.
  std::fill(A.begin(), A.end(), 1.F);

  constexpr float b_value = 0.02F;
  std::fill(B.begin(), B.end(), b_value);

  const size_t a_bytes = sizeof(float) * A.size();
  const size_t b_bytes = sizeof(float) * B.size();
  const size_t c_bytes = sizeof(float) * C.size();
  float *d_A{};
  float *d_B{};
  float *d_C{};
  HIP_CHECK(hipMalloc(&d_A, a_bytes));
  HIP_CHECK(hipMalloc(&d_B, b_bytes));
  HIP_CHECK(hipMalloc(&d_C, c_bytes));

  HIP_CHECK(hipMemcpy(d_A, A.data(), a_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, B.data(), b_bytes, hipMemcpyHostToDevice));

  const dim3 block_dim(block_size, block_size, 2);
  const dim3 grid_dim(c_cols / block_size, c_rows / block_size);

  // Launch matrix multiplication kernel.
  std::cout << "Matrix multiplication: [" << a_rows << 'x' << a_cols << "] * ["
            << b_rows << 'x' << b_cols << "], block size: " << block_size << 'x'
            << block_size << std::endl;
  matrix_multiplication_kernel<block_size>
      <<<grid_dim, block_dim, 0, hipStreamDefault>>>(d_A, d_B, d_C, a_cols);
  // Check if the kernel launch was successful.
  HIP_CHECK(hipGetLastError());

  // Copy the resulting matrix to the host. This call synchronizes with the
  // host.
  HIP_CHECK(hipMemcpy(C.data(), d_C, c_bytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  // Check if the resulting elements match the expectation.
  constexpr float tolerance = 0.001F;
  const bool validation_passed =
      std::all_of(C.begin(), C.end(), [=](const float value) {
        return tolerance > std::abs(value - a_cols * b_value);
      });
  if (validation_passed) {
    std::cout << "Validation passed." << std::endl;
  } else {
    std::cout << "Validation failed." << std::endl;
    return error_exit_code;
  }
}
