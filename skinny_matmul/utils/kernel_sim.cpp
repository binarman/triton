#include <cstdint>
#include <iostream>

struct uint3 {
  uint32_t x, y, z;
};

#define QK8_1 32
#define K_SCALE_SIZE 12
#define QK_K 256
#define QI4_K (QK_K / (4 * QR4_K))
#define QR4_K 2
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

struct ggml_half {
  int16_t d;
};

struct ggml_half2 {
  int16_t x, y;
};

using half2 = ggml_half2;

struct block_q8_1 {
  union {
    struct {
      ggml_half d; // delta
      ggml_half s; // d * sum(qs[i])
    } GGML_COMMON_AGGR_S;
    ggml_half2 ds;
  };
  int8_t qs[QK8_1]; // quants
};

struct block_q4_K {
  union {
    struct {
      ggml_half d;    // super-block scale for quantized scales
      ggml_half dmin; // super-block scale for quantized mins
    } GGML_COMMON_AGGR_S;
    ggml_half2 dm;
  };
  uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
  uint8_t qs[QK_K / 2];         // 4--bit quants
};

float __low2float(ggml_half2 a) { return a.x; }

int ggml_cuda_dp4a(const int a, const int b, int c) {
  for (int i = 0; i < 4; ++i) {
    int apart = (a >> (i * 8)) & 0xff;
    int bpart = (b >> (i * 8)) & 0xff;
    c += apart * bpart;
  }
  return c;
}

struct float2 {
  float x, y;
};

float2 __half22float2(half2 v) {
  return {static_cast<float>(v.x), static_cast<float>(v.y)};
}

uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
  // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
  // fastdiv_values.z is unused and optimized away by the compiler.
  // Compute high 32 bits of n * mp
  // const uint32_t hi = __umulhi(n, fastdiv_values.x);
  const uint32_t hi = ((uint64_t)n * fastdiv_values.x) >> 32;
  // add n, apply bit shift
  return (hi + n) >> fastdiv_values.y;
}

int flat_tid;

float vec_dot_q4_K_q8_1_impl_vmmq(const int *v, const int *u, const uint8_t *sc,
                                  const uint8_t *m, const half2 &dm4,
                                  const float *d8) {

  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

    const int dot1 = ggml_cuda_dp4a(
        v1i, u[2 * i + 1],
        ggml_cuda_dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
    const int dot2 =
        ggml_cuda_dp4a(0x01010101, u[2 * i + 1],
                       ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m +=
        d8[i] *
        (dot2 * m[i]); // multiply constant part of q4_K with sum of q8_1 values
  }

  const float2 dm4f = __half22float2(dm4);

  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

float vec_dot_q4_K_q8_1(void *vbq, block_q8_1 *bq8_1, const int &kbx,
                        const int &iqs) {

  block_q4_K *bq4_K = (block_q4_K *)vbq + kbx;

  int v[2];
  int u[2 * QR4_K]; // 4 elems
  float d8[QR4_K];  // 2 elems

  // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

  // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
  // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
  // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
  // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

  int *q4 = (int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  q4[0] = flat_tid + (flat_tid << 8) + (flat_tid << 16) + (flat_tid << 24);
  q4[4] = flat_tid + (flat_tid << 8) + (flat_tid << 16) + (flat_tid << 24);
  v[0] = q4[0];
  v[1] = q4[4];

  uint16_t *scales = (uint16_t *)bq4_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    scales[j + 0] = 0xfafa;
    scales[j + 2] = 0xfbfb;
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    scales[j + 0] = 0xffff;
    scales[j + 2] = 0xffff;
    scales[j - 2] = 0xffff;
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;

  for (int i = 0; i < QR4_K; ++i) { // 2 iterations
    block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    bq8i->ds.x = 0xfefe;
    bq8i->ds.y = 0xffff;

    int *q8 = (int *)bq8i->qs + ((iqs / 2) % 4);

    q8[0] = flat_tid + (flat_tid << 8) + (flat_tid << 16) + (flat_tid << 24);
    q8[4] = flat_tid + (flat_tid << 8) + (flat_tid << 16) + (flat_tid << 24);

    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }

  bq4_K->dm.x = 0xfcfc;
  bq4_K->dm.y = 0xfdfd;
  return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

struct {
  int x, y, z;
} threadIdx;

struct {
  int x, y, z;
} blockIdx;

template <int ncols_dst, bool has_fusion, bool is_multi_token_id = false>
void mul_mat_vec_q(void *vx, void *vy, const int32_t *ids, float *dst,
                   const uint32_t ncols_x, const uint3 nchannels_y,
                   const uint32_t stride_row_x, const uint32_t stride_col_y,
                   const uint32_t stride_col_dst, const uint3 channel_ratio,
                   const uint32_t stride_channel_x,
                   const uint32_t stride_channel_y,
                   const uint32_t stride_channel_dst, const uint3 sample_ratio,
                   const uint32_t stride_sample_x,
                   const uint32_t stride_sample_y,
                   const uint32_t stride_sample_dst,
                   const uint32_t ids_stride) {

  constexpr int qk = 256;
  constexpr int qi = 32;
  constexpr int vdr = 2; // get_vdr_mmvq(type); // 2
  // constexpr mmvq_parameter_table_id table_id = get_device_table_id();
  constexpr int nwarps = 2; // calc_nwarps(ncols_dst, table_id); // 2
  constexpr int rows_per_cuda_block =
      1; // calc_rows_per_block(ncols_dst, table_id); // 1
  constexpr int warp_size = 64; // ggml_cuda_get_physical_warp_size(); // 64

  // constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

  const int tid = warp_size * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk; //  14336 / 256 = 56
  constexpr int blocks_per_iter =
      vdr * nwarps * warp_size / qi; // 2 * 2 * 64 / 32 = 8

  const uint32_t channel_dst = blockIdx.y;

  uint32_t token_idx = 0;
  uint32_t channel_x;
  uint32_t channel_y;
  uint32_t sample_dst;

  // if constexpr (is_multi_token_id) {
  //     // Multi-token MUL_MAT_ID path, adding these in the normal path causes
  //     a perf regression for n_tokens=1 case token_idx  = blockIdx.z;
  //     channel_x  = ids[channel_dst + token_idx * ids_stride];
  //     channel_y  = fastmodulo(channel_dst, nchannels_y);
  //     sample_dst = 0;
  // } else {
  //     channel_x  = ncols_dst == 1 && ids ? ids[channel_dst] :
  //     fastdiv(channel_dst, channel_ratio); channel_y  = ncols_dst == 1 && ids
  //     ? fastmodulo(channel_dst, nchannels_y) : channel_dst; sample_dst =
  //     blockIdx.z;
  // }

  const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
  const uint32_t sample_y = sample_dst;

  bool use_gate = false;
  bool use_bias = false;
  bool use_gate_bias = false;
  const void *vgate = nullptr;
  const float *x_bias = nullptr;
  const float *gate_bias = nullptr;
  // ggml_glu_op active_glu;

  // if constexpr (has_fusion) {
  //     use_gate      = fusion.gate      != nullptr;
  //     use_bias      = fusion.x_bias    != nullptr;
  //     use_gate_bias = fusion.gate_bias != nullptr && use_gate;
  //     vgate         = fusion.gate;
  //     x_bias        = (const float *) fusion.x_bias;
  //     gate_bias     = (const float *) fusion.gate_bias;
  //     active_glu    = fusion.glu_op;
  // }

  float x_biases[ncols_dst] = {0.0f};
  float gate_biases[ncols_dst] = {0.0f};
  // if constexpr (has_fusion) {
  //     const uint32_t channel_bias = ids ? channel_x : channel_dst;
  //     if (use_bias) {
  //         x_bias = x_bias + sample_dst*stride_sample_dst +
  //         channel_bias*stride_channel_dst + row0;
  //         // 1. Hide latency by prefetching bias and gate here
  //         // 2. load only on threads that won't die after partial sum
  //         calculation if (threadIdx.x < rows_per_cuda_block && threadIdx.y ==
  //         0 &&
  //             (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) <
  //             stride_col_dst)) { for (int j = 0; j < ncols_dst; ++j) {
  //                 x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
  //             }
  //         }
  //     }
  //     if (use_gate_bias) {
  //         gate_bias = gate_bias + sample_dst*stride_sample_dst +
  //         channel_bias*stride_channel_dst + row0; if (threadIdx.x <
  //         rows_per_cuda_block && threadIdx.y == 0 &&
  //             (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) <
  //             stride_col_dst)) { for (int j = 0; j < ncols_dst; ++j) {
  //                 gate_biases[j] = gate_bias[j * stride_col_dst +
  //                 threadIdx.x];
  //             }
  //         }
  //     }
  // }

  // partial sum for each thread
  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
  float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  block_q8_1 *y = ((block_q8_1 *)vy) + sample_y * stride_sample_y +
                  channel_y * stride_channel_y;
  if constexpr (is_multi_token_id) {
    y += token_idx * stride_col_y;
  }
  const int kbx_offset = sample_x * stride_sample_x +
                         channel_x * stride_channel_x + row0 * stride_row_x;

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1); // y block index that aligns with kbx

    // x block quant index when casting the quants to int
    const int kqs = vdr * (tid % (qi / vdr));

    for (int j = 0; j < ncols_dst; ++j) {
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp[j][i] +=
            vec_dot_q4_K_q8_1(vx, &y[j * stride_col_y + kby],
                              kbx_offset + i * stride_row_x + kbx, kqs);
        if constexpr (has_fusion) {
          if (use_gate) {
            tmp_gate[j][i] +=
                vec_dot_q4_K_q8_1(vgate, &y[j * stride_col_y + kby],
                                  kbx_offset + i * stride_row_x + kbx, kqs);
          }
        }
      }
    }
  }
}

int main() {
  // c_ncols_dst: 1
  // is_multi_token_id: 0
  // block_nums: 4096 1 1
  // block_dims: 64 2 1
  // nbytes_shared: 0
  // ncols_x: 14336
  // nchannels_y: 0 0 0

  blockIdx.x = 0;
  blockIdx.y = 0;
  blockIdx.z = 0;

  // type: 12
  // c_ncols_dst: 1
  // is_multi_token_id: 0
  // block_nums: 4096 1 1
  // block_dims: 64 2 1
  // nbytes_shared: 0
  // ncols_x: 14336
  // nchannels_y: 0 0 0
  // stride_row_x: 56
  // stride_col_y: 448
  // stride_col_dst: 4096
  // channel_ratio: 1 0 1
  // stride_channel_x: 229376
  // stride_channel_y: 448
  // stride_channel_dst: 4096
  // sample_ratio: 1 0 1
  // stride_sample_x: 229376
  // stride_sample_y: 448
  // stride_sample_dst: 4096
  // ids_stride: 0

  constexpr int c_ncols_dst = 1;
  constexpr uint32_t ncols_x = 14336;
  uint3 nchannels_y = {0, 0, 0};
  uint32_t stride_row_x = 56;
  uint32_t stride_col_y = 448;
  uint32_t stride_col_dst = 4096;
  uint3 channel_ratio = {1, 0, 1};
  uint32_t stride_channel_x = 229376;
  uint32_t stride_channel_y = 448;
  uint32_t stride_channel_dst = 4096;
  uint3 sample_ratio = {1, 0, 1};
  uint32_t stride_sample_x = 229376;
  uint32_t stride_sample_y = 448;
  uint32_t stride_sample_dst = 4096;
  uint32_t ids_stride = 0;

  constexpr uint32_t nrows_x = 2;
  void *vx = new char[nrows_x * sizeof(block_q4_K) * (ncols_x / QK_K)];
  void *vy = new char[c_ncols_dst * sizeof(block_q8_1) * (ncols_x / QK8_1)];
  int32_t *ids = nullptr;
  float *dst = new float[c_ncols_dst * nrows_x];
  flat_tid = 0;

  for (threadIdx.z = 0; threadIdx.z < 1; threadIdx.z++)
    for (threadIdx.y = 0; threadIdx.y < 2; threadIdx.y++)
      for (threadIdx.x = 0; threadIdx.x < 64; threadIdx.x++) {
        flat_tid++;
        mul_mat_vec_q<c_ncols_dst, false, false>(
            vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y,
            stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,
            stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
            stride_sample_dst, ids_stride);
      }
  std::cout << "X tensor contents: ";
  int start_x_idx =
      144 * (ncols_x / QK_K) *
      0 /* row id */; // size of block_q4_K * number of these blocks * row
  for (int i = start_x_idx; i < start_x_idx + 144 * 4; ++i) {
    std::cout << (int32_t)((int8_t *)vx)[i] << " ";
  }

  std::cout << "\n\n";
  std::cout << "Y tensor contents: ";
  //   int start_y_idx = 36*(ncols_x/QK_K);
  int start_y_idx = 0;
  for (int i = start_y_idx; i < start_y_idx + 1024; ++i) {
    std::cout << (int32_t)((int8_t *)vy)[i] << " ";
  }
  std::cout << "\n";
}
