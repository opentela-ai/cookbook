// vkernels/kernels/dsa_topk_transform.hip
//
// DeepseekSparseAttn (DSA) kpool top-k *transform* — gfx942 / MI300A, vkernels #56.
//
// DRAFT: a faithful HIP port of sglang's `kernels/jit/csrc/dsa/
// kpool_topk_transform.cuh` (radix top-k + page-table transform), re-hosted as
// a vkernels C-ABI kernel in the style of `dsa.hip` (PR #52). The body is a
// near-line-for-line port of the validated CUDA source; gfx942 tuning (register
// pressure, LDS layout, the radix SMEM_INPUT_SIZE cap) is the open work, to be
// profiled once the kernel links. The pure-PyTorch drop-in in the cookbook
// (vkernels_dsa_topk.py, 7/7 vs kpool_topk_transform.cuh:249-310) is the oracle.
//
// See dsa_topk_transform.hpp for the full contract + the #57 precondition checks.

#include <hip/hip_runtime.h>

#include <algorithm>  // std::min
#include <bit>
#include <cstddef>
#include <cstdint>

#include <torch/extension.h>

#include "dsa_topk_transform.hpp"

namespace vkernels {
namespace kernels {
namespace {

// Compile-time pool-level top-k (number of groups selected per row). The host
// launch dispatches over the validated set (128, 160, 192, 224, 256, 512).
inline constexpr int kThreadsPerBlock = 1024;

// Reduced from 128KB to 32KB to improve occupancy (mirrors the sglang source).
// Each radix pass needs at most ~K candidates in the threshold bin, so 4K
// entries per round (2 rounds = 8K entries = 32KB) is sufficient.
inline constexpr std::size_t kSmem = 8 * 1024 * sizeof(uint32_t);

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B] or nullptr
  int32_t* __restrict__ indices;           // unused here (kept for layout parity)
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

#ifndef C10_LIKELY
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#endif

// Radix key transforms (sign-flip so larger float -> larger key). HIP provides
// __float2half_rn / __half_as_ushort / __float_as_uint identically to CUDA.
__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// fast_topk_cuda_tl_impl: radix top-k over input[row_start : row_start + length],
// writing the K largest-valued indices to index[0..K). Adapted from tilelang's
// deepseek_v32/topk_selector.py (see the sglang source header). Assumes
// length > K (the caller's `length <= K` branch below is a complete early-return
// that NEVER reaches here — the documented `:72` precondition applies only to
// this helper, exactly as flagged in vkernels #57).
template <int K>
__device__ void fast_topk_cuda_tl_impl(const float* __restrict__ input,
                                       int* __restrict__ index,
                                       int row_start, int length) {
  int topk = K;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input =
        (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[K - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ int32_t transform_kpool_token(
    int32_t raw_token,
    const int32_t* __restrict__ page_table_entry,
    const int32_t* __restrict__ topk_indices_offset,
    int32_t offset) {
  if (page_table_entry != nullptr) {
    return page_table_entry[raw_token];  // GATHER only (no *pool_size, no +offset)
  }
  if (topk_indices_offset != nullptr) {
    return raw_token + offset;
  }
  return raw_token;
}

// One block per batch row (grid = (B, 1, 1)); K = group_topk is a compile-time
// constant selected by the host launch. Body is kpool_topk_transform.cuh:249-310
// verbatim (see dsa_topk_transform.hpp for the contract).
template <int K>
__global__ __launch_bounds__(kThreadsPerBlock) void kpool_topk_transform_kernel(
    const FastTopKParams params,
    int32_t* __restrict__ dst_token_indices,
    int64_t dst_stride,
    int32_t pool_size,
    int32_t token_topk,
    int32_t out_cols,
    const int32_t* __restrict__ page_table,
    int64_t page_table_stride,
    const int32_t* __restrict__ page_table_row_index,
    const int32_t* __restrict__ topk_indices_offset,
    const int32_t* __restrict__ seq_lens) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = blockIdx.x;
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto score = input + bid * input_stride;
  const auto dst = dst_token_indices + bid * dst_stride;
  const auto page_table_row =
      page_table_row_index == nullptr ? bid : page_table_row_index[bid];
  const auto page_table_entry =
      page_table == nullptr ? nullptr : page_table + page_table_row * page_table_stride;
  const auto offset = topk_indices_offset == nullptr ? 0 : topk_indices_offset[bid];
  const bool append_tail = seq_lens != nullptr;
  const auto full_pool_token_len = length * pool_size;
  const auto history_len =
      full_pool_token_len < token_topk ? full_pool_token_len : token_topk;
  const auto tail_count = append_tail ? seq_lens[bid] % pool_size : 0;

  if (length <= K) {
    // LINEAR branch (GLM-5.3 decode: short pooled history). Every group in
    // order; NO radix top-k. This is the hot path for the unblock.
    for (int col = tid; col < out_cols; col += kThreadsPerBlock) {
      if (col < history_len) {
        const auto group_rank = col / pool_size;
        const auto slot = col % pool_size;
        const auto raw_token = group_rank * pool_size + slot;
        dst[col] = transform_kpool_token(raw_token, page_table_entry,
                                         topk_indices_offset, offset);
      } else if (append_tail && col < history_len + tail_count) {
        const auto raw_token = length * pool_size + (col - history_len);
        dst[col] = transform_kpool_token(raw_token, page_table_entry,
                                         topk_indices_offset, offset);
      } else {
        dst[col] = -1;
      }
    }
    return;
  }

  // RADIX branch (length > K): select the K highest-scoring groups.
  __shared__ int s_indices[K];
  fast_topk_cuda_tl_impl<K>(score, s_indices, row_start, length);
  for (int col = tid; col < out_cols; col += kThreadsPerBlock) {
    if (col < history_len) {
      const auto group_rank = col / pool_size;
      const auto group_id = s_indices[group_rank];
      const auto slot = col % pool_size;
      const auto raw_token = group_id * pool_size + slot;
      dst[col] = transform_kpool_token(raw_token, page_table_entry,
                                       topk_indices_offset, offset);
    } else if (append_tail && col < history_len + tail_count) {
      const auto raw_token = length * pool_size + (col - history_len);
      dst[col] = transform_kpool_token(raw_token, page_table_entry,
                                       topk_indices_offset, offset);
    } else {
      dst[col] = -1;
    }
  }
}

// --- host launch: K = group_topk, one block per batch row ---
template <int K>
void launch(const float* score, const int32_t* lengths, const int32_t* row_starts,
            int32_t* out, int64_t out_stride, int64_t input_stride, int B,
            int pool_size, int topk, const int32_t* page_table,
            int64_t page_table_stride, const int32_t* page_table_row_index,
            const int32_t* topk_indices_offset, const int32_t* seq_lens) {
  int token_topk = topk;
  int out_cols =
      topk + (seq_lens != nullptr ? (pool_size - 1) : 0);
  // input_stride == S (row-major (B, S)); threaded through from the C ABI so the
  // kernel reads score + bid * S (NOT bid * 0 — the placeholder bug the first
  // draft shipped with, fixed here).
  kpool_topk_transform_kernel<K><<<B, kThreadsPerBlock, kSmem, 0>>>(
      FastTopKParams{score, row_starts, /*indices=*/nullptr, lengths, input_stride},
      out, out_stride, pool_size, token_topk, out_cols, page_table,
      page_table_stride, page_table_row_index, topk_indices_offset, seq_lens);
}

}  // namespace
}  // namespace kernels
}  // namespace vkernels

// =============================================================================
// C ABI (mirrors dsa.hip's vk_hip_dsa_sparse_fwd). Precondition checks per #57.
// =============================================================================
extern "C" void vk_hip_dsa_topk_transform(int B, int S,
                                          const void* score,
                                          const void* lengths,
                                          const void* row_starts,
                                          int pool_size,
                                          int topk,
                                          const void* page_table,
                                          const void* topk_indices_offset,
                                          const void* seq_lens,
                                          const void* page_table_row_index,
                                          void* out) {
  using namespace vkernels::kernels;

  // --- #57 precondition checks (named errors, not silent hangs) ---
  TORCH_CHECK(pool_size > 1, "pool_size must be > 1 (got ", pool_size, ")");
  TORCH_CHECK(topk % pool_size == 0,
              "topk (", topk, ") must be a multiple of pool_size (", pool_size, ")");
  int group_topk = topk / pool_size;
  TORCH_CHECK(group_topk == 128 || group_topk == 160 || group_topk == 192 ||
                  group_topk == 224 || group_topk == 256 || group_topk == 512,
              "group_topk (", group_topk, "=topk/pool_size) must be in "
              "(128, 160, 192, 224, 256, 512); got topk=", topk,
              " pool_size=", pool_size);
  TORCH_CHECK(!(page_table != nullptr && topk_indices_offset != nullptr),
              "page_table and topk_indices_offset are mutually exclusive "
              "(both must not be present)");
  TORCH_CHECK(page_table_row_index == nullptr || page_table != nullptr,
              "page_table_row_index requires page_table to be present");
  TORCH_CHECK(score != nullptr, "score must be non-null");
  TORCH_CHECK(lengths != nullptr, "lengths must be non-null");
  TORCH_CHECK(out != nullptr, "out must be non-null");
  TORCH_CHECK(B > 0, "B must be > 0 (got ", B, ")");
  TORCH_CHECK(S > 0, "S must be > 0 (got ", S, ")");

  const auto* score_p = static_cast<const float*>(score);
  const auto* lengths_p = static_cast<const int32_t*>(lengths);
  const auto* row_starts_p =
      row_starts == nullptr ? nullptr : static_cast<const int32_t*>(row_starts);
  auto* out_p = static_cast<int32_t*>(out);
  const auto* page_table_p =
      page_table == nullptr ? nullptr : static_cast<const int32_t*>(page_table);
  const auto* topk_indices_offset_p = topk_indices_offset == nullptr
                                          ? nullptr
                                          : static_cast<const int32_t*>(topk_indices_offset);
  const auto* seq_lens_p =
      seq_lens == nullptr ? nullptr : static_cast<const int32_t*>(seq_lens);
  const auto* page_table_row_index_p =
      page_table_row_index == nullptr ? nullptr
                                      : static_cast<const int32_t*>(page_table_row_index);

  int out_cols = topk + (seq_lens_p != nullptr ? (pool_size - 1) : 0);
  int64_t out_stride = out_cols;
  // page_table is (B, P) row-major -> stride = P (the row length). The wrapper
  // MUST pass the real P; until the ABI grows a `P` argument we CANNOT infer it
  // here (a wrong stride misindexes page_table[bid][raw_token]). This is the
  // single open correctness TODO before this draft can serve: either (a) add
  // `P` to vk_hip_dsa_topk_transform, or (b) have the wrapper pass a strided
  // tensor whose stride the ABI reads via the PyTorch descriptor. Until then the
  // gather branch is BLOCKED (identity + topk_indices_offset paths are fine).
  int64_t page_table_stride = page_table_p == nullptr ? 0 : -1;  // -1 = UNSAFE
  TORCH_CHECK(page_table_p == nullptr, "gather (page_table) branch is blocked on "
              "a page_table_stride ABI arg (see TODO); use the identity or "
              "topk_indices_offset paths for now");

  // One block per batch row. input_stride = S (row-major (B, S)).
  switch (group_topk) {
    case 128:
      launch<128>(score_p, lengths_p, row_starts_p, out_p, out_stride, S, B,
                  pool_size, topk, page_table_p, page_table_stride,
                  page_table_row_index_p, topk_indices_offset_p, seq_lens_p);
      break;
    case 160:
      launch<160>(score_p, lengths_p, row_starts_p, out_p, out_stride, S, B,
                  pool_size, topk, page_table_p, page_table_stride,
                  page_table_row_index_p, topk_indices_offset_p, seq_lens_p);
      break;
    case 192:
      launch<192>(score_p, lengths_p, row_starts_p, out_p, out_stride, S, B,
                  pool_size, topk, page_table_p, page_table_stride,
                  page_table_row_index_p, topk_indices_offset_p, seq_lens_p);
      break;
    case 224:
      launch<224>(score_p, lengths_p, row_starts_p, out_p, out_stride, S, B,
                  pool_size, topk, page_table_p, page_table_stride,
                  page_table_row_index_p, topk_indices_offset_p, seq_lens_p);
      break;
    case 256:
      launch<256>(score_p, lengths_p, row_starts_p, out_p, out_stride, S, B,
                  pool_size, topk, page_table_p, page_table_stride,
                  page_table_row_index_p, topk_indices_offset_p, seq_lens_p);
      break;
    case 512:
      launch<512>(score_p, lengths_p, row_starts_p, out_p, out_stride, S, B,
                  pool_size, topk, page_table_p, page_table_stride,
                  page_table_row_index_p, topk_indices_offset_p, seq_lens_p);
      break;
    default:
      TORCH_CHECK(false, "unreachable: group_topk ", group_topk,
                  " not in the validated set");
  }
}

extern "C" void vk_hip_dsa_topk_transform_config(int topk, int pool_size,
                                                 int* group_topk,
                                                 int* out_cols,
                                                 int* append_tail_present) {
  TORCH_CHECK(pool_size > 1, "pool_size must be > 1 (got ", pool_size, ")");
  TORCH_CHECK(topk % pool_size == 0,
              "topk (", topk, ") must be a multiple of pool_size (", pool_size, ")");
  int g = topk / pool_size;
  TORCH_CHECK(g == 128 || g == 160 || g == 192 || g == 224 || g == 256 || g == 512,
              "group_topk (", g, ") must be in (128, 160, 192, 224, 256, 512)");
  if (group_topk) *group_topk = g;
  // append_tail_present is set by the WRAPPER (it knows whether seq_lens was
  // passed); the kernel itself cannot tell without the tensor. Mirror the
  // existing convention: the host sizes `out` using the wrapper's flag.
  int atp = append_tail_present ? *append_tail_present : 0;
  if (out_cols) *out_cols = topk + (atp ? (pool_size - 1) : 0);
}
