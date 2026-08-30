// vkernels/kernels/dsa_topk_transform.hpp
//
// DeepseekSparseAttn (DSA) kpool top-k *transform* (gfx942 / MI300A), vkernels #56.
//
// The MIDDLE step of the GLM-5.3 DSA decode path, between the two bookend
// kernels already in libvkernels_hip.so:
//
//   query --> vk_hip_dsa_topk_logits   (#51, paged-MQA gated scorer)
//         --> vk_hip_dsa_topk_transform (THIS FILE — radix top-k + page-table
//                                        transform of the selected group tokens)
//         --> vk_hip_dsa_sparse_fwd     (#52, gather over selected tokens)
//
// sglang ships the middle step as a TVM JIT (sgl_kernel_jit_kpool_topk_transform_
// {group_topk}, kernels/ops/moe/kpool_topk_transform.py <- jit/csrc/dsa/
// kpool_topk_transform.cuh). On the ROCm container it compiles only after a
// cuda_fp16.h->hip_runtime.h guard (cookbook 9b6ab33) and on the GLM-5.3 decode
// shape the launched kernel DOES NOT RETURN (beverin job 613311: every
// scheduler_TP{0..3}_EP{0..3} stuck Dl, ENTER/EXIT 584/0, no crash). A
// pure-PyTorch drop-in (cookbook vkernels_dsa_topk.py, validated 7/7 vs
// kpool_topk_transform.cuh:249-310) UNBLOCKS decode and confirms the JIT was the
// hang (job 613339 progressed past the old layer-3 stall). This HIP kernel is
// the long-term, graph-safe, precondition-checked replacement.
//
// CONTRACT (mirrors sglang's fast_kpool_topk_transform_fused wrapper +
// kpool_topk_transform.cuh:249-310; the torch fallback in the cookbook is the
// reference oracle):
//
//   score         : (B, S)             fp32, one score per pool group
//   lengths       : (B,)               int32, valid group count per row (>=0)
//   row_starts    : (B,) int32 or NULL  score-row start offset per batch
//   pool_size     : int > 1            tokens per pool group
//   topk          : int, % pool_size==0  token-level top-k (== group_topk*pool_size)
//   page_table    : (B, P) int32 or NULL raw-group-id -> real-token map (GATHER)
//   topk_indices_offset : (B,) int32 or NULL  per-row +offset (RAGGED)
//                         (page_table and topk_indices_offset are mutually exclusive;
//                          both NULL -> identity)
//   page_table_row_index : (B,) int32 or NULL  row into page_table
//                          (requires page_table; NULL -> batch index b)
//   seq_lens      : (B,) int32 or NULL  when present, appends
//                   seq_lens[b] % pool_size tail tokens (out_cols grows by pool_size-1)
//   out           : (B, out_cols) int32 where
//                   out_cols = topk + (pool_size - 1 if seq_lens else 0)
//
// Returns void. Host launch picks K = group_topk = topk/pool_size via a template
// dispatch over the validated set (128, 160, 192, 224, 256, 512). One block of
// 1024 threads owns exactly one batch row (grid = (B, 1, 1)).
//
// PRECONDITIONS (vkernels #57 — enforced here via TORCH_CHECK, unlike the
// existing dsa.{cpp,hip} which document but do not check them):
//   * topk % pool_size == 0
//   * group_topk in (128, 160, 192, 224, 256, 512)
//   * NOT (page_table present AND topk_indices_offset present)  (mutually exclusive)
//   * page_table_row_index present => page_table present
//   * lengths[b] >= 0 for all b
// An unmet precondition raises a named torch error instead of launching a kernel
// that silently never returns (the exact signature that made bring-up painful).
//
// STREAM (mirrors dsa.hip): the kernel launches on HIP stream 0. The host
// wrapper (vkernels_dsa_topk_transform.py, to be added) MUST torch.cuda.synchronize()
// before and after, exactly as vkernels_dsa.py does for the two bookends, until a
// stream parameter is threaded through (follow-up, mirroring PR #44 vs #52).

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// void vk_hip_dsa_topk_transform(int B, int S,
//   const void* score, const void* lengths, const void* row_starts,
//   int pool_size, int topk,
//   const void* page_table, const void* topk_indices_offset,
//   const void* seq_lens, const void* page_table_row_index,
//   void* out)
//
// See the file-level contract above. score is row-major (B, S) with input_stride
// = S; out is row-major (B, out_cols). All pointers are device pointers of the
// declared dtypes (score/seq_lens/lenses/etc fp32 or int32 as above). Any NULL
// optional is honoured as documented.
void vk_hip_dsa_topk_transform(int B, int S,
                               const void* score,
                               const void* lengths,
                               const void* row_starts,
                               int pool_size,
                               int topk,
                               const void* page_table,
                               const void* topk_indices_offset,
                               const void* seq_lens,
                               const void* page_table_row_index,
                               void* out);

// void vk_hip_dsa_topk_transform_config(int topk, int pool_size,
//   int* group_topk, int* out_cols, int* append_tail_present)
//
// Host helper mirroring vk_hip_dsa_config: resolves the per-shape tiling for the
// transform. group_topk = topk/pool_size (must be in the validated set or the
// call raises). out_cols = topk + (pool_size - 1) * (seq_lens present ? 1 : 0);
// the wrapper passes append_tail_present so the host can size `out` without
// re-deriving. Kept for ABI parity with the other two kernels.
void vk_hip_dsa_topk_transform_config(int topk, int pool_size,
                                      int* group_topk,
                                      int* out_cols,
                                      int* append_tail_present);

#ifdef __cplusplus
}
#endif
