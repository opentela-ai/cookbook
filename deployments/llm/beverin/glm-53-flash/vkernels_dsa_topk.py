"""Pure-PyTorch drop-in for sglang's ``fast_kpool_topk_transform_fused`` — the
radix top-k *transform* step of the GLM-5.3-Flash DSA decode path on MI300A.

WHY THIS EXISTS (vkernels issue #56)
------------------------------------
The GLM-5.3 DSA decode path is three HIP stages:

    query --> vk_hip_dsa_topk_logits      (vkernels #51, validated exact)
          --> kpool_topk_transform         (sglang TVM JIT — THE MIDDLE STEP)
          --> vk_hip_dsa_sparse_fwd        (vkernels #52, validated exact)

vkernels ships the two bookend kernels but NOT the middle ``kpool_topk_transform``
(``sgl_kernel_jit_kpool_topk_transform_{group_topk}``). That JIT is built by
sglang's ``kernels/ops/moe/kpool_topk_transform.py`` from
``kernels/jit/csrc/dsa/kpool_topk_transform.cuh``. On the ROCm 7.2 container it
only compiles after a ``cuda_fp16.h``->``hip_runtime.h`` guard (cookbook 9b6ab33),
and on the GLM-5.3 decode shape the launched kernel does not return — every
``scheduler_TP{0..3}_EP{0..3}`` enters ``Dl`` (uninterruptible GPU-driver wait)
with ENTER/EXIT 584/0 and no crash, no ``cudaGetLastError``, no ``SIGQUIT``
(beverin job 613311; vkernels #57 for the silent-hang signature).

This module is a numerically-equivalent pure-PyTorch implementation of the same
contract, rebinding ``sglang.kernels.ops.moe.kpool_topk_transform
.fast_kpool_topk_transform_fused`` via ``sitecustomize.py`` so the decode path
runs end-to-end on MI300A *without the JIT*. Three goals:

  1. UNBLOCK: if the hang is in the JIT, decode serves immediately.
  2. LOCALISE: if the hang persists with this in place, it is definitively in
     one of the two vkernels bookend kernels (``vk_hip_dsa_topk_logits`` /
     ``vk_hip_dsa_sparse_fwd``) — the JIT is eliminated as a suspect.
  3. REFERENCE: a debuggable oracle to validate the future HIP kernel
     (``vk_hip_dsa_topk_transform``, #56) against — same role as
     ``dsv4/indexer.py::topk_transform_512_pytorch_vectorized`` for the C4 path.

CONTRACT (mirrors sglang's wrapper + ``kpool_topk_transform.cuh:249-310``)
-------------------------------------------------------------------------
    score                : (B, S)            fp32, one score per pool group
    lengths              : (B,)              int32, valid group count per row
    pool_size            : int > 1           tokens per pool group
    topk                 : int % pool_size==0  token-level top-k (= group_topk*pool_size)
    page_table   (opt)   : (B, P)            int32 raw-token -> real-token map
    topk_indices_offset (opt): (B,)          int32 per-row offset (ragged)
                         (page_table and topk_indices_offset are mutually exclusive)
    row_starts   (opt)   : (B,)              int32 score-row start offsets
    seq_lens     (opt)   : (B,)              int32; when present, appends
                         ``seq_lens[b] % pool_size`` tail tokens
    page_table_row_index (opt): (B,)         int32 row into page_table
                         (requires page_table)

Returns: (B, out_cols) int32 where
    out_cols = topk + (pool_size - 1 if seq_lens is not None else 0)

Semantics (exactly the HIP kernel):
  row_start       = row_starts[b] if present else 0
  length          = lengths[b]
  offset          = topk_indices_offset[b] if present else 0
  ptr             = page_table_row_index[b] if present else b
  history_len     = min(length * pool_size, token_topk)   # token_topk == topk
  tail_count      = seq_lens[b] % pool_size  if seq_lens else 0

  For col in [0, history_len):
      group_rank = col // pool_size
      if length <= group_topk:  group_id = group_rank        # LINEAR: all groups
      else:                     group_id = topk(score[b, row_start:row_start+length])
      raw_token = group_id * pool_size + (col % pool_size)
      dst[col] = transform_kpool_token(raw_token)            # see below
  For col in [history_len, history_len + tail_count):
      raw_token = length * pool_size + (col - history_len)
      dst[col] = transform_kpool_token(raw_token)
  Else: dst[col] = -1

  transform_kpool_token(raw_token):
      if page_table is not None:  return page_table[ptr][raw_token]    # GATHER only
      if topk_indices_offset:     return raw_token + offset           # +offset
      return raw_token                                          # identity

GRAPH-SAFETY
------------
This implementation uses a per-batch Python loop with ``.item()`` CPU syncs and
``torch.topk`` for the ``length > group_topk`` path. It is therefore NOT
CUDA-graph-capturable. For the GLM-5.3 decode smoke (B=1, no graph capture during
``GEN_CORRECTNESS_SMOKE``) this is correct and fast enough. PRODUCTION serving
under CUDA graphs needs the HIP kernel (#56) — this bridge exists to unblock +
localise the hang, and to serve tokens while #56 lands. The rebind is gated by
``GLM53_TOPK_TRANSFORM_BACKEND=torch`` (default off -> JIT path is the control),
so the two paths are A/B-testable.

TIE-BREAKING (length > group_topk only)
---------------------------------------
The radix kernel breaks ties by LOWER group index (threshold-bin scan +
``atomicAdd`` places lower-index entries first). ``torch.topk(largest=True,
sorted=True)`` is not index-stable; for the GLM-5.3 decode case this branch is
NOT taken (``length <= group_topk`` for short pooled history), so tie-breaking
is irrelevant to the immediate unblock. The ``length > group_topk`` branch is
provided for completeness/prefill parity and is correct up to ties.
"""

from __future__ import annotations

import torch

__all__ = ["fast_kpool_topk_transform_fused"]


def fast_kpool_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: torch.Tensor | None = None,
    topk_indices_offset: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    page_table_row_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch ``fast_kpool_topk_transform_fused`` (see module docstring)."""
    # --- contract (mirror of sglang's wrapper asserts) ---
    assert topk % pool_size == 0, f"topk ({topk}) must be a multiple of pool_size ({pool_size})"
    group_topk = topk // pool_size
    assert score.dim() == 2, f"score must be 2-D (got {score.dim()})"
    assert page_table is None or topk_indices_offset is None, (
        "page_table and topk_indices_offset are mutually exclusive"
    )
    assert page_table_row_index is None or page_table is not None, (
        "page_table_row_index requires page_table"
    )

    B = int(score.shape[0])
    device = score.device
    append_tail = seq_lens is not None
    out_cols = topk + (pool_size - 1 if append_tail else 0)
    token_topk = topk  # out_cols - tail_cols; tail_cols = pool_size-1 if append_tail else 0

    # --- per-row scalars (int64 for indexing; the .item() syncs are fine for
    #     the B=1 decode smoke; see module docstring on graph-safety) ---
    lengths_i = lengths.to(torch.int64)
    row_starts_i = row_starts.to(torch.int64) if row_starts is not None else torch.zeros(B, dtype=torch.int64, device=device)
    offsets_i = topk_indices_offset.to(torch.int32) if topk_indices_offset is not None else torch.zeros(B, dtype=torch.int32, device=device)
    ptr_i = page_table_row_index.to(torch.int64) if page_table_row_index is not None else torch.arange(B, dtype=torch.int64, device=device)
    seq_lens_i = seq_lens.to(torch.int64) if seq_lens is not None else None

    dst = torch.full((B, out_cols), -1, dtype=torch.int32, device=device)

    has_page_table = page_table is not None
    has_offset = topk_indices_offset is not None

    for b in range(B):
        length = int(lengths_i[b].item())
        if length <= 0:
            # no pooled history: only tail tokens (if any) below; else dst stays -1
            if append_tail and (tail := int(seq_lens_i[b].item()) % pool_size) > 0:
                raw = torch.arange(length * pool_size, length * pool_size + tail,
                                   dtype=torch.int64, device=device)
                if has_page_table:
                    dst[b, :tail] = page_table[int(ptr_i[b].item()), raw].to(torch.int32)
                elif has_offset:
                    dst[b, :tail] = (raw.to(torch.int32) + int(offsets_i[b].item()))
                else:
                    dst[b, :tail] = raw.to(torch.int32)
            continue

        row_start = int(row_starts_i[b].item())
        offset = int(offsets_i[b].item())
        ptr = int(ptr_i[b].item())
        full_pool_token_len = length * pool_size
        history_len = min(full_pool_token_len, token_topk)
        tail_count = (int(seq_lens_i[b].item()) % pool_size) if append_tail else 0

        # --- group selection ---
        if length <= group_topk:
            # LINEAR branch (kpool_topk_transform.cuh:276-286): every group in
            # order; NO radix top-k. This is the GLM-5.3 decode case (short
            # pooled history), so this is the hot path for the unblock.
            group_ids = torch.arange(length, dtype=torch.int64, device=device)
        else:
            # radix top-k branch (cuh:288-310): pick the group_topk highest-
            # scoring groups from score[b, row_start : row_start+length].
            # Tie-break note: see module docstring (not taken on decode).
            row = score[b, row_start : row_start + length]
            _, sel = torch.topk(row, group_topk, largest=True, sorted=True)
            group_ids = sel.to(torch.int64)

        # --- expand to tokens: cols [0, history_len) ---
        # raw_token = group_id * pool_size + slot, where group_id = group_ids[col // pool_size]
        if history_len > 0:
            cols = torch.arange(history_len, dtype=torch.int64, device=device)
            group_rank = cols // pool_size            # (history_len,)
            slot = cols % pool_size                    # (history_len,)
            # group_rank is < len(group_ids) here: for the linear branch
            #   group_rank < history_len//pool_size <= length == len(group_ids);
            # for the topk branch group_rank < group_topk == len(group_ids).
            raw_token = group_ids[group_rank] * pool_size + slot
            if has_page_table:
                dst[b, :history_len] = page_table[ptr, raw_token].to(torch.int32)
            elif has_offset:
                dst[b, :history_len] = (raw_token.to(torch.int32) + offset)
            else:
                dst[b, :history_len] = raw_token.to(torch.int32)

        # --- append tail tokens: cols [history_len, history_len + tail_count) ---
        if tail_count > 0:
            raw = torch.arange(length * pool_size, length * pool_size + tail_count,
                               dtype=torch.int64, device=device)
            if has_page_table:
                dst[b, history_len : history_len + tail_count] = page_table[ptr, raw].to(torch.int32)
            elif has_offset:
                dst[b, history_len : history_len + tail_count] = (raw.to(torch.int32) + offset)
            else:
                dst[b, history_len : history_len + tail_count] = raw.to(torch.int32)

    return dst
