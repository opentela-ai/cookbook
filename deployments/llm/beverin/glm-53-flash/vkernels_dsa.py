"""vkernels HIP DSA sparse-MLA forward — GLM-5.3-Flash on MI300A (gfx942).

Drop-in replacement for sglang's ``tilelang_sparse_fwd``
(``sglang/kernels/ops/attention/dsa/tilelang_kernel.py``) that routes the
sparse-MLA forward through the gfx942 HIP kernel ``vk_hip_dsa_sparse_fwd``
added to the ``vkernels`` library by PR #52 (closes the tilelang/TVM
``FloorMod(_, 0)`` JIT abort documented in vkernels issue #51).

Layout note: this file is an ENGINE DROP-IN (installed into $OVL/pylib by
build_overlay.sh; the meta/diag/glm53 dispatcher's patch_dsa_vk rebinds
sglang's dsa_backend to it lazily at the first DSA forward). It stays in
the recipe dir next to build_overlay.sh; everything reusable or
diagnostic lives in meta/diag/glm53.

WHY THIS EXISTS
---------------
GLM-5.3-Flash runs DeepseekSparseAttn (DSA) with ``qk_rope_head_dim = 0`` ->
``dim = 256, tail_dim = dim - v_head_dim = 256 - 256 = 0``. sglang's tilelang
``sparse_mla_fwd_decode_partial`` (bf16 path, selected on gfx942 by
``tilelang_sparse_fwd``) then allocates zero-extent ``Q_tail_buf`` /
``K_tail_shared`` and emits a zero-K GEMM; TVM's ``VectorizePlanner``
hit ``Check failed: pb->value != 0 (0 vs. 0) : Divide by zero`` at JIT time
(beverin job 612262, the first forward's DSA kernel). There is no
flag-switch escape: ``tilelang`` is the only ``kpool > 1``-legal DSA decode
backend on MI300A (fa3/trtllm are NVIDIA-only; flashmla_*/aiter fail
``_check_kpool_tail_backend``).

PR #52 re-implements the forward as a plain HIP kernel
(``src/c/vkernels/kernels/dsa.hip``) that takes ``tail_dim == 0`` as a
RUNTIME branch (the rope-tail dot loop runs zero iterations — no zero-size
GEMM is ever issued). It is validated on beverin (``test_dsa_correct``
11/11, both ``tail_dim == 0`` GLM-5.3 and ``tail_dim > 0`` DeepSeek-V3).
This module calls that kernel via the C ABI in ``libvkernels_hip.so``,
exactly as ``vkernels_experts.py`` calls ``vk_hip_fused_moe_mxfp4`` (PR #44)
on the Kimi-K3 recipe.

CONTRACT (mirrors sglang's ``tilelang_sparse_fwd`` public signature)
--------------------------------------------------------------------
    q       : (S_q,  H,  dim + tail_dim)  bf16   (3-D; leading batch is absent
             because sglang's dsa_backend._forward_tilelang passes q_all
             WITHOUT a batch dim — the tilelang wrapper re-adds it via
             .unsqueeze(0) for the @T.prim_func. The C kernel takes raw
             pointers + dims, so 3-D contiguous == 4-D[1,...] layout.)
    kv      : (S_kv, kv_group, dim + tail_dim)  bf16   (kv_group == 1)
    indices : (S_q, kv_group, topk)  int32   (topk padded to a multiple of 64;
             entries < 0 or >= S_kv are masked kpool tail tokens)
    sm_scale: float, the RAW MLA softmax scale = 1/sqrt(dim + tail_dim)
             (NOT yet folded with log2(e); the tilelang kernel folds it
             internally, so we fold it here to match vkernels' expectation
             that sm_scale already includes log2(e)).
    d_v     : int = v_head_dim = dim - tail_dim  (256 for GLM-5.3)
    return_lse: bool (False on the MI300A path — see sglang
             tilelang_sparse_fwd's ``assert not return_lse``; we support
             True too since the HIP kernel does, returning (out, lse).)

Returns:
    out : (1, S_q, H, d_v)  bf16   (combined; 4-D with batch=1, matching the
          tilelang ``sparse_mla_fwd_decode_combine`` output shape that
          _forward_tilelang returns directly when return_lse is False).
    (out, lse) when return_lse, where lse is (1, S_q, H) fp32 base-2 LSE
          (_forward_tilelang does ``lse.squeeze(0)`` on its caller side).

STREAM HANDLING (first integration — correctness over concurrency)
------------------------------------------------------------------
``vk_hip_dsa_sparse_fwd`` (PR #52) takes NO stream argument and launches on
HIP stream 0 (``dsa.hip::launch`` does ``<<<grid, block, shmem, 0>>>``),
unlike ``vk_hip_fused_moe_mxfp4`` (PR #44) which threads the current
stream through. To stay correct under EITHER the legacy default-stream
model (stream 0 auto-synchronizes with PyTorch's compute stream) OR the
per-thread default-stream model (it would not), we issue a device-wide
``torch.cuda.synchronize()`` immediately before the ctypes call (input q/kv
produced by sglang's compute stream are done) and immediately after (the
stream-0 kernel's output is done before sglang reads it). This costs two
full device syncs per DSA layer (45 layers/forward) — acceptable for the
"does PR #52 clear the JIT abort?" smoke; the stream parameter (mirroring
#44) is the follow-up optimization once the kernel is verified live.
"""

from __future__ import annotations

import ctypes
import glob
import os

import torch

__all__ = ["tilelang_sparse_fwd", "tilelang_fp8_paged_mqa_logits"]

# log2(e) = 1.44269504...; the tilelang kernel folds the raw MLA scale by
# this constant (tilelang_kernel.py: ``sm_scale = sm_scale * 1.44269504``),
# converting the natural-exp softmax to base-2 (FlashAttention's log2 trick).
# vkernels expects sm_scale already folded, so we replicate the fold here.
_LOG2E = 1.4426950408889634

_lib_cache: dict = {}


def _find_libvkernels_hip():
    """Locate libvkernels_hip.so.

    Search order (mirrors vkernels_experts.py from the Kimi-K3 recipe):
      1. VKERNELS_LIB env var (exact path)
      2. VKERNELS_DIR env var (dir containing build/hip/**/libvkernels_hip.so)
      3. common staging paths on beverin
    """
    direct = os.environ.get("VKERNELS_LIB")
    if direct and os.path.isfile(direct):
        return direct

    vdir = os.environ.get("VKERNELS_DIR")
    if vdir and os.path.isdir(vdir):
        hits = sorted(glob.glob(os.path.join(vdir, "build", "hip", "**", "libvkernels_hip.so"), recursive=True))
        if hits:
            return hits[-1]

    for cand in (
        "/capstor/scratch/cscs/xyao/vkernels/build/hip/src/c/libvkernels_hip.so",
        "/capstor/scratch/cscs/xyao/kimi-k3-vllm/home/pylib/libvkernels_hip.so",
    ):
        if os.path.isfile(cand):
            return cand

    return None


def _get_lib():
    """Load and cache libvkernels_hip.so (ctypes, global symbols)."""
    if "lib" not in _lib_cache:
        path = _find_libvkernels_hip()
        if path is None:
            raise RuntimeError(
                "libvkernels_hip.so not found. Set VKERNELS_LIB (exact path) or "
                "VKERNELS_DIR (dir with build/hip/**/libvkernels_hip.so), or place "
                "it in $K3/home/pylib/. Rebuild vkernels with PR #52."
            )
        _lib_cache["lib"] = ctypes.CDLL(path)
    return _lib_cache["lib"]


def _bind_dsa_fn(lib):
    """Bind vk_hip_dsa_sparse_fwd (PR #52). Returns None if absent."""
    fn = getattr(lib, "vk_hip_dsa_sparse_fwd", None)
    if fn is None:
        raise RuntimeError(
            "vk_hip_dsa_sparse_fwd not found in libvkernels_hip.so — "
            "rebuild vkernels at PR #52 (commit d76517c) with VKERNELS_HAS_HIP=ON"
        )
    # void vk_hip_dsa_sparse_fwd(int S_q, int S_kv, int H, int dim, int tail_dim,
    #   int topk, int kv_group, int block_I, int inner_iter, float sm_scale,
    #   int return_lse, const void* q, const void* kv, const void* indices,
    #   void* out, void* lse, void* stream)
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # stream
    ]
    fn.restype = None
    return fn


def _bind_dsa_config(lib):
    """Bind vk_hip_dsa_config (per-shape tile selector). Returns the (bq,
    threads, block_I, inner_iter) tuple via out-pointers."""
    fn = getattr(lib, "vk_hip_dsa_config", None)
    if fn is None:
        raise RuntimeError("vk_hip_dsa_config not found in libvkernels_hip.so")
    # void vk_hip_dsa_config(int S_q, int H, int dim, int topk,
    #   int* bq, int* threads, int* block_I, int* inner_iter)
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    fn.restype = None
    return fn


def _supports_current_device() -> bool:
    """True only on gfx942 (MI300A), matching the kernel's build target and
    the VkernelFusedExperts._supports_current_device guard pattern."""
    try:
        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        gcn = getattr(props, "gcnArchName", "") or ""
        return "gfx942" in gcn
    except Exception:
        return False


def tilelang_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    return_lse: bool = False,
):
    """gfx942 HIP DSA sparse-MLA forward — drop-in for sglang's tilelang path.

    See the module docstring for the full contract. Mirrors the public
    signature of ``sglang.kernels.ops.attention.dsa.tilelang_sparse_fwd``
    so the meta/diag/glm53 dispatcher (patch_dsa_vk) can rebind that name
    to this function.
    """
    # --- shape/dtype contract (match the tilelang wrapper's asserts) ---
    assert q.dim() == 3 and kv.dim() == 3 and indices.dim() == 3, (
        f"q/kv/indices must be 3-D (got {q.dim()}/{kv.dim()}/{indices.dim()}); "
        "sglang _forward_tilelang passes q_all/kv/page_table_1.unsqueeze(1) "
        "without a leading batch dim."
    )
    num_heads = q.shape[1]
    dim = q.shape[2]
    tail_dim = dim - d_v
    topk = indices.shape[-1]
    assert topk % 64 == 0, "topk must be padded to a multiple of 64"
    assert tail_dim >= 0 and (dim - tail_dim) > 0, (
        f"need dim>0 tail_dim>=0 d_v>0 (dim={dim} tail_dim={tail_dim} d_v={d_v})"
    )
    kv_group = kv.shape[1]
    assert kv_group == 1, f"DSA forward is kv_group==1 (got {kv_group})"
    S_q = int(q.shape[0])
    S_kv = int(kv.shape[0])
    assert q.shape[0] == indices.shape[0], (
        f"S_q mismatch: q={q.shape[0]} indices={indices.shape[0]}"
    )

    # --- device guard (only run the HIP kernel on gfx942) ---
    if not _supports_current_device():
        raise RuntimeError(
            "vkernels_dsa.tilelang_sparse_fwd requires gfx942 (MI300A); "
            f"current device is not gfx942 (gcn={getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '?')}). "
            "patch_dsa_vk (meta/diag/glm53) should only patch on gfx942."
        )

    # --- pick the kernel's group-tiling (mirrors vk_dsa_config / dsa_config_for) ---
    lib = _get_lib()
    cfg = _bind_dsa_config(lib)
    bq = ctypes.c_int(); threads = ctypes.c_int()
    block_I = ctypes.c_int(); inner_iter = ctypes.c_int()
    cfg(S_q, num_heads, dim, topk,
        ctypes.byref(bq), ctypes.byref(threads),
        ctypes.byref(block_I), ctypes.byref(inner_iter))
    block_I = block_I.value; inner_iter = inner_iter.value
    assert block_I > 0 and inner_iter > 0, (
        f"vk_hip_dsa_config returned non-positive tiling "
        f"(block_I={block_I} inner_iter={inner_iter})"
    )
    assert topk % (block_I * inner_iter) == 0, (
        f"topk ({topk}) must be divisible by block_I*inner_iter "
        f"({block_I}*{inner_iter}={block_I * inner_iter})"
    )

    # --- fold log2(e) into sm_scale (tilelang does this internally; vkernels
    #     expects sm_scale already folded so 2^score is the natural-exp weight) ---
    sm_scale_f = float(sm_scale) * _LOG2E

    # --- contiguity (the C kernel reads raw pointers in row-major order;
    #     3-D contiguous == the 4-D[1,...] layout the tilelang kernel uses) ---
    if not q.is_contiguous():
        q = q.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    # --- output buffers (match tilelang's sparse_mla_fwd_decode_combine shape) ---
    out = torch.empty((1, S_q, num_heads, d_v), dtype=torch.bfloat16, device=q.device)
    # lse: (1, S_q, H) fp32. Allocate even when return_lse=False — the kernel
    # is passed a live pointer (guarded by return_lse internally); a real
    # buffer avoids any NULL-deref risk regardless of the guard.
    lse = torch.empty((1, S_q, num_heads), dtype=torch.float32, device=q.device)

    # Launch on the current HIP stream (the capture stream during
    # cuda-graph capture) so no host sync is needed — same-stream kernels
    # are ordered, and the graph replays the launch in-stream.
    fn = _bind_dsa_fn(lib)
    fn(
        ctypes.c_int(S_q), ctypes.c_int(S_kv), ctypes.c_int(num_heads),
        ctypes.c_int(dim), ctypes.c_int(tail_dim), ctypes.c_int(topk),
        ctypes.c_int(kv_group), ctypes.c_int(block_I), ctypes.c_int(inner_iter),
        ctypes.c_float(sm_scale_f), ctypes.c_int(1 if return_lse else 0),
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(kv.data_ptr()),
        ctypes.c_void_p(indices.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(lse.data_ptr()),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )

    if return_lse:
        return out, lse
    return out


# ---------------------------------------------------------------------------
# DSA paged-MQA gated top-k logits (vkernels issue #51, the kpool>1 indexer).
# Drop-in for sglang's ``tilelang_fp8_paged_mqa_logits``
# (sglang/kernels/ops/attention/dsa/tilelang_kernel.py:1519) — the FIRST stage
# of the DSA decode top-k: score each query against its paged KV tiles and
# return the ``(batch_size, max_seq_len)`` fp32 logits that
# ``topk_from_pooled_history_logits`` selects over. The tilelang kernel
# JIT-compiles on gfx942 but never returns for num_heads in {32, 64}; this
# routes through ``vk_hip_dsa_topk_logits`` (feat/issue-51-dsa-topk).
# ---------------------------------------------------------------------------
def _bind_dsa_topk_fn(lib):
    """Bind vk_hip_dsa_topk_logits (vkernels feat/issue-51-dsa-topk)."""
    fn = getattr(lib, "vk_hip_dsa_topk_logits", None)
    if fn is None:
        raise RuntimeError(
            "vk_hip_dsa_topk_logits not found in libvkernels_hip.so — "
            "rebuild vkernels at feat/issue-51-dsa-topk with VKERNELS_HAS_HIP=ON"
        )
    # void vk_hip_dsa_topk_logits(int batch_size, int num_heads, int head_dim,
    #   int block, int max_table_len, int max_seq_len, int split_kv,
    #   const void* q_fp8, const void* kvcache_u8, const void* weight,
    #   const void* seq_lens, const void* page_table, void* out, void* stream)
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,  # stream
    ]
    fn.restype = None
    return fn


def tilelang_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
    clean_logits: bool = True,
):
    """gfx942 HIP DSA paged-MQA gated top-k logits — drop-in for sglang's
    ``tilelang_fp8_paged_mqa_logits`` (tilelang_kernel.py:1519).

    Contract (mirror of the tilelang wrapper's asserts, L1530-1539):
      q_fp8        (bs, 1, nh, hd)    fp8 e4m3fnuz  -> viewed (bs, nh, hd)
      kvcache_fp8  (num_blocks, block, 1, hd+4) fp8 -> viewed (-1, block*(hd+4)) u8
      weight       (bs, nh)            fp32  (the indexer's gated head weight)
      seq_lens     (bs,)               int   (POOLED valid KV count per batch)
      page_table   (bs, max_table_len) int   (pool_block_tables)
      deep_gemm_metadata  Any  (UNUSED — the wrapper does `_ = deep_gemm_metadata`)
      max_seq_len  int   (= max_table_len * block_size; passed by the caller)
      clean_logits bool  (asserted False — the wrapper asserts the same)

    Output: (bs, max_seq_len) fp32. The tilelang wrapper allocates with
    ``new_empty`` (UNINITIALISED); we use ``torch.zeros`` (strictly safer —
    tokens t >= seq_lens[b] are masked by sglang's
    ``topk_from_pooled_history_logits`` via group_lengths/topk_offsets before
    the top-k, so zero/garbage are all excluded). split_kv (perf only,
    grouping-independent) = max(1, min(max_seq_len//block_size, NUM_CU//bs))
    with NUM_CU = 256, matching the wrapper L1552-1553. See the module
    docstring for the two-sync stream-handling rationale.
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    # Mirror the tilelang wrapper's asserts (tilelang_kernel.py L1532-1539).
    assert head_dim == 128, f"head_dim must be 128 (got {head_dim})"
    assert block_size == 64, f"block_size must be 64 (got {block_size})"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False, (
        "vk_hip_dsa_topk_logits writes only t < seq_lens[b]; pass "
        "clean_logits=False (the tilelang wrapper asserts the same)."
    )

    # Device guard (only on gfx942; patch_dsa_vk only patches there).
    if not _supports_current_device():
        raise RuntimeError(
            "vkernels_dsa.tilelang_fp8_paged_mqa_logits requires gfx942 (MI300A); "
            f"current device is not gfx942 (gcn="
            f"{getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '?')})."
        )

    max_table_len = int(page_table.shape[1])
    # The C kernel reads raw int32. Cast defensively (no-op when already int32);
    # a different dtype would otherwise corrupt the raw int32 reads.
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    if page_table.dtype != torch.int32:
        page_table = page_table.to(torch.int32)

    # The SAME views the tilelang wrapper makes (L1563-1564): q_fp8
    # (bs,1,nh,hd) -> (bs, nh, hd); kvcache_fp8 (num_blocks,block,1,hd+4) ->
    # (-1, block*(hd+4)) raw uint8 (B*D fp8 keys then B fp32 per-token scales).
    q_fp8 = q_fp8.view(batch_size, num_heads, head_dim)
    kvcache_u8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))

    # ZEROED output (strictly safer than the wrapper's new_empty; tokens
    # >= seq_lens[b] are excluded from the top-k by sglang's masking).
    logits = torch.zeros(
        (batch_size, max_seq_len), dtype=torch.float32, device=page_table.device
    )

    # split_kv (perf only): max(1, min(max_seq_len//block_size, NUM_CU//bs)) —
    # mirrors the tilelang wrapper L1552-1553 exactly.
    NUM_CU = 256
    split_kv = max(1, min(max_seq_len // block_size, NUM_CU // batch_size))

    # contiguity (raw-pointer reads require row-major; the views above are
    # contiguous iff the underlying tensors are).
    if not q_fp8.is_contiguous():
        q_fp8 = q_fp8.contiguous()
    if not kvcache_u8.is_contiguous():
        kvcache_u8 = kvcache_u8.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()

    # Launch on the current HIP stream (the capture stream during
    # cuda-graph capture) so no host sync is needed — same-stream kernels
    # are ordered, and the graph replays the launch in-stream.
    lib = _get_lib()
    fn = _bind_dsa_topk_fn(lib)
    fn(
        ctypes.c_int(batch_size), ctypes.c_int(num_heads),
        ctypes.c_int(head_dim), ctypes.c_int(block_size),
        ctypes.c_int(max_table_len), ctypes.c_int(max_seq_len),
        ctypes.c_int(split_kv),
        ctypes.c_void_p(q_fp8.data_ptr()), ctypes.c_void_p(kvcache_u8.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(seq_lens.data_ptr()),
        ctypes.c_void_p(page_table.data_ptr()), ctypes.c_void_p(logits.data_ptr()),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )

    return logits
