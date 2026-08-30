"""vkernels HIP DSA sparse-MLA forward — GLM-5.3-Flash on MI300A (gfx942).

Drop-in replacement for sglang's ``tilelang_sparse_fwd``
(``sglang/kernels/ops/attention/dsa/tilelang_kernel.py``) that routes the
sparse-MLA forward through the gfx942 HIP kernel ``vk_hip_dsa_sparse_fwd``
added to the ``vkernels`` library by PR #52 (closes the tilelang/TVM
``FloorMod(_, 0)`` JIT abort documented in vkernels issue #51).

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

__all__ = ["tilelang_sparse_fwd"]

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
    #   void* out, void* lse)
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
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
    so ``sitecustomize.py`` can rebind that name to this function.
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
            "sitecustomize.py should only patch on gfx942."
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

    # --- stream-correct launch (see module docstring): synchronize before
    #     (input q/kv from sglang's compute stream are done) and after
    #     (stream-0 kernel output is done before sglang reads it) ---
    torch.cuda.synchronize()
    fn = _bind_dsa_fn(lib)
    fn(
        ctypes.c_int(S_q), ctypes.c_int(S_kv), ctypes.c_int(num_heads),
        ctypes.c_int(dim), ctypes.c_int(tail_dim), ctypes.c_int(topk),
        ctypes.c_int(kv_group), ctypes.c_int(block_I), ctypes.c_int(inner_iter),
        ctypes.c_float(sm_scale_f), ctypes.c_int(1 if return_lse else 0),
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(kv.data_ptr()),
        ctypes.c_void_p(indices.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(lse.data_ptr()),
    )
    torch.cuda.synchronize()

    if return_lse:
        return out, lse
    return out
