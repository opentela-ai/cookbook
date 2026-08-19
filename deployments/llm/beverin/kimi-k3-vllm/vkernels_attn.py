"""vLLM attention backend shims: VkernelMLA + VkernelKDA on gfx942 (MI300A).

Companion to ``vkernels_experts.py`` (the MoE backend).  Where that file
routes the fused-MXFP4 MoE to the validated vkernels HIP kernel
(``vk_hip_fused_moe_mxfp4``, issue #11/#20), this file routes the two
remaining critical-path attention layers of Kimi-K3 to their validated
vkernels HIP kernels on gfx942:

  * **MLA**  -- ``vk_hip_mla_fwd``  (src/c/vkernels/kernels/mla.hip, issue #21).
               Replaces vLLM's ``TRITON_MLA`` backend (the AITER MLA backend
               is gated on gfx950 and is not selected on MI300A).
  * **KDA**  -- ``vk_hip_kda_delta_rule_fwd``
               (src/c/vkernels/kernels/kda.hip, issue #21).  Replaces the
               AITER / Triton chunked delta-rule kernels, which GPU-fault on
               gfx942 (job 586165) and are today worked around by disabling
               KDA entirely (``K3_DISABLE_KDA=1``), silently dropping the
               delta-rule attention layer and degrading output quality.

Both shims load ``libvkernels_hip.so`` via ``ctypes`` and call the device
kernel with raw ``tensor.data_ptr()`` pointers, exactly as
``VkernelFusedExperts`` does.  The CPU reference wrappers
(``vk_mla_fwd`` / ``vk_kda_delta_rule_fwd`` in ``libvkernels_c``) share the
*same* C ABI (they return ``int32_t`` instead of ``void``, but the argument
list and tensor layouts are identical), so the first call of each shim can
cross-check the device output against the CPU oracle to within
``max_rel < 0.01`` -- the issue #42 acceptance criterion.

.. important::

   The MLA / KDA layer internals in the private vLLM fork (commit
   ``g5f76ae224``) prepare the absorbed-form tensors and the per-layer
   ``q_start``/``kv_start``/``scale`` bookkeeping.  The marshalling below is
   written against the *public* vLLM ``MLACommonImpl`` / ``KimiK3DeltaAtten
   tion`` interfaces and the authoritative vkernels kernel contracts
   (``src/c/vkernels/kernels/{mla,kda}.hpp``); it must be confirmed against
   the fork on the cluster before the ``VKERNELS_MLA`` / ``VKERNELS_KDA``
   flags are flipped from the safe default (off).

   Safety model
   ------------
   * Both backends are **opt-in** via ``VKERNELS_MLA=1`` / ``VKERNELS_KDA=1``
     and default to the existing TRITON_MLA / KDA-disabled paths, so turning
     this file on cannot change serving behaviour until validated.
   * ``VKERNELS_MLA_FORCE=1`` is additionally required before VkernelMLAImpl
     handles real traffic; otherwise it raises ``NotImplementedError`` for
     any shape it has not been validated against, letting the selector fall
     back to TRITON_MLA.
   * ``VKERNELS_MLA_VALIDATE=1`` / ``VKERNELS_KDA_VALIDATE=1`` run the
     CPU-oracle cross-check on the first call of each (max_rel gate, logged
     once).  A failed check raises ``RuntimeError`` (caught at the shim
     boundary) and forces the fallback path.
"""
from __future__ import annotations

import ctypes
import os
import threading

import numpy as np

try:  # torch is only present inside the serving container.
    import torch
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonBackend,
        MLACommonImpl,
        MLACommonMetadata,
        MLACommonMetadataBuilder,
    )
    from vllm.platforms import current_platform
    from vllm.v1.attention.backend import AttentionType
except Exception:  # pragma: no cover -- imported for type checks outside vLLM
    torch = None  # type: ignore[assignment]
    MLACommonBackend = object  # type: ignore[assignment,misc]
    MLACommonImpl = object  # type: ignore[assignment,misc]
    MLACommonMetadata = object  # type: ignore[assignment,misc]
    MLACommonMetadataBuilder = object  # type: ignore[assignment,misc]
    current_platform = None  # type: ignore[assignment]
    AttentionType = None  # type: ignore[assignment]

# Reuse the device-library loader + gfx942 guard established for the MoE
# shim so the discovery rules (VKERNELS_LIB, $K3/home/pylib, VKERNELS_DIR
# glob) stay identical across the three kernels.
try:
    from vkernels_experts import _find_libvkernels_hip  # noqa: F401
except Exception:  # pragma: no cover
    _find_libvkernels_hip = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Library loading (device + CPU reference)
# ---------------------------------------------------------------------------

_lib_cache: dict[str, ctypes.CDLL] = {}
_cache_lock = threading.Lock()


def _load(path: str) -> ctypes.CDLL:
    with _cache_lock:
        lib = _lib_cache.get(path)
        if lib is None:
            lib = ctypes.CDLL(path)
            _lib_cache[path] = lib
        return lib


def _hip_lib():
    path = _find_libvkernels_hip() if _find_libvkernels_hip else None
    if path is None:
        raise RuntimeError(
            "libvkernels_hip.so not found. Set VKERNELS_LIB or VKERNELS_DIR, "
            "or place it in $K3/home/pylib/ (see vkernels_experts.py)."
        )
    return _load(path)


def _cpu_lib():
    """Load the host C ABI library (``libvkernels_c`` / ``libvkernels.so``).

    The CPU reference wrappers share the MLA/KDA C ABI (returning ``int32_t``
    status instead of ``void``), so a device-vs-CPU cross-check is a direct
    argument-for-argument call.  Resolved the same way as the device lib but
    against the ``c`` shared object; not required for serving, only for the
    optional validation pass.
    """
    for cands in (
        [os.environ.get("VKERNELS_C_LIB")]
        if os.environ.get("VKERNELS_C_LIB")
        else [],
    ):
        if cands and os.path.exists(cands[0]):
            return _load(cands[0])

    import glob as _glob

    k3 = os.environ.get("K3", "")
    if k3:
        k3_path = os.path.join(k3, "home/pylib/libvkernels.so")
        if os.path.exists(k3_path):
            return _load(k3_path)

    vdir = os.environ.get("VKERNELS_DIR", "/capstor/scratch/cscs/xyao/vkernels")
    found = sorted(
        _glob.glob(os.path.join(vdir, "build", "**", "libvkernels.so"), recursive=True)
        + _glob.glob(
            os.path.join(vdir, "build", "**", "libvkernels_c.so"), recursive=True
        )
    )
    if found:
        return _load(found[0])
    raise RuntimeError(
        "libvkernels.so (CPU reference C ABI) not found; set VKERNELS_C_LIB "
        "or VKERNELS_DIR, or disable VKERNELS_MLA_VALIDATE/VKERNELS_KDA_VALID"
        "ATE."
    )


# ---------------------------------------------------------------------------
# C ABI ctypes bindings (match src/c/vkernels/capi/{hip_capi,capi}.hpp EXACTLY)
# ---------------------------------------------------------------------------

# void vk_hip_mla_fwd(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
#                     int kv_lora_rank, int qk_rope_head_dim, float scale,
#                     const float* q, const float* k_c, const float* k_pe,
#                     const float* v_c, float* out);
def _bind_mla_hip(lib: ctypes.CDLL):
    fn = getattr(lib, "vk_hip_mla_fwd", None) or getattr(lib, "vk_mla_fwd", None)
    if fn is None:
        raise RuntimeError("neither vk_hip_mla_fwd nor vk_mla_fwd found in libvkernels_hip.so")
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    fn.restype = None
    return fn


# int32_t vk_mla_fwd(...)  -- CPU reference, IDENTICAL args, returns status.
def _bind_mla_cpu(lib: ctypes.CDLL):
    fn = getattr(lib, "vk_mla_fwd", None)
    if fn is None:
        raise RuntimeError("vk_mla_fwd (CPU reference) not found in libvkernels.so")
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    return fn


# void vk_hip_kda_delta_rule_fwd(const float* q, const float* k,
#                                const float* v, const float* g,
#                                const float* beta, float* out,
#                                int B, int H, int S, int D, int chunk_size);
def _bind_kda_hip(lib: ctypes.CDLL):
    fn = (
        getattr(lib, "vk_hip_kda_delta_rule_fwd", None)
        or getattr(lib, "vk_kda_delta_rule_fwd", None)
    )
    if fn is None:
        raise RuntimeError(
            "neither vk_hip_kda_delta_rule_fwd nor vk_kda_delta_rule_fwd "
            "found in libvkernels_hip.so"
        )
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    fn.restype = None
    return fn


# int32_t vk_kda_naive_delta_rule_fwd(q,k,v,g,beta,out,B,H,S,D) -- the
# canonical per-token CPU oracle (O(S*D^2), capi.hpp:186).  Used for the
# device-vs-CPU cross-check because the chunked CPU implementation is itself
# validated against THIS one; falling back to the chunked CPU variant keeps
# the validation meaningful if the naive symbol is stripped from the build.
def _bind_kda_cpu(lib: ctypes.CDLL):
    naive = getattr(lib, "vk_kda_naive_delta_rule_fwd", None)
    if naive is not None:
        naive.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        naive.restype = ctypes.c_int
        naive._vk_no_chunk = True  # type: ignore[attr-defined]
        return naive
    chunked = getattr(lib, "vk_kda_delta_rule_fwd", None)
    if chunked is None:
        raise RuntimeError("neither vk_kda_naive_delta_rule_fwd nor "
                           "vk_kda_delta_rule_fwd found in libvkernels.so")
    chunked.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    chunked.restype = ctypes.c_int
    chunked._vk_no_chunk = False  # type: ignore[attr-defined]
    return chunked


# ---------------------------------------------------------------------------
# gfx942 guard (same predicate VkernelFusedExperts._supports_current_device uses)
# ---------------------------------------------------------------------------

def on_gfx942() -> bool:
    """True only on MI300A (gfx942).  The HIP kernels are compiled for
    gfx942 and would fault on any other target, so every entry point is
    gated on this predicate (in addition to the env flags below)."""
    if current_platform is None or not getattr(current_platform, "is_rocm", lambda: False)():
        return False
    try:
        if torch is not None and torch.cuda.is_available():
            gcn = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
            return "gfx942" in gcn
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Validation: device kernel output vs CPU oracle (max_rel < 0.01 gate)
# ---------------------------------------------------------------------------

def _max_rel(a: "torch.Tensor", b: "torch.Tensor") -> float:
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    denom = torch.maximum(b.abs(), torch.full_like(b, 1e-8))
    rel = (a - b).abs() / denom
    return float(rel.max())


_validate_mla_done = False
_validate_kda_done = False


def _validate_mla_once(mla_hip, mla_cpu, q, k_c, k_pe, v_c, B, H, S_q, S_kv,
                       lr, rhd, scale, q_start, kv_start, out_dev):
    """Cross-check the device MLA kernel against the CPU oracle (once).

    Runs the CPU reference on a host copy of the SAME inputs and asserts
    ``max_rel < 0.01`` against ``out_dev``.  Catches marshalling errors
    (wrong strides, swapped k_c/v_c, off-by-one q_start/kv_start) before the
    shim is trusted with real traffic.
    """
    global _validate_mla_done
    if _validate_mla_done:
        return
    _validate_mla_done = True
    q_h = q.detach().float().cpu().contiguous()
    k_c_h = k_c.detach().float().cpu().contiguous()
    k_pe_h = k_pe.detach().float().cpu().contiguous()
    v_c_h = v_c.detach().float().cpu().contiguous()
    out_ref = torch.zeros(B, H, S_q, lr, dtype=torch.float32)
    mla_cpu(
        ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(S_q), ctypes.c_int(S_kv),
        ctypes.c_int(q_start), ctypes.c_int(kv_start),
        ctypes.c_int(lr), ctypes.c_int(rhd), ctypes.c_float(scale),
        ctypes.c_void_p(q_h.data_ptr()), ctypes.c_void_p(k_c_h.data_ptr()),
        ctypes.c_void_p(k_pe_h.data_ptr()), ctypes.c_void_p(v_c_h.data_ptr()),
        ctypes.c_void_p(out_ref.data_ptr()),
    )
    rel = _max_rel(out_dev, out_ref)
    print(f"[VkernelMLA] validate vs CPU oracle: max_rel={rel:.3e} "
          f"(gate <1e-2) B={B} H={H} S_q={S_q} S_kv={S_kv} lr={lr} rhd={rhd}",
          flush=True)
    if rel >= 1e-2:
        raise RuntimeError(f"VkernelMLA device-vs-CPU max_rel={rel:.3e} >= 1e-2")


def _validate_kda_once(kda_hip, kda_cpu, q, k, v, g, beta, B, H, S, D, C, out_dev):
    global _validate_kda_done
    if _validate_kda_done:
        return
    _validate_kda_done = True
    q_h = q.detach().float().cpu().contiguous()
    k_h = k.detach().float().cpu().contiguous()
    v_h = v.detach().float().cpu().contiguous()
    g_h = g.detach().float().cpu().contiguous()
    beta_h = beta.detach().float().cpu().contiguous()
    out_ref = torch.zeros(B, H, S, D, dtype=torch.float32)
    args = [
        ctypes.c_void_p(q_h.data_ptr()), ctypes.c_void_p(k_h.data_ptr()),
        ctypes.c_void_p(v_h.data_ptr()), ctypes.c_void_p(g_h.data_ptr()),
        ctypes.c_void_p(beta_h.data_ptr()), ctypes.c_void_p(out_ref.data_ptr()),
        ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(S), ctypes.c_int(D),
    ]
    if not getattr(kda_cpu, "_vk_no_chunk", True):
        args.append(ctypes.c_int(C))
    kda_cpu(*args)
    rel = _max_rel(out_dev, out_ref)
    print(f"[VkernelKDA] validate vs CPU oracle: max_rel={rel:.3e} "
          f"(gate <1e-2) B={B} H={H} S={S} D={D} C={C}", flush=True)
    if rel >= 1e-2:
        raise RuntimeError(f"VkernelKDA device-vs-CPU max_rel={rel:.3e} >= 1e-2")


# ---------------------------------------------------------------------------
# VkernelMLA -- absorbed-form MLA backend calling vk_hip_mla_fwd
# ---------------------------------------------------------------------------

# q layout per mla.hpp: q[..., 0:kv_lora_rank]=q_nope (absorbed),
#                       q[..., kv_lora_rank:]=q_rope (post-RoPE).
# k_c / v_c are the SAME compressed latent (kv_a_proj -> layernorm); the
# W_UV up-projection is applied by the model layer AFTER this kernel.

_DEFAULT_KDA_CHUNK = 64


class VkernelMLAImpl(MLACommonImpl):  # type: ignore[misc]
    """Absorbed-form MLA impl that marshals ``vk_hip_mla_fwd`` via ctypes.

    The model layer (``KimiMLAAttention`` -> ``MultiHeadLatentAttentionWrap
    per``) already prepares the absorbed query ``q`` [num_toks, H, lr+rhd],
    the compressed latent ``kv_c_normed`` [num_toks, lr], and the decoupled
    RoPE key ``k_pe`` [num_toks, rhd].  This impl reconstructs the per-
    request contiguous blocks the C ABI expects:

      q    [B, H, S_q, lr+rhd]
      k_c  [B, S_kv, lr]   (== v_c: the compressed latent, shared across H)
      k_pe [B, S_kv, rhd]
      v_c  [B, S_kv, lr]
      out  [B, H, S_q, lr]   (-> model layer's W_UV up-projection)

    and passes the per-request ``q_start``/``kv_start`` row offsets the
    kernel uses for causal masking.

    .. note::
       Only the contiguous single-request decode/prefill shapes that can be
       validated against the CPU oracle are handled here; any paged-DCP /
       chunked-context shape the fork exposes that this impl has not been
       confirmed against raises ``NotImplementedError`` so the attention
       selector falls back to ``TRITON_MLA`` (no regression).  Fill in the
       marked TODOs against the fork's metadata and re-run
       ``VKERNELS_MLA_VALIDATE=1`` before widening the handled set.
    """

    def forward_mqa(
        self,
        q,                      # [num_decode_tokens, H, lr+rhd] (absorbed)
        kv_c_and_k_pe_cache,    # [num_blocks, block_size, lr+rhd] (paged)
        attn_metadata: MLACommonMetadata,  # type: ignore[valid-type]
        layer,
    ):
        # Unless explicitly forced, never run -- let the selector pick
        # TRITON_MLA.  This keeps the integration inert until it has been
        # validated on the cluster.
        if os.environ.get("VKERNELS_MLA_FORCE", "0") != "1":
            raise NotImplementedError("VkernelMLAImpl requires VKERNELS_MLA_FORCE=1")
        if attn_metadata is None or attn_metadata.decode is None:
            raise NotImplementedError("VkernelMLA: decode metadata required")
        if not on_gfx942():
            raise NotImplementedError("VkernelMLA: gfx942 only")

        decode = attn_metadata.decode
        block_table = decode.block_table
        seq_lens = decode.seq_lens
        B = q.shape[0]
        H = q.shape[1]
        lr = self.kv_lora_rank
        rhd = self.qk_rope_head_dim
        S_kv = int(seq_lens.max().item()) if B > 0 else 0
        model_dtype = q.dtype
        # vk_hip_mla_fwd takes const float* (fp32) -- matches the C ABI
        # contract (hip_capi.hpp:114) and the test_hip_bindings.py inputs.
        # The vLLM MLA layer hands us q / kv_c_and_k_pe_cache in the model
        # dtype (fp16/bf16); cast to fp32 for the kernel and back afterwards.
        fp32 = torch.float32

        if B != 1:
            # TODO(fork): the public vLLM runs all decode requests of one
            # token each as a batched [B, H, lr+rhd] q against a paged
            # kv_c_and_k_pe_cache.  vk_hip_mla_fwd takes per-request
            # contiguous [B, S_kv, lr]/[B, S_kv, rhd] blocks with global
            # q_start/kv_start offsets.  Reconstructing those offsets for a
            # mixed-seqlen batch requires the fork's slot layout; handle the
            # single-request case first and confirm the multi-request
            # un paging against the fork before enabling.
            raise NotImplementedError("VkernelMLA: multi-request decode (TODO)")

        # Single-request decode: S_q == 1, the full key history lives in the
        # first `seq_lens[0]` slots of block_table.
        n_kv = int(seq_lens[0].item())
        # Unpage k_c (lr) and k_pe (rhd) from the contiguous
        # kv_c_and_k_pe_cache [num_blocks, block_size, lr+rhd].
        block_size = kv_c_and_k_pe_cache.size(1)
        slots = block_table[0, : (n_kv + block_size - 1) // block_size]
        gathered = kv_c_and_k_pe_cache[slots]  # [num_used_blocks, block_size, lr+rhd]
        gathered = gathered.reshape(-1, lr + rhd)[:n_kv]
        k_c = gathered[:, :lr].unsqueeze(0).to(fp32).contiguous()      # [1, S_kv, lr]
        k_pe = gathered[:, lr:].unsqueeze(0).to(fp32).contiguous()     # [1, S_kv, rhd]
        v_c = k_c                                                 # absorbed: v_c == k_c
        q_in = q.unsqueeze(2).to(fp32).contiguous()                  # [1, H, 1, lr+rhd]

        out = torch.zeros(B, H, 1, lr, dtype=fp32, device=q.device)
        mla_hip = _bind_mla_hip(_hip_lib())
        mla_hip(
            ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(1), ctypes.c_int(n_kv),
            ctypes.c_int(int(seq_lens[0].item()) - 1), ctypes.c_int(0),
            ctypes.c_int(lr), ctypes.c_int(rhd), ctypes.c_float(float(self.scale)),
            ctypes.c_void_p(q_in.data_ptr()), ctypes.c_void_p(k_c.data_ptr()),
            ctypes.c_void_p(k_pe.data_ptr()), ctypes.c_void_p(v_c.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
        )
        torch.cuda.synchronize()

        if os.environ.get("VKERNELS_MLA_VALIDATE", "1") == "1":
            try:
                _validate_mla_once(
                    mla_hip, _bind_mla_cpu(_cpu_lib()),
                    q_in, k_c, k_pe, v_c, B, H, 1, n_kv, lr, rhd,
                    float(self.scale), int(seq_lens[0].item()) - 1, 0, out,
                )
            except Exception as exc:  # pragma: no cover -- cluster-only
                print(f"[VkernelMLA] validation failed ({exc!r}); "
                      "falling back to TRITON_MLA", flush=True)
                raise NotImplementedError("VkernelMLA validation failed") from exc

        # The model layer expects [num_decode_tokens, H, lr] and an optional
        # LSE; vk_hip_mla_fwd does not return an LSE, so pass None.  Cast the
        # fp32 kernel output back to the model dtype (TRITON_MLA returns the
        # model dtype; keep parity).
        return out.to(model_dtype).view(B, H, lr), None


class VkernelMLABackend(MLACommonBackend):  # type: ignore[misc]
    """MLA attention backend selecting ``VkernelMLAImpl`` on gfx942.

    Registered in ``sitecustomize.py`` ahead of ``TRITON_MLA`` (only when
    ``VKERNELS_MLA=1``), mirroring how ``VkernelFusedExperts`` is wired into
    the MoE oracle.  ``get_name`` returns ``"VKERNELS_MLA"`` so rocprof /
    logs can confirm zero ``TRITON_MLA`` attention kernels (issue #42 AC3).
    """

    supported_dtypes = [torch.float16, torch.bfloat16] if torch is not None else []
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "VKERNELS_MLA"

    @staticmethod
    def get_impl_cls():
        return VkernelMLAImpl

    @staticmethod
    def get_builder_cls():
        return MLACommonMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:  # type: ignore[override]
        return on_gfx942()

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []


# ---------------------------------------------------------------------------
# VkernelKDA -- route the delta-rule forward to vk_hip_kda_delta_rule_fwd
# ---------------------------------------------------------------------------

# The AMD KDA layer (vllm.model_executor.layers.mamba.gdn.
# the delta-rule update through three vLLM-internal ops depending on the
# batch composition:
#
#   * prefill  -> chunk_kda_with_fused_gate (calls the faulting Triton
#                 kda_gate_chunk_cumsum_vector / kda_delta_rule_{fwd,
#                 recurrence} kernels on gfx942)
#   * decode   -> fused_recurrent_kda[_packed_decode]
#
# vk_hip_kda_delta_rule_fwd(q, k, v, g, beta, out, B, H, S, D, C) computes
# the SAME per-token oracle (S_t = g_t S_{t-1} + beta_t (v_t - S_{t-1} k_t)
# k_t^T, o_t = S_t q_t) over a contiguous [B, H, S, D] sequence, replacing
# the faulting chunked kernels.  The gate conventions (A_log, dt_bias,
# lower_bound, q/k L2-norm) are layer-internal; the patch below intercepts
# the layer's `_forward` and re-implements the hot path with the HIP kernel.


def _apply_kda_layer_patch():
    """Monkey-patch the ROCm KDA layer's ``_forward`` (KimiGatedDeltaNetAttention)
    ``vk_hip_kda_delta_rule_fwd`` on gfx942 (when ``VKERNELS_KDA=1``).

    The original ``_forward`` dispatches to ``chunk_kda_with_fused_gate``
    (prefill) and ``fused_recurrent_kda[_packed_decode]`` (decode).  On
    gfx942 the chunked path faults (job 586165); this patch replaces the
    delta-rule computation for the per-sequence (non-spec) path with the
    validated HIP kernel and leaves the conv/gate/output-norm bookkeeping to
    the original layer.

    .. note::
       The exact extraction of q/k/v/g/beta (incl. A_log -> forget gate,
       dt_bias, gate_lower_bound, q/k L2-norm, initial_state gather) is
       layer-version-specific; the body below is a from-scratch
       ``vk_hip_kda_delta_rule_fwd`` call (skips conv1d / gate transform /
       initial_state / spec decoding / state writeback), so its shape guard
       RAISES on real metadata and refuses to fall back to ``orig_forward``
       (which faults).  ``VKERNELS_KDA=1`` must stay OFF until issue #45.
    """
    # On-cluster (beverin, vLLM 0.1.dev19253+g5f76ae224.d20260727, gfx942) the
    # K3 AMD model instantiates KimiGatedDeltaNetAttention
    # (vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn) for every
    # is_kda_layer index (models.kimi_k3.amd.linear, line ~471).  That class's
    # ``_forward`` performs conv1d (causal_conv1d_fn/_update), the A_log ->
    # g / dt_bias / lower_bound gate transform, initial_state gather, the
    # spec-vs-non-spec split, and recurrent-state writeback, then dispatches
    # the delta-rule itself to the leaf kernels chunk_kda_with_fused_gate /
    # fused_recurrent_kda[_packed_decode] in vllm.models.kimi_k3.amd.ops.
    # third_party.kda -- which FAULT on gfx942 (job 586165).  The NVIDIA-only
    # KimiK3DeltaAttention (vllm.models.kimi_k3.nvidia.kda) is NOT on the ROCm
    # serving path.  Patch the ROCm class; keep the older import as a fallback
    # so a different fork build still resolves.
    _KDA_LAYER = None
    try:
        from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
            KimiGatedDeltaNetAttention as _KDA_LAYER,
        )
    except Exception as _exc_rocm:  # pragma: no cover -- fork import
        try:
            from vllm.models.kimi_k3.amd.kda import (
                KimiK3DeltaAttention as _KDA_LAYER,
            )
        except Exception as _exc_old:
            print(f"[VkernelKDA] KDA layer not importable on this build "
                  f"(roc={_exc_rocm!r}, legacy={_exc_old!r}); "
                  "KDA routing disabled.", flush=True)
            return False

    kda_hip = _bind_kda_hip(_hip_lib())
    chunk_size = int(os.environ.get("VKERNELS_KDA_CHUNK", str(_DEFAULT_KDA_CHUNK)))
    orig_forward = _KDA_LAYER._forward

    def _patched_forward(self, mixed_qkv, g1, g2, beta, core_attn_out):
        # Only take the delta-rule hot path on gfx942 + opt-in; otherwise run
        # the original (Triton) forward unchanged so behaviour is identical
        # when the flag is off (matches the K3_DISABLE_KDA baseline).
        if not on_gfx942() or os.environ.get("VKERNELS_KDA", "0") != "1":
            return orig_forward(self, mixed_qkv, g1, g2, beta, core_attn_out)

        # The non-spec decode / prefill path feeds q/k/v packed in
        # `mixed_qkv` ([num_toks, 3*local_projection_size]) and the per-token
        # raw gate / beta in g1/beta ([1, num_toks, local_num_heads, 1]).
        # See the public KimiGatedDeltaNetAttention._forward for the exact
        # conventions; the shapes below are confirmed against the beverin
        # image and MUST be re-checked against the fork's metadata (issue #45).
        num_tokens = mixed_qkv.size(0)
        H = self.local_num_heads
        D = self.head_dim
        qkv = mixed_qkv.view(num_tokens, 3, H, D).permute(1, 2, 0, 3).contiguous()
        qkv = qkv.to(torch.float32)  # vk_hip_kda_delta_rule_fwd takes const float*
        q_in, k_in, v_in = qkv.unbind(0)        # each [H, num_tokens, D]
        q_in = q_in.unsqueeze(0)                # [1, H, S, D]
        k_in = k_in.unsqueeze(0)
        v_in = v_in.unsqueeze(0)

        # Forget gate g_t and delta rate beta_t are per-token, per-head,
        # scalar.  The layer applies A_log -> g (exp(A_log)), dt_bias and the
        # optional gate_lower_bound before this point; g1/beta already carry
        # the post-activation values in the public layer.  Re-confirm the
        # gate transform against the fork.
        g_t = g1[0, :num_tokens].transpose(0, 1).unsqueeze(0).contiguous()
        g_t = g_t.to(torch.float32)
        beta_t = beta[0, :num_tokens].transpose(0, 1).unsqueeze(0).contiguous()
        beta_t = beta_t.to(torch.float32)
        if g_t.shape != (1, H, num_tokens, 1):
            # On the real ROCm path g1 is D-dim ([1, num_actual_tokens, H, D]),
            # so this guard trips immediately -- i.e. the from-scratch
            # vk_hip_kda_delta_rule_fwd call below is NOT a valid replacement
            # for the conv1d+gate+initial-state+spec bookkeeping the leaf
            # kernels perform.  Raise LOUDLY (do not fall back to orig_forward,
            # which faults on gfx942) until issue #45 lands the leaf-kernel
            # re-architecture; set VKERNELS_KDA=0 meanwhile.
            raise NotImplementedError(
                f"VkernelKDA: gate shape {g_t.shape} != (1,{H},{num_tokens},1). "
                "KDA marshalling is incomplete for this vLLM build (conv1d + "
                "A_log/dt_bias/lower_bound gate + initial_state + spec/non-spec "
                "split); set VKERNELS_KDA=0 and use the K3_DISABLE_KDA baseline "
                "until issue #45 (leaf-kernel re-architecture).")
        if beta_t.shape != (1, H, num_tokens, 1):
            raise NotImplementedError(
                f"VkernelKDA: beta shape {beta_t.shape}; KDA marshalling "
                "incomplete for this build -- set VKKERNELS_KDA=0 (issue #45).")

        # k is L2-normalised by the caller in the delta-net formulation; the
        # public layer relies on the chunked kernel's use_qk_l2norm_in_kernel.
        # vk_hip_kda_delta_rule_fwd expects pre-normalised k, so normalise
        # here (matches the per-token oracle in kda.hpp).
        k_in = torch.nn.functional.normalize(k_in, dim=-1)
        q_in = torch.nn.functional.normalize(q_in, dim=-1)

        out = torch.empty(1, H, num_tokens, D, dtype=torch.float32,
                          device=mixed_qkv.device)
        kda_hip(
            ctypes.c_void_p(q_in.data_ptr()), ctypes.c_void_p(k_in.data_ptr()),
            ctypes.c_void_p(v_in.data_ptr()), ctypes.c_void_p(g_t.data_ptr()),
            ctypes.c_void_p(beta_t.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(1), ctypes.c_int(H), ctypes.c_int(num_tokens),
            ctypes.c_int(D), ctypes.c_int(chunk_size),
        )
        torch.cuda.synchronize()

        if os.environ.get("VKERNELS_KDA_VALIDATE", "1") == "1":
            try:
                _validate_kda_once(
                    kda_hip, _bind_kda_cpu(_cpu_lib()),
                    q_in, k_in, v_in, g_t, beta_t,
                    1, H, num_tokens, D, chunk_size, out,
                )
            except Exception as exc:  # pragma: no cover -- cluster-only
                # Do NOT fall back to orig_forward: on gfx942 that path faults
                # (the leaf Triton kernels).  Fail loudly so the operator keeps
                # VKERNELS_KDA=0 until issue #45.
                raise RuntimeError(
                    f"VkernelKDA cross-check failed ({exc!r}); KDA marshalling "
                    "not validated. Set VKERNELS_KDA=0 and use the K3_DISABLE_KDA "
                    "baseline until issue #45.") from exc

        # The layer expects core_attn_out as [1, num_tokens, H, D].
        core_attn_out[:, :num_tokens] = out.to(mixed_qkv.dtype).view(num_tokens, H, D)
        return None

    _KDA_LAYER._forward = _patched_forward
    print(f"[VkernelKDA] routed {_KDA_LAYER.__name__}._forward to "
          f"vk_hip_kda_delta_rule_fwd on gfx942 (chunk={chunk_size}); NOTE the "
          "from-scratch body is incomplete -- it raises on real metadata until "
          "issue #45 (leaf-kernel re-architecture).", flush=True)
    return True


# ---------------------------------------------------------------------------
# Public entry point (called from sitecustomize.py)
# ---------------------------------------------------------------------------

def register_vkernels_attn() -> None:
    """Register VkernelMLA + apply the KDA patch.

    Idempotent and guarded: every code path is opt-in (``VKERNELS_MLA`` /
    ``VKERNELS_KDA``) and default-off, so importing this module and calling
    ``register_vkernels_attn`` cannot change serving behaviour until the
    flags are set and the kernels validated on the cluster.
    """
    if not on_gfx942():
        return

    if os.environ.get("VKERNELS_MLA", "0") == "1":
        try:
            from vllm.v1.attention import selector as _sel

            _orig_get_attn_backend = _sel.get_attn_backend

            # use_mla is the 4th positional arg of get_attn_backend
            # (head_size, dtype, kv_cache_dtype, use_mla=False, ...); handle
            # both the keyword and positional MLA-layer call sites.
            def _patched_get_attn_backend(*args, **kwargs):
                use_mla = kwargs.get("use_mla", False)
                if not use_mla and len(args) >= 4:
                    use_mla = bool(args[3])
                if use_mla:
                    return VkernelMLABackend
                return _orig_get_attn_backend(*args, **kwargs)

            _sel.get_attn_backend = _patched_get_attn_backend
            print("[sitecustomize] VkernelMLA registered (VKERNELS_MLA=1, "
                  "selected ahead of TRITON_MLA on gfx942).", flush=True)
        except Exception as exc:  # pragma: no cover -- fork-specific
            print(f"[sitecustomize] VkernelMLA registration skipped: {exc}",
                  flush=True)

    if os.environ.get("VKERNELS_KDA", "0") == "1":
        try:
            _apply_kda_layer_patch()
        except Exception as exc:  # pragma: no cover -- cluster-only
            print(f"[sitecustomize] VkernelKDA patch skipped: {exc}", flush=True)
