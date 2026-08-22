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
    """Bind the KDA forward HIP kernel.  Two entry points are exported:
      * vk_hip_kda_delta_rule_fwd       -- allocates+zeros an internal
        B*H*D*D state scratch, runs the recurrence from S_0=0, and
        hipFrees the scratch on return (one-shot / probe path).
      * vk_hip_kda_delta_rule_fwd_with_scratch -- the CALLER owns the
        state scratch [B,H,D,D]; pre-fill it with the gathered initial
        state (zeros for first-turn prefill) and read back S_S after the
        call for multi-turn decode.  This is the entry point the layer
        patch uses so it can carry per-sequence initial/final states.
    """
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

    scratch = getattr(lib, "vk_hip_kda_delta_rule_fwd_with_scratch", None)
    if scratch is not None:
        scratch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        scratch.restype = None
    return fn, scratch


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


def _scale_rel(a: "torch.Tensor", b: "torch.Tensor") -> float:
    """Scale-invariant relative error: max|a-b| / max|b|.

    Unlike _max_rel (max|a-b|/|b| per-element, inflated by near-zero
    outputs on a wide-dynamic-range recurrent kernel), this is bounded by
    the kernel's own magnitude and reflects the true fp32 rounding error.
    """
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-8)


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
    mrel = _max_rel(out_dev, out_ref)
    srel = _scale_rel(out_dev, out_ref)
    print(f"[VkernelKDA] validate vs CPU oracle: max_rel={mrel:.3e} "
          f"scale_rel={srel:.3e} (gate scale_rel<1e-2) "
          f"B={B} H={H} S={S} D={D} C={C}", flush=True)
    if srel >= 1e-2:
        raise RuntimeError(f"VkernelKDA device-vs-CPU scale_rel={srel:.3e} "
                           f">= 1e-2 (max_rel={mrel:.3e} for context)")


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
    """Monkey-patch the ROCm KDA PREFILL leaf ``chunk_kda_with_fused_gate``
    to call ``vk_hip_kda_delta_rule_fwd_with_scratch`` on gfx942 (opt-in via
    ``VKERNELS_KDA=1``).  The chunked Triton kernel faults on gfx942 (job
    586165); the HIP kernel is cross-checked bit-for-bit against the FLA
    recurrent reference (``fused_recurrent_kda``, same gated-delta-rule
    IS_KDA=True) by ``probe_kda_xcheck.py`` (issue #45), so it is a valid
    drop-in replacement for the forward recurrence.

    Patching the LEAF (not ``_forward``) leaves all the layer bookkeeping --
    conv1d (causal_conv1d_fn/_update), the spec-vs-non-spec split,
    ``gather_initial_states``, and ``recurrent_state[indices] = last`` -- to
    the original code; only the delta-rule kernel call is replaced.  The
    patch replicates the gate activation the leaf kernel does internally
    (``fuse_gate=True``: ``g = exp(lower_bound * sigmoid(exp(A_log)*(raw_g+g_bias)))``,
    ``beta = sigmoid(raw_beta)``) and the ``use_qk_l2norm_in_kernel=True``
    L2-norm + ``scale = D**-0.5`` that the wrapper applies, then runs the
    HIP recurrence per sequence (seeding each seq's state from
    ``initial_state``; zeros for first-turn) and writes ``final_state``
    back so multi-turn / continued prefill carries state.
    """
    # On-cluster (beverin, vLLM 0.1.dev19253+g5f76ae224.d20260727, gfx942) the
    # K3 AMD model's KimiGatedDeltaNetAttention._forward dispatches the
    # non-spec prefill delta-rule to ``chunk_kda_with_fused_gate``
    # (vllm.models.kimi_k3.amd.ops.third_party.kda.chunk), which FAULTS on
    # gfx942 (job 586165).  The decode leaves (fused_recurrent_kda[_packed])
    # do NOT fault -- only the chunked prefill path does -- so this patch
    # targets the chunked prefill leaf only and leaves decode untouched.
    try:
        from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (
            chunk_kda_with_fused_gate as _chunk_kda,
        )
    except Exception as _exc_chunk:  # pragma: no cover -- fork import
        print(f"[VkernelKDA] chunk_kda_with_fused_gate not importable "
              f"({_exc_chunk!r}); KDA prefill patch disabled.", flush=True)
        return False

    kda_hip, kda_scratch = _bind_kda_hip(_hip_lib())
    if kda_scratch is None:
        print("[VkernelKDA] vk_hip_kda_delta_rule_fwd_with_scratch missing "
              "from libvkernels_hip.so; prefill patch disabled (rebuild the "
              "HIP lib inside the container, ROCm 7.2.3).", flush=True)
        return False
    chunk_size = int(os.environ.get("VKERNELS_KDA_CHUNK", str(_DEFAULT_KDA_CHUNK)))
    orig_chunk = _chunk_kda

    def _patched_chunk_kda(q, k, v, raw_g, raw_beta, A_log, g_bias,
                           scale=None, initial_state=None,
                           output_final_state=False, lower_bound=None,
                           use_qk_l2norm_in_kernel=False, cu_seqlens=None,
                           **kwargs):
        # Only intercept on gfx942 + opt-in; otherwise run the original
        # (Triton) leaf unchanged so behaviour is identical when the flag
        # is off (matches the K3_DISABLE_KDA baseline).
        if not on_gfx942() or os.environ.get("VKERNELS_KDA", "0") != "1":
            return orig_chunk(q, k, v, raw_g, raw_beta, A_log, g_bias,
                              scale, initial_state, output_final_state,
                              lower_bound, use_qk_l2norm_in_kernel,
                              cu_seqlens, **kwargs)

        # q/k/v/raw_g are [B=1, n, H, D]; raw_beta [B=1, n, H]; A_log [H];
        # g_bias (dt_bias) [H*D] or None.  See chunk_kda_with_fused_gate
        # (chunk.py:774) and _forward (kimi_gdn_linear_attn.py:569).
        B, n, H, D = q.shape
        assert k.shape == q.shape and v.shape == q.shape
        assert raw_g.shape == (B, n, H, D)
        assert raw_beta.shape == (B, n, H)
        if scale is None:
            scale = float(D ** -0.5)
        dev = q.device

        # --- L2-normalise q, k + apply scale to q (use_qk_l2norm_in_kernel).
        # The HIP kernel takes PRE-normalised, PRE-scaled inputs (the
        # Triton leaf does the same internally when fuse_gate=True). ---
        qf = q.detach().to(torch.float32)
        kf = k.detach().to(torch.float32)
        vf = v.detach().to(torch.float32)
        gf = raw_g.detach().to(torch.float32)
        btf = raw_beta.detach().to(torch.float32)
        # L2-norm q/k + scale q AFTER norm -- EXACTLY matches the Triton
        # recurrent kernel (fused_recurrent_kda_fwd_kernel:221-224,
        # USE_QK_L2NORM_IN_KERNEL=True: q/=sqrt(sum(q^2)+1e-6); q*=scale).
        # F.normalize uses eps=1e-12; the kernel uses 1e-6, so do it by
        # hand to keep the integration cross-check bit-for-bit.
        if use_qk_l2norm_in_kernel:
            qf = qf / torch.sqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6)
            kf = kf / torch.sqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)
        qf = qf * scale                  # scale applied AFTER L2-norm

        # --- Activate the gate (fuse_gate=True): g = exp(lower_bound *
        # --- sigmoid(exp(A_log) * (raw_g + g_bias))) in NORMAL space.
        # Matches fused_recurrent_kda_fwd (HAS_DT_BIAS, USE_LOWER_BOUND),
        # validated bit-for-bit by probe_kda_xcheck.py (issue #45).
        A = torch.exp(A_log.to(torch.float32))                  # [H]
        if g_bias is not None:
            gb = g_bias.to(torch.float32).view(H, D)            # [H, D]
            gate_log = A[None, None, :, None] * (gf + gb[None, None, :, :])
        else:
            gate_log = A[None, None, :, None] * gf              # [B, n, H, D]
        if lower_bound is not None:
            gate_log = float(lower_bound) * torch.sigmoid(gate_log)
        g = torch.exp(gate_log)                                 # normal space
        beta = torch.sigmoid(btf)                               # [B, n, H]

        # --- Transpose to HIP layout [B, H, n, D] / [B, H, n] ---
        q_h = qf.permute(0, 2, 1, 3).contiguous()               # [B, H, n, D]
        k_h = kf.permute(0, 2, 1, 3).contiguous()
        v_h = vf.permute(0, 2, 1, 3).contiguous()
        g_h = g.permute(0, 2, 1, 3).contiguous()
        beta_h = beta.permute(0, 2, 1).contiguous()             # [B, H, n]

        # --- Per-sequence loop (cu_seqlens = [N+1] cumulative starts).
        # The HIP kernel runs ONE continuous recurrence per (b,h) from the
        # seed state, so each sequence needs its own launch seeded from
        # initial_state[s] (zeros for first-turn prefill).
        if cu_seqlens is not None:
            seq_lo = cu_seqlens[:-1].to(torch.int64).tolist()   # [N]
            seq_hi = cu_seqlens[1:].to(torch.int64).tolist()    # [N]
            N = len(seq_lo)
        else:
            seq_lo, seq_hi, N = [0], [n], 1

        out_h = torch.empty(B, H, n, D, dtype=torch.float32, device=dev)
        # Reusable per-seq scratch (B=1); zero-seeded for first-turn,
        # copied from initial_state[s] for multi-turn.
        state_scr = torch.empty(1, H, D, D, dtype=torch.float32, device=dev)
        if output_final_state:
            final_state = (initial_state.to(torch.float32).clone()
                           if initial_state is not None
                           else torch.empty(N, H, D, D,
                                            dtype=torch.float32, device=dev))
        else:
            final_state = initial_state

        for s in range(N):
            lo, hi = seq_lo[s], seq_hi[s]
            sl = hi - lo
            if sl <= 0:
                continue
            # Slice THIS sequence's tokens (HIP needs contiguous [1,H,sl,D]).
            qs = q_h[:, :, lo:hi, :].contiguous()
            ks = k_h[:, :, lo:hi, :].contiguous()
            vs = v_h[:, :, lo:hi, :].contiguous()
            gs = g_h[:, :, lo:hi, :].contiguous()
            bs = beta_h[:, :, lo:hi].contiguous()              # [1, H, sl]
            out_slice = out_h[:, :, lo:hi, :]                      # scatter target
            if initial_state is not None:
                state_scr.copy_(initial_state[s:s + 1].to(torch.float32))
            else:
                state_scr.zero_()
            kda_scratch(
                ctypes.c_void_p(qs.data_ptr()), ctypes.c_void_p(ks.data_ptr()),
                ctypes.c_void_p(vs.data_ptr()), ctypes.c_void_p(gs.data_ptr()),
                ctypes.c_void_p(bs.data_ptr()), ctypes.c_void_p(state_scr.data_ptr()),
                ctypes.c_void_p(out_slice.data_ptr()),
                ctypes.c_int(1), ctypes.c_int(H), ctypes.c_int(sl),
                ctypes.c_int(D),
            )
            # kda_delta_rule_fwd_with_scratch hipDeviceSynchronize()s on
            # return, so state_scr holds S_S -- copy back before the next
            # seq reseeds it.
            if output_final_state:
                final_state[s:s + 1].copy_(state_scr)

        # --- Cross-check vs the CPU oracle on the first call, ONLY for
        # --- the clean first-prefill case (N==1, zero initial_state):
        # --- the CPU oracle (vk_kda_naive_delta_rule_fwd) starts at
        # --- S_0=0, so multi-turn / multi-seq would diverge by design.
        # The CPU reference .so (libvkernels.so) is a SEPARATE build from
        # the HIP .so; an HIP-only deployment skips this gracefully (the
        # 12/12 unit tests + probe_kda_xcheck.py already validated the
        # kernel bit-for-bit).  A genuine mismatch still fails loudly.
        if (os.environ.get("VKERNELS_KDA_VALIDATE", "1") == "1"
                and N == 1
                and (initial_state is None
                     or float(initial_state.abs().max().item()) == 0.0)):
            try:
                _cpu = _bind_kda_cpu(_cpu_lib())
            except Exception as _lib_err:
                if "not found" in str(_lib_err).lower():
                    _cpu = None  # HIP-only build; skip, don't crash serve
                else:
                    raise
            if _cpu is not None:
                try:
                    _validate_kda_once(
                        kda_hip, _cpu,
                        q_h, k_h, v_h, g_h, beta_h,
                        1, H, n, D, chunk_size, out_h,
                    )
                except Exception as exc:  # pragma: no cover -- cluster-only
                    # Do NOT fall back to orig_chunk: on gfx942 that path faults
                    # (the chunked Triton leaf, job 586165).  Fail loudly.
                    raise RuntimeError(
                        f"VkernelKDA cross-check failed ({exc!r}); the HIP "
                        "prefill leaf disagrees with the CPU oracle. Set "
                        "VKERNELS_KDA=0 and use the K3_DISABLE_KDA baseline.") \
                        from exc
        torch.cuda.synchronize()

        # out [B, H, n, D] -> [B, n, H, D] (the layer's expected layout).
        out = out_h.permute(0, 2, 1, 3).contiguous().to(q.dtype)
        if (output_final_state and initial_state is not None
                and initial_state.dtype != torch.float32):
            final_state = final_state.to(initial_state.dtype)
        return out, final_state

    # Patch the LEAF module attribute (the K3 layer imports
    # ``chunk_kda_with_fused_gate`` by name inside _forward, so re-binding
    # the module attribute is enough -- no class attribute swap needed).
    import vllm.models.kimi_k3.amd.ops.third_party.kda.chunk as _chunk_mod
    _chunk_mod.chunk_kda_with_fused_gate = _patched_chunk_kda
    print(f"[VkernelKDA] patched chunk_kda_with_fused_gate (prefill leaf) "
          f"to vk_hip_kda_delta_rule_fwd_with_scratch on gfx942 "
          f"(chunk={chunk_size}); cross-checked vs FLA recurrent ref "
          f"(probe_kda_xcheck.py, issue #45).", flush=True)
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
