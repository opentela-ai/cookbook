"""DSA one-shot SDPA route on gfx942 (always-on for MI300A; clariden
b8d5296 shape).

GLM-5.3-Flash's DSA layers are PURE MHA (num_attention_heads ==
num_key_value_heads == 64, qk_nope_head_dim == v_head_dim == 256,
qk_rope_head_dim == 0 -> no GQA), so a per-request PyTorch
scaled_dot_product_attention ragged loop over metadata.cu_seqlens_{q,k}
is a correct drop-in for the one-shot prefill path.

On gfx942 the MLA path is broken: tilelang aborts on tail_dim==0 (issue #51)
and the rebound vk_hip_dsa_sparse_fwd (#52) yields garbage (" 1 "). The
one-shot MHA gate (dsa_backend set_dsa_prefill_impl) ALSO excludes gfx942
(only SM90 / SM100-109 / gfx95), so even though _forward_standard_mha
exists, it is never taken on MI300A.

This patcher does BOTH, lazily when sglang imports dsa_backend:
  1. _forward_standard_mha -> PyTorch SDPA ragged loop (FA is absent on
     ROCm exactly as on aarch64 Hopper, so SDPA is the universal kernel).
  2. set_dsa_prefill_impl -> call original, then force use_mha=True on
     is_hip() for extend-without-speculative, KEEPING the dtype / len /
     cp / hisparse guardrails (only the device_sm exclusion is lifted).

ALWAYS-ON on gfx942 (no env gate): the stale run/614856/engine.sh plumbing
drops GLM53_DSA_SDPA from the TP workers; supports_current_device() (checked
at import-hook fire and again at forward time) is the real guard, identical
to the tilelang patcher which fired in job 616424.

The import-timing race is handled by import_hook.run_after_import itself:
whether dsa_backend is imported later (meta-path hook) or was already
imported when we run (direct rebind), the same code path applies the patch
exactly once. That replaces the old TWO-path arrangement (a meta-path
finder plus a separate try-import direct rebind) whose flake in jobs
616583/616594/616601 — sglang pre-imported inside sitecustomize, the cached
sys.modules entry skipped find_spec, and the patch silently never applied —
is what motivates the shared helper.
"""
import sys

from gfx942 import supports_current_device
from import_hook import run_after_import

_DSA_SDPA_TARGET = "sglang.srt.layers.attention.dsa_backend"
_orig_set_dsa_prefill_impl = None


def _forward_standard_mha_sdpa(self, q, k, v, layer, forward_batch, metadata):
    """is_hip + FA-absent: PyTorch SDPA ragged loop (clariden b8d5296 shape)."""
    import torch
    _F = torch.nn.functional
    q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
    k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
    v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
    cu_q, cu_k = metadata.cu_seqlens_q, metadata.cu_seqlens_k
    causal = True
    scale = layer.scaling
    gqa = q.shape[-2] != k.shape[-2]
    out = torch.empty_like(q)
    for _i in range(len(cu_q) - 1):
        _qs, _qe = int(cu_q[_i]), int(cu_q[_i + 1])
        if _qe <= _qs:
            continue
        _ks, _ke = int(cu_k[_i]), int(cu_k[_i + 1])
        _qi = q[_qs:_qe][None]
        _ki = k[_ks:_ke][None]
        _vi = v[_ks:_ke][None]
        _sl_q, _sl_k = _qe - _qs, _ke - _ks
        if causal and _sl_q == _sl_k:
            _oi = _F.scaled_dot_product_attention(
                _qi, _ki, _vi, is_causal=True, scale=scale, enable_gqa=gqa
            )
        elif causal:  # sl_q < sl_k: bottom-right additive mask
            _m = torch.ones(_sl_q, _sl_k, device=q.device, dtype=torch.bool).tril(
                diagonal=_sl_k - _sl_q
            )[None, None]
            _oi = _F.scaled_dot_product_attention(
                _qi, _ki, _vi, attn_mask=_m, scale=scale, enable_gqa=gqa
            )
        else:
            _oi = _F.scaled_dot_product_attention(
                _qi, _ki, _vi, scale=scale, enable_gqa=gqa
            )
        out[_qs:_qe] = _oi[0]
    return out


def _set_dsa_prefill_impl_sdpa(self, forward_batch=None):
    """Call the original impl (sets dsa_prefill_impl + use_mha with the
    device_sm gate), then lift ONLY the device exclusion on gfx942 for
    extend-without-speculative, keeping every other guardrail."""
    _orig_set_dsa_prefill_impl(self, forward_batch)
    # Device gate at FORWARD time (torch.cuda reliably ready here, unlike import
    # time). Only force use_mha=True on gfx942; non-MI300A keeps native MLA.
    if not supports_current_device()[0]:
        return
    if getattr(self, "use_mha", False):
        return
    if not forward_batch or not forward_batch.forward_mode.is_extend_without_speculative():
        return
    try:
        import torch
        from sglang.srt.layers.attention.dsa_backend import (
            envs,
            is_dsa_enable_prefill_cp,
        )

        _sl = forward_batch.seq_lens_cpu
        _ok = (
            self.supports_mha_one_shot
            and _sl.max().item()
            <= envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()
            and self.token_to_kv_pool.dtype in (torch.bfloat16, torch.float8_e4m3fn)
            and sum(_sl) <= forward_batch.get_max_chunk_capacity()
            and (not is_dsa_enable_prefill_cp())
            and (self.hisparse_coordinator is None)
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[sitecustomize] DSA-SDPA use_mha gate re-check failed ({exc!r}); "
            "NOT forcing (engine keeps native MLA path)\n"
        )
        return
    if _ok:
        self.use_mha = True
        sys.stderr.write(
            "[sitecustomize] DSA-SDPA: forced use_mha=True on gfx942 "
            "(pure-MHA DSA -> PyTorch SDPA one-shot, clariden b8d5296 shape)\n"
        )


def _install(module):
    global _orig_set_dsa_prefill_impl
    try:
        _, gcn = supports_current_device()
    except Exception:  # noqa: BLE001
        gcn = ""
    gcn = gcn or "(no-cuda-device-0-here; gated again at forward time)"
    try:
        _orig_set_dsa_prefill_impl = (
            module.DeepseekSparseAttnBackend.set_dsa_prefill_impl
        )
        module.DeepseekSparseAttnBackend._forward_standard_mha = (
            _forward_standard_mha_sdpa
        )
        module.DeepseekSparseAttnBackend.set_dsa_prefill_impl = (
            _set_dsa_prefill_impl_sdpa
        )
        sys.stderr.write(
            "[sitecustomize] DSA-SDPA patch APPLIED on "
            + gcn
            + ": _forward_standard_mha->PyTorch SDPA ragged, "
            "use_mha forced on is_hip (pure-MHA DSA).\n"
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[sitecustomize] DSA-SDPA patch FAILED ({exc!r}); "
            "engine keeps native MLA path.\n"
        )


run_after_import(_DSA_SDPA_TARGET, _install)
