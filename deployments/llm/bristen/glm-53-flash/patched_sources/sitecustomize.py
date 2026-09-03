"""SM80 (A100) torch fallbacks for the DeepGEMM DSA-indexer logits kernels.

deep_gemm's ``fp8_paged_mqa_logits`` / ``fp8_mqa_logits`` JIT only for SM90+
(bristen engine.sh: "DeepGEMM / FlashInfer TRT-LLM / CUTLASS require SM90+"),
but the DSA kpool indexer routes SM80 to them unconditionally
(``_should_use_tilelang_paged_mqa_logits`` gates only HIP and arch==9).

apply_sm80_patch.sh copies this file to ``$SGLANG_PATCH_DIR/sitecustomize.py``;
patches_full is first on PYTHONPATH, so CPython imports it automatically at
interpreter startup and, when running on an SM80 device, rebinds the two
logits entry points (plus the paged metadata builder) to pure-torch
fallbacks. Semantics replicate the DeepGEMM/tilelang reference
(sglang/kernels/ops/attention/dsa/tilelang_kernel.py,
``fp8_paged_mqa_logits_kernel`` / ``fp8_index``):

    logits[r, t] = sum_h relu(q_fp8[r,h] . k_fp8[t]) * weight[r,h] * k_scale[t]

computed in fp32 from the fp8 (de)quantized values -- the same products as
the fp8 tensor-core dot with fp32 accumulation, so numerics differ from the
reference only in accumulation order. The kpool cache byte layout consumed
here (per page: [64 x 128B fp8 K][64 x 4B fp32 scale]) is the one the write
side produces and the tilelang reference reads (BLOCK_BYTES = 64*132,
SCALE_OFFSET = 64*128).

No effect on non-SM80 devices: beverin (MI300A) / clariden (SM90) keep the
real deep_gemm. Set ``SGLANG_SM80_DG_SHIM_DISABLE=1`` to veto.

Validated by tests_local/test_dsa_kpool_wiring.py T8 (vs a per-element loop
reference of the same semantics, incl. page-table mapping and ragged masks).
"""

import os as _os

import torch


def _sm80_logits_core(q, k, weights):
    """relu(q.k) per head, weighted sum over heads.

    q: [R, H, D] (fp8 or float), k: [N, D] (fp8 or float), weights: [R, H]
    fp32 -> [R, N] fp32. Matches tilelang ``fp8_index``: relu BEFORE the
    per-head weight, k-scale applied by the caller.
    """
    dots = torch.einsum("rhd,nd->rhn", q.to(torch.float32), k.to(torch.float32))
    dots = torch.relu_(dots).mul(weights.unsqueeze(-1))
    return dots.sum(dim=1)


def _sm80_fp8_paged_mqa_logits(
    q_fp8,
    kv_cache,
    weights,
    context_lens=None,
    block_tables=None,
    schedule_metadata=None,
    max_model_len=None,
    clean_logits=False,
    *args,
    **kwargs,
):
    """Paged variant (decode path).

    q_fp8: [R, 1, H, D] float8_e4m3fn (next_n collapsed via reshape).
    kv_cache: [P, B*(D+4)] uint8 (any view/shape that reshapes to it), per
    page [B x 128B fp8 K][B x 4B fp32 scale].
    block_tables: [R, L] int32 page ids; logits column for table col i is
    i*B + j (tilelang writes o[bx, i*B .. i*B+B)).
    Returns [R, S] fp32 with -1e30 where never written (deep_gemm's
    clean_logits=False leaves junk there; the downstream
    ``_topk_from_kpool_logits`` masks by pooled seqlens either way -- a
    finite sentinel is strictly safer than junk).
    """
    dev = q_fp8.device
    q = q_fp8.reshape(-1, q_fp8.shape[-2], q_fp8.shape[-1])
    w = weights.reshape(q.shape[0], q.shape[1])
    R, H, D = q.shape
    B = 64
    cache = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
    cache = cache.reshape(cache.shape[0], -1)
    P, nbytes = cache.shape
    assert nbytes % B == 0, f"kpool page bytes {nbytes} not divisible by {B}"
    D4 = nbytes // B  # 132 in the vintage layout
    assert D4 * B - B * D == B * 4, "unexpected kpool cache layout"
    k_all = cache[:, : B * D].contiguous().view(torch.float8_e4m3fn).reshape(P, B, D)
    ks_all = cache[:, B * D :].view(torch.float32).reshape(P, B)
    L = block_tables.shape[1]
    S = int(max_model_len) if max_model_len is not None else L * B
    out = torch.full((R, S), -1e30, dtype=torch.float32, device=dev)
    bt = block_tables.to(torch.long).clamp_(0, P - 1)
    CH = 256  # pages per chunk: dots buffer R*H*CH*B*4 ~ 33MB at R=8
    for p0 in range(0, L, CH):
        p1 = min(p0 + CH, L)
        pages = bt[:, p0:p1]  # [R, c]
        k = k_all[pages].to(torch.float32)  # [R, c, B, D]
        ks = ks_all[pages]  # [R, c, B]
        dots = torch.einsum("rhd,rcbd->rhcb", q.to(torch.float32), k)
        dots = torch.relu_(dots).mul(w.unsqueeze(-1).unsqueeze(-1))
        lg = dots.sum(dim=1).mul(ks)  # [R, c, B]
        flat = lg.reshape(R, -1)
        cols = torch.arange(p0 * B, p1 * B, device=dev)
        ok = cols < S
        out[:, cols[ok]] = flat[:, ok]
    return out


def _sm80_fp8_mqa_logits(
    q_fp8,
    kv,
    weights,
    ks_per_q=None,
    ke_per_q=None,
    clean_logits=True,
    *args,
    **kwargs,
):
    """Ragged variant (extend path).

    q_fp8: [R, H, D] float8_e4m3fn; kv = (k_fp8 [N, D], k_scale [N]);
    ks/ke_per_q: [R] int row ranges. Returns [R, N] fp32; zero outside
    [ks, ke) when clean_logits (deep_gemm's clean_logits=True behavior).
    """
    k_fp8, k_scale = kv[0], kv[1]
    q = q_fp8.reshape(-1, q_fp8.shape[-2], q_fp8.shape[-1])
    w = weights.reshape(q.shape[0], q.shape[1])
    R, H, D = q.shape
    N = k_fp8.shape[0]
    dev = q.device
    if clean_logits:
        out = torch.zeros((R, N), dtype=torch.float32, device=dev)
    else:
        out = torch.empty((R, N), dtype=torch.float32, device=dev)
    if ks_per_q is not None:
        ks = ks_per_q.to(torch.long).reshape(R)
        ke = ke_per_q.to(torch.long).reshape(R)
    else:
        ks = torch.zeros(R, dtype=torch.long, device=dev)
        ke = torch.full((R,), N, dtype=torch.long, device=dev)
    kf = k_fp8.to(torch.float32)
    ksc = k_scale.to(torch.float32)
    CH_N, CH_R = 8192, 128  # dots buffer CH_R*H*CH_N*4 ~ 268MB worst case
    for n0 in range(0, N, CH_N):
        n1 = min(n0 + CH_N, N)
        for r0 in range(0, R, CH_R):
            r1 = min(r0 + CH_R, R)
            lg = _sm80_logits_core(q[r0:r1], kf[n0:n1], w[r0:r1])
            lg = lg.mul(ksc[n0:n1].unsqueeze(0))
            if clean_logits:
                ar = torch.arange(n0, n1, device=dev)
                lo = ks[r0:r1, None]
                hi = ke[r0:r1, None]
                lg = lg.masked_fill((ar < lo) | (ar >= hi), 0.0)
            out[r0:r1, n0:n1] = lg
    return out


def _sm80_get_paged_mqa_logits_metadata(*args, **kwargs):
    # The real builder feeds deep_gemm's own kernel launch; the torch
    # fallback ignores it. Callers only pass the object through.
    return torch.empty(0, dtype=torch.int32, device="cuda")


def _install():
    if _os.environ.get("SGLANG_SM80_DG_SHIM_DISABLE") == "1":
        return
    try:
        if not torch.cuda.is_available():
            return
        if torch.cuda.get_device_capability(torch.device("cuda"))[0] != 8:
            return
    except Exception:
        return
    import sys
    import types

    try:
        import deep_gemm
    except Exception:
        # deep_gemm itself unusable on this arch: provide a minimal module
        deep_gemm = types.ModuleType("deep_gemm")
        sys.modules["deep_gemm"] = deep_gemm
        deep_gemm.get_num_sms = lambda: torch.cuda.get_device_properties(
            torch.device("cuda")
        ).multi_processor_count
    deep_gemm.fp8_paged_mqa_logits = _sm80_fp8_paged_mqa_logits
    deep_gemm.fp8_mqa_logits = _sm80_fp8_mqa_logits
    deep_gemm.get_paged_mqa_logits_metadata = _sm80_get_paged_mqa_logits_metadata


if __name__ != "__main__":
    _install()
