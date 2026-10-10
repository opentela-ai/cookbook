"""MoE fp8-expert bf16-dequant (gfx942 in-kernel fp8 GEMM divergence).

GLM-5.3-Flash routed MoE experts are fp8 block-quant ([128,128]). On MI300A
(gfx942) the in-kernel TRITON fp8 dequant+GEMM diverges catastrophically from
clariden at the first MoE layer: mlp_out cos ~0.55, max_abs ~8.0, with
IDENTICAL routing (topk_ids/topk_weights match to ~1.0 — confirmed by
comp_capture beverin_l3_moeinner vs clariden, Sep 5-6). SGLANG_USE_AITER
defaults False and is NOT exported by the sbatch/launcher, so _use_aiter=False
(fp8.py:130) and the fnuz block's `if _use_aiter: shuffle(16,16)` is SKIPPED
-> the experts are in STANDARD (unshuffled) layout. The bug is the fp8 GEMM
path (in-kernel dequant or GEMM), NOT a shuffle/runner mismatch and NOT the
router (routing already matches).

FIX (GLM53_MOE_DEQUANT_BF16=1, gfx942 only): after Fp8MoEMethod's
process_weights_after_loading_block_quant (which fnuz-normalizes w13/w2+
scales on gfx942, keeping them consistent; _use_aiter=False so NO shuffle),
dequant the expert weights fp8->bf16 in torch using the standard block form
  bf16 = w.to(f32).view(E, nbn, bn, nbk, bk) * scale.view(E, nbn, 1, nbk, 1)
(robust to the e4m3fn vs e4m3fnuz state: fnuz keeps w*scale consistent),
rebind as bf16 Parameters, and set a per-method flag. Then override
get_triton_quant_info to report the UNQUANTIZED TritonMoeQuantInfo
(use_fp8_w8a8=False, w13/w2_scale=None, a13/a2_scale=None, no block). The
TRITON runner then executes the SAME bf16 fused_moe that
UnquantizedFusedMoEMethod uses (the tested unquantized path on gfx942,
unquant.py:869-876), bypassing the broken in-kernel fp8 dequant+GEMM
entirely. No layout change (standard), no runner swap, no new kernel. The
fp8 scales are left in place (unused by the bf16 GEMM; safer than `del`).

The dequant runs EAGERLY in process_weights_after_loading_block_quant
(after checkpoint load, before ANY forward/CUDA-graph capture) so there is
zero graph-allocation risk. Validated by comp_capture vs clariden: with this
patch the layer-3 mlp_out cos ->1.0 (attn_pre/ffn_pre also ->1.0 once the
accumulated broken-MoE residual is removed).

Gate: GLM53_MOE_DEQUANT_BF16=1 (default off — the fp8 path is correct on
CUDA/other GPUs) AND gfx942. Lazily hooked via import_hook.run_after_import
on the first import of sglang.srt.layers.quantization.fp8 (during sglang
import for quant config); applied exactly once.
"""
import os
import sys

from gfx942 import supports_current_device
from import_hook import run_after_import

_MOE_DEQ_TARGET = "sglang.srt.layers.quantization.fp8"


def _dequant_block_fp8_to_bf16(w_fp8, scale_inv, block_n, block_k):
    """Standard block-fp8 dequant, CHUNKED over experts.

    w_fp8:     [E, N, K] float8_e4m3fn(uz)
    scale_inv: [E, N//block_n, K//block_k] float32 (per-block scale)
    returns:   [E, N, K] bfloat16  with real = fp8_value * scale_per_block
    (Correct regardless of e4m3fn vs e4m3fnuz: normalize_e4m3fn_to_e4m3fnuz
    scales w by *2 and scale by /2, keeping the product constant.)

    Chunked because a full-size fp32 temp (E,N,K) is 4 B/elem — job 625235
    OOM-killed the node (513 GB cgroup) with 4 ranks x ~8 GB transient at the
    end of a full-shard real-weight load. Chunking caps the fp32 temp at
    ~0.5 GB (8 experts), so the per-rank peak stays ~fp8 + bf16-out.
    """
    import torch

    E, N, K = w_fp8.shape
    n_blk, k_blk = N // block_n, K // block_k
    assert N % block_n == 0 and K % block_k == 0, (
        f"[MoE-dequant] weight {N}x{K} not divisible by block {block_n}x{block_k}"
    )
    assert tuple(scale_inv.shape) == (E, n_blk, k_blk), (
        f"[MoE-dequant] scale {tuple(scale_inv.shape)} != ({E},{n_blk},{k_blk})"
    )
    out = torch.empty((E, N, K), dtype=torch.bfloat16, device=w_fp8.device)
    chunk = max(1, min(E, 8))
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        w32 = w_fp8[e0:e1].to(torch.float32).view(
            e1 - e0, n_blk, block_n, k_blk, block_k
        )
        s32 = scale_inv[e0:e1].to(torch.float32).view(
            e1 - e0, n_blk, 1, k_blk, 1
        )
        out[e0:e1] = (w32 * s32).reshape(e1 - e0, N, K).to(torch.bfloat16)
        del w32, s32
    return out


def _apply_moe_dequant(fp8_module, gcn):
    import torch  # noqa: F401  (kept for parity with the inlined original)

    if os.environ.get("GLM53_MOE_DEQUANT_BF16", "0") != "1":
        return
    Fp8MoEMethod = getattr(fp8_module, "Fp8MoEMethod", None)
    if Fp8MoEMethod is None:
        sys.stderr.write(
            "[sitecustomize] MoE-dequant: Fp8MoEMethod not found; NOT patching\n"
        )
        return
    try:
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[sitecustomize] MoE-dequant: TritonMoeQuantInfo import failed "
            f"({exc!r}); NOT patching\n"
        )
        return

    _orig_pwblk = Fp8MoEMethod.process_weights_after_loading_block_quant
    _orig_gqi = Fp8MoEMethod.get_triton_quant_info

    # STREAMING dequant state: ONE shared bf16 slot for the whole process.
    # Load-time dequant of every layer needs 2x the fp8 expert weights
    # (~163 GB/rank bf16 vs ~82 GB/rank fp8 on GLM-5.3-Flash; jobs 625235 and
    # 625245 were OOM-killed at the 513 GB node cgroup right after the first
    # layers dequanted). Instead: dequant the layer's experts to bf16 the
    # first time get_triton_quant_info asks for it, keep ONLY that layer in
    # the cache, and free it when the NEXT layer asks. Layers run strictly
    # sequentially per forward, so the allocator reuses the same block ->
    # net steady-state cost is one layer's bf16 (~3.6 GB/rank).
    # REQUIRES eager decode (DISABLE_CUDA_GRAPH=1): captured decode graphs
    # would bake the shared buffer's pointer and replay with the WRONG
    # layer's weights (silent corruption).
    _deq_cache = {"key": None, "w13": None, "w2": None}
    _deq_logged = False

    def process_weights_after_loading_block_quant(self, layer):
        # sglang's own post-load processing: on gfx942 (_is_fp8_fnuz) it
        # normalizes w13/w2+scales to e4m3fnuz (kept consistent);
        # _use_aiter=False so NO (16,16) shuffle -> standard layout.
        # NO load-time dequant here (does not fit in node memory) — the
        # bf16 conversion happens per-forward in get_triton_quant_info.
        _orig_pwblk(self, layer)

    def get_triton_quant_info(self, layer):
        if not getattr(self, "block_quant", False) or not (
            hasattr(layer, "w13_weight_scale_inv")
            and hasattr(layer, "w2_weight_scale_inv")
        ):
            return _orig_gqi(self, layer)

        key = id(layer)
        if _deq_cache["key"] != key:
            # free the previous layer's bf16 FIRST so this layer's
            # allocations reuse the same allocator blocks (no growth)
            _deq_cache["w13"] = None
            _deq_cache["w2"] = None
            bn, bk = (self.weight_block_size or [128, 128])
            _deq_cache["w13"] = _dequant_block_fp8_to_bf16(
                layer.w13_weight.data, layer.w13_weight_scale_inv.data, bn, bk
            )
            _deq_cache["w2"] = _dequant_block_fp8_to_bf16(
                layer.w2_weight.data, layer.w2_weight_scale_inv.data, bn, bk
            )
            _deq_cache["key"] = key
            nonlocal _deq_logged
            if not _deq_logged:
                _deq_logged = True
                sys.stderr.write(
                    "[sitecustomize] MoE-dequant STREAMING: dequant w13 "
                    f"{tuple(_deq_cache['w13'].shape)} + w2 "
                    f"{tuple(_deq_cache['w2'].shape)} per layer per forward "
                    f"(bf16 fused_moe, eager decode required) on {gcn}\n"
                )
        # bf16 weights, no fp8 scales, no block. The TRITON runner runs the
        # SAME bf16 fused_moe that UnquantizedFusedMoEMethod uses (tested
        # correct on gfx942).
        return TritonMoeQuantInfo(
            w13_weight=_deq_cache["w13"],
            w2_weight=_deq_cache["w2"],
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
            use_mxfp8=False,
            use_fp8_w8a8=False,
            w13_scale=None,
            w2_scale=None,
            a13_scale=None,
            a2_scale=None,
        )

    Fp8MoEMethod.process_weights_after_loading_block_quant = (
        process_weights_after_loading_block_quant
    )
    Fp8MoEMethod.get_triton_quant_info = get_triton_quant_info
    sys.stderr.write(
        "[sitecustomize] MoE-dequant patch APPLIED on "
        f"{gcn} (GLM53_MOE_DEQUANT_BF16=1; Fp8MoEMethod."
        "process_weights_after_loading_block_quant + get_triton_quant_info "
        "overridden)\n"
    )


def _install(fp8_module):
    # Env gate at fire time (mirrors patch_topk_torch): the dispatcher imports
    # us unconditionally, so GLM53_MOE_DEQUANT_BF16 is the real switch and
    # is re-checked inside _apply_moe_dequant too.
    if os.environ.get("GLM53_MOE_DEQUANT_BF16", "0") != "1":
        sys.stderr.write(
            "[sitecustomize] MoE-dequant: GLM53_MOE_DEQUANT_BF16 != 1; "
            "fp8 module loaded UNPATCHED\n"
        )
        return
    ok, gcn = supports_current_device("MoE-dequant")
    if ok:
        _apply_moe_dequant(fp8_module, gcn)
    else:
        sys.stderr.write(
            f"[sitecustomize] MoE-dequant: not gfx942 (gcn={gcn!r}); "
            "fp8 module loaded UNPATCHED\n"
        )


# Install at startup. Zero work: no sglang/torch import. The hook fires only
# when sglang imports sglang.srt.layers.quantization.fp8 (its quant-config
# import); a CPU-only preflight never imports it, so the gate stays fast.
run_after_import(_MOE_DEQ_TARGET, _install)
