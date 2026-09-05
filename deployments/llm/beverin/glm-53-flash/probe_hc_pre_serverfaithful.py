#!/usr/bin/env python3
"""Decisive hc_attn_pre replication on beverin MI300A (gfx942).

Reproduces the TP4 server's EXACT hc_attn_pre call (layer 0, 2505-token prefill)
using:
  * the SAVED bit-exact server input  comp_layer0_attn_pre_in  [2505,16384] bf16
  * the SAVED server output          comp_layer0_attn_pre_out [2505,4096]  bf16
  * the REAL model weights            hc_attn_fn [24,16384]  (bf16 ckpt -> fp32 at runtime)
                                     hc_attn_scale [3] fp32,  hc_attn_base [24] fp32
                                     input_layernorm.weight [4096] bf16

Server args (from glm5_next.py:794 _hc_pre / mhc.py:1728 hc_pre, routed via
communicator_mhc.py:100 with out_norm = self.input_layernorm):
  post_mult_value=2.0, hc_norm_weight=None,
  out_norm_weight=input_layernorm.weight, out_norm_eps=1e-5

With SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 (server path) and 2505>2048 tokens, this
routes to mhc_pre_gemm_sqrsum_tilelang (hidden_block=128) THEN, because
out_norm_weight is not None, to mhc_pre_big_fuse_with_norm_tilelang -- the
FUSED out-norm kernel the server actually uses. (The earlier probe_gdn_kernels
passed out_norm_weight=None, taking the NON-fused mhc_pre_big_fuse_tilelang and
never exercising this kernel.)

Tests:
  (A) tilelang li_t  vs  saved attn_pre_out    -- single-rank faithfulness
  (B) run1 vs run2 determinism                 -- is the kernel racy?
  (C) out_norm(torch-ref li_r) vs li_t         -- fused-norm kernel correctness
  (D) None-variant li_n vs torch-ref li_r      -- isolates the fused-norm kernel
"""
import os

# MUST be set before importing sglang (read at import time). The EDF already
# sets these, but belt-and-suspenders for the single-rank run.
os.environ["SGLANG_OPT_DEEPGEMM_HC_PRENORM"] = "0"
os.environ.setdefault("SGLANG_USE_AITER", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29519")

import sys

OVL = "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay"
for _p in (
    f"{OVL}/pylib",
    f"{OVL}/pkgs310",
    f"{OVL}/sgl-workspace/sglang/python",
    f"{OVL}/sgl-workspace/transformers/src",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import torch
from safetensors import safe_open

DEV = "cuda"
MD = "/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash"
CAP = "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture/beverin_l0ops"
HID, N, EPS = 4096, 4, 1e-6
NORM_EPS = 1e-5


def err(a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        return f"SHAPE-MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a - b).abs()
    rel = d.mean() / b.abs().mean().clamp_min(1e-9)
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()
    return f"max_abs={d.max().item():.6f} rel_mean={rel.item():.6f} cos={cos:.6f}"


def stats(name, t):
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        print(f"    {name}: <non-tensor {type(t).__name__}>", flush=True)
        return
    tf = t.float()
    nan_pct = float((tf != tf).float().mean()) * 100
    inf_pct = float(torch.isinf(tf).float().mean()) * 100
    print(
        f"    {name}: min={tf.min().item():.4f} max={tf.max().item():.4f} "
        f"mean={tf.mean().item():.5f} std={tf.std().item():.5f} "
        f"nan%={nan_pct:.4f} inf%={inf_pct:.4f}",
        flush=True,
    )


def load_weight(key):
    wm = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
    sh = wm[key]
    with safe_open(f"{MD}/{sh}", framework="pt", device=DEV) as sf:
        return sf.get_tensor(key)


def main():
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}", flush=True)

    # --- saved bit-exact server tensors (plain tensors, comp_capture.py:185) ---
    inp = torch.load(f"{CAP}/comp_layer0_attn_pre_in.pt", map_location=DEV)
    saved_out = torch.load(f"{CAP}/comp_layer0_attn_pre_out.pt", map_location=DEV)
    if isinstance(inp, dict):
        inp = max(inp.values(), key=lambda v: getattr(v, "numel", lambda: 0)())
    if isinstance(saved_out, dict):
        saved_out = max(saved_out.values(), key=lambda v: getattr(v, "numel", lambda: 0)())
    print(
        f"saved attn_pre_in: {tuple(inp.shape)} {inp.dtype} (abs_mean={inp.float().abs().mean():.5f})\n"
        f"saved attn_pre_out: {tuple(saved_out.shape)} {saved_out.dtype} (abs_mean={saved_out.float().abs().mean():.5f})",
        flush=True,
    )

    # --- real layer-0 weights (ckpt bf16 -> runtime fp32 for hc_fn, per glm5_next.py:730) ---
    hc_fn = load_weight("model.language_model.layers.0.hc_attn_fn").float()  # [24,16384] fp32
    hc_scale = load_weight("model.language_model.layers.0.hc_attn_scale")  # [3] fp32
    hc_base = load_weight("model.language_model.layers.0.hc_attn_base")  # [24] fp32
    inorm_w = load_weight("model.language_model.layers.0.input_layernorm.weight")  # [4096] bf16
    print(
        f"weights: hc_fn={tuple(hc_fn.shape)} {hc_fn.dtype}  hc_scale={tuple(hc_scale.shape)} {hc_scale.dtype}  "
        f"hc_base={tuple(hc_base.shape)} {hc_base.dtype}  inorm_w={tuple(inorm_w.shape)} {inorm_w.dtype}",
        flush=True,
    )

    # --- 1-rank dist init so get_tp_group() works (hc_pre allocates layer_input
    #     under use_symmetric_memory(get_tp_group())). Single rank = no-op alloc. ---
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method="env://"
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    print("dist: 1-rank TP group initialized", flush=True)

    from sglang.kernels.ops.layernorm.mhc import hc_pre, _mhc_pre_torch

    # ===== SERVER-FAITHFUL args (out_norm_weight = input_layernorm.weight) =====
    args_server = dict(
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        hc_mult=N,
        rms_eps=NORM_EPS,
        hc_eps=EPS,
        sinkhorn_iters=20,
        post_mult_value=2.0,
        hc_norm_weight=None,
        out_norm_weight=inorm_w,
        out_norm_eps=NORM_EPS,
    )
    # None variant = the path the earlier probe_gdn_kernels used (NON-fused)
    args_none = dict(args_server)
    args_none["out_norm_weight"] = None
    args_none["out_norm_eps"] = None

    # ===== (A)(B) server-faithful tilelang, twice for determinism =====
    li_t, cr_t, pm_t, nf_t = hc_pre(inp.clone(), **args_server)
    li_t2, cr_t2, pm_t2, nf_t2 = hc_pre(inp.clone(), **args_server)
    print(
        f"\n[SERVER-FAITHFUL] norm_fused={bool(nf_t)}  li={tuple(li_t.shape)} {li_t.dtype} "
        f"cr={tuple(cr_t.shape)} pm={tuple(pm_t.shape)}",
        flush=True,
    )
    print(f"  (A) tilelang li_t   vs  saved attn_pre_out : {err(li_t, saved_out)}", flush=True)
    print(f"  (B) run1 vs run2  (li det)                  : {err(li_t, li_t2)}", flush=True)
    print(f"      run1 vs run2  (cr det)                  : {err(cr_t, cr_t2)}", flush=True)
    print(f"      run1 vs run2  (pm det)                  : {err(pm_t, pm_t2)}", flush=True)

    # ===== (C) torch reference + fused out_norm (RMSNorm with inorm_w, eps=1e-5) =====
    residual = inp.view(inp.shape[0], N, HID).float()
    pm_r, cr_r, li_r = _mhc_pre_torch(
        residual=residual,
        fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=NORM_EPS,
        hc_pre_eps=EPS,
        hc_sinkhorn_eps=EPS,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )

    def rms_norm(x, w, eps):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)

    li_r_normed = rms_norm(li_r, inorm_w, NORM_EPS)
    print(f"\n  (C) out_norm(torch-ref li_r)  vs tilelang li_t : {err(li_r_normed, li_t)}", flush=True)
    print(f"      torch-ref li_r (NO norm)  vs tilelang li_t : {err(li_r, li_t)}", flush=True)
    print(f"      torch-ref cr_r            vs tilelang cr_t : {err(cr_r.reshape(cr_t.shape), cr_t)}", flush=True)
    print(f"      torch-ref pm_r            vs tilelang pm_t : {err(pm_r.reshape(pm_t.shape), pm_t)}", flush=True)

    # ===== (D) None variant: isolate the fused-norm kernel =====
    li_n, cr_n, pm_n, nf_n = hc_pre(inp.clone(), **args_none)
    print(f"\n[NONE-VARIANT] norm_fused={bool(nf_n)}", flush=True)
    print(f"  (D) None-variant li_n  vs torch-ref li_r (no norm) : {err(li_n, li_r)}", flush=True)
    print(f"      None-variant li_n  vs SERVER-FAITHFUL li_t     : {err(li_n, li_t)}", flush=True)
    print(f"      None-variant cr_n  vs torch-ref cr_r           : {err(cr_n, cr_r.reshape(cr_n.shape))} ", flush=True)

    # ===== (E) FIX VALIDATION: beverin non-fused + external torch RMSNorm
    #         vs clariden's SAVED server output (clariden = reference machine).
    # If (E1)/(E2) match clariden to ~bf16 while (E3) shows the ~30% gap,
    # the bug is confirmed AND disabling the fused norm restores correct output.
    CLAR_CAP = "/capstor/scratch/cscs/xyao/glm-53-flash/comp_capture/clariden_l0ops"
    try:
        cl_in = torch.load(f"{CLAR_CAP}/comp_layer0_attn_pre_in.pt", map_location=DEV)
        cl_out = torch.load(f"{CLAR_CAP}/comp_layer0_attn_pre_out.pt", map_location=DEV)
        if isinstance(cl_in, dict):
            cl_in = max(cl_in.values(), key=lambda v: getattr(v, "numel", lambda: 0)())
        if isinstance(cl_out, dict):
            cl_out = max(cl_out.values(), key=lambda v: getattr(v, "numel", lambda: 0)())
        print(f"\n[CLARIDEN REFERENCE] cl_in={tuple(cl_in.shape)} {cl_in.dtype} (abs_mean={cl_in.float().abs().mean():.5f})", flush=True)
        print(f"                    cl_out={tuple(cl_out.shape)} {cl_out.dtype} (abs_mean={cl_out.float().abs().mean():.5f})", flush=True)
        in_diff = (inp.float() - cl_in.float()).abs().max().item()
        print(f"  beverin attn_pre_in vs clariden attn_pre_in (max_abs): {in_diff:.8f}" + ("" if in_diff == 0 else " *** INPUTS DIFFER ***"), flush=True)
        # ===== POST-FIX VALIDATION (the patch above forces non-fused on ROCm) =====
        # Pre-patch: nf_t was True and li_t diverged from clariden by ~30%
        # (E3). Post-patch: nf_t must be False and out_norm(li_t) must match
        # clariden to ~2e-3 (the kernel fix restores correct output).
        print(f"\n[POST-FIX] norm_fused now = {bool(nf_t)} (was True pre-patch)", flush=True)
        print(f"  (P1) li_t (server-faithful) vs li_n (none-variant) : {err(li_t, li_n)}  (both non-fused now)", flush=True)
        print(f"  (P2) out_norm(li_t) vs clariden saved out         : {err(rms_norm(li_t, inorm_w, NORM_EPS), cl_out)}  (SUCCESS: ~2e-3)", flush=True)
        li_fixed = rms_norm(li_n, inorm_w, NORM_EPS)  # beverin non-fused + external torch RMSNorm
        print(f"  (E1) BEVERIN-FIXED  out_norm(li_n)    vs clariden saved out : {err(li_fixed, cl_out)}", flush=True)
        print(f"  (E2) BEVERIN-FIXED  out_norm(torch_r) vs clariden saved out : {err(li_r_normed, cl_out)}", flush=True)
        print(f"  (E3) BEVERIN-SERVER li_t              vs clariden saved out : {err(li_t, cl_out)}  (the ~30% divergence)", flush=True)
        li_t_on_cl, cr_on_cl, pm_on_cl, nf_on_cl = hc_pre(cl_in.clone(), **args_server)
        print(f"  (E4) beverin hc_pre on CLARIDEN input  vs clariden saved out : {err(li_t_on_cl, cl_out)}", flush=True)
        print(f"  (E5) beverin hc_pre on CLARIDEN input  vs BEVERIN-FIXED      : {err(li_t_on_cl, li_fixed)}", flush=True)
        stats("cl_out    (clariden server)", cl_out)
        stats("li_fixed  (beverin fixed)", li_fixed)
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[CLARIDEN-REF] skipped: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    # ===== value-range diagnostics =====
    print("\n[value ranges]", flush=True)
    for nm, t in [
        ("li_t        (server-faithful)", li_t),
        ("saved_out   (server)", saved_out),
        ("li_r_normed (torch+fused)", li_r_normed),
        ("li_r        (torch, no norm)", li_r),
        ("li_n        (none-variant)", li_n),
    ]:
        stats(nm, t)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
