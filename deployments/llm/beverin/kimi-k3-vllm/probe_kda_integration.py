#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Issue #45 INTEGRATION cross-check: the *patched* chunk_kda_with_fused_gate
(vkernels_attn.py routes it to vk_hip_kda_delta_rule_fwd_with_scratch on
gfx942 when VKERNELS_KDA=1) vs the working FLA Triton recurrent reference
``fused_recurrent_kda`` (same gated-delta-rule, IS_KDA=True).

This validates the PYTHON MARSHALLING in the leaf patch -- the gate
activation (lower_bound * sigmoid(exp(A_log) * (raw_g + g_bias))),
beta sigmoid, q/k L2-norm + scale = D**-0.5, the [B,H,n,D] transpose,
the per-sequence loop over cu_seqlens, initial_state seeding and
final_state writeback -- against the kernel the layer actually ships
with (and which probe_kda_xcheck.py already validated the HIP kernel
against bit-for-bit).

Run INSIDE the kimi-k3-vllm container on one MI300A, with
``VKERNELS_KDA=1`` so sitecustomize.py applies the leaf patch:

  python3 probe_kda_integration.py [H] [S] [D]
defaults: H=4 S=64 D=32  (fast; bump to 12 256 128 for the real head)
"""
import os
import sys

import torch

H = int(sys.argv[1]) if len(sys.argv) > 1 else 4
S = int(sys.argv[2]) if len(sys.argv) > 2 else 64
D = int(sys.argv[3]) if len(sys.argv) > 3 else 32
dev = "cuda"
dt = torch.float32
torch.manual_seed(42)

print(f"[kda-int] B=1 H={H} S={S} D={D}  VKERNELS_KDA="
      f"{os.environ.get('VKERNELS_KDA', '0')}", flush=True)

# sitecustomize.py has already applied the leaf patch (VKERNELS_KDA=1), so
# importing chunk_kda_with_fused_gate here yields the *patched* function.
from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (  # noqa: E402
    chunk_kda_with_fused_gate,
)
from vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent import (  # noqa: E402
    fused_recurrent_kda,
)

# raw inputs (B=1, n=S, H, D); A_log [H]; g_bias (dt_bias) [H*D]
q = (torch.randn(1, S, H, D, device=dev, dtype=dt) * 0.1)
k = (torch.randn(1, S, H, D, device=dev, dtype=dt) * 0.1)
v = (torch.randn(1, S, H, D, device=dev, dtype=dt) * 0.1)
raw_g = (torch.randn(1, S, H, D, device=dev, dtype=dt) * 0.3)
raw_beta = (torch.randn(1, S, H, device=dev, dtype=dt) * 0.3)
A_log = (torch.randn(H, device=dev, dtype=dt) * 0.2).contiguous()
g_bias = (torch.randn(H * D, device=dev, dtype=dt) * 0.1).contiguous()
lower_bound = -5.0
cu_seqlens = torch.tensor([0, S], device=dev, dtype=torch.int32)
initial_state = torch.zeros(1, H, D, D, device=dev, dtype=dt)

# --- patched leaf (HIP kernel on gfx942, or original Triton otherwise) ---
out_patch, fs_patch = chunk_kda_with_fused_gate(
    q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
    A_log=A_log, g_bias=g_bias, scale=None,
    initial_state=initial_state, output_final_state=True,
    lower_bound=lower_bound, use_qk_l2norm_in_kernel=True,
    cu_seqlens=cu_seqlens,
)
torch.cuda.synchronize()
print(f"[kda-int] patch out {tuple(out_patch.shape)} max|o|="
      f"{out_patch.abs().max().item():.6f}  fs {tuple(fs_patch.shape)}",
      flush=True)

# --- FLA Triton recurrent reference (activates gate internally, fuse_gate) ---
init_ref = torch.zeros(2, H, D, D, device=dev, dtype=dt)   # slot 1 = zeros
ssm_idx = torch.ones(1, S, device=dev, dtype=torch.int32)  # all -> slot 1
out_ref, fs_ref = fused_recurrent_kda(
    q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
    A_log=A_log, dt_bias=g_bias, lower_bound=lower_bound,
    initial_state=init_ref, cu_seqlens=cu_seqlens,
    ssm_state_indices=ssm_idx,
)
torch.cuda.synchronize()
print(f"[kda-int] ref   out {tuple(out_ref.shape)} max|o|="
      f"{out_ref.abs().max().item():.6f}  fs {tuple(fs_ref.shape)}",
      flush=True)


# --- compare ---
# max-relative (max|a-b|/|b|) is inflated by near-zero output elements on a
# wide-dynamic-range recurrent output.  Report scale_rel (max|a-b|/max|b|,
# scale-invariant), max_abs, mean_rel, and rel_top1pct (relative error only
# at the top-1% largest |b|, the significant elements) to distinguish a real
# mismatch from fp32 rounding.
def _metrics(a, b):
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    abs_diff = (a - b).abs()
    denom = torch.maximum(b.abs(), torch.full_like(b, 1e-8))
    rel = abs_diff / denom
    max_abs = float(abs_diff.max())
    max_b = float(b.abs().max())
    top1 = rel[b.abs() > 0.01 * max_b] if max_b > 0 else rel
    return {
        "max_rel": float(rel.max()),
        "max_abs": max_abs,
        "scale_rel": max_abs / max(max_b, 1e-8),
        "mean_rel": float(rel.mean()),
        "rel_top1pct": float(top1.max()) if top1.numel() else 0.0,
    }


mo = _metrics(out_patch, out_ref)
mf = _metrics(fs_patch[0], fs_ref[1])
print(f"[kda-int] OUT  max_rel={mo['max_rel']:.3e} scale_rel="
      f"{mo['scale_rel']:.3e} max_abs={mo['max_abs']:.3e} "
      f"mean_rel={mo['mean_rel']:.3e} rel_top1pct={mo['rel_top1pct']:.3e}",
      flush=True)
print(f"[kda-int] FS   max_rel={mf['max_rel']:.3e} scale_rel="
      f"{mf['scale_rel']:.3e} max_abs={mf['max_abs']:.3e} "
      f"mean_rel={mf['mean_rel']:.3e} rel_top1pct={mf['rel_top1pct']:.3e}",
      flush=True)
# Pass on scale_rel (scale-invariant, not inflated by near-zero).  max_rel is
# reported for context (a recurrent kernel naturally has a wide dynamic
# range, so max_rel can be >> scale_rel purely from near-zero elements).
THRESH = 1e-2
if mo['scale_rel'] < THRESH and mf['scale_rel'] < THRESH:
    print(f"[kda-int] PASS: patched leaf == FLA recurrent ref "
          f"(scale_rel < {THRESH})", flush=True)
    sys.exit(0)
print(f"[kda-int] FAIL: patched leaf disagrees with FLA recurrent ref "
      f"(scale_rel >= {THRESH}); set VKERNELS_KDA=0 for the K3_DISABLE_KDA "
      f"baseline.", flush=True)
sys.exit(1)
