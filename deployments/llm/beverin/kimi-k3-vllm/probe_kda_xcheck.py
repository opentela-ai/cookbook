#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Issue #45 cross-check: vk_hip_kda_delta_rule_fwd vs the vLLM Triton
KDA decode kernel (fused_recurrent_kda) at K3 head shapes.

The vLLM Triton kernel is a faithful copy of the FLA gated-delta-rule
(IS_KDA=True): per-key-dim gate, applied BEFORE the prediction (post-gate
prediction). The vkernels HIP kernel must produce the SAME output (up to
fp round-off) when fed the same raw inputs — the gate activation
(lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))), beta sigmoid, q/k
L2-norm and q *= scale are done by the CALLER for the HIP kernel and by the
TRITON kernel internally, so the probe replicates the activation for the
HIP call and passes raw_g/raw_beta to the Triton call (fuse_gate=True).

Usage (inside the beverin vLLM container, one MI300A GPU):
  python3 probe_kda_xcheck.py [H] [S] [D]
defaults: H=12 S=256 D=128  (one K3 KDA head block)
"""
import os
import sys
import ctypes
import math

import torch

# ---------------------------------------------------------------------------
# 1. Load the HIP shared lib + bind vk_hip_kda_delta_rule_fwd
# ---------------------------------------------------------------------------
HIP_SO = os.environ.get(
    "VKERNELS_HIP_LIB",
    "/capstor/scratch/cscs/xyao/vkernels/build/hip/src/c/libvkernels_hip.so",
)
lib = ctypes.CDLL(HIP_SO)
vk_hip_kda = lib.vk_hip_kda_delta_rule_fwd
vk_hip_kda.restype = None
vk_hip_kda.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5

# ---------------------------------------------------------------------------
# 2. K3 shapes (one head block; small enough for a quick GPU run)
# ---------------------------------------------------------------------------
H = int(sys.argv[1]) if len(sys.argv) > 1 else 12
S = int(sys.argv[2]) if len(sys.argv) > 2 else 256
D = int(sys.argv[3]) if len(sys.argv) > 3 else 128
B = 1
dev = "cuda"
dt = torch.float32
torch.manual_seed(42)

print(f"[xcheck] B={B} H={H} S={S} D={D}  (scale={D**-0.5:.6f})", flush=True)

# ---------------------------------------------------------------------------
# 3. Raw inputs (matching the vLLM KimiGatedDeltaNetAttention contract)
#    raw_g [1,S,H,D], raw_beta [1,S,H], A_log [H], dt_bias [H*D]
# ---------------------------------------------------------------------------
raw_q = torch.randn(B, S, H, D, device=dev, dtype=dt) * 0.1
raw_k = torch.randn(B, S, H, D, device=dev, dtype=dt) * 0.1
raw_v = torch.randn(B, S, H, D, device=dev, dtype=dt) * 0.1
raw_g = torch.randn(B, S, H, D, device=dev, dtype=dt) * 0.3
raw_beta = torch.randn(B, S, H, device=dev, dtype=dt) * 0.3
A_log = (torch.randn(H, device=dev, dtype=dt) * 0.2).contiguous()
dt_bias = (torch.randn(H * D, device=dev, dtype=dt) * 0.1).contiguous()
lower_bound = -5.0  # self.gate_lower_bound (K3 default)

# ---------------------------------------------------------------------------
# 4. Triton oracle: fused_recurrent_kda (fuse_gate=True activates internally)
# ---------------------------------------------------------------------------
from vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent import (  # noqa: E402
    fused_recurrent_kda,
)

initial_state = torch.zeros(2, H, D, D, device=dev, dtype=dt)  # slot 0=NULL, 1=zeros
cu_seqlens = torch.tensor([0, S], device=dev, dtype=torch.int32)
ssm_state_indices = torch.ones(1, S, device=dev, dtype=torch.int32)  # all -> slot 1

print("[xcheck] launching vLLM Triton oracle (first call compiles ~10-30s)...",
      flush=True)
out_triton, _ = fused_recurrent_kda(
    q=raw_q, k=raw_k, v=raw_v,
    raw_g=raw_g, raw_beta=raw_beta,
    A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
    initial_state=initial_state,
    cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
)
# out_triton: [1, S, H, D]
torch.cuda.synchronize()
print(f"[xcheck] Triton done: out shape={tuple(out_triton.shape)} "
      f"max|o|={out_triton.abs().max().item():.6f}", flush=True)

# ---------------------------------------------------------------------------
# 5. HIP kernel: replicate the activation + normalization the Triton kernel
#    does internally, then call vk_hip_kda_delta_rule_fwd (S_0=0, own state).
# ---------------------------------------------------------------------------
# gate[t,h,k] = exp(lower_bound * sigmoid(exp(A_log[h]) * (raw_g[t,h,k] + dt_bias[h,k])))
A = torch.exp(A_log)                                   # [H]
dt_bias_2d = dt_bias.view(H, D)                        # [H, D]
gate_log = lower_bound * torch.sigmoid(
    A[None, None, :, None] * (raw_g + dt_bias_2d[None, None, :, :])
)                                                      # [1, S, H, D]
gate = torch.exp(gate_log)                             # [1, S, H, D] normal space
beta = torch.sigmoid(raw_beta)                         # [1, S, H]
scale = D ** -0.5

# q: L2-norm + scale ; k: L2-norm only  (matches use_qk_l2norm_in_kernel=True)
q_norm = raw_q / (raw_q.norm(dim=-1, keepdim=True) + 1e-6) * scale   # [1, S, H, D]
k_norm = raw_k / (raw_k.norm(dim=-1, keepdim=True) + 1e-6)           # [1, S, H, D]

# Transpose [1, S, H, D] -> [1, H, S, D] (HIP layout) ; beta [1, S, H] -> [1, H, S]
q_hip = q_norm.permute(0, 2, 1, 3).contiguous()
k_hip = k_norm.permute(0, 2, 1, 3).contiguous()
v_hip = raw_v.permute(0, 2, 1, 3).contiguous()
g_hip = gate.permute(0, 2, 1, 3).contiguous()
beta_hip = beta.permute(0, 2, 1).contiguous()           # [1, H, S]
out_hip = torch.empty(1, H, S, D, device=dev, dtype=dt)
chunk_size = 64 if S >= 64 else S

vk_hip_kda(
    ctypes.c_void_p(q_hip.data_ptr()), ctypes.c_void_p(k_hip.data_ptr()),
    ctypes.c_void_p(v_hip.data_ptr()), ctypes.c_void_p(g_hip.data_ptr()),
    ctypes.c_void_p(beta_hip.data_ptr()), ctypes.c_void_p(out_hip.data_ptr()),
    ctypes.c_int(1), ctypes.c_int(H), ctypes.c_int(S), ctypes.c_int(D),
    ctypes.c_int(chunk_size),
)
torch.cuda.synchronize()
out_hip_vs = out_hip.permute(0, 2, 1, 3).contiguous()   # [1, S, H, D]
print(f"[xcheck] HIP done: out shape={tuple(out_hip_vs.shape)} "
      f"max|o|={out_hip_vs.abs().max().item():.6f}", flush=True)

# ---------------------------------------------------------------------------
# 6. Compare
# ---------------------------------------------------------------------------
diff = (out_triton - out_hip_vs).abs()
maxd = diff.max().item()
maxabs = out_triton.abs().max().item()
rel = maxd / (maxabs + 1e-9)
# per-element relative (guard the small denominators)
rel_elem = (diff / (out_triton.abs() + 1e-6)).clamp(max=1e9)
p99 = torch.quantile(rel_elem.flatten().to(torch.float64), 0.99).item()

print(f"[xcheck] max_abs_diff={maxd:.6e}  max_abs={maxabs:.6e}  "
      f"rel(max)={rel:.6e}  rel(p99)={p99:.6e}", flush=True)
THRESH = 1e-2
if rel < THRESH:
    print(f"[xcheck] PASS: HIP matches vLLM Triton KDA "
          f"(rel={rel:.4e} < {THRESH}). Issue #45 forward validated.", flush=True)
    sys.exit(0)
print(f"[xcheck] FAIL: HIP disagrees with vLLM Triton KDA "
      f"(rel={rel:.4e} >= {THRESH}).", flush=True)
# show the worst element
worst = diff.argmax().item()
idx = torch.unravel_index(torch.tensor(worst), out_triton.shape)
print(f"[xcheck] worst at {tuple(int(i) for i in idx)}: "
      f"triton={out_triton[idx].item():.6f}  hip={out_hip_vs[idx].item():.6f}",
      flush=True)
sys.exit(1)
