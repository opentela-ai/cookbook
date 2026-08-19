#!/usr/bin/env python3
"""Smoke test for vkernels HIP C ABI on gfx942 (MI300A).

Phase 1 (C++) already validated kernel correctness against CPU reference.
This script verifies the `libvkernels_hip.so` shared library:
  1. Loads with ctypes
  2. Calls each kernel with known inputs
  3. Verifies output is non-trivial (not all zeros / no crash)
  4. Reports output statistics

This is the integration surface for the vLLM MoE backend shim.
"""
import ctypes
import glob
import os
import sys

import numpy as np
import torch

VKERNELS_DIR = os.environ.get("VKERNELS_DIR", "/capstor/scratch/cscs/xyao/vkernels")

# Find libvkernels_hip.so
_cands = sorted(glob.glob(os.path.join(VKERNELS_DIR, "build", "hip", "**", "libvkernels_hip.so"), recursive=True))
HIP_LIB = _cands[0] if _cands else os.path.join(VKERNELS_DIR, "build", "hip", "src", "c", "libvkernels_hip.so")

if not os.path.exists(HIP_LIB):
    print(f"[FAIL] {HIP_LIB} not found (candidates: {_cands})", flush=True)
    sys.exit(1)

lib = ctypes.CDLL(HIP_LIB)
print(f"[0] C ABI loaded from {HIP_LIB}", flush=True)

dev = torch.device("cuda")
print(f"    GPU: {torch.cuda.get_device_name(0)}", flush=True)

PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond: PASS += 1
    else: FAIL += 1
    print(f"  {status} {name} {detail}", flush=True)

def bf16_np(arr_f32):
    u32 = np.array(arr_f32, dtype=np.float32).view(np.uint32)
    rounded = u32 + 0x7FFF + ((u32 >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)

# Replicate the C++ test's rnd() + e2m1_nibble() for reproducibility
def rnd(seed, i):
    h = np.uint32((seed * 2654435761 + i * 40503) & 0xFFFFFFFF)
    h = np.uint32(h ^ (h >> np.uint32(13)))
    h = np.uint32(h * np.uint32(2654435761))
    h = np.uint32(h ^ (h >> np.uint32(15)))
    return float(np.uint32(h & np.uint32(0x7FFF))) / 0x7FFF

E2M1_VALS = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
def e2m1_nibble(v):
    a = abs(v)
    best = int(np.argmin(np.abs(E2M1_VALS - a)))
    return best | (8 if v < 0 else 0)

def pack_e2m1(seed, size):
    """Pack size/2 bytes of E2M1 nibbles, matching the C++ test."""
    out = np.zeros(size, dtype=np.uint8)
    for i in range(size // 2):
        v0 = rnd(seed, 2 * i) * 0.5
        v1 = rnd(seed + 1, 2 * i + 1) * 0.5
        out[2 * i] = e2m1_nibble(v0) | (e2m1_nibble(v1) << 4)
        v2 = rnd(seed + 2, i) * 0.5
        v3 = rnd(seed + 3, i) * 0.5
        out[2 * i + 1] = e2m1_nibble(v2) | (e2m1_nibble(v3) << 4)
    return out

# ---------------------------------------------------------------------------
# 1. vk_fused_moe_mxfp4 (SwiGLU + SiTU) — verify non-trivial output
# ---------------------------------------------------------------------------
print("[1] vk_fused_moe_mxfp4", flush=True)

for act_name, activation in [("swiglu", 0), ("situ", 1)]:
    np.random.seed(42)
    M, hidden, ispp, top_k = 8, 256, 128, 2
    E, BLOCK_M, group_size, LIMIT = 4, 16, 32, 4.0

    # Same data as C++ test_moe_fused_correct.hip (replicate exactly)
    hA = np.array([bf16_np([rnd(1, i) * 0.1])[0] for i in range(M * hidden)], dtype=np.uint16)
    hw13 = pack_e2m1(2, E * 2 * ispp * (hidden // 2))
    hw13s = np.array([125 + (i % 5) for i in range(E * 2 * ispp * (hidden // group_size))], dtype=np.uint8)
    hw2 = pack_e2m1(6, E * hidden * (ispp // 2))
    hw2s = np.array([125 + (i % 5) for i in range(E * hidden * (ispp // group_size))], dtype=np.uint8)
    h_tk_ids = np.array([[i % E, (i + 1) % E] for i in range(M)], dtype=np.int32).reshape(-1)
    h_tk_w = np.array([[0.7, 0.3] for _ in range(M)], dtype=np.float32).reshape(-1)

    # moe_align_block_size (pure host, matching vkernels C++ impl exactly)
    N = M * top_k  # total flat indices; also the pad value for out-of-bounds
    per_expert = [[] for _ in range(E)]
    for i in range(N):
        per_expert[h_tk_ids[i]].append(i)
    EM = sum(((len(v) + BLOCK_M - 1) // BLOCK_M) * BLOCK_M for v in per_expert)
    sids = np.zeros(EM, dtype=np.int32)
    eids = np.zeros(EM // BLOCK_M, dtype=np.int32)
    idx = 0
    for e in range(E):
        for i in per_expert[e]: sids[idx] = i; idx += 1
        padded_nt = ((len(per_expert[e]) + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        for _ in range(len(per_expert[e]), padded_nt): sids[idx] = N; idx += 1
    idx = 0
    for e in range(E):
        nt = len(per_expert[e])
        padded_blocks = (nt + BLOCK_M - 1) // BLOCK_M
        for b in range(padded_blocks):
            eids[idx] = e if (b * BLOCK_M < nt) else -1; idx += 1

    # Transfer to device — HIP kernel takes RAW topk_ids/topk_w, plus sorted_ids/expert_ids
    dA = torch.from_numpy(hA.astype(np.int16)).to(dev).view(torch.bfloat16)
    dw13 = torch.from_numpy(hw13).to(dev); dw13s = torch.from_numpy(hw13s).to(dev)
    dw2 = torch.from_numpy(hw2).to(dev); dw2s = torch.from_numpy(hw2s).to(dev)
    dTids = torch.from_numpy(h_tk_ids).to(dev)   # raw [M*top_k] expert IDs
    dTw = torch.from_numpy(h_tk_w).to(dev)        # raw [M*top_k] routing weights
    dSids = torch.from_numpy(sids[:EM].copy()).to(dev)
    dEids = torch.from_numpy(eids[:EM // BLOCK_M].copy()).to(dev)
    # Biases (matching C++ test: rnd(12, i)*0.3f, rnd(13, i)*0.3f)
    hb13 = np.array([rnd(12, i) * 0.3 for i in range(E * 2 * ispp)], dtype=np.float32)
    hb2 = np.array([rnd(13, i) * 0.3 for i in range(E * hidden)], dtype=np.float32)
    db13 = torch.from_numpy(hb13).to(dev)
    db2 = torch.from_numpy(hb2).to(dev)
    dact = torch.zeros(EM * ispp, dtype=torch.bfloat16).to(dev)
    dout = torch.zeros(M * hidden, dtype=torch.float32).to(dev)

    # PR #44 names the device C ABI vk_hip_* (namespaced away from the CPU
    # reference vk_* in capi.hpp); older local builds exported the bare name.
    fused_moe = getattr(lib, "vk_hip_fused_moe_mxfp4", None) or getattr(lib, "vk_fused_moe_mxfp4", None)
    fused_moe(
        ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dw13.data_ptr()),
        ctypes.c_void_p(dw13s.data_ptr()), ctypes.c_void_p(dw2.data_ptr()),
        ctypes.c_void_p(dw2s.data_ptr()),
        ctypes.c_void_p(dTids.data_ptr()), ctypes.c_void_p(dTw.data_ptr()),
        ctypes.c_void_p(dact.data_ptr()), ctypes.c_void_p(dout.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(hidden), ctypes.c_int(ispp),
        ctypes.c_int(top_k),
        ctypes.c_void_p(dSids.data_ptr()), ctypes.c_void_p(dEids.data_ptr()),
        ctypes.c_int(EM), ctypes.c_float(LIMIT),
        ctypes.c_int(activation), ctypes.c_float(4.0), ctypes.c_float(25.0),
        ctypes.c_void_p(db13.data_ptr()), ctypes.c_void_p(db2.data_ptr()),
        ctypes.c_int(BLOCK_M))
    torch.cuda.synchronize()

    out_np = dout.cpu().numpy()
    out_abs = float(np.max(np.abs(out_np[~np.isnan(out_np)]))) if not np.all(np.isnan(out_np)) else float('nan')
    out_nonzero = int(np.count_nonzero(out_np))
    n_nan = int(np.count_nonzero(np.isnan(out_np)))
    print(f"    EM={EM} sids[:8]={sids[:8]} eids[:4]={eids[:4]}", flush=True)
    print(f"    out[:8]={out_np[:8]} nan={n_nan}/{out_np.size}", flush=True)
    check(f"fused_moe ({act_name})", (n_nan == 0) and (out_abs > 1e-6),
          f"max_abs={out_abs:.6f} nan={n_nan}/{out_np.size}")

# ---------------------------------------------------------------------------
# 2. vk_mla_fwd
# ---------------------------------------------------------------------------
print("[2] vk_mla_fwd", flush=True)
np.random.seed(99)
B, H, S_q, S_kv = 1, 4, 16, 16
lr, rhd = 64, 32
scale = 1.0 / np.sqrt(lr + rhd)
q = np.random.randn(B, H, S_q, lr + rhd).astype(np.float32)
k_c = np.random.randn(B, S_kv, lr).astype(np.float32)
k_pe = np.random.randn(B, S_kv, rhd).astype(np.float32)
v_c = np.random.randn(B, S_kv, lr).astype(np.float32)

dq, dk_c, dk_pe, dv_c = [torch.from_numpy(x).to(dev) for x in (q, k_c, k_pe, v_c)]
dout = torch.zeros(B * H * S_q * lr, dtype=torch.float32).to(dev)
mla_fwd = getattr(lib, "vk_hip_mla_fwd", None) or getattr(lib, "vk_mla_fwd", None)
mla_fwd(ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(S_q), ctypes.c_int(S_kv),
    ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(lr), ctypes.c_int(rhd),
    ctypes.c_float(scale), ctypes.c_void_p(dq.data_ptr()), ctypes.c_void_p(dk_c.data_ptr()),
    ctypes.c_void_p(dk_pe.data_ptr()), ctypes.c_void_p(dv_c.data_ptr()),
    ctypes.c_void_p(dout.data_ptr()))
torch.cuda.synchronize()
out_np = dout.cpu().numpy()
out_abs = float(np.max(np.abs(out_np)))
check("vk_mla_fwd", (out_abs > 1e-6) and (out_abs < 1e4) and not np.any(np.isnan(out_np)),
      f"max_abs={out_abs:.6f} nnz={np.count_nonzero(out_np)}/{out_np.size}")

# ---------------------------------------------------------------------------
# 3. vk_kda_delta_rule_fwd
# ---------------------------------------------------------------------------
print("[3] vk_kda_delta_rule_fwd", flush=True)
np.random.seed(77)
B, H, S, D = 1, 4, 64, 32
q = (np.random.randn(B, H, S, D) * 0.1).astype(np.float32)
k = (np.random.randn(B, H, S, D) * 0.1).astype(np.float32)
v = (np.random.randn(B, H, S, D) * 0.1).astype(np.float32)
g = np.random.uniform(0.1, 0.9, (B, H, S, 1)).astype(np.float32)
beta = np.full((B, H, S, 1), 4.0, dtype=np.float32)

dq, dk, dv, dg, dbeta = [torch.from_numpy(x).to(dev) for x in (q, k, v, g, beta)]
dout = torch.zeros(B * H * S * D, dtype=torch.float32).to(dev)
kda_fwd = getattr(lib, "vk_hip_kda_delta_rule_fwd", None) or getattr(lib, "vk_kda_delta_rule_fwd", None)
kda_fwd(ctypes.c_void_p(dq.data_ptr()), ctypes.c_void_p(dk.data_ptr()),
    ctypes.c_void_p(dv.data_ptr()), ctypes.c_void_p(dg.data_ptr()),
    ctypes.c_void_p(dbeta.data_ptr()), ctypes.c_void_p(dout.data_ptr()),
    ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(S), ctypes.c_int(D), ctypes.c_int(64))
torch.cuda.synchronize()
out_np = dout.cpu().numpy()
out_abs = float(np.max(np.abs(out_np)))
check("vk_kda_delta_rule_fwd", (out_abs > 1e-6) and (out_abs < 1e4) and not np.any(np.isnan(out_np)),
      f"max_abs={out_abs:.6f} nnz={np.count_nonzero(out_np)}/{out_np.size}")

print(f"\n[done] {PASS} passed, {FAIL} failed", flush=True)
sys.exit(1 if FAIL else 0)
