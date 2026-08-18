#!/usr/bin/env python3
"""Minimal test: replicate test_hip_bindings.py exactly but load lib through
_get_lib() (same as VkernelFusedExperts.apply()) to isolate the crash."""
import ctypes
import os
import sys
import numpy as np
import torch

_K3 = os.environ.get("K3", "/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm")
sys.path.insert(0, _K3)
sys.path.insert(0, os.path.join(_K3, "home/pylib"))

# Load through _get_lib() — same as VkernelFusedExperts.apply()
from vkernels_experts import _get_lib, _find_libvkernels_hip

lib_path = _find_libvkernels_hip()
print(f"[test] Library found at: {lib_path}", flush=True)
lib = _get_lib()
print(f"[test] Library loaded", flush=True)

# Also load directly for comparison
lib_direct = ctypes.CDLL("/capstor/scratch/cscs/xyao/vkernels/build/hip/src/c/libvkernels_hip.so")
print(f"[test] Direct library loaded", flush=True)

dev = torch.device("cuda")

def bf16_np(arr_f32):
    u32 = np.array(arr_f32, dtype=np.float32).view(np.uint32)
    rounded = u32 + 0x7FFF + ((u32 >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)

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
    out = np.zeros(size, dtype=np.uint8)
    for i in range(size // 2):
        v0 = rnd(seed, 2 * i) * 0.5
        v1 = rnd(seed + 1, 2 * i + 1) * 0.5
        out[2 * i] = e2m1_nibble(v0) | (e2m1_nibble(v1) << 4)
        v2 = rnd(seed + 2, i) * 0.5
        v3 = rnd(seed + 3, i) * 0.5
        out[2 * i + 1] = e2m1_nibble(v2) | (e2m1_nibble(v3) << 4)
    return out

# Same data as test_hip_bindings.py (SwiGLU)
np.random.seed(42)
M, hidden, ispp, top_k = 8, 256, 128, 2
E, BLOCK_M, group_size, LIMIT = 4, 16, 32, 4.0
activation = 0  # SwiGLU

hA = np.array([bf16_np([rnd(1, i) * 0.1])[0] for i in range(M * hidden)], dtype=np.uint16)
hw13 = pack_e2m1(2, E * 2 * ispp * (hidden // 2))
hw13s = np.array([125 + (i % 5) for i in range(E * 2 * ispp * (hidden // group_size))], dtype=np.uint8)
hw2 = pack_e2m1(6, E * hidden * (ispp // 2))
hw2s = np.array([125 + (i % 5) for i in range(E * hidden * (ispp // group_size))], dtype=np.uint8)
h_tk_ids = np.array([[i % E, (i + 1) % E] for i in range(M)], dtype=np.int32).reshape(-1)
h_tk_w = np.array([[0.7, 0.3] for _ in range(M)], dtype=np.float32).reshape(-1)

# moe_align_block_size (inline, same as test_hip_bindings.py)
N = M * top_k
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

hb13 = np.array([rnd(12, i) * 0.3 for i in range(E * 2 * ispp)], dtype=np.float32)
hb2 = np.array([rnd(13, i) * 0.3 for i in range(E * hidden)], dtype=np.float32)

# Transfer to device — EXACTLY as test_hip_bindings.py (1D tensors)
dA = torch.from_numpy(hA.astype(np.int16)).to(dev).view(torch.bfloat16)
dw13 = torch.from_numpy(hw13).to(dev); dw13s = torch.from_numpy(hw13s).to(dev)
dw2 = torch.from_numpy(hw2).to(dev); dw2s = torch.from_numpy(hw2s).to(dev)
dTids = torch.from_numpy(h_tk_ids).to(dev)
dTw = torch.from_numpy(h_tk_w).to(dev)
dSids = torch.from_numpy(sids[:EM].copy()).to(dev)
dEids = torch.from_numpy(eids[:EM // BLOCK_M].copy()).to(dev)
db13 = torch.from_numpy(hb13).to(dev)
db2 = torch.from_numpy(hb2).to(dev)
dact = torch.zeros(EM * ispp, dtype=torch.bfloat16).to(dev)
dout = torch.zeros(M * hidden, dtype=torch.float32).to(dev)

print(f"[test] About to call lib.vk_fused_moe_mxfp4 (from _get_lib)", flush=True)
lib.vk_fused_moe_mxfp4(
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
print(f"[test] _get_lib call PASSED: max_abs={np.max(np.abs(dout.cpu().numpy()))}", flush=True)

# Now test with lib_direct
dout2 = torch.zeros(M * hidden, dtype=torch.float32).to(dev)
print(f"[test] About to call lib_direct.vk_fused_moe_mxfp4", flush=True)
lib_direct.vk_fused_moe_mxfp4(
    ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dw13.data_ptr()),
    ctypes.c_void_p(dw13s.data_ptr()), ctypes.c_void_p(dw2.data_ptr()),
    ctypes.c_void_p(dw2s.data_ptr()),
    ctypes.c_void_p(dTids.data_ptr()), ctypes.c_void_p(dTw.data_ptr()),
    ctypes.c_void_p(dact.data_ptr()), ctypes.c_void_p(dout2.data_ptr()),
    ctypes.c_int(M), ctypes.c_int(hidden), ctypes.c_int(ispp),
    ctypes.c_int(top_k),
    ctypes.c_void_p(dSids.data_ptr()), ctypes.c_void_p(dEids.data_ptr()),
    ctypes.c_int(EM), ctypes.c_float(LIMIT),
    ctypes.c_int(activation), ctypes.c_float(4.0), ctypes.c_float(25.0),
    ctypes.c_void_p(db13.data_ptr()), ctypes.c_void_p(db2.data_ptr()),
    ctypes.c_int(BLOCK_M))
torch.cuda.synchronize()
print(f"[test] lib_direct call PASSED: max_abs={np.max(np.abs(dout2.cpu().numpy()))}", flush=True)

print("[test] DONE")
