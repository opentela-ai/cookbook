#!/usr/bin/env python3
"""Test if importing vLLM modules before calling the C ABI causes the crash."""
import ctypes
import os
import sys
import numpy as np
import torch

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

# Same data as test_hip_bindings.py
np.random.seed(42)
M, hidden, ispp, top_k = 8, 256, 128, 2
E, BLOCK_M, group_size, LIMIT = 4, 16, 32, 4.0

hA = np.array([bf16_np([rnd(1, i) * 0.1])[0] for i in range(M * hidden)], dtype=np.uint16)
hw13 = pack_e2m1(2, E * 2 * ispp * (hidden // 2))
hw13s = np.array([125 + (i % 5) for i in range(E * 2 * ispp * (hidden // group_size))], dtype=np.uint8)
hw2 = pack_e2m1(6, E * hidden * (ispp // 2))
hw2s = np.array([125 + (i % 5) for i in range(E * hidden * (ispp // group_size))], dtype=np.uint8)
h_tk_ids = np.array([[i % E, (i + 1) % E] for i in range(M)], dtype=np.int32).reshape(-1)
h_tk_w = np.array([[0.7, 0.3] for _ in range(M)], dtype=np.float32).reshape(-1)

N = M * top_k
per_expert = [[] for _ in range(E)]
for i in range(N): per_expert[h_tk_ids[i]].append(i)
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

HIP_LIB = "/capstor/scratch/cscs/xyao/vkernels/build/hip/src/c/libvkernels_hip.so"

# Step 1: Load library and call WITHOUT any vLLM imports
print("[step1] Loading library and calling WITHOUT vLLM imports", flush=True)
lib = ctypes.CDLL(HIP_LIB)
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
    ctypes.c_int(0), ctypes.c_float(4.0), ctypes.c_float(25.0),
    ctypes.c_void_p(db13.data_ptr()), ctypes.c_void_p(db2.data_ptr()),
    ctypes.c_int(BLOCK_M))
torch.cuda.synchronize()
print(f"[step1] PASSED: max_abs={np.max(np.abs(dout.cpu().numpy())):.6f}", flush=True)

# Step 2: Now import vLLM modules
print("[step2] Importing vLLM modules...", flush=True)
from vllm.model_executor.layers.fused_moe.config import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    UnfusedOAITritonExperts,
)
from vllm.platforms import current_platform
print(f"[step2] Imports done. current_platform={current_platform}", flush=True)

# Step 3: Call the SAME kernel again with the SAME data
print("[step3] Calling same kernel again AFTER vLLM imports", flush=True)
dout2 = torch.zeros(M * hidden, dtype=torch.float32).to(dev)
lib.vk_fused_moe_mxfp4(
    ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dw13.data_ptr()),
    ctypes.c_void_p(dw13s.data_ptr()), ctypes.c_void_p(dw2.data_ptr()),
    ctypes.c_void_p(dw2s.data_ptr()),
    ctypes.c_void_p(dTids.data_ptr()), ctypes.c_void_p(dTw.data_ptr()),
    ctypes.c_void_p(dact.data_ptr()), ctypes.c_void_p(dout2.data_ptr()),
    ctypes.c_int(M), ctypes.c_int(hidden), ctypes.c_int(ispp),
    ctypes.c_int(top_k),
    ctypes.c_void_p(dSids.data_ptr()), ctypes.c_void_p(dEids.data_ptr()),
    ctypes.c_int(EM), ctypes.c_float(LIMIT),
    ctypes.c_int(0), ctypes.c_float(4.0), ctypes.c_float(25.0),
    ctypes.c_void_p(db13.data_ptr()), ctypes.c_void_p(db2.data_ptr()),
    ctypes.c_int(BLOCK_M))
torch.cuda.synchronize()
print(f"[step3] PASSED: max_abs={np.max(np.abs(dout2.cpu().numpy())):.6f}", flush=True)

print("[done] ALL STEPS PASSED")
