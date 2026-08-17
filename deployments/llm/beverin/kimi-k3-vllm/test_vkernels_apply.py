#!/usr/bin/env python3
"""End-to-end test of VkernelFusedExperts.apply() through the vLLM interface.

Uses the same synthetic data as test_hip_bindings.py (which validated the
C ABI directly) to verify that VkernelFusedExperts correctly maps vLLM's
tensor format to the C ABI's expected format.
"""
import sys
import os
import ctypes
import numpy as np
import torch

_K3 = os.environ.get("K3", "/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm")
sys.path.insert(0, _K3)
sys.path.insert(0, os.path.join(_K3, "home/pylib"))

from vllm.model_executor.layers.fused_moe.config import MoEActivation
from vkernels_experts import VkernelFusedExperts, _get_lib, _moe_align_block_size_cpu

# Replicate the C++ test's rnd() + e2m1_nibble() (same as test_hip_bindings.py)
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

def bf16_np(arr_f32):
    u32 = np.array(arr_f32, dtype=np.float32).view(np.uint32)
    rounded = u32 + 0x7FFF + ((u32 >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)

# ---------------------------------------------------------------------------
# Create synthetic data (same as test_hip_bindings.py, SwiGLU)
# ---------------------------------------------------------------------------
np.random.seed(42)
M, hidden, ispp, top_k = 8, 256, 128, 2
E, BLOCK_M, group_size, LIMIT = 4, 16, 32, 4.0
activation = 0  # SwiGLU

# Weights and scales
hw13 = pack_e2m1(2, E * 2 * ispp * (hidden // 2))
hw13s = np.array([125 + (i % 5) for i in range(E * 2 * ispp * (hidden // group_size))], dtype=np.uint8)
hw2 = pack_e2m1(6, E * hidden * (ispp // 2))
hw2s = np.array([125 + (i % 5) for i in range(E * hidden * (ispp // group_size))], dtype=np.uint8)
h_tk_ids = np.array([[i % E, (i + 1) % E] for i in range(M)], dtype=np.int32).reshape(-1)
h_tk_w = np.array([[0.7, 0.3] for _ in range(M)], dtype=np.float32).reshape(-1)
hA = np.array([bf16_np([rnd(1, i) * 0.1])[0] for i in range(M * hidden)], dtype=np.uint16)
hb13 = np.array([rnd(12, i) * 0.3 for i in range(E * 2 * ispp)], dtype=np.float32)
hb2 = np.array([rnd(13, i) * 0.3 for i in range(E * hidden)], dtype=np.float32)

# Move to GPU
dev = torch.device("cuda")
dA = torch.from_numpy(hA.astype(np.int16)).to(dev).view(torch.bfloat16).reshape(M, hidden)
dw13 = torch.from_numpy(hw13).to(dev).reshape(E, 2*ispp, hidden//2)
dw13s = torch.from_numpy(hw13s).to(dev).reshape(E, 2*ispp, hidden//group_size)
dw2 = torch.from_numpy(hw2).to(dev).reshape(E, hidden, ispp//2)
dw2s = torch.from_numpy(hw2s).to(dev).reshape(E, hidden, ispp//group_size)
dTkIds = torch.from_numpy(h_tk_ids.reshape(M, top_k)).to(dev)
dTkW = torch.from_numpy(h_tk_w.reshape(M, top_k)).to(dev)
db13 = torch.from_numpy(hb13).to(dev)
db2 = torch.from_numpy(hb2).to(dev)

# ---------------------------------------------------------------------------
# Create VkernelFusedExperts instance (bypass __init__)
# ---------------------------------------------------------------------------
inst = VkernelFusedExperts.__new__(VkernelFusedExperts)

# Create a mock quant_config with the needed properties
class MockQuantConfig:
    w1_scale = dw13s
    w2_scale = dw2s
    w1_bias = db13
    w2_bias = db2
    gemm1_clamp_limit = LIMIT
    gemm1_alpha = 1.0
    gemm1_beta = 0.0
    quant_dtype = None
    weight_quant_dtype = None
    block_shape = [1, 32]
    per_act_token_quant = False
    per_out_ch_quant = False
    use_int4_w4a16 = True
    use_int8_w8a16 = False
    use_fp8_w8a8 = False
    use_fp8_w8a16 = False
    config_name = lambda self, dtype: "mxfp4_static"
    has_bias = True

class MockMoeConfig:
    activation = MoEActivation.SILU  # SwiGLU
    activation_situ_beta = None
    activation_situ_linear_beta = None
    experts_per_token = top_k
    num_experts = E
    hidden_dim = hidden
    intermediate_size = ispp * 2
    is_lora_enabled = False
    routing_method = None
    moe_parallel_config = None

inst.quant_config = MockQuantConfig()
inst.moe_config = MockMoeConfig()
inst.gemm1_clamp_limit = LIMIT
inst.gemm1_alpha = 1.0
inst.gemm1_beta = 0.0
inst._lora_context = None

# ---------------------------------------------------------------------------
# Call apply()
# ---------------------------------------------------------------------------
output = torch.zeros(M, hidden, dtype=torch.bfloat16, device=dev)
# workspace13: [EM_max, ispp] bf16
EM_max = M * top_k + E
EM_max = ((EM_max + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
workspace13 = torch.zeros(EM_max, ispp, dtype=torch.bfloat16, device=dev)
workspace2 = torch.zeros(0, dtype=torch.bfloat16, device=dev)

print("=== Calling VkernelFusedExperts.apply() ===")
try:
    inst.apply(
        output=output,
        hidden_states=dA,
        w1=dw13,
        w2=dw2,
        topk_weights=dTkW,
        topk_ids=dTkIds,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    print("apply() returned without error")
except Exception as e:
    print(f"apply() FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check output
out_np = output.cpu().to(torch.float32).numpy()
n_nan = int(np.count_nonzero(np.isnan(out_np)))
n_zero = int(np.count_nonzero(out_np == 0))
max_abs = float(np.max(np.abs(out_np[~np.isnan(out_np)]))) if not np.all(np.isnan(out_np)) else float('nan')
print(f"  output shape: {out_np.shape}")
print(f"  nan: {n_nan}/{out_np.size}")
print(f"  zero: {n_zero}/{out_np.size}")
print(f"  max_abs: {max_abs}")
print(f"  out[:4]: {out_np[0, :4]}")

if n_nan == 0 and max_abs > 1e-6:
    print("\nPASS: VkernelFusedExperts.apply() produces valid output")
else:
    print(f"\nFAIL: output has {n_nan} NaN, max_abs={max_abs}")

# ---------------------------------------------------------------------------
# Also test SiTU activation
# ---------------------------------------------------------------------------
print("\n=== Testing SiTU activation ===")
output_situ = torch.zeros(M, hidden, dtype=torch.bfloat16, device=dev)
inst.moe_config.activation = MoEActivation.SITU
try:
    inst.apply(
        output=output_situ,
        hidden_states=dA,
        w1=dw13,
        w2=dw2,
        topk_weights=dTkW,
        topk_ids=dTkIds,
        activation=MoEActivation.SITU,
        global_num_experts=E,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    out_situ = output_situ.cpu().to(torch.float32).numpy()
    n_nan_s = int(np.count_nonzero(np.isnan(out_situ)))
    max_abs_s = float(np.max(np.abs(out_situ[~np.isnan(out_situ)]))) if not np.all(np.isnan(out_situ)) else float('nan')
    print(f"  nan: {n_nan_s}/{out_situ.size}, max_abs: {max_abs_s}")
    if n_nan_s == 0 and max_abs_s > 1e-6:
        print("  PASS: SiTU activation works")
    else:
        print(f"  FAIL: SiTU has {n_nan_s} NaN")
except Exception as e:
    print(f"  SiTU FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nDONE")
