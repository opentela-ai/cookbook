#!/usr/bin/env python3
"""Test VkernelFusedExperts.is_supported_config with a realistic K3 config.

Must be run with PYTHONPATH including $K3/home/pylib so sitecustomize.py
is auto-imported at startup (which patches _get_priority_backends and
backend_to_kernel_cls).
"""
import sys
import os

import torch
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    MoEActivation,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    kMxfp4Static,
    _get_priority_backends,
    backend_to_kernel_cls,
)
from vkernels_experts import VkernelFusedExperts, _get_lib, _find_libvkernels_hip

from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as _oracle_mod

print("=== _supports_quant_scheme ===")
print(f"  (kMxfp4Static, None): {VkernelFusedExperts._supports_quant_scheme(kMxfp4Static, None)}")

print("\n=== Priority backends (after sitecustomize patch) ===")
backends = _get_priority_backends()
print(f"  {backends}")

print("\n=== backend_to_kernel_cls for VKERNELS_MXFP4_BF16 ===")
try:
    cls_list = backend_to_kernel_cls(_oracle_mod.Mxfp4MoeBackend.VKERNELS_MXFP4_BF16)
    print(f"  VKERNELS_MXFP4_BF16 -> {[c.__name__ for c in cls_list]}")
except Exception as e:
    print(f"  VKERNELS_MXFP4_BF16 -> ERROR: {e}")

print("\n=== is_supported_config for each backend ===")
mpc = FusedMoEParallelConfig(
    tp_size=8, pcp_size=1, dp_size=1, ep_size=1,
    tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
    sp_size=1, use_ep=False, all2all_backend=None, enable_eplb=False,
)
moe_config = FusedMoEConfig(
    num_experts=256,
    experts_per_token=8,
    hidden_dim=5120,
    intermediate_size=1536,
    num_local_experts=256,
    num_logical_experts=256,
    activation=MoEActivation.SITU,
    device=torch.device("cuda"),
    routing_method=RoutingMethodType.DeepSeekV3,
    moe_parallel_config=mpc,
    in_dtype=torch.bfloat16,
    has_bias=True,
)
activation_format = mk.FusedMoEActivationFormat.Standard

for backend in backends:
    for k_cls in backend_to_kernel_cls(backend):
        supported, reason = k_cls.is_supported_config(
            k_cls, moe_config, kMxfp4Static, None, activation_format
        )
        status = "PASS" if supported else "FAIL"
        print(f"  {status} {backend} -> {k_cls.__name__}: {reason}")

print("\n=== VkernelFusedExperts workspace_shapes ===")
inst = VkernelFusedExperts.__new__(VkernelFusedExperts)
inst.moe_config = moe_config
inst.gemm1_clamp_limit = 4.0
ws13, ws2, out = inst.workspace_shapes(
    M=8, N=2*192, K=640, topk=8,
    global_num_experts=256, local_num_experts=256,
    expert_tokens_meta=None,
    activation=MoEActivation.SITU,
)
print(f"  workspace13: {ws13} (for act_scratch [EM_max, ispp])")
print(f"  workspace2: {ws2}")
print(f"  output: {out}")

print("\n=== moe_config.activation fields ===")
print(f"  activation: {moe_config.activation}")
print(f"  activation_situ_beta: {getattr(moe_config, 'activation_situ_beta', 'N/A')}")
print(f"  activation_situ_linear_beta: {getattr(moe_config, 'activation_situ_linear_beta', 'N/A')}")

print("\nALL CHECKS DONE")
