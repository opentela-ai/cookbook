#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== gpt_oss_triton_kernels_moe.py: UnfusedOAITritonExperts ==="
grep -n "class UnfusedOAITritonExperts\|def apply\|def workspace\|def _supports\|def __init__" "$V/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py" 2>/dev/null | head -20
echo
echo "=== UnfusedOAITritonExperts full class ==="
sed -n "$(grep -n 'class UnfusedOAITritonExperts' "$V/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py" 2>/dev/null | head -1 | cut -d: -f1),+80p" "$V/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py" 2>/dev/null
