#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== UnfusedOAITritonExperts.apply() ==="
sed -n '1152,1310p' "$V/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py" 2>/dev/null
