#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== w1_scale/w2_scale/w1_bias/w2_bias on UnfusedOAITritonExperts ==="
grep -n "w1_scale\|w2_scale\|w1_bias\|w2_bias\|def.*scale\|def.*bias" "$V/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py" | head -20
echo
echo "=== Same on FusedMoEExpertsModular ==="
grep -n "w1_scale\|w2_scale\|w1_bias\|w2_bias" "$V/model_executor/layers/fused_moe/modular_kernel.py" | head -20
