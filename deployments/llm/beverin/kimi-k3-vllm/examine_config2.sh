#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== FusedMoEConfig: activation_situ_beta, swiglu_limit ==="
grep -n "activation_situ_beta\|activation_situ_linear_beta\|swiglu_limit\|swiglu_alpha\|gemm1_clamp\|gemm1_alpha\|gemm1_beta" "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null | head -15
echo
echo "=== FusedMoEQuantConfig: gemm1_clamp_limit ==="
grep -n "gemm1_clamp_limit\|gemm1_alpha\|gemm1_beta" "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null | head -10
