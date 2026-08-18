#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== BaseOAITritonExperts._supports_quant_scheme ==="
sed -n '900,930p' "$V/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py" 2>/dev/null
echo
echo "=== Mxfp4MoEMethod weight loading (convert call) ==="
grep -n "convert_gpt_oss_weight\|process_weights_after_loading\|def select" "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null | head -15
echo
echo "=== convert_gpt_oss_weight: TRITON/TRITON_UNFUSED branch ==="
sed -n '700,760p' "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null
