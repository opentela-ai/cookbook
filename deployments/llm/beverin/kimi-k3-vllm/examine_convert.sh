#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== convert_gpt_oss_weight_to_mxfp4_moe_kernel_format ==="
grep -n "def convert_gpt_oss_weight_to_mxfp4_moe_kernel_format" "$V/model_executor/layers/fused_moe/quant/mxfp4.py"
echo
echo "=== Body (first 80 lines after def) ==="
LINE=$(grep -n "def convert_gpt_oss_weight_to_mxfp4_moe_kernel_format" "$V/model_executor/layers/fused_moe/quant/mxfp4.py" | head -1 | cut -d: -f1)
if [ -n "$LINE" ]; then
    sed -n "${LINE},$((LINE+80))p" "$V/model_executor/layers/fused_moe/quant/mxfp4.py"
fi
