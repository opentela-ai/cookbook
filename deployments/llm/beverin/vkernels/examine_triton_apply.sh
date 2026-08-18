#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== TritonExperts.apply ==="
sed -n '202,310p' "$V/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null
echo
echo "=== workspace_shapes abstract ==="
sed -n '850,900p' "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null
echo
echo "=== select_deepseek_v4_mxfp4_moe_backend return ==="
sed -n '620,700p' "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null
