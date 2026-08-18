#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== triton_moe.py: classes and key methods ==="
grep -n "^class\|def apply\|def create_weights\|def _supports\|def workspace_shapes" "$V/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null | head -20
echo
echo "=== FusedMoEExperts abstract methods ==="
sed -n '472,570p' "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null
echo
echo "=== FusedMoEExpertsModular apply signature ==="
grep -n "def apply" "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null | head -10
