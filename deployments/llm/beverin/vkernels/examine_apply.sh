#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== FusedMoEExpertsModular.apply signature ==="
sed -n '763,850p' "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null
echo
echo "=== TritonExperts full class ==="
sed -n '57,95p' "$V/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null
echo
echo "=== TritonExperts._supports_activation ==="
sed -n '126,150p' "$V/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null
