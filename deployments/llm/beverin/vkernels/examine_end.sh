#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== TritonExperts.apply: end (routing weight + output) ==="
sed -n '490,545p' "$V/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null
echo
echo "=== FusedMoEExpertsModular.apply (abstract or base?) ==="
sed -n '900,970p' "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null
