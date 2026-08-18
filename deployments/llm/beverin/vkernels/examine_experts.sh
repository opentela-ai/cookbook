#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== experts dir ==="
ls "$V/model_executor/layers/fused_moe/experts/" 2>/dev/null
echo
echo "=== FusedMoEExperts base class ==="
grep -n "class FusedMoEExperts" "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null | head -3
echo
echo "=== modular_kernel.py FusedMoEExperts signature ==="
grep -n -A30 "class FusedMoEExperts" "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null | head -40
echo
echo "=== UNFUSED experts implementation ==="
grep -n "class.*Unfused\|def apply\|_supports_activation\|activation" "$V/model_executor/layers/fused_moe/experts/unfused.py" 2>/dev/null | head -20
echo
echo "=== AITER experts ==="
grep -rn "class.*Aiter\|class.*ATITER.*Experts\|_supports_activation" "$V/model_executor/layers/fused_moe/experts/" 2>/dev/null | head -10
echo
echo "=== activation.py MoEActivation enum ==="
grep -n "class MoEActivation\|SITU\|SITU_AND_MUL" "$V/model_executor/layers/fused_moe/activation.py" 2>/dev/null | head -10
