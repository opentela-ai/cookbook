#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== fused_moe directory ==="
ls "$V/model_executor/layers/fused_moe/" 2>/dev/null | head -20
echo
echo "=== oracle directory ==="
ls "$V/model_executor/layers/fused_moe/oracle/" 2>/dev/null
echo
echo "=== mxfp4 backend enum ==="
grep -n "class Mxfp4MoeBackend\|AITER\|TRITON\|UNFUSED" "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null | head -15
echo
echo "=== _get_priority_backends ==="
grep -n -A25 "_get_priority_backends" "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null | head -35
echo
echo "=== select_deepseek_v4_mxfp4_moe_backend ==="
grep -n -A30 "def select_deepseek_v4_mxfp4" "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null | head -40
echo
echo "=== Mxfp4MoEMethod class ==="
grep -n "class Mxfp4MoEMethod\|class Experts\|def create_weights\|def apply\|def _supports\|def get_fused_experts\|def get_unfused" "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null | head -15
