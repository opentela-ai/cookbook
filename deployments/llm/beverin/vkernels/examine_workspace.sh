#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== TritonExperts.workspace_shapes ==="
sed -n '185,205p' "$V/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null
echo
echo "=== _prepare_expert_assignment ==="
grep -n "def _prepare_expert_assignment" "$V/model_executor/layers/fused_moe/fused_moe.py" 2>/dev/null
sed -n "$(grep -n 'def _prepare_expert_assignment' "$V/model_executor/layers/fused_moe/fused_moe.py" 2>/dev/null | head -1 | cut -d: -f1),+40p" "$V/model_executor/layers/fused_moe/fused_moe.py" 2>/dev/null
echo
echo "=== moe_align_block_size ==="
grep -n "def moe_align_block_size\|def invoke_moe_align_block" "$V/model_executor/layers/fused_moe/moe_align_block_size.py" 2>/dev/null | head -5
sed -n "$(grep -n 'def moe_align_block_size' "$V/model_executor/layers/fused_moe/moe_align_block_size.py" 2>/dev/null | head -1 | cut -d: -f1),+30p" "$V/model_executor/layers/fused_moe/moe_align_block_size.py" 2>/dev/null
