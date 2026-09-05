#!/bin/bash
# moe_align_probe.sh — find UnfusedOAITritonExperts + the GPU moe_align_block_size kernel sig.
set -uo pipefail
PY=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== class UnfusedOAITritonExperts (which file) ==="
grep -rln "class UnfusedOAITritonExperts" "$PY" 2>/dev/null
echo
echo "=== moe_align_block_size.py: signatures + docstring hints ==="
grep -nE "def moe_align_block_size|def align_topk_ids|invoked=|sorted_ids|expert_ids|num_tokens_post_pad|@triton|out_dtype|ndarray|tl\." "$PY/model_executor/layers/fused_moe/moe_align_block_size.py" 2>/dev/null | head -30
echo
echo "=== how triton_moe / parents call moe_align_block_size (the GPU path) ==="
grep -nE "moe_align_block_size|align_topk_ids|sorted_ids|expert_ids|num_tokens_post_pad" "$PY/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null | head -15
