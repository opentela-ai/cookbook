#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== mxfp4.py: classes ==="
grep -n "^class\|def create_weights\|def process_weights\|class Mxfp4MoEMethod\|class.*Experts" "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null | head -20
echo
echo "=== convert_gpt_oss_weight format ==="
grep -n -A30 "def convert_gpt_oss_weight" "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null | head -40
echo
echo "=== k3_patch.py: MoE rectification (existing) ==="
grep -n "convert_gpt_oss\|create_weights\|Mxfp4\|w13_weight\|w2_weight\|w13_scale\|w2_scale\|SITU\|situ" deployments/llm/beverin/kimi-k3-vllm/k3_patch.py 2>/dev/null | head -20
