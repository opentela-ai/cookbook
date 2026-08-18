#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== FusedMoEQuantConfig fields ==="
grep -n "class FusedMoEQuantConfig" "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null | head -3
grep -n "w1_scale\|w2_scale\|w13_weight_scale\|w2_weight_scale\|w13_bias\|w2_bias\|weighted" "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null | head -20
echo
echo "=== mxfp4.py convert: TRITON branch ==="
grep -n -A5 "TRITON_UNFUSED\|TRITON\b.*w13\|triton.*w13" "$V/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null | head -20
echo
echo "=== Mxfp4MoEMethod.create_weights ==="
grep -n -B2 -A50 "def create_weights" "$V/model_executor/layers/quantization/mxfp4.py" 2>/dev/null | head -60
