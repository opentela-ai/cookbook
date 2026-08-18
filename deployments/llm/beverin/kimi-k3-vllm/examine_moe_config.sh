#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== FusedMoEConfig __init__ signature ==="
grep -n "class FusedMoEConfig" "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null | head -3
sed -n "$(grep -n 'class FusedMoEConfig' "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null | head -1 | cut -d: -f1),+40p" "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null
