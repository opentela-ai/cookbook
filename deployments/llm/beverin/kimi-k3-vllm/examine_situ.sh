#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== FusedMoEConfig: search for activation_situ ==="
grep -n "activation_situ" "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null | head -10
echo
echo "=== Kimi-K3 model: search for SiTU beta ==="
find "$V" -name "*.py" -path "*kimi*" 2>/dev/null | head -5
grep -rn "situ_beta\|situ_linear_beta\|MoEActivation.SITU\|activation.*situ" "$V/models/" 2>/dev/null | grep -i kimi | head -10
echo
echo "=== FusedMoEConfig: activation_situ_beta property or attribute ==="
grep -n "activation_situ_beta\|activation_situ_linear_beta\|situ_beta\|situ_linear" "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null | head -10
echo
echo "=== FusedMoEConfig full fields (lines 1274-1340) ==="
sed -n '1274,1340p' "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null
