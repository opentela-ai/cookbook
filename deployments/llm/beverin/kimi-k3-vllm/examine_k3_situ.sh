#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== K3 model config: activation_situ ==="
find "$V/models" -name "*.py" -path "*kimi*" 2>/dev/null | while read f; do
    if grep -l "activation_situ" "$f" 2>/dev/null; then
        grep -n "activation_situ\|hidden_act\|situ_beta\|situ_linear" "$f" 2>/dev/null
        echo "---"
    fi
done
echo
echo "=== K3 linear.py: full context (190-230) ==="
sed -n '190,230p' "$V/models/kimi_k3/amd/linear.py" 2>/dev/null
echo
echo "=== K3 config files: search for situ ==="
find "$V" -name "*.json" -path "*kimi*" 2>/dev/null | while read f; do
    if grep -l "situ" "$f" 2>/dev/null; then
        grep "situ\|activation" "$f" 2>/dev/null
    fi
done
