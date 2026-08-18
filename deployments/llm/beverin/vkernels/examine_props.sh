#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== FusedMoEExperts: w1_scale/w2_scale/w1_bias/w2_bias properties ==="
grep -n "def w1_scale\|def w2_scale\|def w1_bias\|def w2_bias\|def a1_scale\|def a2_scale\|self.w1_scale\|self.w2_scale" "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null | head -20
echo
echo "=== FusedMoEExpertsModular: full property list ==="
sed -n '550,762p' "$V/model_executor/layers/fused_moe/modular_kernel.py" 2>/dev/null
