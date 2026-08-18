#!/bin/bash
V=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== FusedMoEQuantConfig w1/w2/bias properties ==="
sed -n '330,400p' "$V/model_executor/layers/fused_moe/config.py" 2>/dev/null
echo
echo "=== TritonExperts.apply: how scales accessed ==="
grep -n "self.w1_scale\|self.w2_scale\|self.quant_config\|self.w13\|self.w2_bias\|self.w1_bias\|self.a1_scale\|self.a2_scale\|quant_config\." "$V/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null | head -25
echo
echo "=== TritonExperts.__init__ full ==="
sed -n '57,90p' "$V/model_executor/layers/fused_moe/experts/triton_moe.py" 2>/dev/null
