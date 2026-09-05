#!/bin/bash
# Find where KimiK3DeltaAttention actually lives in the .sqsh image's vLLM.
set -uo pipefail
T=/capstor/scratch/cscs/xyao/vk-attn-test
export PYTHONPATH="$T/pylib:${PYTHONPATH:-}"
export K3="/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin"
export VKERNELS_DIR="/capstor/scratch/cscs/xyao/vkernels"
export VKERNELS_LIB="$VKERNELS_DIR/build/hip/src/c/libvkernels_hip.so"

python3 - <<'PY'
import os, vllm
vd = os.path.dirname(vllm.__file__)
print("vllm.__file__:", vllm.__file__, "v:", getattr(vllm, "__version__", "?"))
print("=== kimi_k3 package layout ===")
k3root = os.path.join(vd, "models", "kimi_k3")
for root, dirs, files in os.walk(k3root):
    for f in files:
        if f.endswith(".py"):
            print(os.path.relpath(os.path.join(root, f), vd))
print("=== grep for KDA / DeltaAttention / GatedDelta across vllm ===")
import subprocess
for pat in ("class KimiK3DeltaAttention", "class GatedDelta", "KimiK3DeltaAttention",
            "is_kda_layer", "KDA", "delta_rule"):
    r = subprocess.run(["grep", "-rIl", pat, vd], capture_output=True, text=True)
    hits = [x for x in r.stdout.split() if x.endswith(".py")]
    print(f"\n--- {pat!r} -> {len(hits)} file(s) ---")
    for h in hits[:12]:
        try:
            print("  ", os.path.relpath(h, vd))
        except Exception:
            print("  ", h)
PY
