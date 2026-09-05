#!/bin/bash
# Runs INSIDE the kimi-k3-vllm container on one gfx942 node.
set -uo pipefail

T=/capstor/scratch/cscs/xyao/vk-attn-test
# Isolated copy of the NEW files FIRST (live $K3/home/pylib untouched).
export PYTHONPATH="$T/pylib:${PYTHONPATH:-}"
export K3="/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin"
export VKERNELS_DIR="/capstor/scratch/cscs/xyao/vkernels"
export VKERNELS_LIB="$VKERNELS_DIR/build/hip/src/c/libvkernels_hip.so"
# Exercise the registration + KDA patch paths (registration does NOT invoke
# the kernels, so the absence of the CPU oracle lib is irrelevant here).
export VKERNELS_MLA="1"
export VKERNELS_KDA="1"

echo "========== (A) GPU + gcnArchName =========="
rocm-smi --showproductname 2>/dev/null | head -12 || true
python3 - <<'PY'
import torch
print("torch:", torch.__version__, "cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gcnArchName:", p.gcnArchName, "name:", p.name)
PY

echo; echo "========== (B) vkernels_attn import + on_gfx942 + register (real fork) =========="
python3 - <<'PY'
import vkernels_attn as v
print("vkernels_attn.__file__:", v.__file__)
print("on_gfx942():", v.on_gfx942())
v.register_vkernels_attn()
print("register_vkernels_attn() returned cleanly (no exception)")
try:
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    print("KimiK3DeltaAttention._forward patched? name =",
          KimiK3DeltaAttention._forward.__name__)
except Exception as e:
    print("KDA class recheck failed:", repr(e))
PY

echo; echo "========== (C) test_hip_bindings.py: kernels EXECUTE on gfx942 =========="
python3 "$T/pylib/test_hip_bindings.py" 2>&1 | tail -45
echo "========== DONE =========="
