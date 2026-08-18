#!/bin/bash
# Test VkernelFusedExperts import + backend registration inside K3 container
set -e
export K3="${K3:-/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm}"
export VKERNELS_DIR="${VKERNELS_DIR:-/capstor/scratch/cscs/xyao/vkernels}"

echo "=== K3=$K3 ==="
echo "=== VKERNELS_DIR=$VKERNELS_DIR ==="
echo "=== PYTHONPATH=${PYTHONPATH:-} ==="

echo "=== Find libvkernels_hip.so ==="
ls -la "$VKERNELS_DIR/build/hip/src/c/libvkernels_hip.so" 2>/dev/null || echo "NOT FOUND at expected path"

echo "=== Test VkernelFusedExperts import ==="
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.environ['K3'], 'home/pylib'))
sys.path.insert(0, os.environ['K3'])  # also try K3 root
# Try current dir as fallback
sys.path.insert(0, '/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm')
from vkernels_experts import VkernelFusedExperts, _find_libvkernels_hip
lib = _find_libvkernels_hip()
print(f'libvkernels_hip.so: {lib}')
print(f'VkernelFusedExperts: {VkernelFusedExperts}')
print(f'  base: {VkernelFusedExperts.__bases__}')
print(f'  _supports_current_device: {VkernelFusedExperts._supports_current_device()}')
from vllm.model_executor.layers.fused_moe.config import MoEActivation
print(f'  _supports_activation(SITU): {VkernelFusedExperts._supports_activation(MoEActivation.SITU)}')
print(f'  activation_format: {VkernelFusedExperts.activation_format()}')
print('IMPORT OK')
"

echo "=== Test sitecustomize backend_to_kernel_cls patch ==="
PYTHONPATH="$K3/home/pylib:$K3:${PYTHONPATH:-}" python3 -c "
import sys, os
os.environ['K3'] = os.environ.get('K3', '')
os.environ['VKERNELS_DIR'] = os.environ.get('VKERNELS_DIR', '')
# Import sitecustomize (auto-imported if on path, but force it)
import importlib
try:
    sc = importlib.import_module('sitecustomize')
except Exception as e:
    print(f'sitecustomize import error: {e}')

from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle
result = oracle.backend_to_kernel_cls(oracle.Mxfp4MoeBackend.TRITON_UNFUSED)
print(f'backend_to_kernel_cls(TRITON_UNFUSED): {result}')
if result and result[0].__name__ == 'VkernelFusedExperts':
    print('BACKEND REGISTRATION OK')
else:
    print(f'BACKEND NOT REGISTERED (got {result})')
"
