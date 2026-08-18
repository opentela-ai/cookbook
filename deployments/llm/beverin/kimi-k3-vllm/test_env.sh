#!/bin/bash
echo "=== gfx942 code in library ==="
strings /capstor/scratch/cscs/xyao/vkernels/build/hip/src/c/libvkernels_hip.so | grep -i "gfx942\|gfx900\|amdgcn" | head -10
echo
echo "=== torch version ==="
python3 -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('hip:', torch.version.hip)"
echo
echo "=== Simple torch CUDA test ==="
python3 -c "
import torch
x = torch.randn(8, 256, device='cuda', dtype=torch.bfloat16)
y = x @ x.T
torch.cuda.synchronize()
print('torch CUDA works:', y.shape, 'max=', y.abs().max().item())
"
echo
echo "=== Device info ==="
python3 -c "
import torch
print('Device:', torch.cuda.get_device_name(0))
print('Arch:', torch.cuda.get_device_properties(0))
"
