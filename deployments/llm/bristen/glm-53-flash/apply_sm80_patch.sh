#!/bin/bash
# Apply SM80 (A100) FP8 compute patches to a copy of the Beverin GLM-5.3
# SGLang overlay.  The patched tree is placed under $DEPLOY_DIR/patches_full
# and should be prepended to PYTHONPATH.
set -euo pipefail

: "${DEPLOY_DIR:=/capstor/scratch/cscs/xyao/glm-53-flash-bristen}"
: "${OVL:=/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay/sgl-workspace/sglang/python}"
PATCH_DIR="$DEPLOY_DIR/patches_full"
SRC_DIR="$OVL/sglang"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$PATCH_DIR"

if [ ! -f "$SRC_DIR/__init__.py" ]; then
    echo "FATAL: overlay source not found: $SRC_DIR" >&2
    exit 1
fi

echo "[$(date -Is)] copying overlay sglang tree to $PATCH_DIR ..."
rm -rf "$PATCH_DIR/sglang"
cp -a "$SRC_DIR" "$PATCH_DIR/sglang"

echo "[$(date -Is)] applying SM80 FP8 compute patched sources ..."
cp "$SCRIPT_DIR/patched_sources/sglang/kernels/ops/quantization/fp8_kernel.py" \
   "$PATCH_DIR/sglang/kernels/ops/quantization/fp8_kernel.py"
cp "$SCRIPT_DIR/patched_sources/sglang/kernels/ops/moe/fused_moe_triton_kernels.py" \
   "$PATCH_DIR/sglang/kernels/ops/moe/fused_moe_triton_kernels.py"

echo "[$(date -Is)] SM80 patch tree ready at $PATCH_DIR/sglang"
