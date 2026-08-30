#!/bin/bash
# build_overlay.sh — reproducibly build the GLM-5.3 Python overlay on Beverin.
#
# WHAT and WHY (see sglang-rocm.toml for the full design): the only image that
# ships GLM-5.3 (sglang/srt/configs/glm5_next.py, models/glm5_next{,_nextn}.py,
# srt/layers/communicator_mhc*.py, transformers/models/glm5_next/) is the
# Clariden aarch64/CUDA build (cp312). It cannot run on MI300A, but its sglang
# + transformers are PURE PYTHON, so we extract them to $OVL and prepend them
# on PYTHONPATH over the upstream ROCm image (which supplies the compiled
# torch/sgl_kernel/aiter/tilelang). A few pure/compiled deps must be bumped to
# clariden transformers 5.16's pins as cp310 wheels into $OVL/pkgs310.
#
# Idempotent: skips extraction/installs already present. Re-runs are cheap.
# Run from the Beverin LOGIN node (unsquashfs needs no container; the pip step
# runs INSIDE the v0.5.18 container via srun — see step 2). ~3-5 min cold.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVL="${OVL:-/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay}"
# Source of the GLM-5.3 Python (cp312/CUDA; only its Python is extracted).
CLARIDEN_IMG="${CLARIDEN_IMG:-/capstor/scratch/cscs/xyao/glm-53-flash/images/sglang-glm-5.3-flash.aarch64.sqsh}"

echo "[$(date -Is)] build_overlay: OVL=$OVL"
echo "[$(date -Is)] build_overlay: CLARIDEN_IMG=$CLARIDEN_IMG"
[ -f "$CLARIDEN_IMG" ] || { echo "FATAL: clariden image not found: $CLARIDEN_IMG" >&2; exit 1; }

# 1. Extract the clariden sglang + transformers source (pure Python, cp312 —
#    imports fine on cp310; no ABI). ~6.5k .py, ~80 MB on /capstor.
if [ -f "$OVL/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py" ] && \
   [ -f "$OVL/sgl-workspace/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py" ]; then
  echo "[$(date -Is)] sglang + transformers already extracted; skipping unsquashfs"
else
  mkdir -p "$OVL"
  echo "[$(date -Is)] extracting clariden sglang + transformers source ..."
  unsquashfs -f -d "$OVL" "$CLARIDEN_IMG" \
    "sgl-workspace/sglang/python" \
    "sgl-workspace/transformers/src" 2>&1 | tail -2
fi

# 1b. Apply Beverin MI300A patches to the extracted sglang source. These are
#     the minimal, version-controlled deviations from the Clariden CUDA build
#     needed to run on MI300A. Each patch lives next to this script; apply
#     idempotently (patch -p1 --forward skips already-applied hunks).
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for p in "$PATCH_DIR"/tilelang-mhc-reduce-hidden_block-for-mi300a-64KB-LDS.patch \
         "$PATCH_DIR"/dsa-kpool-hip-paged-mqa-logits.patch \
         "$PATCH_DIR"/dsa-kpool-tilelang-probe.patch \
         "$PATCH_DIR"/dsa-topk-transform-relax-2048-assert-for-always-select-tail.patch \
         "$PATCH_DIR"/sglang-dsa-kpool-topk-transform-cuda_fp16-to-hip-guard.patch; do
  [ -f "$p" ] || continue
  echo "[$(date -Is)] applying patch: $(basename "$p")"
  ( cd "$OVL" && patch -p1 --forward < "$p" ) 2>&1 | grep -viE '^(patching file|Reversed.*previously applied|hunk.*succeeded at| hunk ignored)$' || true
  # Re-check: the last line from patch is normally empty/grep-filtered; a real
  # failure (FAILED / can't find file) would have printed above.
done

# 1c. Install the PR #52 DSA-vkernels shim (sitecustomize.py + vkernels_dsa.py)
#     onto $OVL/pylib, which the sglang-rocm EDF prepends FIRST on PYTHONPATH
#     (sglang-rocm.toml [env]). sitecustomize.py is auto-imported by CPython at
#     startup (before any sglang import) and, on gfx942, rebinds sglang's DSA
#     tilelang_sparse_fwd -> vkernels_dsa.tilelang_sparse_fwd (a ctypes adapter
#     for vk_hip_dsa_sparse_fwd in libvkernels_hip.so), bypassing the
#     tilelang/TVM FloorMod(_, 0) JIT abort (vkernels issue #51, the job-612262
#     blocker). VKERNELS_DIR (exported by the sbatch) points the adapter at the
#     rebuilt .so. Mirrors the Kimi-K3 recipe's $K3/home/pylib/sitecustomize.py.
mkdir -p "$OVL/pylib"
cp -f "$PATCH_DIR/sitecustomize.py" "$PATCH_DIR/vkernels_dsa.py" "$OVL/pylib/"
echo "[$(date -Is)] installed DSA-vkernels shim -> $OVL/pylib/{sitecustomize,vkernels_dsa}.py"

# 2. Bump the skewed deps to clariden transformers 5.16's pins as cp310 wheels.
#    The login node's only pip is python3.6's (20.0.2) — too old for
#    --python-version/--ignore-requires-python, and uv is a wrong-arch binary
#    copied from Clariden. So the dep install runs INSIDE the v0.5.18
#    container via srun (cp310 + modern pip + /capstor + internet), exactly as
#    the verified import gate did. ~1-2 min.
#    The upstream v0.5.18 ROCm image ships transformers 5.12 / tokenizers 0.22 /
#    huggingface_hub 0.3x / safetensors 0.4x / older regex — ALL too old for
#    clariden transformers 5.16 (tokenizers>=0.23.1, huggingface_hub>=1.5.0,
#    safetensors>=0.8.0, regex>=2025.10.22, accelerate>=1.1.0). torch/numpy/etc.
#    are NOT installed here (they come from the image, already ROCm/cp310).
#    `kernels` is intentionally NOT installed: clariden transformers guards the
#    `from kernels import ...` behind is_kernels_available(); absent -> skipped.
export EDF_PATH="${DEPLOY_DIR}:${EDF_PATH:-${HOME}/.edf}"
PKGS=(
  "tokenizers>=0.23.1,<0.24.0"
  "huggingface_hub>=1.5.0,<2.0"
  "safetensors>=0.8.0"
  "regex>=2025.10.22"
  "accelerate>=1.1.0"
  "jinja2>=3.1.0" "tqdm>=4.60"
  "requests>=2.0" "pyyaml>=5.1" "packaging>=20.0"
  "typing_extensions>=4.0" "filelock" "fsspec"
)
mkdir -p "$OVL/pkgs310"
if [ -f "$OVL/pkgs310/tokenizers/__init__.py" ] && [ -f "$OVL/pkgs310/safetensors/__init__.py" ]; then
  echo "[$(date -Is)] pkgs310 deps already installed (tokenizers+ safetensors present); skipping"
else
  echo "[$(date -Is)] installing cp310 dep bumps INSIDE the container (srun) ..."
  srun -p mi300 -A a-infra02 --gres=gpu:1 -N1 -n1 -t0:10:00 \
    --environment=sglang-rocm \
    python3 -m pip install --no-cache-dir --no-deps --target="$OVL/pkgs310" \
      "${PKGS[@]}" 2>&1 | tail -8
fi

echo "[$(date -Is)] build_overlay DONE"
echo "  sglang .py : $(find "$OVL/sgl-workspace/sglang/python/sglang" -name '*.py' ! -path '*__pycache__*' 2>/dev/null | wc -l)"
echo "  tf .py     : $(find "$OVL/sgl-workspace/transformers/src/transformers" -name '*.py' ! -path '*__pycache__*' 2>/dev/null | wc -l)"
echo "  pkgs310    : $(ls "$OVL"/pkgs310/*.dist-info -d 2>/dev/null | sed 's#.*/##;s/.dist-info//' | tr '\n' ' ')"
