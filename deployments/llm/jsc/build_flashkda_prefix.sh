#!/usr/bin/env bash
#
# build_flashkda_prefix.sh — install MoonshotAI/FlashKDA into a prefix for
# the sglang Kimi-K3 container on JSC (SM90a / GH200).
#
# The prefix is bind-mounted into the container at runtime (the container's
# /e/scratch is already bind-mounted) and auto-detected by serve_llm_otela_jsc.sbatch
# at $DEPLOY_DIR/flashkda_prefix/flash_kda/__init__.py.
#
# Run on a LOGIN node — the install does not need a GPU (only the
# FLASH_KDA_CUDA_ARCHS=90a env is needed so the build targets GH200, not
# the login node's absent GPU). The actual kernels are JIT-compiled on the
# compute node at first use.
#
set -euo pipefail

PROJECT="${PROJECT:-reformo}"
USER="${USER:-$(whoami)}"
DEPLOY_DIR="${DEPLOY_DIR:-/e/scratch/$PROJECT/$USER/otela-llm}"
PREFIX="$DEPLOY_DIR/flashkda"

if [ -f "$PREFIX/flash_kda/__init__.py" ]; then
  echo "[build_flashkda] FlashKDA prefix already exists at $PREFIX — skipping."
  echo "[build_flashkda] Remove it first if you want to rebuild:"
  echo "    rm -rf $PREFIX"
  exit 0
fi

IMG="${IMG:-/e/scratch/$PROJECT/$USER/kimi-k3/images/sglang-kimi-k3.sif}"
if [ ! -f "$IMG" ]; then
  echo "[build_flashkda] ERROR: container image not found at $IMG" >&2
  echo "[build_flashkda]        Run build_kimi_k3_image.sh first." >&2
  exit 1
fi

echo "[build_flashkda] Installing FlashKDA into $PREFIX (inside container)..."
echo "[build_flashkda]   image:  $IMG"
echo "[build_flashkda]   arch:   SM90a (GH200)"
echo "[build_flashkda]   source: https://github.com/MoonshotAI/FlashKDA.git"

mkdir -p "$PREFIX"

# Install INSIDE the container so the wheel matches the container's Python
# (3.12) and PyTorch (2.11.0+cu130). The prefix is on /e/scratch, which is
# already bind-mounted into the container at the same path.
apptainer exec \
  --bind /e/scratch:/e/scratch --bind /e/home:/e/home \
  "$IMG" bash -c "
    set -euo pipefail
    export FLASH_KDA_CUDA_ARCHS=90a
    export PIP_CACHE_DIR=/e/scratch/$PROJECT/$USER/.pip-cache
    mkdir -p \"\$PIP_CACHE_DIR\"
    pip install --no-build-isolation --prefix '$PREFIX' \
      git+https://github.com/MoonshotAI/FlashKDA.git
  "

echo
echo "[build_flashkada] Verifying import in container..."
apptainer exec \
  --bind /e/scratch:/e/scratch --bind /e/home:/e/home \
  "$IMG" python3 -c "
import sys; sys.path.insert(0, '$PREFIX')
import flash_kda
print('flash_kda OK:', flash_kda.__file__[:100])
print('exports:', [x for x in dir(flash_kda) if not x.startswith('_')][:10])
" 2>&1 || { echo "[build_flashkada] IMPORT FAILED" >&2; exit 1; }

echo
echo "[build_flashkada] Done. FlashKDA prefix at: $PREFIX"
echo "[build_flashkada] serve_llm_otela_jsc.sbatch will auto-detect it."
