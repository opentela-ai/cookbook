#!/bin/bash
# Build the GLM-5.3-Flash serving image on a JSC LOGIN node (~10-25 min, one
# time). The compute sbatch consumes the .sif this produces:
#   $DEPLOY_DIR/images/sglang-glm-5.3-flash.sif
#
# Image: docker://lmsysorg/sglang:glm-5.3-flash — the GLM-5.3-specific SGLang
# build (multi-arch; apptainer pulls the aarch64 manifest on JSC login).
# Validated on clariden GH200 (see deployments/llm/clariden/glm-53-flash).
#
# Usage (login node):  bash build_glm_5_3_flash_image.sh
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/e/scratch/reformo/yao4/glm-53-flash}"
IMAGE="$DEPLOY_DIR/images/sglang-glm-5.3-flash.sif"

# Keep apptainer's docker-layer cache + tmp OFF /e/home (NFS quota) and on
# /e/scratch; a partially-failed build leaves junk in these dirs.
export APPTAINER_CACHEDIR="$DEPLOY_DIR/cache/apptainer"
export APPTAINER_TMPDIR="$DEPLOY_DIR/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$IMAGE")"

echo "[$(date -Is)] building $IMAGE"
echo "[$(date -Is)]   from docker://lmsysorg/sglang:glm-5.3-flash"
apptainer build --force "$IMAGE" docker://lmsysorg/sglang:glm-5.3-flash
echo "[$(date -Is)] done: $(du -sh "$IMAGE" | cut -f1)  $IMAGE"

# Quick sanity: the container's python must import the GLM-5.3 arch class
# (CPU-only import; runs anywhere).
apptainer exec --bind /e/scratch:/e/scratch "$IMAGE" python3 -c '
import importlib
m = importlib.import_module("sglang.srt.models.glm5_next")
print("glm5_next EntryClass:", getattr(m, "EntryClass", None))
import sglang
print("sglang:", getattr(sglang, "__version__", "?"))
'
echo "OK"
