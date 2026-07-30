#!/bin/bash
# Build the Kimi-K3 serving image for JSC Jupiter Booster (GH200 / aarch64) as
# an Apptainer .sif, from docker://lmsysorg/sglang:kimi-k3 (CUDA 13, sm_90 cubins
# present). The resulting .sif is what serve_llm_otela_jsc.sbatch runs via
# `apptainer exec --nv <sif> sglang serve ...`.
#
# JSC has Apptainer only — no Enroot/Pyxis/EDF like CSCS — so this is a plain
# `apptainer build <sif> docker://...`. Run on a LOGIN node: it has the outbound
# internet the pull needs, Apptainer, and (conveniently) GH200 for a quick
# `sglang --version` smoke check. No Slurm allocation required.
#
#   bash deployments/llm/jsc/build_kimi_k3_image.sh
#
# Env overrides:
#   PROJECT=reformo                       Slurm account / scratch project
#   KIMI_K3_DIR=/e/scratch/$PROJECT/$USER/kimi-k3   deploy root (matches the sbatch $DEPLOY_DIR)
#   IMAGE_SOURCE=docker://lmsysorg/sglang:kimi-k3   source image
#   IMAGE=/e/scratch/.../sglang-kimi-k3.sif         output path (matches the sbatch $IMAGE)
#
# ~20 GiB image: expect 10-25 min depending on the link. The .sif lands at the
# SAME path the sbatch reads by default, so the two scripts are wired together
# with no extra config.
set -euo pipefail

# ---------------------------------------------------------------- deployment --
PROJECT="${PROJECT:-reformo}"
export DEPLOY_DIR="${KIMI_K3_DIR:-/e/scratch/$PROJECT/$USER/kimi-k3}"
export IMAGE="${IMAGE:-$DEPLOY_DIR/images/sglang-kimi-k3.sif}"
SRC="${IMAGE_SOURCE:-docker://lmsysorg/sglang:kimi-k3}"
IMG_DIR="$(dirname "$IMAGE")"

# Keep the Apptainer OCI-layer cache + mksquashfs temp OFF the small, node-local
# /tmp and ON the persistent GPFS scratch (/e/scratch = EXASCRATCH; the
# /p/scratch JUSTSCRATCH is NOT mounted on compute nodes, so every artefact the
# job reads must live under /e/scratch). APPTAINER_TMPDIR emits a harmless
# WARNING but does not block a docker->sif build.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$DEPLOY_DIR/cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$DEPLOY_DIR/tmp}"
export TMPDIR="${TMPDIR:-$DEPLOY_DIR/tmp}"

mkdir -p "$IMG_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$DEPLOY_DIR/logs"

echo "[build $(date -Is)] src=$SRC"
echo "[build ] dst=$IMAGE"
echo "[build ] arch=$(uname -m)  cache=$APPTAINER_CACHEDIR  tmp=$TMPDIR"
echo "[build ] /e/scratch capacity:"; df -h /e/scratch | tail -1 | sed 's/^/        /'

command -v apptainer >/dev/null || { echo "FATAL: apptainer not found (run on a JSC login node)" >&2; exit 1; }

# Back up any prior build so a re-run never clobbers a known-good image.
[ -f "$IMAGE" ] && { echo "[build ] backing up existing $(basename "$IMAGE") -> .prev"; mv -f "$IMAGE" "$IMAGE.prev"; }

# Build straight from the Docker registry to a SIF. --force re-pulls layers so a
# stale cached layer can't pin an old sglang.
apptainer build --force "$IMAGE" "$SRC"
rc=$?

# apptainer can exit non-zero after writing a usable SIF — verify by size.
if [ -f "$IMAGE" ]; then
  SZ=$(stat -c %s "$IMAGE")
  echo "[build ] wrote $IMAGE : $(numfmt --to=iec "$SZ")"
  if [ "$SZ" -lt $((5 * 1024 * 1024 * 1024)) ]; then
    echo "FAIL: SIF under 5 GiB — build truncated (rc=$rc)" >&2; exit 1
  fi
  echo "OK: image looks complete (rc=$rc)"
else
  echo "FAIL: no SIF written (rc=$rc)" >&2; exit 1
fi

echo "[build $(date -Is)] done. Verify with:"
echo "  apptainer inspect $IMAGE | head"
echo "  apptainer exec --nv $IMAGE sglang --version"
