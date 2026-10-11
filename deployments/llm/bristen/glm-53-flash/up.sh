#!/usr/bin/env bash
# One-shot bring-up: stage the recipe's helper files (engine.sh, preflight,
# SM80 patches, probe, otela config) to the cluster deploy dir, then submit.
# The sbatch runs them from $DEPLOY_DIR, not from this repo checkout.
# Prerequisites (overlay, image, weights): README.md → Hardware / image.
set -euo pipefail
cd "$(dirname "$0")"

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash-bristen}"
mkdir -p "$DEPLOY_DIR/logs"

# Repo checkout is the source of truth — refresh the staged copies every time.
rsync -a \
  --exclude='*.sbatch' --exclude='up.sh' --exclude='README.md' \
  --exclude='CORRECTNESS_BUG_INVESTIGATION.md' --exclude='tests_local/' \
  ./ "$DEPLOY_DIR/"
cp ../../../../meta/common/gen_correctness.py "$DEPLOY_DIR/gen_correctness.py"

# Stage the latest official OpenTela (amd64) on /capstor — best effort: a
# stale staged binary still serves; a missing one FATALs in the sbatch preflight.
OTELA_DIR=/capstor/scratch/cscs/xyao/opentela OTELA_ARCH=amd64 \
  bash ../../../../meta/common/download_otela.sh \
  || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"
sbatch "$@" serve_glm_53_flash_sglang.sbatch
