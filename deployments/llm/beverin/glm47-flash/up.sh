#!/usr/bin/env bash
# One-shot bring-up: submit the serving job (the image comes from the EDF and
# is pulled on first run — pre-warm command in README.md → Submit).
set -euo pipefail
cd "$(dirname "$0")"
# Stage the latest official OpenTela (amd64) — best effort: a stale staged
# binary still serves; a missing one FATALs in the sbatch preflight.
OTELA_DIR=/capstor/scratch/cscs/xyao/opentela OTELA_ARCH=amd64 \
  bash ../../../../meta/common/download_otela.sh \
  || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"
sbatch "$@" serve_glm_47_flash_sglang.sbatch
