#!/usr/bin/env bash
# One-shot bring-up: ensure the login-node relay is running, then submit the
# serving job. One-time prerequisites (model, SIF image, otela binary) are
# documented in README.md → Prerequisites; knobs: README.md → Knobs.
set -euo pipefail
cd "$(dirname "$0")"

bash start_relay_euler.sh status >/dev/null 2>&1 || bash start_relay_euler.sh
# Stage the latest official OpenTela (amd64) into $DEPLOY_DIR/bin — best
# effort: a stale staged binary still serves; a missing one FATALs in-preflight.
DEPLOY_DIR="${DEPLOY_DIR:-/cluster/scratch/$USER/otela-qwen36}" OTELA_ARCH=amd64 \
  bash ../../../../meta/common/download_otela.sh \
  || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"
sbatch serve_qwen36_otela_euler.sbatch
