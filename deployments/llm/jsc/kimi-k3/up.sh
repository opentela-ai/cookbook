#!/bin/bash
# One-shot: build the image if missing, ensure the login-node relay is up
# (idempotent), then submit the serving sbatch. The three Start-here steps
# from README.md as a single command; extra args are passed to sbatch, e.g.:
#   bash up.sh --nodes=1 --export=ALL,NNODES=1,PP_SIZE=1
set -e
cd "$(dirname "$0")"
export PROJECT="${PROJECT:-reformo}"
IMAGE_DEFAULT="/e/scratch/$PROJECT/$USER/kimi-k3/images/sglang-kimi-k3.sif"
[ -f "${IMAGE:-$IMAGE_DEFAULT}" ] || bash build_kimi_k3_image.sh
bash start_relay_jsc.sh
# Stage the latest official OpenTela (arm64) on /e/scratch — best effort: a
# stale staged binary still serves; a missing one FATALs in the sbatch preflight.
OTELA_DIR="${OTELA_DIR:-/e/scratch/$PROJECT/$USER/otela-llm}" OTELA_ARCH=arm64 \
  bash ../../../../meta/common/download_otela.sh \
  || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"
sbatch "$@" serve_llm_otela_jsc.sbatch
