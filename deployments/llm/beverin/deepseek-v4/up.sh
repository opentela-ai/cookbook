#!/usr/bin/env bash
# One-shot bring-up: probe gate → submit the serve job.
# NOTE the README banner: DeepSeek-V4-Flash does not generate tokens on this
# image yet (aiter bug) — this runs the bring-up log, not a working deployment.
set -euo pipefail
cd "$(dirname "$0")"

sbatch probe_sglang.sbatch
# Stage the latest official OpenTela (amd64) — best effort: a stale staged
# binary still serves; a missing one FATALs in the sbatch preflight.
OTELA_DIR=/capstor/scratch/cscs/xyao/opentela OTELA_ARCH=amd64 \
  bash ../../../../meta/common/download_otela.sh \
  || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"
sbatch "$@" serve_deepseek_v4_flash.sbatch
