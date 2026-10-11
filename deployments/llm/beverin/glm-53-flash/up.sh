#!/usr/bin/env bash
# One-shot bring-up. Default = fast smoke (dummy weights, forward-path gate,
# no otela). Full serve + OpenTela registration:
#   bash up.sh --export=ALL,LOAD_FORMAT=auto,SMOKE=0
# Prerequisites (overlay, otela binary): README.md
set -euo pipefail
cd "$(dirname "$0")"

# The sbatch requires `rcc run --cwd <ABSOLUTE recipe dir>` so BASH_SOURCE /
# SCRIPT_DIR resolve here (plain `rcc job submit` submits from the remote_dir
# root and the siblings are not found) — see the sbatch header note.
# Stage the latest official OpenTela (amd64) on the shared scratch — best
# effort: a stale staged binary still serves; a missing one FATALs in-preflight.
rcc --profile beverin run --cwd "$PWD" -- bash -c 'OTELA_DIR=/capstor/scratch/cscs/xyao/opentela OTELA_ARCH=amd64 bash ../../../../meta/common/download_otela.sh || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"'
rcc --profile beverin run --cwd "$PWD" -- sbatch "$@" serve_glm_53_flash_sglang.sbatch
