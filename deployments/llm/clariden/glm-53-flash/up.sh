#!/usr/bin/env bash
# One-shot bring-up through rcc (mirrors README "Quick start").
#   bash up.sh                                # default: 1 node, TP4 (validated)
#   bash up.sh --sbatch-args='--export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1'   # fast smoke
#   bash up.sh --sbatch-args='--nodes=2 --export=ALL,NNODES=2,PP_SIZE=2'                # TP4 x PP2
# Prerequisites (image, weights, otela v0.2.6): README → Quick start / preflight prints staging cmds.
set -euo pipefail
cd "$(dirname "$0")"
# Stage the latest official OpenTela (arm64) on /iopsstor — best effort: a
# stale staged binary still serves; a missing one FATALs in the sbatch preflight.
rcc --profile clariden run --cwd "$PWD" -- bash -c 'OTELA_DIR=/iopsstor/scratch/cscs/xyao/opentela OTELA_ARCH=arm64 bash ../../../../meta/common/download_otela.sh || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"'
rcc --profile clariden job submit deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch "$@"
