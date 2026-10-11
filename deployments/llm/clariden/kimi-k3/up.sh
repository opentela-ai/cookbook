#!/usr/bin/env bash
# One-shot submit of the 8-node TP4×PP8 Kimi-K3 serve job through rcc.
#   bash up.sh                                # default: 8 nodes (README "Submit" for variants)
#   bash up.sh --sbatch-args='--nodes=4 --export=ALL,NNODES=4,PP_SIZE=4'
# Prerequisites (arm64 image import, otela-arm64 staging, servekit checkout):
# README → Submit. First run: cold start ~105 min; `rcc --profile clariden job tail <JOBID> -f`.
set -euo pipefail
cd "$(dirname "$0")"
# Stage the latest official OpenTela (arm64) on /capstor — best effort: a
# stale staged binary still serves; a missing one FATALs in the sbatch preflight.
rcc --profile clariden run --cwd "$PWD" -- bash -c 'OTELA_DIR=/capstor/scratch/cscs/xyao/opentela OTELA_ARCH=arm64 bash ../../../../meta/common/download_otela.sh || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"'
rcc --profile clariden job submit deployments/llm/clariden/kimi-k3/serve_kimi_k3_otela_clariden.sbatch "$@"
