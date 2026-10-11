#!/usr/bin/env bash
# One-shot bring-up for Kimi-K3 on beverin (6 nodes x 4 MI300A, TP8 x PP3).
# First run = fast smoke (dummy weights, no real generation):
#   bash up.sh --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1
# Then real weights (~cold start; the mandatory correctness probe gates
# otela registration):
#   bash up.sh
# Prerequisites (image import, otela binary, restriped weights): README.
set -euo pipefail
cd "$(dirname "$0")"

# SLURM_SUBMIT_DIR must be this recipe dir so the sbatch finds its siblings
# (kimi-k3-vllm.toml, k3_patch.py, sitecustomize.py) — `rcc run --cwd` does
# that; `rcc job submit` runs from the repo root and fails preflight
# (observed, job 598873 — see README → Submit).
# Stage the latest official OpenTela (amd64) on the shared scratch — best
# effort: a stale staged binary still serves; a missing one FATALs in-preflight.
rcc --profile beverin run --cwd "$PWD" -- bash -c 'OTELA_DIR=/capstor/scratch/cscs/xyao/opentela OTELA_ARCH=amd64 bash ../../../../meta/common/download_otela.sh || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"'
rcc --profile beverin run --cwd "$PWD" -- sbatch "$@" serve_kimi_k3_otela_beverin.sbatch
