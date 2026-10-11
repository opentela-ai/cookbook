#!/usr/bin/env bash
# One-shot bring-up. First run = fast smoke (dummy weights, ~10 min); after the
# smoke passes, run with real weights (~4 h cold start):
#   bash up.sh --export=ALL,GEN_CORRECTNESS_SMOKE=1,LOAD_FORMAT=dummy   # smoke
#   bash up.sh --export=ALL,GEN_CORRECTNESS_SMOKE=0                     # real weights
# Default (no args) submits with in-sbatch defaults (real weights, probe gate on).
# Prerequisites (image import, otela binary, lfs setstripe): README → Submission.
set -euo pipefail
cd "$(dirname "$0")"

# `rcc run --cwd <ABSOLUTE recipe dir>` is required so BASH_SOURCE/SCRIPT_DIR
# resolve here and sibling files are found (see the sbatch header note).
# Stage the latest official OpenTela (amd64) on the shared scratch — best
# effort: a stale staged binary still serves; a missing one FATALs in-preflight.
rcc --profile beverin run --cwd "$PWD" -- bash -c 'OTELA_DIR=/capstor/scratch/cscs/xyao/opentela OTELA_ARCH=amd64 bash ../../../../meta/common/download_otela.sh || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"'
rcc --profile beverin run --cwd "$PWD" -- sbatch "$@" serve_glm_52_sglang.sbatch
