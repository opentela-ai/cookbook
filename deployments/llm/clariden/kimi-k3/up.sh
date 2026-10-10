#!/usr/bin/env bash
# One-shot submit of the 8-node TP4×PP8 Kimi-K3 serve job through rcc.
#   bash up.sh                                # default: 8 nodes (README "Submit" for variants)
#   bash up.sh --sbatch-args='--nodes=4 --export=ALL,NNODES=4,PP_SIZE=4'
# Prerequisites (arm64 image import, otela-arm64 staging, servekit checkout):
# README → Submit. First run: cold start ~105 min; `rcc --profile clariden job tail <JOBID> -f`.
set -euo pipefail
cd "$(dirname "$0")"
rcc --profile clariden job submit deployments/llm/clariden/kimi-k3/serve_kimi_k3_otela_clariden.sbatch "$@"
