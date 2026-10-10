#!/usr/bin/env bash
# One-shot bring-up through rcc (mirrors README "Quick start").
#   bash up.sh                                # default: 1 node, TP4 (validated)
#   bash up.sh --sbatch-args='--export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1'   # fast smoke
#   bash up.sh --sbatch-args='--nodes=2 --export=ALL,NNODES=2,PP_SIZE=2'                # TP4 x PP2
# Prerequisites (image, weights, otela v0.2.6): README → Quick start / preflight prints staging cmds.
set -euo pipefail
cd "$(dirname "$0")"
rcc --profile clariden job submit deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch "$@"
