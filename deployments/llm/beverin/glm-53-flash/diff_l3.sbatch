#!/bin/bash
#SBATCH --job-name=diff-l3-beverin
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/%x-%j.out
set -uo pipefail

DEPLOY_DIR=/capstor/scratch/cscs/xyao/glm-53-flash-beverin
RECIPE=/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/glm-53-flash
OVL=$DEPLOY_DIR/overlay
export EDF_PATH="${RECIPE}:${EDF_PATH:-${HOME}/.edf}"
export OTELA_EDF_NAME="sglang-rocm"
export HSA_NO_SCRATCH_RECLAIM=1
export TRITON_CACHE_DIR="$DEPLOY_DIR/cache/triton"
export TILELANG_CACHE_DIR="$DEPLOY_DIR/cache/tilelang"
export OMP_NUM_THREADS=1
# NOTE: deliberately NO PYTHONPATH override -- the container's stock python3
# has torch in site-packages; adding OVL/pylib would run sitecustomize.py
# (the DSA-SDPA rebind) -> aiter import -> hang. The .pt files are plain
# torch tensors (weights_only=False), loadable with stock torch.
export PYTHONPATH="${OVL}/pylib:${OVL}/pkgs310:${OVL}/sgl-workspace/sglang/python:${OVL}/sgl-workspace/transformers/src"

A="${A:-beverin_l3_s65}"
B="${B:-clariden_l3_s65}"
echo "[$(date -Is)] diff-l3 A=$A B=$B on $(hostname) EDF=$OTELA_EDF_NAME"
srun --environment="$OTELA_EDF_NAME" -N1 -n1 --gpus-per-task=1 -t0:12:00 \
  python3 -S -u "$RECIPE/diff_l3_components.py" "$A" "$B"
echo "[$(date -Is)] diff rc=$?"
