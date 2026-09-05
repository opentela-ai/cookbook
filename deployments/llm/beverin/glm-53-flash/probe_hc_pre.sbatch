#!/bin/bash
#SBATCH --job-name=probe-hcpre-beverin
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/%x-%j.out
set -uo pipefail

# Recipe dir on /capstor (holds sglang-rocm.toml EDF + overlay + probe script).
DEPLOY_DIR=/capstor/scratch/cscs/xyao/glm-53-flash-beverin
RECIPE=/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/glm-53-flash
OVL=$DEPLOY_DIR/overlay
export EDF_PATH="${RECIPE}:${EDF_PATH:-${HOME}/.edf}"
export OTELA_EDF_NAME="sglang-rocm"
export OVL
export MODEL_PATH="/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash"
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0     # server path (tilelang, NOT deep_gemm)
export SGLANG_USE_AITER=1
export TRITON_CACHE_DIR="$DEPLOY_DIR/cache/triton"
export TILELANG_CACHE_DIR="$DEPLOY_DIR/cache/tilelang"
export OMP_NUM_THREADS=1
export PYTHONPATH="${OVL}/pylib:${OVL}/pkgs310:${OVL}/sgl-workspace/sglang/python:${OVL}/sgl-workspace/transformers/src"
mkdir -p "$TRITON_CACHE_DIR" "$TILELANG_CACHE_DIR" 2>/dev/null || true

echo "[$(date -Is)] probe-hcpre on $(hostname) | EDF=$OTELA_EDF_NAME | cache=$TILELANG_CACHE_DIR"
srun --environment="$OTELA_EDF_NAME" -N1 -n1 --gpus-per-task=1 -t0:25:00 \
  python3 -u "$RECIPE/probe_hc_pre_serverfaithful.py"
rc=$?
echo "[$(date -Is)] probe rc=$rc"
exit $rc
