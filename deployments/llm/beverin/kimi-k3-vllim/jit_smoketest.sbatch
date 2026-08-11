#!/bin/bash
#SBATCH --job-name=k3-jit-smoke
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/%x-%j.out
set -euo pipefail

DEPLOY_DIR="/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/deploy"
export K3="/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin"
export OTELA_EDF_NAME="kimi-k3-vllm"
export PYTHONPATH="${K3}/home/pylib:${PYTHONPATH:-}"

echo "[$(date -Is)] k3-jit-smoke job=$SLURM_JOB_ID node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
echo "[$(date -Is)] K3=$K3 PYTHONPATH=$PYTHONPATH"
echo "[$(date -Is)] EDF=$OTELA_EDF_NAME"
echo "[$(date -Is)] starting jit_smoketest.py (expect 10-30 min for cold JIT)..."

srun --ntasks-per-node=1 --environment="$OTELA_EDF_NAME" \
  bash -c 'export PYTHONPATH="'"$PYTHONPATH"'"; python3 -u "'"$DEPLOY_DIR"'/jit_smoketest.py"'

echo "[$(date -Is)] jit_smoketest exited rc=$?"
