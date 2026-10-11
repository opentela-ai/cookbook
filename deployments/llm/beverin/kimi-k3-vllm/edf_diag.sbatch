#!/bin/bash
#SBATCH --job-name=edf-diag
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:03:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/%x-%j.out
set -uo pipefail
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
echo "SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
echo "BASH_SOURCE=${BASH_SOURCE[0]}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "EDF_PATH=${EDF_PATH}"
echo "PWD=$(pwd)"
ls -l "${SCRIPT_DIR}/ofi-rccl-smoke.toml" 2>&1 | head -2
