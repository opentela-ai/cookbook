#!/bin/bash
#SBATCH --job-name=moefix-build
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/vkernels/logs/%x-%j.out
#
# Lean in-container rebuild of libvkernels_hip.so after the MoE
# stream-aware-launch change, plus the two MoE correctness tests vs the
# CPU oracle. The build runs INSIDE the vkernels-rocm container (ROCm 7.2.3)
# so the .so links libamdhip64.so.7 (a login-node ROCm-6 build links .so.6,
# which fails to load in the serving container).
#
# Submit:  rcc --profile beverin run --cwd "$PWD" -- sbatch build_moefix.sbatch
set -uo pipefail

export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
export OTELA_EDF_NAME="${OTELA_EDF_NAME:-vkernels-rocm}"
export VKERNELS_DIR="${VKERNELS_DIR:-/capstor/scratch/cscs/xyao/vkernels}"
export BUILD_DIR="$VKERNELS_DIR/build/hip"
export NODE_CPUS="${SLURM_CPUS_ON_NODE:-192}"
mkdir -p "$VKERNELS_DIR/logs"

echo "[$(date -Is)] moefix in-container build on $SLURM_JOB_NODELIST"
srun --nodes=1 --ntasks=1 --cpus-per-task="$NODE_CPUS" --gpus-per-task=1 \
     --environment="$OTELA_EDF_NAME" bash "$SCRIPT_DIR/build_moefix_inner.sh"
RC=$?
echo "[$(date -Is)] srun exited code=$RC"
exit "$RC"
