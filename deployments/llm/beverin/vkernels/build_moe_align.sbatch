#!/bin/bash
#SBATCH --job-name=moe-align-build
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/opentela-cookbook/logs/%x-%j.out
#
# In-container rebuild of libvkernels_hip.so + the new on-device
# moe_align_block_size parity test (PR #47, commit 42b79bd), plus the
# existing test_capi_moe as a regression guard. The build runs INSIDE
# the vkernels-rocm container (ROCm 7.2.3) so the .so links
# libamdhip64.so.7 (a login-node ROCm-6 build links .so.6, which fails
# to load in the serving container).
#
# Submit:  rcc --profile beverin run --cwd "$PWD" -- sbatch build_moe_align.sbatch
set -uo pipefail

# rcc job submit runs sbatch from remote_dir, so SLURM_SUBMIT_DIR is the
# project root on the cluster. Resolve this recipe's dir from it (SLURM
# copies only the sbatch to its spool dir, so BASH_SOURCE is unreliable
# on the compute node). Override VKERNELS_DIR / the relative path only if
# you are re-rooting the checkout.
REL="${RECIPE_REL:-deployments/llm/beverin/vkernels}"
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/$REL"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
export OTELA_EDF_NAME="${OTELA_EDF_NAME:-vkernels-rocm}"
export VKERNELS_DIR="${VKERNELS_DIR:-/capstor/scratch/cscs/xyao/vkernels}"
export BUILD_DIR="$VKERNELS_DIR/build/hip"
export NODE_CPUS="${SLURM_CPUS_ON_NODE:-192}"
mkdir -p "$VKERNELS_DIR/logs"

echo "[$(date -Is)] moe-align in-container build on $SLURM_JOB_NODELIST"
# Tee the srun output into the vkernels scratch logs too (in-container, so
# it survives even if the batch #SBATCH --output redirect has a problem).
srun --nodes=1 --ntasks=1 --cpus-per-task="$NODE_CPUS" --gpus-per-task=1 \
     --environment="$OTELA_EDF_NAME" bash "$SCRIPT_DIR/build_moe_align_inner.sh" \
  2>&1 | tee "$VKERNELS_DIR/logs/moe-align-build-${SLURM_JOB_ID:-run}.run.log"
RC=${PIPESTATUS[0]}
echo "[$(date -Is)] srun exited code=$RC"
exit "$RC"
