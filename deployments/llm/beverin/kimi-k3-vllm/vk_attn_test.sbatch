#!/bin/bash
#SBATCH --job-name=vk-attn-test
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:18:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vk-attn-test/logs/%x-%j.out

# Issue #42 on-cluster foundation test (1 GPU, ~18 min).  Validates that the
# new vkernels_attn.py registration/patch machinery resolves against the REAL
# vLLM fork (MLACommonImpl / KimiK3DeltaAttention) and that the validated HIP
# kernels (vk_hip_mla_fwd / vk_hip_kda_delta_rule_fwd) actually EXECUTE on
# gfx942.  The live serving deployment ($K3/home/pylib) is NOT touched:
# PYTHONPATH points at an isolated copy of the new files.  See issue #45 for
# the remaining marshalling + full-correctness work.
set -uo pipefail

export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# kimi-k3-vllm.toml lives in the cookbook recipe dir, not here:
export EDF_PATH="/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm:${EDF_PATH:-${HOME}/.edf}"
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-vllm}"

srun --ntasks=1 --environment="$OTELA_EDF_NAME" bash "$SCRIPT_DIR/test.sh"
echo "[$(date -Is)] srun_exited code=$?"
