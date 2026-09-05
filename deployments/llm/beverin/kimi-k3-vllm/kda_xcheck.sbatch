#!/bin/bash
#SBATCH --job-name=kda-xcheck
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:14:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vk-attn-test/logs/%x-%j.out
#
# Issue #45 cross-check: vk_hip_kda_delta_rule_fwd vs the vLLM Triton
# KDA decode kernel (fused_recurrent_kda) at K3 head shapes.  Runs the
# probe INSIDE the kimi-k3-vllm container (one MI300A GPU, ~1 min incl.
# Triton JIT).  Pass H S D as args (default 12 256 128).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-vllm}"
ARGS="${KDA_XCHECK_ARGS:-12 256 128}"
export VKERNELS_HIP_LIB="/capstor/scratch/cscs/xyao/vkernels/build/hip/src/c/libvkernels_hip.so"
# libamdhip64.so.6 (HIP runtime) is needed by libvkernels_hip.so; the EDF
# env does not add /opt/rocm/lib to LD_LIBRARY_PATH for this ctypes probe.
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
echo "[$(date -Is)] kda-xcheck start (EDF=$OTELA_EDF_NAME args=$ARGS)"
srun --ntasks=1 --environment="$OTELA_EDF_NAME" \
    python3 /capstor/scratch/cscs/xyao/vkernels/probe_kda_xcheck.py $ARGS
rc=$?
echo "[$(date -Is)] kda-xcheck srun_exited code=$rc"
exit $rc
