#!/bin/bash
#SBATCH --job-name=kda-int
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:14:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vk-attn-test/logs/%x-%j.out
#
# Issue #45 INTEGRATION cross-check: the *patched* chunk_kda_with_fused_gate
# (vkernels_attn.py routes it to vk_hip_kda_delta_rule_fwd_with_scratch on
# gfx942 when VKERNELS_KDA=1) vs the working FLA Triton recurrent reference
# fused_recurrent_kda.  Runs INSIDE the kimi-k3-vllm container on one MI300A.
#
# MINIMAL overlay (no aiter MoE patches -- the integration probe only
# exercises the KDA attention leaf).  Installs into $K3/home/pylib/ (where
# sitecustomize.py expects it) exactly the three modules + the rebuilt .so,
# then runs the probe with VKERNELS_KDA=1 so sitecustomize.py applies the
# leaf patch at startup.
#
# Pass H S D as args (default 4 64 32 fast; bump to 12 256 128 for the head).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Unique, writable capstor overlay (NOT the inherited K3=/var/spool/slurmd/...
# which is read-only).  capstor is mounted inside the container, so the
# probe sees the install.  Per-job dir avoids colliding with the real
# serve's $DEPLOY_DIR/home/pylib overlay.
export K3="/capstor/scratch/cscs/xyao/vk-attn-test/k3int-${SLURM_JOB_ID}"
PYLIB="$K3/home/pylib"
HIP_SO="/capstor/scratch/cscs/xyao/vkernels/build/hip/src/c/libvkernels_hip.so"
ARGS="${KDA_INT_ARGS:-4 64 32}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-vllm}"
# libamdhip64.so.7 (ROCm 7) is needed by libvkernels_hip.so
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export VKERNELS_KDA="1"
export VKERNELS_KDA_VALIDATE="1"
export VKERNELS_LIB="$HIP_SO"                        # direct path for _find_libvkernels_hip

echo "[$(date -Is)] kda-int start (EDF=$OTELA_EDF_NAME K3=$K3 PYLIB=$PYLIB args=$ARGS)"
echo "[$(date -Is)] HIP so: $HIP_SO ($(stat -c %s "$HIP_SO" 2>/dev/null || echo MISSING) bytes)"

srun --ntasks=1 --environment="$OTELA_EDF_NAME" bash -c '
set -euo pipefail
# Hard-code the deploy dir: SLURM copies the sbatch to /var/spool/slurmd/job...
# so ${BASH_SOURCE[0]} is the spool path, NOT the deploy dir.
_DEPLOY='/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm'
echo "[in-container] K3=$K3  PYLIB=$K3/home/pylib  VKERNELS_KDA=$VKERNELS_KDA"
echo "[in-container] device: $(rocminfo 2>/dev/null | awk "/Name:/{print \$2; exit}")"
mkdir -p "$K3/home/pylib"
cp "$_DEPLOY/sitecustomize.py"     "$K3/home/pylib/"
cp "$_DEPLOY/vkernels_attn.py"     "$K3/home/pylib/"
cp "$_DEPLOY/vkernels_experts.py"  "$K3/home/pylib/"
cp "'"$HIP_SO"'"                    "$K3/home/pylib/libvkernels_hip.so"
ls -la "$K3/home/pylib/"
echo "[in-container] running probe_kda_integration.py '"$ARGS"' ..."
export PYTHONPATH="$K3/home/pylib:${PYTHONPATH:-}"
python3 "$_DEPLOY/probe_kda_integration.py" '"$ARGS"'
'
rc=$?
echo "[$(date -Is)] kda-int srun_exited code=$rc"
exit $rc
