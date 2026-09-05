#!/bin/bash
#SBATCH --job-name=kda-build
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:14:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vk-attn-test/logs/%x-%j.out
#
# Rebuild libvkernels_hip.so + vkernels_test_kda INSIDE the kimi-k3-vllm
# container so the HIP runtime linkage matches (container has ROCm 7 /
# libamdhip64.so.7; the host has ROCm 6.3 / .so.6 — building on the host
# produces a lib that fails to dlopen inside the container).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-vllm}"
echo "[$(date -Is)] kda-build start (EDF=$OTELA_EDF_NAME)"
srun --ntasks=1 --environment="$OTELA_EDF_NAME" bash -c '
set -e
export PATH=/usr/local/bin:$PATH
cd /capstor/scratch/cscs/xyao/vkernels/build/hip
# the Makefile cache (from a host build) points at /usr/bin/cmake; the
# container has cmake at /usr/local/bin.  Symlink so the cached path
# resolves, avoiding a full reconfigure.
[ -e /usr/bin/cmake ] || ln -sf /usr/local/bin/cmake /usr/bin/cmake 2>/dev/null || true
CMAKE=$(command -v cmake)
echo "## cmake=$CMAKE  ROCm=$(readlink -f /opt/rocm) ##"
echo "## incremental build: libvkernels_hip.so (kda.hip + hip_capi.cpp) ##"
"$CMAKE" --build . --target vkernels_hip -j 8 2>&1 | tail -8
echo "## incremental build: vkernels_test_kda (kda.cpp + test_kda.cpp) ##"
"$CMAKE" --build . --target vkernels_test_kda -j 8 2>&1 | tail -6
echo "## ldd the new lib (must find libamdhip64.so.7) ##"
ldd src/c/libvkernels_hip.so 2>&1 | grep -iE "amdhip|not found" | head -3
echo "## exported symbols ##"
nm -D src/c/libvkernels_hip.so 2>/dev/null | grep -iE "vk_hip_kda_delta_rule"
echo "## run the CPU tests (hand-checked + zero-gate) ##"
./tests/vkernels_test_kda 2>&1 | tail -8
echo "KDA_BUILD_OK"
'
rc=$?
echo "[$(date -Is)] kda-build srun_exited code=$rc"
exit $rc
