#!/bin/bash
#SBATCH --job-name=vkernels-build-test
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/vkernels/logs/%x-%j.out
#
# Build + test vkernels (gfx942 kernel library for Kimi-K3) on one MI300A node.
#
# Phase 1 (hip preset):     Build HIP device kernels + C ABI shared library
#                            + GPU correctness tests (C++).
# Phase 2 (host preset):    Build CPU reference pybind11 module (for the
#                            Python smoke test to compare against).
# Phase 3 (Python smoke):   Load libvkernels_hip.so via ctypes and validate
#                            the HIP device kernels vs CPU reference.
#
# Submit:
#   rcc --profile beverin run --cwd "$PWD" -- sbatch build_test_vkernels.sbatch

set -uo pipefail

export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
export OTELA_EDF_NAME="${OTELA_EDF_NAME:-vkernels-rocm}"

export VKERNELS_DIR="${VKERNELS_DIR:-/capstor/scratch/cscs/xyao/vkernels}"
export BUILD_DIR="$VKERNELS_DIR/build/hip"
mkdir -p "$VKERNELS_DIR/logs"

echo "[$(date -Is)] vkernels build+test on $SLURM_JOB_NODELIST (gfx942)"
echo "[$(date -Is)] source: $VKERNELS_DIR"

NODE_CPUS="${SLURM_CPUS_ON_NODE:-192}"

srun --nodes=1 --ntasks=1 --cpus-per-task="$NODE_CPUS" --gpus-per-task=4 \
     --environment="$OTELA_EDF_NAME" bash -c '
set -euo pipefail
cd "'"$VKERNELS_DIR"'"

# ---- Phase 1: HIP build + C ABI + GPU correctness tests ----
echo "[phase1] cmake configure (hip preset + benchmarks) ..."
cmake --preset hip -DVKERNELS_BUILD_BENCHMARKS=ON 2>&1 | tail -15

echo "[phase1] cmake build (HIP device kernels + C ABI + tests) ..."
cmake --build --preset hip -j"'"$NODE_CPUS"'" 2>&1

echo "[phase1] GPU correctness tests (C++) ..."
B="'"$BUILD_DIR"'"
P1_PASS=0; P1_FAIL=0
run_correct() {
  local name="$1"; shift
  local bin="$B/meta/benchmarks/$name"
  [ -x "$bin" ] || { echo "  SKIP $name (not built)"; return; }
  echo "  --- $name $* ---"
  if "$bin" "$@" 2>&1 | tail -20; then
    echo "  PASS $name"; P1_PASS=$((P1_PASS+1))
  else
    echo "  FAIL $name"; P1_FAIL=$((P1_FAIL+1))
  fi
}
run_correct test_kda_correct
run_correct test_mla_correct
run_correct test_gemm_bf16_correct
run_correct test_moe_aux_correct
run_correct test_moe_fused_correct
run_correct test_moe_fused_correct situ
run_correct test_capi_moe
run_correct test_moe_fused_prefill_correct
run_correct test_moe_fused_dist_correct
echo "[$(date -Is)] Phase 1 (C++ GPU): $P1_PASS passed, $P1_FAIL failed"

echo "[phase1] C ABI shared library:"
HIP_SO=$(find "$B" -name "libvkernels_hip.so" 2>/dev/null | head -1)
if [ -n "$HIP_SO" ]; then
  echo "  found: $HIP_SO"
  nm -D "$HIP_SO" | grep -E "vk_fused|vk_mla|vk_kda" || echo "  WARNING: no vk_* symbols"
else
  echo "  WARNING: libvkernels_hip.so not found under $B"
fi

# ---- Phase 2: Python smoke test (ctypes + C ABI) ----
echo "[phase2] Python smoke test (ctypes: libvkernels_hip.so) ..."
export VKERNELS_DIR="'"$VKERNELS_DIR"'"
python3 "'"$VKERNELS_DIR"'/test_hip_bindings.py" 2>&1

echo "[done] all phases complete"
'

RC=$?
echo "[$(date -Is)] srun exited code=$RC"
exit "$RC"
