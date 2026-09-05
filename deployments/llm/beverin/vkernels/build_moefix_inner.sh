#!/bin/bash
# build_moefix_inner.sh — runs INSIDE the vkernels-rocm container (via the
# sbatch's `srun --environment=$OTELA_EDF_NAME`). Inherits VKERNELS_DIR,
# BUILD_DIR, NODE_CPUS from the submitting sbatch's environment.
#
# Force-recompiles the two files the login-node (ROCm-6) build touched, then
# rebuilds libvkernels_hip.so (→ libamdhip64.so.7) and the two MoE
# correctness tests, and runs them vs the CPU oracle.
set -uo pipefail

: "${VKERNELS_DIR:?VKERNELS_DIR not set}"
: "${BUILD_DIR:?BUILD_DIR not set}"
: "${NODE_CPUS:=192}"

cd "$VKERNELS_DIR"
B="$BUILD_DIR"

# Stale ROCm-6 objects (from the login-node make) must be recompiled in the
# ROCm-7 container; rm them so cmake --build cannot skip them as up-to-date.
rm -f "$B/src/c/CMakeFiles/vkernels.dir/vkernels/kernels/moe_fused.hip.o" \
      "$B/src/c/CMakeFiles/vkernels_hip.dir/vkernels/capi/hip_capi.cpp.o" 2>/dev/null \
  && echo "[moefix] removed stale moe_fused.hip.o + hip_capi.cpp.o (force recompile)"

echo "[build] cmake --build --preset hip --target vkernels_hip test_capi_moe test_moe_fused_correct -j$NODE_CPUS"
cmake --build --preset hip --target vkernels_hip test_capi_moe test_moe_fused_correct -j"$NODE_CPUS" 2>&1 | tail -20
RC=${PIPESTATUS[0]}
echo "[build] exit=$RC"

HIP_SO=$(find "$B" -name "libvkernels_hip.so" 2>/dev/null | head -1)
echo "[build] .so: $HIP_SO"
if [ -n "$HIP_SO" ]; then
  ls -la "$HIP_SO"
  ldd "$HIP_SO" 2>/dev/null | grep -E "libamdhip64" || echo "[build] WARN: no libamdhip64 linkage"
fi

if [ "$RC" -ne 0 ]; then
  echo "[build] FAILED - aborting tests"
  exit "$RC"
fi

echo "[test1] test_capi_moe (C ABI, nullptr stream) vs CPU oracle ..."
"$B/meta/benchmarks/test_capi_moe" 2>&1 | tail -8
echo "[test1] exit=${PIPESTATUS[0]}"

echo "[test2] test_moe_fused_correct (kernel launch, default stream) vs CPU oracle ..."
"$B/meta/benchmarks/test_moe_fused_correct" 2>&1 | tail -8
echo "[test2] exit=${PIPESTATUS[0]}"

echo "[done] moefix build+test complete"
