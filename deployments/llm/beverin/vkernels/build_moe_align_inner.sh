#!/bin/bash
# build_moe_align_inner.sh — runs INSIDE the vkernels-rocm container (via the
# sbatch's `srun --environment=$OTELA_EDF_NAME`). Inherits VKERNELS_DIR,
# BUILD_DIR, NODE_CPUS from the submitting sbatch's environment.
#
# Fetches PR #47 (perf/moe-on-device-align, on-device moe_align_block_size,
# no topk_ids.cpu() sync), force-recompiles the touched translation units,
# rebuilds libvkernels_hip.so + the new test_capi_moe_align parity target
# (and test_capi_moe as a regression guard), and runs them vs the CPU oracle.
set -uo pipefail

: "${VKERNELS_DIR:?VKERNELS_DIR not set}"
: "${BUILD_DIR:?BUILD_DIR not set}"
: "${NODE_CPUS:=192}"

cd "$VKERNELS_DIR"
B="$BUILD_DIR"
BRANCH="perf/moe-on-device-align"
EXPECT_COMMIT="${EXPECT_COMMIT:-2672f7a}"

# The scratch checkout is synced out-of-band (login node has SSH to github;
# the ROCm container does NOT, so a git fetch here would fail). Just assert
# we are on the expected commit before building — fail loud if not, so a
# stale tree is never silently built.
ACTUAL_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
ACTUAL_COMMIT="$(git rev-parse --short=9 HEAD 2>/dev/null)"
echo "[git] branch=$ACTUAL_BRANCH  HEAD=$ACTUAL_COMMIT  expect=$EXPECT_COMMIT/$BRANCH"
if [ "$ACTUAL_BRANCH" != "$BRANCH" ] || ! git merge-base --is-ancestor "$EXPECT_COMMIT" HEAD 2>/dev/null; then
  echo "[git] FATAL: scratch checkout is NOT at $BRANCH >= $EXPECT_COMMIT."
  echo "[git] Sync it on the login node (git -C $VKERNELS_DIR fetch origin $BRANCH;"
  echo "[git]   git -C $VKERNELS_DIR checkout $BRANCH) and resubmit. Aborting."
  exit 2
fi
echo "[git] OK — scratch checkout at $ACTUAL_COMMIT, building."

# Stale ROCm-6 objects (from a prior login-node make) must be recompiled in
# the ROCm-7 container; rm them so cmake --build cannot skip them as
# up-to-date. moe_fused.hpp changed (new decl) -> both moe_fused.hip.o and
# hip_capi.cpp.o (which include it) rebuild; rm them explicitly to be safe.
rm -f "$B/src/c/CMakeFiles/vkernels.dir/vkernels/kernels/moe_fused.hip.o" \
      "$B/src/c/CMakeFiles/vkernels_hip.dir/vkernels/capi/hip_capi.cpp.o" 2>/dev/null \
  && echo "[build] removed stale moe_fused.hip.o + hip_capi.cpp.o (force recompile)"

echo "[build] cmake --build --preset hip --target vkernels_hip test_capi_moe_align test_capi_moe -j$NODE_CPUS"
cmake --build --preset hip --target vkernels_hip test_capi_moe_align test_capi_moe -j"$NODE_CPUS" 2>&1 | tail -30
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

echo "[test0] test_capi_moe_align (GPU moe_align vs CPU ref; issue #46) ..."
"$B/meta/benchmarks/test_capi_moe_align" 2>&1 | tail -20
RC0=${PIPESTATUS[0]}
echo "[test0] exit=$RC0"

echo "[test1] test_capi_moe (regression: C ABI fused MoE vs CPU oracle) ..."
"$B/meta/benchmarks/test_capi_moe" 2>&1 | tail -8
RC1=${PIPESTATUS[0]}
echo "[test1] exit=$RC1"

echo "[done] moe-align build+test complete (align=$RC0 moe=$RC1)"
[ "$RC0" -eq 0 ] && [ "$RC1" -eq 0 ]
