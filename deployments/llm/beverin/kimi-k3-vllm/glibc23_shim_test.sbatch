#!/bin/bash
#SBATCH --job-name=glibc23-shim-test
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=00:05:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/%x-%j.out
#
# glibc23-shim-test: cheap, isolated proof that a ~25-line shim makes the
# CSCS Cray libfabric 1.29.1 (with the cxi provider builtin) LOAD inside the
# existing vllm+vllm-openai-rocm+kimi-k3 image (Ubuntu 22.04, glibc 2.35) -
# WITHOUT a 30 GiB image rebuild or any from-source libfabric/cxi build.
#
# ROOT CAUSE (jobs 603817/603820/603821): every object in the host Cray
# libfabric stack imports EXACTLY these glibc-2.38 symbols and nothing else
# 2.38-specific (verified: objdump -T .../libfabric.so.1 | grep GLIBC_2.38):
#     (GLIBC_2.38) __isoc23_fscanf  __isoc23_sscanf
#                   __isoc23_strtol  __isoc23_strtoul
# These are glibc's C23 stdlib redirections - an artifact of CSCS compiling
# with a C23-capable GCC, NOT a runtime feature. Forwarding them to the
# plain functions (which glibc 2.35 provides) is functionally identical.
#
# WHY the shim is built IN-CONTAINER (not on the compute head, glibc 2.38):
# on glibc 2.35 the headers do not redirect, so the shim's own fscanf/strtol
# resolve to plain 2.35 symbols (no recursion, no new __isoc23_* dep). Built
# on the head (glibc 2.38 + a C23 gcc) the shim would reintroduce the very
# redirection we are shimming.
#
# WHY the test links only -ldl: without the shim the program still STARTS
# (no glibc-2.38 dep of its own), so dlopen(RTLD_NOW) runs and prints a clean
# dlerror("...GLIBC_2.38 not found...") instead of the dynamic loader
# aborting before main(). With the shim, dlopen RTLD_NOW resolves every
# undefined symbol up front and runs libfabric's constructors -> PASS.
#
# Reuses the proven ofi-rccl-smoke EDF (mounts /opt/cray/libfabric). This
# job is standalone: it touches neither the smoke nor the serving recipe.
#
# Submit:
#   rcc -p beverin push
#   rcc -p beverin job submit deployments/llm/beverin/kimi-k3-vllm/glibc23_shim_test.sbatch
#   rcc -p beverin job tail <JOBID> -f
set -uo pipefail

export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
OTELA_EDF_NAME="${OTELA_EDF_NAME:-ofi-rccl-smoke}"
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin}"
export RUNDIR="${RUNDIR:-$DEPLOY_DIR/glibc23-shim-test-$SLURM_JOB_ID}"
export CRAY_OFI="${CRAY_OFI:-/opt/cray/libfabric/2.3.1}"
mkdir -p "$RUNDIR" "$DEPLOY_DIR/logs" 2>/dev/null

echo "[$(date -Is)] job=$SLURM_JOB_ID edf=$OTELA_EDF_NAME test=$RUNDIR"
[ -f "$SCRIPT_DIR/glibc23_shim.c" ] || { echo "FATAL: $SCRIPT_DIR/glibc23_shim.c missing" >&2; exit 1; }
[ -f "$SCRIPT_DIR/glibc23_shim.map" ] || { echo "FATAL: $SCRIPT_DIR/glibc23_shim.map missing" >&2; exit 1; }
[ -f "$SCRIPT_DIR/dlopen_cxi_test.c" ] || { echo "FATAL: $SCRIPT_DIR/dlopen_cxi_test.c missing" >&2; exit 1; }
cp "$SCRIPT_DIR/glibc23_shim.c" "$SCRIPT_DIR/glibc23_shim.map" "$SCRIPT_DIR/dlopen_cxi_test.c" "$RUNDIR/"

# Mirror ofi_rccl_smoke.sbatch's build.sh: copy the host Cray DSOs the EDF
# does NOT mount (libcxi + the OpenLDAP-2.4 trio the host libcurl needs) into
# $RUNDIR, which run.sh prepends to LD_LIBRARY_PATH. WITHOUT these the dlopen
# fails on "libcxi.so.1: cannot open" BEFORE the glibc-2.38 wall is reached,
# so the shim/no-shim contrast is invisible. With them, the gate becomes the
# __isoc23_*@GLIBC_2.38 resolution (exactly what the shim provides).
CXI_SRC="$(ls /usr/lib64/libcxi.so.1.* 2>/dev/null | head -1)"
if [ -n "$CXI_SRC" ]; then
  cp -L "$CXI_SRC" "$RUNDIR/libcxi.so.1.5.0"
  ln -sf libcxi.so.1.5.0 "$RUNDIR/libcxi.so.1"
  ln -sf libcxi.so.1 "$RUNDIR/libcxi.so"
  echo "[$(date -Is)] copied libcxi: $CXI_SRC -> $RUNDIR/libcxi.so.1.5.0 (+symlinks)"
else
  echo "[$(date -Is)] WARN: libcxi.so.1.* not found under /usr/lib64" >&2
fi
cp -L /usr/lib64/libldap_r-2.4.so.2.10.9 "$RUNDIR/" 2>/dev/null && ln -sf libldap_r-2.4.so.2.10.9 "$RUNDIR/libldap_r-2.4.so.2"
cp -L /usr/lib64/liblber-2.4.so.2.10.9 "$RUNDIR/" 2>/dev/null && ln -sf liblber-2.4.so.2.10.9 "$RUNDIR/liblber-2.4.so.2"
cp -L /usr/lib64/libsasl2.so.3.0.0 "$RUNDIR/" 2>/dev/null && ln -sf libsasl2.so.3.0.0 "$RUNDIR/libsasl2.so.3"
echo "[$(date -Is)] copied OpenLDAP-2.4 trio (libldap_r-2.4, liblber-2.4, libsasl2.so.3)"

# run.sh executes IN-CONTAINER (glibc 2.35) via srun --environment=ofi-rccl-smoke.
cat > "$RUNDIR/run.sh" <<RUN
#!/bin/bash
set -uo pipefail
export LD_LIBRARY_PATH="$RUNDIR:$CRAY_OFI/lib64:\${LD_LIBRARY_PATH:-}"
echo "[r0] container glibc: \$(ldd --version | head -1)"
echo "[r0] target libfabric: $CRAY_OFI/lib64/libfabric.so.1"

echo "[r0] compiling shim against container glibc (must NOT emit __isoc23_*) ..."
gcc -std=c11 -D_DEFAULT_SOURCE -shared -fPIC -O2 \\
    -Wl,--version-script="$RUNDIR/glibc23_shim.map" \\
    "$RUNDIR/glibc23_shim.c" -o "$RUNDIR/libglibc23_shim.so" || {
  echo "G23SHIM_BUILD_FAIL rc=\$?"; exit 2; }
echo "[r0] shim dynamic symbols (expect __isoc23_* under GLIBC_2.38):"
nm -D "$RUNDIR/libglibc23_shim.so" | grep -E "isoc23|GLIBC" || echo "[r0] (no isoc23 exported?!)"

echo "[r0] compiling dlopen test (links -ldl only, no libfabric) ..."
gcc -std=c11 "$RUNDIR/dlopen_cxi_test.c" -ldl -o "$RUNDIR/dlopen_cxi_test" || {
  echo "G23SHIM_TEST_BUILD_FAIL rc=\$?"; exit 3; }

echo "############ WITHOUT shim (expect G23SHIM_DLOPEN_FAIL: GLIBC_2.38) ############"
TARGET_LIBFABRIC="$CRAY_OFI/lib64/libfabric.so.1" "$RUNDIR/dlopen_cxi_test"; rc_no=\$?
echo "[r0] no-shim rc=\$rc_no"

echo "############ WITH shim (expect G23SHIM_PASS) ############"
LD_PRELOAD="$RUNDIR/libglibc23_shim.so" TARGET_LIBFABRIC="$CRAY_OFI/lib64/libfabric.so.1" \\
    GLIBC_TEST_LABEL="2.35+shim" "$RUNDIR/dlopen_cxi_test"; rc_shim=\$?
echo "[r0] shim rc=\$rc_shim"

if [ "\$rc_no" -ne 0 ] && [ "\$rc_shim" -eq 0 ]; then
  echo "G23SHIM_HYPOTHESIS_CONFIRMED: Cray libfabric 1.29.1 fails to load on glibc 2.35 without the shim, loads WITH it."
  echo "G23SHIM_NEXT: wire LD_PRELOAD=libglibc23_shim.so into ofi_rccl_smoke.sbatch and re-run for real CXI transport."
  exit 0
else
  echo "G23SHIM_INCONCLUSIVE: no-shim rc=\$rc_no shim rc=\$rc_shim - inspect above."
  exit 1
fi
RUN
chmod +x "$RUNDIR/run.sh"

srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:03:00 \
     --environment="$OTELA_EDF_NAME" bash "$RUNDIR/run.sh"
RC=$?
echo "[$(date -Is)] run rc=$RC"
exit $RC
