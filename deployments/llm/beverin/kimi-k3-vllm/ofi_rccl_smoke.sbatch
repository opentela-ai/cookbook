#!/bin/bash
#SBATCH --job-name=ofi-rccl-smoke
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/%x-%j.out
#
# vkernels OFI/CXI RCCL net-plugin smoke test (issue #19) on Beverin MI300A.
#
# WHY: serve_kimi_k3_otela_beverin.sbatch (README fix 4) neutralises the EDF's
# CUDA-built aws_ofi_nccl plugin and falls back to RCCL Socket over Slingshot
# because no HIP-aware OFI plugin exists. vkernels now ships one
# (plugins/rccl-net-ofi/librccl-net-ofi.so). The open question before wiring
# it into the 3h Kimi-K3 cold start is ABI: the plugin redeclares a v6/v7-style
# 11-field ncclNet v-table (name/init/devices/getHandle/listen/connect/accept/
# send/recv/close*), with NO getDeviceMR/regHandle/flush. The container's RCCL
# is 1.0.70203 / NCCL-compat 2.27.7 — if its loader expects a newer vN with
# getDeviceMR/regHandle, the plugin is rejected (or crashes) and the
# integration is blocked upstream. This smoke answers that cheaply, inside the
# EXACT image/RCCL the serving recipe uses, WITHOUT touching the recipe.
#
# WHAT: (1) build librccl-net-ofi.so inside the kimi-k3-vllm container against
# the host Cray libfabric 1.29.1 (mounted at /opt/cray/libfabric; the image
# only has libfabric 1.14.0, which has no cxi provider); (2) run a 2-rank
# torch.distributed RCCL allreduce across both nodes with
# NCCL_NET=.../librccl-net-ofi.so FI_PROVIDER=cxi; (3) print
# OFI_SMOKE_PASS/FAIL and grep the log for the transport RCCL actually used.
#
# ACCEPTANCE: the allreduce completes with the correct sum (OFI_SMOKE_PASS).
# NCCL_DEBUG=INFO (from the EDF) makes the transport visible. A crash / ABI
# reject is a clear FAIL and we stop before the real K3 run.
#
# This is a STANDALONE job. It does NOT modify serve_kimi_k3_otela_beverin.sbatch
# or kimi-k3-vllm.toml. Follow-up (gated K3_OFI_NET knob + EDF mount) only if
# this prints OFI_SMOKE_PASS.
#
# Submit:
#   rcc -p beverin push
#   rcc -p beverin job submit deployments/llm/beverin/kimi-k3-vllm/ofi_rccl_smoke.sbatch
#   rcc -p beverin job tail <JOBID> -f
set -uo pipefail

# ----------------------------------------------------------- script / dirs --
# SLURM runs the sbatch from a spool copy; the real submit dir wins.
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
OTELA_EDF_NAME="${OTELA_EDF_NAME:-ofi-rccl-smoke}"

export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin}"
export RUNDIR="${RUNDIR:-$DEPLOY_DIR/ofi-smoke-$SLURM_JOB_ID}"
export VKERNELS="${VKERNELS:-/capstor/scratch/cscs/xyao/vkernels}"
# Cray libfabric with the cxi provider BUILTIN (image's 1.14.0 has none).
export CRAY_OFI="${CRAY_OFI:-/opt/cray/libfabric/2.3.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
mkdir -p "$RUNDIR" "$DEPLOY_DIR/logs" 2>/dev/null

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export HEAD="${NODES[0]}"
export HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"

echo "[$(date -Is)] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST head=$HEAD ($HEAD_IP)"
echo "[$(date -Is)] vkernels=$VKERNELS  cray_ofi=$CRAY_OFI  edf=$OTELA_EDF_NAME  smoke=$RUNDIR"

# ----------------------------------------------------------------- preflight --
[ -d "$VKERNELS" ] || {
  echo "FATAL: vkernels not found at $VKERNELS." >&2
  echo "       Clone it there first:" >&2
  echo "         rcc -p beverin run -s 'git clone https://github.com/opentela-ai/vkernels $VKERNELS'" >&2
  exit 1
}
[ -f "$VKERNELS/plugins/rccl-net-ofi/rccl_net_ofi.c" ] || {
  echo "FATAL: plugin source missing: $VKERNELS/plugins/rccl-net-ofi/rccl_net_ofi.c" >&2
  exit 1
}
[ -f "$CRAY_OFI/include/rdma/fabric.h" ] || { echo "FATAL: Cray libfabric headers missing: $CRAY_OFI/include/rdma/fabric.h" >&2; exit 1; }
[ -f "$CRAY_OFI/lib64/libfabric.so.1" ] || { echo "FATAL: Cray libfabric lib missing: $CRAY_OFI/lib64/libfabric.so.1" >&2; exit 1; }

# ------------------------------------------------------------- build (rank 0) --
# Single C translation unit; links libfabric ONLY (no rccl.h, no HIP runtime).
# Built inside the container for a toolchain that matches the image; the
# mounted /opt/cray/libfabric provides rdma/fabric.h + libfabric.so.1.29.1
# (cxi builtin). Output is a portable .so dropped on /capstor (shared by both
# ranks via the /capstor mount).
cat > "$RUNDIR/build.sh" <<BUILD
#!/bin/bash
set -euo pipefail
# Built on the HOST (first compute node's batch shell), which has /usr/bin/gcc
# + /opt/cray/libfabric (shared across nodes). The .so is pure C and links
# libfabric.so.1 (soname) only; at runtime inside the container the rpath +
# LD_LIBRARY_PATH resolve it to the mounted Cray 1.29.1 (cxi builtin),
# shadowing the image's 1.14.0. An rpath makes the resolution robust even if
# the launcher forgets LD_LIBRARY_PATH.
CC=gcc; command -v gcc >/dev/null 2>&1 || CC=cc
echo "[build] host=$(hostname) CC=\$CC (\$(\$CC --version | head -1))"
echo "[build] src=$VKERNELS/plugins/rccl-net-ofi/rccl_net_ofi.c"
echo "[build] ofi_inc=$CRAY_OFI/include  ofi_lib=$CRAY_OFI/lib64"
mkdir -p "$RUNDIR"
\$CC -shared -fPIC -fvisibility=hidden -O2 \
    -I"$CRAY_OFI/include" \
    "$VKERNELS/plugins/rccl-net-ofi/rccl_net_ofi.c" \
    -o "$RUNDIR/librccl-net-ofi.so" \
    -L"$CRAY_OFI/lib64" -lfabric \
    -Wl,-rpath,"$CRAY_OFI/lib64"
echo "[build] wrote $RUNDIR/librccl-net-ofi.so (\$(stat -c%s "$RUNDIR/librccl-net-ofi.so") bytes)"
# The RCCL net-plugin loader resolves the symbol nccl_ofi_net (NCCL_NET_SYMBOL
# in the source); -fvisibility=hidden hides everything else. If it is missing,
# dlopen succeeds but getSymbol fails -> clear, not silent.
# nccl_ofi_net is a const struct (a v-table), so nm lists it as D or R
# (global data), never T (text). Match both so the check is correct.
if nm -D "$RUNDIR/librccl-net-ofi.so" 2>/dev/null | grep -qE ' [DR] nccl_ofi_net'; then
  echo "[build] SYMBOL_OK nccl_ofi_net exported"
else
  echo "[build] SYMBOL_MISSING nccl_ofi_net — dlopen will not find the v-table" >&2
  nm -D "$RUNDIR/librccl-net-ofi.so" 2>/dev/null | head -20 >&2
  exit 2
fi
echo "[build] ldd fabric:"; ldd "$RUNDIR/librccl-net-ofi.so" 2>/dev/null | grep -i fabric || true
# libfabric.so.1 (Cray 1.29.1) NEEEDS libcxi.so.1, a host DSO at /usr/lib64
# that the EDF does NOT mount into the container. Without it the plugin's
# dlopen fails there with "libcxi.so.1: cannot open shared object file" and
# RCCL silently falls back to the built-in Socket transport (job 603818:
# every cross-node channel went "via NET/Socket" despite FI_PROVIDER=cxi).
# Copy the real libcxi.so.1.* into $RUNDIR and recreate the .so / .so.1
# symlinks; run.sh prepends $RUNDIR to LD_LIBRARY_PATH so libfabric finds
# them in-container. (Runs on the head host where /usr/lib64 is present.)
CXI_SRC=\$(ls /usr/lib64/libcxi.so.1.* 2>/dev/null | head -1)
if [ -n "\$CXI_SRC" ]; then
  cp -L "\$CXI_SRC" "$RUNDIR/libcxi.so.1.5.0"
  ln -sf libcxi.so.1.5.0 "$RUNDIR/libcxi.so.1"
  ln -sf libcxi.so.1 "$RUNDIR/libcxi.so"
  echo "[build] copied libcxi: \$CXI_SRC -> $RUNDIR/libcxi.so.1.5.0 (+symlinks)"
  echo "[build] ldd cxi+fabric:"; ldd "$RUNDIR/librccl-net-ofi.so" 2>/dev/null | grep -iE "cxi|fabric" || true
else
  echo "[build] WARN: libcxi.so.1.* not found under /usr/lib64 - plugin will fail to dlopen in-container" >&2
fi
# OpenLDAP 2.4 trio: host Cray libfabric -> libcurl -> {libldap_r-2.4,
# liblber-2.4, libsasl2.so.3}. The container ships OpenLDAP 2.5 (different
# SONAME), so the host libcurl cannot resolve in-image -> "libldap_r-2.4.so.2:
# cannot open" (job 603820) -> RCCL silently falls back to Socket. Copy the
# host trio into $RUNDIR (first on LD_LIBRARY_PATH). An in-container ldd in
# run.sh names any LATER gap so we stop guessing.
cp -L /usr/lib64/libldap_r-2.4.so.2.10.9 "$RUNDIR/" 2>/dev/null && ln -sf libldap_r-2.4.so.2.10.9 "$RUNDIR/libldap_r-2.4.so.2" && echo "[build] copied libldap_r-2.4.so.2.10.9 (+soname)"
cp -L /usr/lib64/liblber-2.4.so.2.10.9 "$RUNDIR/" 2>/dev/null && ln -sf liblber-2.4.so.2.10.9 "$RUNDIR/liblber-2.4.so.2" && echo "[build] copied liblber-2.4.so.2.10.9 (+soname)"
cp -L /usr/lib64/libsasl2.so.3.0.0 "$RUNDIR/" 2>/dev/null && ln -sf libsasl2.so.3.0.0 "$RUNDIR/libsasl2.so.3" && echo "[build] copied libsasl2.so.3.0.0 (+soname)"
BUILD
chmod +x "$RUNDIR/build.sh"

echo "[$(date -Is)] building plugin on head host ($HEAD) ..."
if ! bash "$RUNDIR/build.sh"; then
  echo "[$(date -Is)] BUILD_FAILED — inspect the log for the compiler/link error." >&2
  echo "[$(date -Is)]   (host /usr/bin/gcc + /opt/cray/libfabric headers; if gcc is" >&2
  echo "[$(date -Is)]    missing on the compute node, build in-container instead.)" >&2
  exit 1
fi
echo "[$(date -Is)] BUILD_DONE"

# ------------------------------------------------------------- run (2 ranks) --
# 2-rank torch.distributed RCCL allreduce across both nodes with the plugin
# active. Crucially we do NOT set the Socket fallback (NCCL_IB_DISABLE /
# NCCL_SOCKET_IFNAME / unset NCCL_NET) the way engine.sh does — the whole
# point is to let RCCL load librccl-net-ofi over CXI. LD_LIBRARY_PATH prepends
# the Cray libfabric (1.29.1, cxi) so it shadows the image's 1.14.0 (no cxi);
# the image's ROCm libs stay discoverable via ldconfig.
cat > "$RUNDIR/run.sh" <<RUN
#!/bin/bash
set -uo pipefail
export LD_LIBRARY_PATH="$RUNDIR:$CRAY_OFI/lib64:\${LD_LIBRARY_PATH:-}"
# NCCL_NET/RCCL_NET select a network *type* (IB/Socket) in this RCCL 2.27.7
# build; they do NOT load the external plugin (proven: RCCL still tried
# IB+Socket with NCCL_NET=<path> set). The plugin is loaded below as
# libnccl-net.so via LD_LIBRARY_PATH (+ NCCL_NET_PLUGIN for compat builds),
# so we LEAVE NCCL_NET/RCCL_NET UNSET to avoid any type restriction and let
# RCCL pick whatever loads (the OFI plugin, else fall back to IB/Socket).
unset NCCL_NET RCCL_NET 2>/dev/null || true
export FI_PROVIDER="cxi"
export MASTER_ADDR="$HEAD"
export MASTER_PORT="$MASTER_PORT"
export RANK="\${SLURM_NODEID:-0}"
export WORLD_SIZE="\${SLURM_NTASKS:-2}"
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
# NCCL_DEBUG comes from the EDF (INFO) by default; override to TRACE at
# submit via SMOKE_NCCL_DEBUG (baked BELOW at heredoc-write time — the EDF
# replaces the inherited env at srun --environment, so SMOKE_* must be baked
# here, not read inside the container). SMOKE_PG_TIMEOUT makes init / the
# first cross-node collective fail fast with a torch.distributed TimeoutError
# (and a stack trace) instead of hanging to the SLURM wall. FI_LOG_LEVEL +
# FI_LOG_PROV drive the Cray libfabric (cxi) provider logs.
#   rcc -p beverin run -s 'cd <SCRIPT_DIR> &&'
#   sbatch --export=ALL,SMOKE_NCCL_DEBUG=TRACE,SMOKE_FI_LOG_LEVEL=DEBUG,SMOKE_PG_TIMEOUT=90
#   ofi_rccl_smoke.sbatch
# Do NOT set NCCL_IB_DISABLE/NCCL_SOCKET_IFNAME (would force the Socket
# fallback, defeating the purpose). NCCL_DEBUG_SUBSYS narrows the RCCL
# trace; unset = all subsystems at the NCCL_DEBUG level.
export NCCL_DEBUG="${SMOKE_NCCL_DEBUG:-${NCCL_DEBUG:-INFO}}"
export SMOKE_PG_TIMEOUT="${SMOKE_PG_TIMEOUT:-1800}"
export FI_LOG_LEVEL="${SMOKE_FI_LOG_LEVEL:-${FI_LOG_LEVEL:-warn}}"
export FI_LOG_PROV="${SMOKE_FI_LOG_PROV:-cxi}"
[ -n "${SMOKE_NCCL_DEBUG_SUBSYS:-}" ] && export NCCL_DEBUG_SUBSYS="${SMOKE_NCCL_DEBUG_SUBSYS:-}"
echo "[r\$RANK] NCCL_NET=\${NCCL_NET:-<unset>} FI_PROVIDER=\$FI_PROVIDER MASTER=\$MASTER_ADDR:\$MASTER_PORT CUDA=\$CUDA_VISIBLE_DEVICES debug: NCCL_DEBUG=\$NCCL_DEBUG SUBSYS=\${NCCL_DEBUG_SUBSYS:-<all>} FI_LOG=\$FI_LOG_LEVEL/\$FI_LOG_PROV PG_TIMEOUT=\$SMOKE_PG_TIMEOUT s"
# RCCL 2.27.7 dlopens the external net plugin by the HARDCODED name
# libnccl-net.so (resolved via LD_LIBRARY_PATH) and IGNORES NCCL_NET=<path>
# for plugin loading (NCCL_NET selects a network *type* here — the log still
# tried IB+Socket with NCCL_NET=<path> set). So expose the vkernels plugin AS
# libnccl-net.so (symlink in $RUNDIR, on LD_LIBRARY_PATH) AND offer it via
# NCCL_NET_PLUGIN=<path> for builds that honour that env. (Both $RUNDIR
# below are baked at heredoc-write time, like the NCCL_NET line above.)
ln -sf "$RUNDIR/librccl-net-ofi.so" "$RUNDIR/libnccl-net.so"
export NCCL_NET_PLUGIN="$RUNDIR/librccl-net-ofi.so"
echo "[r\$RANK] plugin: NCCL_NET_PLUGIN=\$NCCL_NET_PLUGIN LD_LIBRARY_PATH=\$LD_LIBRARY_PATH"
# Authoritative in-container dependency check: ldd the plugin with THIS
# run.sh's LD_LIBRARY_PATH so any still-missing host DSO is named in the
# log (instead of RCCL's one-line "could not open" + silent Socket fallback).
echo "[r\$RANK] in-container ldd (remaining not-found deps):"
ldd "\$NCCL_NET_PLUGIN" 2>&1 | grep -i "not found" || echo "[r\$RANK] ldd: all plugin deps resolved"
python3 "$SCRIPT_DIR/ofi_rccl_smoke.py"
rc=\$?
echo "[r\$RANK] ofi_rccl_smoke rc=\$rc"
exit \$rc
RUN
chmod +x "$RUNDIR/run.sh"

echo "[$(date -Is)] running 2-rank RCCL allreduce with librccl-net-ofi over CXI ..."
# --gres=gpu:1 per task so each rank gets one GPU (we override
# CUDA_VISIBLE_DEVICES to SLURM_LOCALID in run.sh regardless).
srun --jobid="$SLURM_JOB_ID" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --gres=gpu:1 --environment="$OTELA_EDF_NAME" bash "$RUNDIR/run.sh"
SMOKE_RC=$?

# --------------------------------------------------------------- verdict --
LOG="$DEPLOY_DIR/logs/ofi-rccl-smoke-${SLURM_JOB_ID}.out"
echo "[$(date -Is)] ---- transport / plugin evidence (NCCL_DEBUG=INFO) ----"
grep -iE "ofi|cxi|net plugin|librccl|transport|Socket|version|getDeviceMR|regHandle|NET/|plugin" "$LOG" 2>/dev/null | head -50 || true
echo "[$(date -Is)] ---- OFI_SMOKE result ----"
grep -E "OFI_SMOKE_(PASS|FAIL)" "$LOG" 2>/dev/null || true
# Transport assertion: a correct allreduce over the built-in Socket fallback
# is a SILENT FAIL - Socket can move a 1-elem tensor over TCP, so the sum
# being right does NOT prove the OFI plugin carried the traffic. If every
# cross-node channel went "via NET/Socket" and none via a non-Socket net, the
# plugin did not engage even if it dlopened (job 603818 was exactly this:
# libcxi.so.1 missing -> Socket fallback -> OFI_SMOKE_PASS over TCP).
SOCK=$(grep -aoE "via NET/Socket" "$LOG" 2>/dev/null | sort -u | wc -l)
NONSOCK=$(grep -aoE "via NET/[A-Za-z0-9_.-]+" "$LOG" 2>/dev/null | grep -avE "via NET/Socket" | sort -u | wc -l)
echo "[$(date -Is)] transport assertion: socket_channels=$SOCK non_socket_channels=$NONSOCK"
if [ "$SOCK" -gt 0 ] && [ "$NONSOCK" -eq 0 ]; then
  echo "[$(date -Is)] OFI_SMOKE_FAIL transport=Socket-only - plugin did not carry cross-node traffic (see $LOG)." >&2
  SMOKE_RC=1
fi

if [ "$SMOKE_RC" = 0 ] && grep -q "OFI_SMOKE_PASS" "$LOG" 2>/dev/null; then
  echo "[$(date -Is)] OFI_INTEGRATION_PASS — plugin loaded, RCCL accepted the ABI, allreduce completed."
  echo "[$(date -Is)] NEXT: wire the gated K3_OFI_NET knob into serve_kimi_k3_otela_beverin.sbatch"
  echo "[$(date -Is)]       + add the /opt/cray/libfabric mount to kimi-k3-vllm.toml, then"
  echo "[$(date -Is)]       sbatch --export=ALL,K3_OFI_NET=1 serve_kimi_k3_otela_beverin.sbatch"
  exit 0
else
  echo "[$(date -Is)] OFI_INTEGRATION_FAIL smoke_rc=$SMOKE_RC — plugin did not load or allreduce wrong." >&2
  echo "[$(date -Is)]   If RCCL rejected the ABI (needs getDeviceMR/regHandle for a newer vN than" >&2
  echo "[$(date -Is)]   the plugin's v6/v7 11-field v-table), extend plugins/rccl-net-ofi/rccl_net_ofi.c" >&2
  echo "[$(date -Is)]   before any K3 integration. Inspect $LOG for the exact rcclNet error." >&2
  exit 1
fi
