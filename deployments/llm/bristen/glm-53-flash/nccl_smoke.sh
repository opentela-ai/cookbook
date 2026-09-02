#!/bin/bash
# In-container wrapper for the 2-node NCCL allreduce smoke. Mirrors the
# Slingshot/CXI NCCL env that engine.sh will use for the real PP2 run, then
# execs the Python probe. The plugin .so is bind-mounted via CONTAINER_OPTS;
# NCCL_NET_PLUGIN is a full path so NCCL dlopens it directly (no rename needed).
set -uo pipefail
RANK="${SLURM_NODEID:-0}"
WORLD="${SLURM_JOB_NUM_NODES:-1}"

# --- Slingshot NCCL (mirrors clariden glm-53-flash engine.sh, verified libs) -
# Plugin .so was copied to the SHARED /capstor (every node mounts /capstor),
# so it does not depend on the per-node-local /opt/cscs/aws-ofi-ccl-plugin path
# (which is unpopulated on bad-boot nodes like nid002293). NCCL dlopens the .so
# directly; it resolves libfabric.so.1 (system) + libcxi.so.1 (auto-mounted
# /opt/cscs/netstack, present on every healthy node) at dlopen time. ldd clean.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-/capstor/scratch/cscs/xyao/glm-53-flash-bristen/cache/nccl-plugin/libnccl-net.so}"
export NCCL_NET="${NCCL_NET:-AWS Libfabric}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
export FI_CXI_RDZV_THRESHOLD="${FI_CXI_RDZV_THRESHOLD:-0}"
export FI_CXI_RDZV_GET_MIN="${FI_CXI_RDZV_GET_MIN:-0}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
# hsn0 carries the routable 172.28/16 IPv4 used for the TCPStore bootstrap.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-hsn0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-IPv4}"
export GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-IPv4}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

echo "[rank $RANK/$WORLD] $(hostname) plugin=$NCCL_NET_PLUGIN net=$NCCL_NET iface=$NCCL_SOCKET_IFNAME head=$HEAD_IP:$MASTER_PORT"
exec python3 "$(dirname "$0")/nccl_smoke.py"
