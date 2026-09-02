#!/bin/bash
#SBATCH --job-name=nccl-smoke
#SBATCH --account=infra02
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --exclude=nid002293,nid002297
#SBATCH --time=0:08:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-53-flash-bristen/logs/%x-%j.out
#
# 2-node NCCL allreduce smoke over Slingshot/CXI on Bristen (x86_64 A100).
# Validates the aws_ofi_nccl CUDA13 plugin + FI_CXI_* env that the real
# PP2 serve will use, WITHOUT the ~22 min weight load.
#
#   sbatch nccl_smoke.sbatch
#   NCCL_DEBUG=INFO sbatch nccl_smoke.sbatch   # verbose transport (default)
# =============================================================================
set -uo pipefail

export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash-bristen}"
export IMAGE="${IMAGE:-$DEPLOY_DIR/cache/enroot/sglang-dev-cu13.sqsh}"
export GLM53_NIC="${GLM53_NIC:-hsn0}"
export MASTER_PORT="${MASTER_PORT:-30011}"

mkdir -p "$DEPLOY_DIR/logs"
cp "$DEPLOY_DIR/nccl_smoke.sh" "$DEPLOY_DIR/nccl_smoke.py" "$DEPLOY_DIR/" 2>/dev/null || true

# Rank 0's hsn0 IPv4 (routable on 172.28/16) is the TCPStore bootstrap addr.
HEAD_IP="$(ip -o -4 addr show "$GLM53_NIC" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
[ -z "$HEAD_IP" ] && HEAD_IP="$(hostname -I | awk '{print $1}')"
export HEAD_IP

echo "[$(date -Is)] nccl-smoke head=$(hostname) ip=$HEAD_IP iface=$GLM53_NIC nodes=$SLURM_JOB_NUM_NODES"

CONTAINER_OPTS=(
  --container-image="$IMAGE"
  --container-name=nccl-smoke
  # /capstor is shared across all nodes and already carries the NCCL plugin
  # copy (cache/nccl-plugin/libnccl-net.so). No per-node-local mount needed,
  # so the container launches on any healthy node (bad-boot nodes like
  # nid002293 are excluded above).
  --container-mounts=/capstor:/capstor
  --container-env=HEAD_IP,MASTER_PORT,GLM53_NIC,NCCL_NET_PLUGIN,NCCL_NET,NCCL_CROSS_NIC,FI_CXI_DISABLE_HOST_REGISTER,FI_CXI_DEFAULT_CQ_SIZE,FI_CXI_RDZV_THRESHOLD,FI_CXI_RDZV_GET_MIN,FI_MR_CACHE_MONITOR,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,NCCL_SOCKET_FAMILY,GLOO_SOCKET_FAMILY,NCCL_DEBUG
)

srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
     --cpus-per-task=4 --gpus-per-task=1 \
     --export=ALL "${CONTAINER_OPTS[@]}" \
  bash "$DEPLOY_DIR/nccl_smoke.sh"
RC=$?
echo "[$(date -Is)] srun rc=$RC (0=PASS-EXIT, result lines above)"
exit "$RC"
