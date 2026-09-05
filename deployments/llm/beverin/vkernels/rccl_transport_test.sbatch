#!/bin/bash
#SBATCH --job-name=rccl-xport
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:03:00
#SBATCH --signal=B:TERM@20
#SBATCH --output=/capstor/scratch/cscs/xyao/vkernels/logs/%x-%j.out
#
# 2-node, 1-GPU-per-node cross-node RCCL transport A/B (Socket vs OFI/CXI).
# Forces the all_to_all over the network (no intra-node shm shortcut) and
# captures the transport RCCL chose via NCCL_DEBUG=INFO. The gate for running
# the full K3 serving benchmark on the OFI transport instead of the slow
# Socket fallback.
#
# Submit:  rcc --profile beverin run -- sbatch rccl_transport_test.sbatch
# Then run the OFI variant: sbatch --export=ALL,RCCL_TRANSPORT=ofi ...
set -uo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-vllm}"
RCCL_TRANSPORT="${RCCL_TRANSPORT:-socket}"
export RCCL_TRANSPORT
mkdir -p /capstor/scratch/cscs/xyao/vkernels/logs

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${NODES[0]}"
HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TESTPY="$SCRIPT_DIR/rccl_transport_test.py"
[ -f "$TESTPY" ] || { echo "FATAL: $TESTPY not found" >&2; exit 1; }
export HEAD HEAD_IP
export MASTER_ADDR="$HEAD_IP"
export MASTER_PORT="${MASTER_PORT:-6381}"

echo "[$(date -Is)] rccl-xport job=$SLURM_JOB_ID nodes=${NODES[*]} head=$HEAD($HEAD_IP) transport=$RCCL_TRANSPORT"

# Transport selection (mirrors serve_kimi_k3... RCCL block, flipped by
# RCCL_TRANSPORT). Both keep NCCL_SOCKET_IFNAME=hsn0 for the RCCL bootstrap
# (gloo is not used here; this is pure torch.distributed/rccl).
unset NCCL_NET NCCL_IB_DISABLE NCCL_NET_PLUGIN
export NCCL_SOCKET_IFNAME="${K3_NIC:-hsn0}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
# libfabric + CXI live in the CSCS netstack; set it inside the srun via
# `env` (belt-and-suspenders: --environment may or may not carry host vars).
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"   # WARN for speed; INFO to see transport negotiation

case "$RCCL_TRANSPORT" in
  ofi)
    # RCCL's own (non-CUDA) OFI net plugin -> libfabric/CXI over Slingshot.
    # Keep IB ENABLED (OFI/CXI is not IB verbs; the unset above lets the
    # plugin pick CXI). If the plugin fails to init, NCCL falls back to Socket
    # only if NCCL_NET is unset; to force visibility of init failure we rely
    # on NCCL_DEBUG=INFO + the test's explicit PASS check.
    export NCCL_NET_PLUGIN="librccl-net-ofi.so"
    echo "[rccl-xport] OFI/CXI transport: NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN"
    # K3_GDR=0 disables GPU Direct RDMA (hypothesis: GDR over MI300A CXI
    # hangs the first cross-node collective; the EDF sets GDR on by default).
    if [ "${K3_GDR:-1}" = "0" ]; then
      export NCCL_NET_GDR_READ=0 NCCL_NET_GDR_C2C=0 NCCL_NET_GDR_LEVEL=LOC
      echo "[rccl-xport] GDR DISABLED (K3_GDR=0): READ=0 C2C=0 LEVEL=LOC"
    else
      echo "[rccl-xport] GDR ENABLED (K3_GDR=1, EDF default)"
    fi
    ;;
  socket|*)
    # The verified working (slow) path: Socket over hsn0, IB disabled.
    export NCCL_IB_DISABLE=1
    echo "[rccl-xport] Socket transport: NCCL_IB_DISABLE=1 (baseline)"
    ;;
esac

srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --environment="$OTELA_EDF_NAME" \
  -t 180 env LD_LIBRARY_PATH="/opt/cscs/netstack:${LD_LIBRARY_PATH:-}" \
  python3 -u "$TESTPY"
RC=$?
echo "[$(date -Is)] srun exited code=$RC"
exit "$RC"
