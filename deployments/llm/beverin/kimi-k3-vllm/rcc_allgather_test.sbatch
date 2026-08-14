#!/bin/bash
#SBATCH --job-name=rcc-ag-test
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/rcc-ag-test-%j.out

# Minimal cross-node RCCL all-gather repro. Mirrors the TP8 group of the K3
# recipe (8 ranks over 2 nodes) and the exact env (aws_ofi_nccl neutralised,
# RCCL Socket over hsn0). The python script sets the NCCL_* vars itself
# (before importing torch) so the EDF [env] cannot override them.

set -euo pipefail

SCRIPT_DIR="/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm"
export EDF_PATH="$SCRIPT_DIR"
OTELA_EDF_NAME="kimi-k3-vllm"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${NODES[0]}"
HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"
export MASTER_ADDR="$HEAD_IP"
export MASTER_PORT="29500"
echo "[$(date -Is)] head=$HEAD ($HEAD_IP) nodes=$SLURM_JOB_NODELIST world=$SLURM_NTASKS"

srun --environment="$OTELA_EDF_NAME" python3 "$SCRIPT_DIR/rcc_allgather_test.py"
