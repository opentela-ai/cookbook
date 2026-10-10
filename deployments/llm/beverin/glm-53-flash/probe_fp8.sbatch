#!/bin/bash
#SBATCH --job-name=gdn-probe
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/gdn-probe-%j.out
export PYTHONPATH=/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay/pylib:/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay/sgl-workspace/sglang/python:${PYTHONPATH}
export PYTHONUNBUFFERED=1
srun --environment=sglang-rocm python3 /capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/glm-53-flash/probe_fp8.py
