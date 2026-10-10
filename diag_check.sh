#!/bin/bash
echo "=== my jobs ==="
squeue -u $USER -o "%.10i %.20j %.8T %.10M" 2>/dev/null
J=625610
echo "=== job $J recent log (load progress) ==="
# the engine log for this job lives in the recipe dir; find it
ls -t /capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/*.out 2>/dev/null | head -3
echo "=== grep load/health in latest slurm out ==="
# find the running sbatch out file
for F in $(ls -t /capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/*.out 2>/dev/null | head -3); do
  echo "--- $F (last 3 lines):"
  tail -3 "$F" 2>/dev/null | cut -c1-150
done
