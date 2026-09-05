#!/bin/bash
# Submit a pinned 2-node NCCL smoke to two known-idle nodes and watch it.
set -uo pipefail
D=/capstor/scratch/cscs/xyao/glm-53-flash-bristen
cd "$D" || { echo "FATAL: cannot cd $D"; exit 1; }
# cancel any lingering nccl-smoke of mine
for j in $(squeue -u "$USER" -n nccl-smoke -h -o %i 2>/dev/null); do scancel "$j"; done
# pick two currently-idle non-drain gpu nodes
GRP=$(sinfo -p normal -o '%N %t' -h 2>/dev/null | awk '$2=="idle"{print $1}' | head -1)
N1=$(scontrol show hostnames "$GRP" 2>/dev/null | sed -n 1p)
N2=$(scontrol show hostnames "$GRP" 2>/dev/null | sed -n 2p)
echo "idle-group=$GRP pinned=$N1,$N2"
[ -z "$N1" ] || [ -z "$N2" ] && { echo "FATAL: no 2 idle nodes"; exit 2; }
J=$(sbatch --parsable -w "$N1,$N2" nccl_smoke.sbatch 2>&1)
echo "submitted job=$J $(date +%H:%M:%S)"
for w in $(seq 1 16); do
  sleep 7
  st=$(squeue -j "$J" -h -o '%T' 2>/dev/null)
  nodes=$(squeue -j "$J" -h -o '%N' 2>/dev/null)
  echo "  $(date +%H:%M:%S): ${st:-DONE} nodes=$nodes"
  case "$st" in
    RUNNING*) echo "  >>> RUNNING, waiting 40s for result"; sleep 40
              echo "  --- log ---"; tail -24 "logs/nccl-smoke-$J.out" 2>/dev/null
              break;;
    COMPLETED|CANCELLED|FAILED)
              echo "  --- log ---"; tail -24 "logs/nccl-smoke-$J.out" 2>/dev/null
              echo "  --- job reason ---"; scontrol show job "$J" 2>/dev/null | grep -iE 'JobState|Reason|StartTime|EligibleTime|Restarts|NodeList'
              break;;
  esac
done
echo "=== readable slurmd log? ==="
tail -30 /var/log/slurmd.log 2>/dev/null | grep -iE 'error|launch|node|gpu|enroot|pyxis' | tail -15 || echo 'slurmd.log not readable'
