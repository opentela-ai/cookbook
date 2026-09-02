#!/bin/bash
# Probe every idle GPU node on the normal partition for boot readiness and
# print the bad ones as a comma-separated --exclude list (suitable for:
#   sbatch --exclude="$(find_bad_nodes.sh)" serve_glm_53_flash_sglang.sbatch
# ). A node is "bad" when scontrol shows IDLE/gpu:4 but a 1-task launch fails
# with "Something is wrong with the boot of the nodes" -- Slurm keeps offering
# such nodes, so multi-node jobs pinned/allocated to them fail to launch.
# Use right before submit; the set changes as nodes recover or reboot.
set -uo pipefail
ACCT="${SLURM_ACCOUNT:-infra02}"
mapfile -t NODES < <(
  sinfo -p normal -o '%N %t' -h 2>/dev/null |
    awk '$2=="idle"{print $1}' |
    xargs -r -n1 scontrol show hostnames 2>/dev/null | sort -u
)
bad=()
for n in "${NODES[@]}"; do
  out=$(timeout 22 srun -A "$ACCT" --partition=normal -w "$n" \
        --gres=none --nodes=1 --ntasks=1 -t0:00:15 --overlap \
        --job-name=bootchk bash -c 'echo OK' 2>&1 | tail -1)
  case "$out" in
    OK) ;;  # good
    *) printf '%s\n' "$n" >&2; bad+=("$n") ;;  # bad (prints node to stderr)
  esac
done
printf '%s\n' "${bad[*]}" 2>/dev/null | tr ' ' ','
[ "${#bad[@]}" -gt 0 ] && printf 'found %d bad-boot nodes\n' "${#bad[@]}" >&2 ||
  printf 'all idle nodes boot OK\n' >&2
